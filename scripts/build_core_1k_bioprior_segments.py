#!/usr/bin/env python3
"""Build Core-1K BioPrior-guided candidate deletion segments."""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bioprior.prior_types import MOTIF_SHADOW_RISK, GEOMETRIC_CLOSURE  # noqa: E402
from bioprior.scoring import compute_bioprior_score  # noqa: E402
from bioprior.segment_proposal import propose_bioprior_segments  # noqa: E402
from bioprior.utils import feature_bool, feature_float, load_bioprior_config, make_segment, mean  # noqa: E402


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def read_core_csv(path):
    rows = {}
    order = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            rows[accession] = row
            order.append(accession)
    return rows, order


def read_residue_csv(path, limit_accessions=None):
    grouped = defaultdict(list)
    limit_set = set(limit_accessions or [])
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row["accession"]
            if limit_set and accession not in limit_set:
                continue
            grouped[accession].append(row)
    for accession in grouped:
        grouped[accession].sort(key=lambda row: int(row["residue_index_0based"]))
    return dict(grouped)


def parse_pdb_ca_coords(path):
    coords = []
    seen = set()
    with open(path, "rb") as handle:
        for raw_line in handle:
            if not raw_line.startswith(b"ATOM"):
                continue
            line = raw_line.decode("ascii", "ignore")
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            chain = line[21:22]
            resseq = line[22:26].strip()
            icode = line[26:27]
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def canonical_residue_features(rows):
    features = []
    for row in rows:
        features.append(
            {
                "position": int(row["residue_index_0based"]),
                "aa": row["aa"],
                "is_protected": row["is_protected"],
                "is_motif": row["is_protected"],
                "is_motif_shadow_8A": row["is_motif_shadow_8A"],
                "is_shadow": row["is_motif_shadow_8A"],
                "distance_to_motif": row["distance_to_nearest_protected"],
                "distance_to_nearest_protected": row["distance_to_nearest_protected"],
                "contact_density_8A": row["contact_density_8A"],
                "contact_density": row["contact_density_8A"],
                "motif_contact_count_8A": row["motif_contact_count_8A"],
                "motif_contact_count": row["motif_contact_count_8A"],
                "plddt": row["pLDDT"],
                "pLDDT": row["pLDDT"],
                "is_terminal_region": row["is_terminal_region"],
                "is_disulfide": row.get("is_disulfide_residue", "False"),
                "is_metal_binding": row.get("is_metal_binding", "False"),
                "is_ptm": row.get("is_modified_residue", "False"),
            }
        )
    return features


def runs_from_mask(mask, min_len, max_len):
    runs = []
    start = None
    for idx, value in enumerate(mask + [False]):
        if value and start is None:
            start = idx
        if not value and start is not None:
            end = idx - 1
            if end - start + 1 >= min_len:
                while end - start + 1 > max_len:
                    runs.append((start, start + max_len - 1))
                    start += max_len
                if end - start + 1 >= min_len:
                    runs.append((start, end))
            start = None
    return runs


def add_shadow_heavy_negatives(features, config):
    min_len = int(config.get("min_segment_len", 3))
    max_len = int(config.get("max_segment_len", 30))
    mask = [
        feature_bool(row, "is_motif_shadow_8A", "is_shadow")
        and not feature_bool(row, "is_motif", "is_protected")
        for row in features
    ]
    segments = []
    for start, end in runs_from_mask(mask, min_len, max_len):
        segments.append(
            make_segment(
                start,
                end,
                MOTIF_SHADOW_RISK,
                "motif-shadow-heavy support-neighborhood hard negative",
            )
        )
    return segments


def add_closure_unfriendly_negatives(features, coords, config):
    if coords is None:
        return []
    min_len = int(config.get("min_segment_len", 3))
    max_len = min(int(config.get("max_segment_len", 30)), 15)
    closure_cutoff = float(config.get("closure_cutoff", 8.0))
    stride = max(3, min_len)
    segments = []
    n = len(features)
    for start in range(min_len, max(0, n - min_len - 1), stride):
        end = min(n - 2, start + min_len - 1)
        if end - start + 1 > max_len:
            continue
        if any(feature_bool(features[pos], "is_motif", "is_protected") for pos in range(start, end + 1)):
            continue
        left = start - 1
        right = end + 1
        dx = coords[left][0] - coords[right][0]
        dy = coords[left][1] - coords[right][1]
        dz = coords[left][2] - coords[right][2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist > closure_cutoff:
            segments.append(
                make_segment(
                    start,
                    end,
                    GEOMETRIC_CLOSURE,
                    "non-terminal closure-unfriendly hard negative",
                )
            )
        if len(segments) >= 10:
            break
    return segments


def segment_rows(segment, features):
    return features[segment["seg_start"] : segment["seg_end"] + 1]


def aggregate_segment(accession, protein_length, segment, features, coords, config):
    rows = segment_rows(segment, features)
    score = compute_bioprior_score(segment, features, coords, config)
    components = score["components"]
    closure = score["closure"]

    protected_flags = [feature_bool(row, "is_protected", "is_motif") for row in rows]
    shadow_flags = [feature_bool(row, "is_motif_shadow_8A", "is_shadow") for row in rows]
    distances = [feature_float(row, "distance_to_nearest_protected", "distance_to_motif", default=999.0) for row in rows]
    contacts = [feature_float(row, "contact_density_8A", "contact_density") for row in rows]
    motif_contacts = [feature_float(row, "motif_contact_count_8A", "motif_contact_count") for row in rows]
    plddts = [feature_float(row, "pLDDT", "plddt") for row in rows]
    terminal_flags = [feature_bool(row, "is_terminal_region") for row in rows]
    low_plddt_threshold = float(config.get("low_plddt_threshold", 70))
    high_plddt_threshold = float(config.get("high_plddt_threshold", 90))
    seg_len = segment["seg_len"]

    n_protected = sum(1 for value in protected_flags if value)
    n_shadow = sum(1 for value in shadow_flags if value)
    if closure["is_terminal_deletion"]:
        closure_type = "terminal"
    elif closure["closure_friendly"]:
        closure_type = "internal_friendly"
    else:
        closure_type = "internal_unfriendly"

    return {
        "accession": accession,
        "protein_length": protein_length,
        "seg_start": segment["seg_start"],
        "seg_end": segment["seg_end"],
        "seg_len": seg_len,
        "proposal_source": segment["proposal_source"],
        "biological_rationale": segment["biological_rationale"],
        "n_protected_overlap": n_protected,
        "protected_overlap_fraction": "{:.6f}".format(n_protected / float(seg_len)),
        "n_shadow_overlap": n_shadow,
        "shadow_overlap_fraction": "{:.6f}".format(n_shadow / float(seg_len)),
        "min_distance_to_protected": "{:.4f}".format(min(distances) if distances else 0.0),
        "mean_distance_to_protected": "{:.4f}".format(mean(distances)),
        "mean_contact_density_8A": "{:.4f}".format(mean(contacts)),
        "max_contact_density_8A": "{:.4f}".format(max(contacts) if contacts else 0.0),
        "mean_motif_contact_count_8A": "{:.4f}".format(mean(motif_contacts)),
        "max_motif_contact_count_8A": "{:.4f}".format(max(motif_contacts) if motif_contacts else 0.0),
        "mean_pLDDT": "{:.4f}".format(mean(plddts)),
        "min_pLDDT": "{:.4f}".format(min(plddts) if plddts else 0.0),
        "low_pLDDT_fraction": "{:.6f}".format(sum(1 for value in plddts if value < low_plddt_threshold) / float(seg_len)),
        "high_pLDDT_fraction": "{:.6f}".format(sum(1 for value in plddts if value >= high_plddt_threshold) / float(seg_len)),
        "terminal_overlap_fraction": "{:.6f}".format(sum(1 for value in terminal_flags if value) / float(seg_len)),
        "boundary_ca_distance": "" if closure["boundary_ca_distance"] is None else "{:.4f}".format(closure["boundary_ca_distance"]),
        "closure_type": closure_type,
        "closure_friendly_8A": str(closure["closure_friendly"]),
        "terminal_tail_score": "{:.6f}".format(components["terminal_tail"]),
        "surface_flexible_loop_score": "{:.6f}".format(components["surface_flexible_loop"]),
        "disorder_like_score": "{:.6f}".format(components["disorder_like"]),
        "linker_compressibility_score": "{:.6f}".format(components["linker_compressibility"]),
        "functional_core_risk_score": "{:.6f}".format(components["functional_core_risk"]),
        "structural_core_risk_score": "{:.6f}".format(components["structural_core_risk"]),
        "motif_shadow_risk_score": "{:.6f}".format(components["motif_shadow_risk"]),
        "geometric_closure_score": "{:.6f}".format(components["geometric_closure"]),
        "final_bioprior_score": "{:.6f}".format(score["score"]),
        "hard_reject": str(score["hard_reject"]),
        "reject_reason": ";".join(score["hard_reject_reasons"]),
    }


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(len(values) - 1, lo + 1)
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def write_summary(path, total_proteins, processed_proteins, rows):
    source_counts = Counter(row["proposal_source"] for row in rows)
    rationale_counts = Counter(row["biological_rationale"] for row in rows)
    hard_reject_count = sum(1 for row in rows if row["hard_reject"] == "True")
    closure_friendly_count = sum(1 for row in rows if row["closure_friendly_8A"] == "True")
    scores = [float(row["final_bioprior_score"]) for row in rows]
    shadow_fracs = [float(row["shadow_overlap_fraction"]) for row in rows]
    protected_overlap = [int(row["n_protected_overlap"]) for row in rows]

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as handle:
        handle.write("Core-1K BioPrior segment candidate summary\n\n")
        handle.write("total_proteins: {}\n".format(total_proteins))
        handle.write("processed_proteins_with_segments: {}\n".format(processed_proteins))
        handle.write("total_segments: {}\n".format(len(rows)))
        handle.write(
            "avg_segments_per_processed_protein: {:.6f}\n".format(
                len(rows) / float(processed_proteins or 1)
            )
        )
        handle.write("\nProposal source counts:\n")
        for key, value in source_counts.most_common():
            handle.write("- {}: {}\n".format(key, value))
        handle.write("\nBiological rationale counts:\n")
        for key, value in rationale_counts.most_common():
            handle.write("- {}: {}\n".format(key, value))
        handle.write("\nCategory counts:\n")
        for key in [
            "terminal_tail_prior",
            "surface_flexible_loop_prior",
            "disorder_like_prior",
            "linker_compressibility_prior",
            "functional_core_protection_prior",
            "structural_core_protection_prior",
            "motif_neighborhood_support_prior",
            "geometric_closure_prior",
        ]:
            handle.write("- {}: {}\n".format(key, source_counts[key]))
        handle.write("\n")
        handle.write("hard_reject_count: {}\n".format(hard_reject_count))
        handle.write("hard_reject_fraction: {:.6f}\n".format(hard_reject_count / float(len(rows) or 1)))
        handle.write("closure_friendly_count: {}\n".format(closure_friendly_count))
        handle.write("closure_friendly_fraction: {:.6f}\n".format(closure_friendly_count / float(len(rows) or 1)))
        handle.write(
            "protected_overlap_segments: {}\n".format(
                sum(1 for value in protected_overlap if value > 0)
            )
        )
        handle.write("protected overlap check: favorable proposals should mostly avoid protected overlap; hard negatives are expected to include it.\n")
        handle.write("\nShadow overlap fraction distribution:\n")
        handle.write("- min: {:.6f}\n".format(min(shadow_fracs) if shadow_fracs else 0.0))
        handle.write("- p25: {:.6f}\n".format(percentile(shadow_fracs, 0.25)))
        handle.write("- median: {:.6f}\n".format(percentile(shadow_fracs, 0.50)))
        handle.write("- p75: {:.6f}\n".format(percentile(shadow_fracs, 0.75)))
        handle.write("- max: {:.6f}\n".format(max(shadow_fracs) if shadow_fracs else 0.0))
        handle.write("\nFinal BioPrior score distribution:\n")
        handle.write("- min: {:.6f}\n".format(min(scores) if scores else 0.0))
        handle.write("- p25: {:.6f}\n".format(percentile(scores, 0.25)))
        handle.write("- median: {:.6f}\n".format(percentile(scores, 0.50)))
        handle.write("- p75: {:.6f}\n".format(percentile(scores, 0.75)))
        handle.write("- max: {:.6f}\n".format(max(scores) if scores else 0.0))
        handle.write("- mean: {:.6f}\n".format(mean(scores)))
        handle.write("\nCORE_1K_BIOPRIOR_SEGMENTS_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Core-1K BioPrior segment proposals.")
    parser.add_argument("--residue_csv", required=True)
    parser.add_argument("--core_csv", required=True)
    parser.add_argument("--structure_dir", required=True)
    parser.add_argument("--config", default="configs/bioprior_v1.yaml")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--mode", default="proposal", choices=["proposal"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_bioprior_config(args.config)
    core_rows, core_order = read_core_csv(args.core_csv)
    selected_accessions = core_order[: args.limit] if args.limit else core_order
    residue_groups = read_residue_csv(args.residue_csv, selected_accessions)

    out_rows = []
    processed = 0
    skipped = []
    for accession in selected_accessions:
        residue_rows = residue_groups.get(accession)
        if not residue_rows:
            skipped.append((accession, "no_residue_features"))
            continue
        core_row = core_rows[accession]
        protein_length = int(core_row["length"])
        if len(residue_rows) != protein_length:
            skipped.append((accession, "residue_length_mismatch"))
            continue
        pdb_path = os.path.join(args.structure_dir, accession + ".pdb")
        if not os.path.exists(pdb_path):
            skipped.append((accession, "missing_pdb"))
            continue
        coords = parse_pdb_ca_coords(pdb_path)
        if len(coords) != protein_length:
            skipped.append((accession, "coord_length_mismatch"))
            continue

        features = canonical_residue_features(residue_rows)
        segments = propose_bioprior_segments(features, coords, config, include_hard_negatives=True)
        segments.extend(add_shadow_heavy_negatives(features, config))
        segments.extend(add_closure_unfriendly_negatives(features, coords, config))

        unique = {}
        for segment in segments:
            key = (segment["seg_start"], segment["seg_end"], segment["proposal_source"])
            unique[key] = segment

        if unique:
            processed += 1
        for segment in unique.values():
            out_rows.append(aggregate_segment(accession, protein_length, segment, features, coords, config))

    fields = [
        "accession",
        "protein_length",
        "seg_start",
        "seg_end",
        "seg_len",
        "proposal_source",
        "biological_rationale",
        "n_protected_overlap",
        "protected_overlap_fraction",
        "n_shadow_overlap",
        "shadow_overlap_fraction",
        "min_distance_to_protected",
        "mean_distance_to_protected",
        "mean_contact_density_8A",
        "max_contact_density_8A",
        "mean_motif_contact_count_8A",
        "max_motif_contact_count_8A",
        "mean_pLDDT",
        "min_pLDDT",
        "low_pLDDT_fraction",
        "high_pLDDT_fraction",
        "terminal_overlap_fraction",
        "boundary_ca_distance",
        "closure_type",
        "closure_friendly_8A",
        "terminal_tail_score",
        "surface_flexible_loop_score",
        "disorder_like_score",
        "linker_compressibility_score",
        "functional_core_risk_score",
        "structural_core_risk_score",
        "motif_shadow_risk_score",
        "geometric_closure_score",
        "final_bioprior_score",
        "hard_reject",
        "reject_reason",
    ]
    parent = os.path.dirname(args.out_csv)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    write_summary(args.summary_txt, len(selected_accessions), processed, out_rows)
    print("Input proteins: {}".format(len(selected_accessions)))
    print("Processed proteins with segments: {}".format(processed))
    print("Skipped proteins: {}".format(len(skipped)))
    if skipped:
        print("Skipped examples: {}".format("; ".join("{}:{}".format(a, r) for a, r in skipped[:10])))
    print("Segments: {}".format(len(out_rows)))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
