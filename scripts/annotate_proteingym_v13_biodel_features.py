#!/usr/bin/env python3
"""Annotate ProteinGym v1.3 deletion rows with structural BioPrior features."""

import argparse
import csv
import difflib
import gzip
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC = os.path.join(ROOT, "src")
for path in (SCRIPT_DIR, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

from annotate_proteingym_with_swissprot_features import (  # noqa: E402
    FEATURE_PRIORITY,
    direct_child,
    entry_name,
    entry_sequence,
    feature_positions,
    first_accession,
    local_name,
)
from build_core_1k_bioprior_segments import (  # noqa: E402
    aggregate_segment,
    canonical_residue_features,
    parse_pdb_ca_coords,
)
from compute_core_1k_residue_biopriors import compute_residue_rows  # noqa: E402
from bioprior.utils import load_bioprior_config  # noqa: E402


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_temp_pdb_from_zip(zip_path, member, tmp_dir):
    ensure_parent(os.path.join(tmp_dir, "x"))
    out_path = os.path.join(tmp_dir, os.path.basename(member))
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    with open(out_path, "wb") as handle:
        handle.write(data)
    return out_path


def zip_basename_map(path):
    out = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith("/"):
                out[os.path.basename(name)] = name
    return out


def extract_swissprot_features(xml_path, wanted_ids, max_range_len):
    features = {}
    stats = Counter()
    if not xml_path or not os.path.exists(xml_path):
        stats["xml_scan_skipped"] += 1
        return features, stats
    wanted = set(item for item in wanted_ids if item)
    opener = gzip.open if xml_path.endswith(".gz") else open
    with opener(xml_path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            if local_name(elem.tag) != "entry":
                continue
            stats["entries_scanned"] += 1
            accession = first_accession(elem)
            name = entry_name(elem)
            keys = {accession, name}
            if not (keys & wanted):
                elem.clear()
                continue
            sequence = entry_sequence(elem)
            positions = set()
            types = set()
            descriptions = []
            for feature in elem.iter():
                if local_name(feature.tag) != "feature":
                    continue
                used_positions, desc = feature_positions(feature, len(sequence), max_range_len, stats)
                if not used_positions:
                    continue
                positions.update(used_positions)
                types.add(feature.get("type", ""))
                if desc:
                    descriptions.append(desc)
            item = {
                "accession": accession,
                "entry_name": name,
                "sequence": sequence,
                "length": len(sequence),
                "protected_positions": sorted(positions),
                "protected_types": ";".join(value for value in FEATURE_PRIORITY if value in types),
                "protected_feature_descriptions": "|".join(descriptions[:50]),
            }
            for key in keys:
                if key:
                    features[key] = item
            stats["matched_entries"] += 1
            elem.clear()
    return features, stats


def join_positions(values):
    return ";".join(str(value) for value in sorted(values))


def safe_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_positions(value):
    if not value:
        return []
    return [int(item) for item in str(value).split(";") if item != ""]


def load_feature_cache(path):
    cache = {}
    if not path or not os.path.exists(path):
        return cache
    rows, _ = read_csv(path)
    for row in rows:
        key = row.get("target_sha1", "")
        if not key or key in cache:
            continue
        if str(row.get("has_swissprot_features", "")).lower() not in ("true", "1", "yes"):
            continue
        cache[key] = {
            "accession": row.get("swissprot_feature_accessions", ""),
            "entry_name": row.get("swissprot_feature_entry_names", ""),
            "protected_positions": parse_positions(row.get("protected_positions", "")),
            "protected_types": row.get("protected_types", ""),
            "protected_feature_descriptions": row.get("protected_feature_descriptions", ""),
        }
    return cache


def project_feature_item(feature_item, target, min_target_coverage):
    """Project Swiss-Prot protected positions onto a ProteinGym target sequence."""

    if not feature_item:
        return {
            "status": "missing_swissprot_features",
            "positions": set(),
            "types": "",
            "descriptions": "",
            "accession": "",
            "entry_name": "",
            "target_coverage": 0.0,
        }

    source = feature_item["sequence"]
    protected = set(feature_item["protected_positions"])
    if source == target:
        return {
            "status": "success",
            "positions": protected,
            "types": feature_item["protected_types"],
            "descriptions": feature_item["protected_feature_descriptions"],
            "accession": feature_item["accession"],
            "entry_name": feature_item["entry_name"],
            "target_coverage": 1.0,
        }

    src_to_tgt = {}
    status = "swissprot_sequence_mismatch"
    target_coverage = 0.0
    offset = source.find(target)
    if offset >= 0:
        src_to_tgt = {offset + idx: idx for idx in range(len(target))}
        status = "substring_feature_projection_success"
        target_coverage = 1.0
    else:
        offset = target.find(source)
        if offset >= 0:
            src_to_tgt = {idx: offset + idx for idx in range(len(source))}
            status = "source_embedded_feature_projection_success"
            target_coverage = len(source) / float(max(1, len(target)))
        else:
            matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
            matched_target = 0
            for block in matcher.get_matching_blocks():
                if block.size <= 0:
                    continue
                matched_target += block.size
                for delta in range(block.size):
                    src_to_tgt[block.a + delta] = block.b + delta
            target_coverage = matched_target / float(max(1, len(target)))
            if target_coverage >= min_target_coverage:
                status = "aligned_feature_projection_success"

    if status.endswith("_success"):
        projected = {src_to_tgt[pos] for pos in protected if pos in src_to_tgt}
    else:
        projected = set()
    return {
        "status": status,
        "positions": projected,
        "types": feature_item["protected_types"],
        "descriptions": feature_item["protected_feature_descriptions"],
        "accession": feature_item["accession"],
        "entry_name": feature_item["entry_name"],
        "target_coverage": target_coverage,
    }


def fmt(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def terminal_proximity(row):
    midpoint = safe_float(row.get("normalized_midpoint"))
    if midpoint is None:
        return None
    return abs(midpoint - 0.5) * 2.0


def full_biodel_score(row, stage1_weight, bioprior_weight):
    stage1 = safe_float(row.get("stage1_utility_score"))
    bioprior = safe_float(row.get("final_bioprior_score"))
    if stage1 is None or bioprior is None:
        return None
    return stage1_weight * stage1 + bioprior_weight * bioprior


def risk_only_score(row):
    values = [
        safe_float(row.get("functional_core_risk_score")),
        safe_float(row.get("structural_core_risk_score")),
        safe_float(row.get("motif_shadow_risk_score")),
    ]
    values = [value for value in values if value is not None]
    closure = 1.0 if row.get("closure_type") == "internal_unfriendly" else 0.0
    values.append(closure)
    return sum(values) / float(len(values)) if values else None


def annotate_rows(rows, feature_by_id, feature_cache, af2_zip, af2_members, config, tmp_dir, args):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["UniProt_ID"], []).append(row)

    out_rows = []
    stats = Counter()
    for uniprot_id, protein_rows in grouped.items():
        stats["proteins"] += 1
        target = protein_rows[0]["target_sequence"]
        feature_item = feature_by_id.get(uniprot_id)
        protected_positions = set()
        protected_types = ""
        protected_descriptions = ""
        protected_accession = ""
        protected_entry_name = ""
        feature_projection_coverage = 0.0
        feature_status = "missing_swissprot_features"
        if feature_item:
            projected = project_feature_item(feature_item, target, args.min_feature_projection_coverage)
            protected_positions = projected["positions"]
            protected_types = projected["types"]
            protected_descriptions = projected["descriptions"]
            protected_accession = projected["accession"]
            protected_entry_name = projected["entry_name"]
            feature_projection_coverage = projected["target_coverage"]
            feature_status = projected["status"]
        if feature_status != "success":
            cached = feature_cache.get(protein_rows[0].get("target_sha1", ""))
            if cached:
                protected_positions = set(cached["protected_positions"])
                protected_types = cached["protected_types"]
                protected_descriptions = cached["protected_feature_descriptions"]
                protected_accession = cached["accession"]
                protected_entry_name = cached["entry_name"]
                feature_projection_coverage = 1.0
                feature_status = "sequence_exact_feature_cache_success"

        pdb_member = af2_members.get("{}.pdb".format(uniprot_id))
        residue_rows = None
        coords = None
        structure_status = "missing_af2_structure"
        if pdb_member:
            try:
                pdb_path = write_temp_pdb_from_zip(af2_zip, pdb_member, tmp_dir)
                coords = parse_pdb_ca_coords(pdb_path)
                if len(coords) == len(target):
                    residues = []
                    with open(pdb_path, "rb") as handle:
                        seen = set()
                        for raw_line in handle:
                            if not raw_line.startswith(b"ATOM"):
                                continue
                            line = raw_line.decode("ascii", "ignore")
                            if line[12:16].strip() != "CA":
                                continue
                            altloc = line[16:17]
                            if altloc not in (" ", "A"):
                                continue
                            key = (line[21:22], line[22:26].strip(), line[26:27])
                            if key in seen:
                                continue
                            seen.add(key)
                            residues.append({
                                "aa": target[len(residues)] if len(residues) < len(target) else "X",
                                "coord": (
                                    float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54]),
                                ),
                                "plddt": float(line[60:66]),
                            })
                    residue_rows = compute_residue_rows(
                        uniprot_id,
                        target,
                        sorted(protected_positions),
                        protected_types,
                        residues,
                        float(config.get("shadow_cutoff", 8.0)),
                        float(config.get("contact_cutoff", 8.0)),
                    )
                    structure_status = "success"
                else:
                    structure_status = "structure_length_mismatch"
            except Exception as exc:  # noqa: BLE001
                structure_status = "structure_parse_error:{}".format(str(exc)[:120])

        features = canonical_residue_features(residue_rows) if residue_rows else None
        for row in protein_rows:
            out = dict(row)
            start = int(row["deletion_start"])
            end = int(row["deletion_end"])
            deleted = set(range(start, end + 1))
            protected_overlap = deleted & protected_positions
            out["swissprot_feature_status"] = feature_status
            out["swissprot_accession"] = protected_accession
            out["swissprot_entry_name"] = protected_entry_name
            out["swissprot_feature_projection_coverage"] = fmt(feature_projection_coverage)
            out["protected_positions"] = join_positions(protected_positions)
            out["protected_types"] = protected_types
            out["protected_feature_descriptions"] = protected_descriptions
            out["n_protected_positions"] = len(protected_positions)
            out["protected_overlap_count"] = len(protected_overlap)
            out["protected_overlap_fraction_step3"] = fmt(len(protected_overlap) / float(max(1, end - start + 1)))
            out["protected_overlap_positions"] = join_positions(protected_overlap)
            out["af2_zip_member"] = pdb_member or ""
            out["structure_feature_status"] = structure_status
            out["terminal_proximity_score"] = fmt(terminal_proximity(row))

            if features and coords:
                segment = {
                    "seg_start": start,
                    "seg_end": end,
                    "seg_len": end - start + 1,
                    "proposal_source": "proteingym_observed_deletion",
                    "biological_rationale": "ProteinGym observed single-segment deletion",
                }
                agg = aggregate_segment(uniprot_id, len(target), segment, features, coords, config)
                for key, value in agg.items():
                    if key in ("accession", "protein_length", "seg_start", "seg_end", "seg_len"):
                        continue
                    out[key] = value
                risk = risk_only_score(out)
                out["bioprior_risk_only_score"] = fmt(risk)
                out["bioprior_risk_only_favorable_score"] = fmt(-risk if risk is not None else None)
                out["full_biodel_score"] = fmt(full_biodel_score(out, args.stage1_weight, args.bioprior_weight))
                stats["rows_with_structure_features"] += 1
            else:
                for key in STRUCTURAL_FIELDS:
                    out.setdefault(key, "")
                out["bioprior_risk_only_score"] = ""
                out["bioprior_risk_only_favorable_score"] = ""
                out["full_biodel_score"] = ""
            out_rows.append(out)
            stats["rows"] += 1
            if protected_overlap:
                stats["rows_with_protected_overlap"] += 1
            if feature_status == "success":
                stats["rows_with_swissprot_features"] += 1
        stats["protein_structure_{}".format(structure_status.split(":", 1)[0])] += 1
        stats["protein_feature_{}".format(feature_status)] += 1
        stats["protein_feature_with_projected_positions"] += 1 if protected_positions else 0
    return out_rows, stats


STRUCTURAL_FIELDS = [
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


def write_summary(path, args, stats, extraction_stats):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("ProteinGym v1.3 BioDel Step-3 feature annotation\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("af2_zip: {}\n".format(args.af2_zip))
        handle.write("swissprot_xml: {}\n".format(args.swissprot_xml))
        handle.write("out_csv: {}\n\n".format(args.out_csv))
        handle.write("Swiss-Prot extraction stats: {}\n".format(dict(extraction_stats)))
        handle.write("Annotation stats: {}\n\n".format(dict(stats)))
        handle.write("PROTEINGYM_V13_BIODEL_FEATURES_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Annotate ProteinGym v1.3 deletions with BioPrior structural features.")
    parser.add_argument("--input_csv", default="results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_zero_shot.csv")
    parser.add_argument("--out_csv", default="results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_biodel_features.csv")
    parser.add_argument("--summary_txt", default="results/proteingym_v13_indel/proteingym_v13_biodel_features_summary.txt")
    parser.add_argument("--af2_zip", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/external/proteingym_v13/ProteinGym_AF2_structures.zip")
    parser.add_argument("--swissprot_xml", default="data/raw/uniprot/uniprot_sprot.xml.gz")
    parser.add_argument("--feature_cache_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_mapped_features.csv")
    parser.add_argument("--config", default="configs/bioprior_v1.yaml")
    parser.add_argument("--tmp_dir", default="results/proteingym_v13_indel/tmp_af2_structures")
    parser.add_argument("--max_range_feature_len", type=int, default=30)
    parser.add_argument("--min_feature_projection_coverage", type=float, default=0.50)
    parser.add_argument("--stage1_weight", type=float, default=1.0)
    parser.add_argument("--bioprior_weight", type=float, default=1.0)
    args = parser.parse_args()

    rows, fields = read_csv(args.input_csv)
    wanted = sorted(set(row.get("UniProt_ID", "") for row in rows))
    feature_cache = load_feature_cache(args.feature_cache_csv)
    feature_by_id, extraction_stats = extract_swissprot_features(args.swissprot_xml, wanted, args.max_range_feature_len)
    config = load_bioprior_config(args.config)
    af2_members = zip_basename_map(args.af2_zip)
    out_rows, stats = annotate_rows(rows, feature_by_id, feature_cache, args.af2_zip, af2_members, config, args.tmp_dir, args)

    added = [
        "swissprot_feature_status",
        "swissprot_accession",
        "swissprot_entry_name",
        "swissprot_feature_projection_coverage",
        "protected_positions",
        "protected_types",
        "protected_feature_descriptions",
        "n_protected_positions",
        "protected_overlap_count",
        "protected_overlap_fraction_step3",
        "protected_overlap_positions",
        "af2_zip_member",
        "structure_feature_status",
        "terminal_proximity_score",
    ] + STRUCTURAL_FIELDS + [
        "bioprior_risk_only_score",
        "bioprior_risk_only_favorable_score",
        "full_biodel_score",
    ]
    out_fields = fields + [field for field in added if field not in fields]
    write_csv(args.out_csv, out_rows, out_fields)
    write_summary(args.summary_txt, args, stats, extraction_stats)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
