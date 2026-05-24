#!/usr/bin/env python3
"""Prepare current-best BioDel case-study and structure-reprediction inputs.

This script intentionally does not run a structure predictor.  It exports the
original/deleted FASTA files, deletion block summaries, breakpoint summaries,
and original AFDB paths for a small set of representative current-best planner
outputs.  A separate AF2/ESMFold/OmegaFold/ColabFold run can consume these
FASTA files and then this table can be used to inspect the predicted structures.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def safe_float(value, default=0.0):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def parse_positions(text):
    if not text:
        return set()
    return {int(float(token)) for token in str(text).split(";") if token.strip() != ""}


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_core(path):
    rows = {}
    for row in read_csv(path):
        accession = row.get("accession") or row.get("protein_id")
        if not accession:
            continue
        rows[accession] = {
            "accession": accession,
            "sequence": row.get("sequence", ""),
            "length": safe_int(row.get("length") or len(row.get("sequence", ""))),
            "protected_positions": parse_positions(row.get("protected_positions", "")),
            "primary_protected_type": row.get("primary_protected_type", ""),
            "protected_types": row.get("protected_types", ""),
            "protein_name": row.get("protein_name", ""),
            "organism": row.get("organism", ""),
        }
    return rows


def read_structure_mapping(path):
    mapping = {}
    if not path or not os.path.exists(path):
        return mapping
    for row in read_csv(path):
        accession = row.get("accession")
        if not accession:
            continue
        structure_path = row.get("structure_path", "")
        found = truthy(row.get("found"))
        if structure_path and not os.path.isabs(structure_path):
            structure_path = os.path.abspath(structure_path)
        mapping[accession] = {
            "structure_path": structure_path,
            "structure_found": found and bool(structure_path) and os.path.exists(structure_path),
            "structure_length": row.get("structure_length", ""),
            "notes": row.get("notes", ""),
        }
    return mapping


def selected_by_accession(path, setting, profile):
    grouped = defaultdict(list)
    for row in read_csv(path):
        if setting and row.get("setting") != setting:
            continue
        if profile and row.get("auto_profile") != profile:
            continue
        grouped[row["accession"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (safe_int(row.get("seg_start")), safe_int(row.get("seg_end"))))
    return grouped


def segment_positions(rows, length):
    positions = set()
    for row in rows:
        start = safe_int(row.get("seg_start"))
        end = safe_int(row.get("seg_end"))
        for pos in range(start, end + 1):
            if 0 <= pos < length:
                positions.add(pos)
    return positions


def deletion_blocks(positions):
    ordered = sorted(positions)
    if not ordered:
        return []
    blocks = []
    start = prev = ordered[0]
    for pos in ordered[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        blocks.append((start, prev))
        start = prev = pos
    blocks.append((start, prev))
    return blocks


def block_text(blocks):
    return ";".join("{}-{}".format(start, end) for start, end in blocks)


def deleted_sequence(sequence, positions):
    return "".join(aa for idx, aa in enumerate(sequence) if idx not in positions)


def wrap_fasta(sequence, width=80):
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def write_fasta(path, header, sequence):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write(">{}\n{}\n".format(header, wrap_fasta(sequence)))


def segment_agg(rows):
    agg = Counter()
    vals = defaultdict(list)
    for row in rows:
        seg_len = safe_int(row.get("seg_len"))
        agg["segments"] += 1
        agg["deleted_len"] += seg_len
        agg["protected_overlap"] += safe_int(row.get("n_protected_overlap"))
        agg["shadow_overlap"] += safe_int(row.get("n_shadow_overlap"))
        if row.get("closure_type") == "terminal":
            agg["terminal_segments"] += 1
        else:
            agg["internal_segments"] += 1
        if row.get("closure_type") != "terminal" and not truthy(row.get("closure_friendly_8A")):
            agg["closure_unfriendly_segments"] += 1
            agg["closure_unfriendly_len"] += seg_len
        for field in [
            "boundary_ca_distance",
            "mean_pLDDT",
            "mean_contact_density_8A",
            "stage1_utility_score",
            "final_bioprior_score",
            "risk_upper",
            "risk_point",
            "evidence_confidence",
        ]:
            if row.get(field) not in ("", None):
                vals[field].append(safe_float(row.get(field)))
    out = dict(agg)
    for field, values in vals.items():
        out["mean_" + field] = sum(values) / float(len(values)) if values else 0.0
        out["max_" + field] = max(values) if values else 0.0
    return out


def choose_cases(grouped, core, structures, limit):
    candidates = []
    for accession, rows in grouped.items():
        info = core.get(accession)
        if not info:
            continue
        sequence = info["sequence"]
        if not sequence:
            continue
        positions = segment_positions(rows, len(sequence))
        if not positions:
            continue
        agg = segment_agg(rows)
        delete_ratio = len(positions) / float(len(sequence))
        structure_ok = structures.get(accession, {}).get("structure_found", False)
        score = (
            100.0 * delete_ratio
            + 4.0 * min(agg.get("internal_segments", 0), 4)
            - 12.0 * agg.get("closure_unfriendly_segments", 0)
            - 10.0 * agg.get("protected_overlap", 0)
            - 2.0 * agg.get("shadow_overlap", 0)
            + (6.0 if structure_ok else 0.0)
        )
        candidates.append((score, delete_ratio, accession, rows, positions, agg))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[:limit]


def write_case_outputs(args, chosen, core, structures):
    ensure_dir(args.out_dir)
    case_rows = []
    segment_rows = []
    for rank, (_, delete_ratio, accession, rows, positions, agg) in enumerate(chosen, start=1):
        info = core[accession]
        sequence = info["sequence"]
        deleted = deleted_sequence(sequence, positions)
        blocks = deletion_blocks(positions)
        case_dir = os.path.join(args.out_dir, accession)
        ensure_dir(case_dir)
        original_fasta = os.path.join(case_dir, "{}_original.fasta".format(accession))
        deleted_fasta = os.path.join(case_dir, "{}_deleted.fasta".format(accession))
        write_fasta(original_fasta, "{}|original|len={}".format(accession, len(sequence)), sequence)
        write_fasta(
            deleted_fasta,
            "{}|current_best|setting={}|profile={}|deleted={}".format(
                accession, args.setting, args.auto_profile, len(positions)
            ),
            deleted,
        )
        structure = structures.get(accession, {})
        protected_kept = len(positions & info["protected_positions"]) == 0
        case_rows.append(
            {
                "case_rank": rank,
                "accession": accession,
                "protein_name": info.get("protein_name", ""),
                "organism": info.get("organism", ""),
                "primary_protected_type": info.get("primary_protected_type", ""),
                "protected_types": info.get("protected_types", ""),
                "original_length": len(sequence),
                "deleted_length": len(positions),
                "new_length": len(deleted),
                "delete_ratio": "{:.6f}".format(delete_ratio),
                "selected_segments": agg.get("segments", 0),
                "internal_segments": agg.get("internal_segments", 0),
                "terminal_segments": agg.get("terminal_segments", 0),
                "deletion_blocks_0based": block_text(blocks),
                "protected_positions_deleted": len(positions & info["protected_positions"]),
                "protected_kept": int(protected_kept),
                "shadow_overlap_residues": agg.get("shadow_overlap", 0),
                "closure_unfriendly_segments": agg.get("closure_unfriendly_segments", 0),
                "closure_unfriendly_len": agg.get("closure_unfriendly_len", 0),
                "mean_boundary_ca_distance": "{:.6f}".format(agg.get("mean_boundary_ca_distance", 0.0)),
                "max_boundary_ca_distance": "{:.6f}".format(agg.get("max_boundary_ca_distance", 0.0)),
                "mean_pLDDT": "{:.6f}".format(agg.get("mean_mean_pLDDT", 0.0)),
                "mean_stage1_utility": "{:.6f}".format(agg.get("mean_stage1_utility_score", 0.0)),
                "mean_final_bioprior_score": "{:.6f}".format(agg.get("mean_final_bioprior_score", 0.0)),
                "mean_risk_upper": "{:.6f}".format(agg.get("mean_risk_upper", 0.0)),
                "mean_evidence_confidence": "{:.6f}".format(agg.get("mean_evidence_confidence", 0.0)),
                "original_afdb_pdb": structure.get("structure_path", ""),
                "original_afdb_found": int(bool(structure.get("structure_found", False))),
                "original_fasta": original_fasta,
                "deleted_fasta": deleted_fasta,
                "prediction_status": "not_run",
            }
        )
        for seg_rank, row in enumerate(rows, start=1):
            out = {
                "case_rank": rank,
                "case_accession": accession,
                "case_segment_rank": seg_rank,
            }
            out.update(row)
            segment_rows.append(out)
    return case_rows, segment_rows


def write_csv(path, rows, preferred_fields=None):
    ensure_parent(path)
    fields = list(preferred_fields or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, case_rows, segment_rows):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("# Current-Best Structure Case Studies\n\n")
        handle.write("This current-best case-study block exports FASTA inputs and deletion summaries for the selected BioDel planner output.  It does not claim structure re-prediction metrics unless predicted PDBs are supplied later.\n\n")
        handle.write("## Inputs\n\n")
        handle.write("- selected_segments_csv: `{}`\n".format(args.selected_segments_csv))
        handle.write("- core_csv: `{}`\n".format(args.core_csv))
        handle.write("- structure_mapping_csv: `{}`\n".format(args.structure_mapping_csv))
        handle.write("- setting/profile: `{}` / `{}`\n\n".format(args.setting, args.auto_profile))
        handle.write("## Selected Cases\n\n")
        handle.write("| Rank | Accession | Delete Ratio | Segments | Internal | Protected Deleted | Shadow | Closure Unfriendly | AFDB |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in case_rows:
            handle.write(
                "| {case_rank} | {accession} | {delete_ratio} | {selected_segments} | {internal_segments} | {protected_positions_deleted} | {shadow_overlap_residues} | {closure_unfriendly_segments} | {original_afdb_found} |\n".format(
                    **row
                )
            )
        handle.write("\n## Status\n\n")
        handle.write("- FASTA files for original and deleted sequences were exported under `{}`.\n".format(args.out_dir))
        handle.write("- Original AFDB structure paths are recorded when available.\n")
        handle.write("- Deleted-sequence structure prediction is marked `not_run`; run AF2/ESMFold/OmegaFold/ColabFold externally on the exported deleted FASTA files to populate structure metrics.\n")
        handle.write("- Segment rows exported: {}\n\n".format(len(segment_rows)))
        handle.write("CURRENT_BEST_CASE_STUDY_PREP_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare current-best case-study structure inputs.")
    parser.add_argument("--selected_segments_csv", default="results/current_best_experiments/experiment_family_split_current_certified/bioprior_10k_test_auto_budget_certified_selected_segments.csv")
    parser.add_argument("--core_csv", default="data/processed/swissprot_motif_bioprior_10k.csv")
    parser.add_argument("--structure_mapping_csv", default="data/structures/afdb_bioprior_10k_mapping.csv")
    parser.add_argument("--setting", default="safe")
    parser.add_argument("--auto_profile", default="default")
    parser.add_argument("--limit_cases", type=int, default=5)
    parser.add_argument("--out_dir", default="results/current_best_experiments/experiment6_case_studies")
    parser.add_argument("--out_cases_csv", default="results/current_best_experiments/experiment6_case_studies/current_best_case_studies.csv")
    parser.add_argument("--out_segments_csv", default="results/current_best_experiments/experiment6_case_studies/current_best_case_study_segments.csv")
    parser.add_argument("--out_report", default="results/current_best_experiments/experiment6_case_studies/CURRENT_BEST_CASE_STUDIES_REPORT.md")
    args = parser.parse_args()

    core = read_core(args.core_csv)
    structures = read_structure_mapping(args.structure_mapping_csv)
    grouped = selected_by_accession(args.selected_segments_csv, args.setting, args.auto_profile)
    chosen = choose_cases(grouped, core, structures, args.limit_cases)
    case_rows, segment_rows = write_case_outputs(args, chosen, core, structures)
    write_csv(args.out_cases_csv, case_rows)
    write_csv(args.out_segments_csv, segment_rows)
    write_report(args.out_report, args, case_rows, segment_rows)
    print("Wrote {}".format(args.out_cases_csv))
    print("Wrote {}".format(args.out_segments_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
