#!/usr/bin/env python3
"""Build per-protein evidence profiles for BioDel.

The profile is intentionally lightweight: it records which evidence channels
are available for each protein and assigns an evidence level/confidence that can
drive evidence-adaptive planning.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def read_metadata(path):
    if not path:
        return {}
    out = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if accession:
                out[accession] = row
    return out


def read_accession_filter(path, accession_column):
    if not path:
        return None
    allowed = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get(accession_column) or row.get("accession") or row.get("protein_id")
            if accession:
                allowed.add(accession)
    return allowed


def has_value(row, fields):
    return any(row.get(field) not in ("", None) for field in fields)


def evidence_level(has_stage1, has_structure, has_annotation, has_msa=False, has_zero_shot=False):
    if has_stage1 and has_structure and has_annotation and (has_msa or has_zero_shot):
        return "full_evidence"
    if has_stage1 and has_structure and has_annotation:
        return "annotation_structure"
    if has_stage1 and has_structure:
        return "structure_aware"
    if has_stage1 and has_annotation:
        return "annotation_aware"
    if has_stage1:
        return "sequence_only"
    return "insufficient"


def confidence_for(level):
    return {
        "full_evidence": 1.00,
        "annotation_structure": 0.92,
        "structure_aware": 0.78,
        "annotation_aware": 0.70,
        "sequence_only": 0.50,
        "insufficient": 0.20,
    }.get(level, 0.20)


def build_profiles(args):
    metadata = read_metadata(args.metadata_csv)
    allowed = read_accession_filter(args.accession_filter_csv, args.accession_column)
    grouped = defaultdict(list)
    with open(args.segments_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession")
            if not accession:
                continue
            if allowed and accession not in allowed:
                continue
            grouped[accession].append(row)

    rows = []
    counts = Counter()
    for accession, segments in sorted(grouped.items()):
        meta = metadata.get(accession, {})
        protein_length = to_int(segments[0].get("protein_length") or meta.get("length"))
        stage1_success = sum(1 for row in segments if row.get("stage1_scoring_status") == "success")
        structure_rows = sum(
            1
            for row in segments
            if has_value(row, ["mean_pLDDT", "mean_contact_density_8A", "boundary_ca_distance", "closure_type"])
        )
        annotation_rows = sum(
            1
            for row in segments
            if has_value(row, ["n_protected_overlap", "protected_overlap_fraction", "n_shadow_overlap", "shadow_overlap_fraction"])
        )
        n_protected = to_int(meta.get("n_protected"))
        protected_positions = meta.get("protected_positions", "")
        feature_counts = meta.get("feature_counts", "")
        has_stage1 = stage1_success > 0
        has_structure = structure_rows > 0
        has_annotation = bool(n_protected > 0 or protected_positions or feature_counts or annotation_rows > 0)
        has_msa = bool(meta.get("MSA_filename") or meta.get("MSA_N_eff"))
        has_zero_shot = bool(meta.get("zero_shot_scores_found"))
        level = evidence_level(has_stage1, has_structure, has_annotation, has_msa, has_zero_shot)
        confidence = confidence_for(level)
        counts[level] += 1
        rows.append(
            {
                "accession": accession,
                "protein_length": protein_length,
                "candidate_segments": len(segments),
                "has_stage1": int(has_stage1),
                "stage1_success_rows": stage1_success,
                "has_structure": int(has_structure),
                "structure_feature_rows": structure_rows,
                "structure_feature_coverage": structure_rows / float(len(segments) or 1),
                "has_annotation": int(has_annotation),
                "annotation_feature_rows": annotation_rows,
                "annotation_feature_coverage": annotation_rows / float(len(segments) or 1),
                "n_protected_positions": n_protected,
                "has_msa": int(has_msa),
                "has_zero_shot": int(has_zero_shot),
                "evidence_level": level,
                "evidence_confidence": confidence,
            }
        )
    return rows, counts


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "accession",
        "protein_length",
        "candidate_segments",
        "has_stage1",
        "stage1_success_rows",
        "has_structure",
        "structure_feature_rows",
        "structure_feature_coverage",
        "has_annotation",
        "annotation_feature_rows",
        "annotation_feature_coverage",
        "n_protected_positions",
        "has_msa",
        "has_zero_shot",
        "evidence_level",
        "evidence_confidence",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, rows, counts):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("BioDel evidence profile report\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("metadata_csv: {}\n".format(args.metadata_csv or ""))
        handle.write("accession_filter_csv: {}\n\n".format(args.accession_filter_csv or ""))
        handle.write("proteins: {}\n".format(len(rows)))
        handle.write("evidence_level_counts: {}\n\n".format(dict(counts)))
        handle.write("BIODEL_EVIDENCE_PROFILE_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build BioDel per-protein evidence profiles.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--accession_filter_csv", default=None)
    parser.add_argument("--accession_column", default="accession")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows, counts = build_profiles(args)
    write_csv(args.out_csv, rows)
    write_report(args.out_report, args, rows, counts)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
