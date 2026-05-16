#!/usr/bin/env python3
"""Merge previously computed Stage-1 ProteinGym scores into v1.3 rows."""

import argparse
import csv
import os


STAGE1_FIELDS = [
    "stage1_mean_prob_b10",
    "stage1_sum_prob_b10",
    "stage1_max_prob_b10",
    "stage1_mean_logit_b10",
    "stage1_mean_prob_b20",
    "stage1_sum_prob_b20",
    "stage1_max_prob_b20",
    "stage1_mean_logit_b20",
    "stage1_mean_prob_b30",
    "stage1_sum_prob_b30",
    "stage1_max_prob_b30",
    "stage1_mean_logit_b30",
    "stage1_utility_score",
    "stage1_utility_budget",
    "stage1_scoring_status",
]


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def key(row):
    return (
        row.get("target_sha1", ""),
        row.get("mutated_sequence", ""),
        str(row.get("deletion_start", "")),
        str(row.get("deletion_end", "")),
    )


def safe_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def recompute_full_biodel(row, stage1_weight, bioprior_weight):
    stage1 = safe_float(row.get("stage1_utility_score"))
    bioprior = safe_float(row.get("final_bioprior_score"))
    if stage1 is None or bioprior is None:
        return ""
    return fmt(stage1_weight * stage1 + bioprior_weight * bioprior)


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Merge existing Stage-1 scores into ProteinGym v1.3 Step-3 rows.")
    parser.add_argument("--input_csv", default="results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_biodel_features.csv")
    parser.add_argument("--stage1_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_mapped_features.csv")
    parser.add_argument("--out_csv", default="results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_biodel_features_stage1.csv")
    parser.add_argument("--summary_txt", default="results/proteingym_v13_indel/proteingym_v13_stage1_merge_summary.txt")
    parser.add_argument("--stage1_weight", type=float, default=1.0)
    parser.add_argument("--bioprior_weight", type=float, default=1.0)
    args = parser.parse_args()

    rows, fields = read_csv(args.input_csv)
    stage_rows, _ = read_csv(args.stage1_csv)
    by_key = {}
    for row in stage_rows:
        item_key = key(row)
        if item_key[0] and item_key not in by_key:
            by_key[item_key] = row

    matched = 0
    for row in rows:
        match = by_key.get(key(row))
        if match:
            matched += 1
            for field in STAGE1_FIELDS:
                row[field] = match.get(field, "")
            row["full_biodel_score"] = recompute_full_biodel(row, args.stage1_weight, args.bioprior_weight)
        else:
            for field in STAGE1_FIELDS:
                row.setdefault(field, "")
    out_fields = fields + [field for field in STAGE1_FIELDS if field not in fields]
    write_csv(args.out_csv, rows, out_fields)
    ensure_parent(args.summary_txt)
    with open(args.summary_txt, "w") as handle:
        handle.write("ProteinGym v1.3 Stage-1 score merge\n\n")
        handle.write("input_rows: {}\n".format(len(rows)))
        handle.write("stage1_source_rows: {}\n".format(len(stage_rows)))
        handle.write("matched_rows: {}\n".format(matched))
        handle.write("unmatched_rows: {}\n".format(len(rows) - matched))
        handle.write("\nPROTEINGYM_V13_STAGE1_MERGE_PASS\n")
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
