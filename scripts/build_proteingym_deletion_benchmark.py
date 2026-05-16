#!/usr/bin/env python3
"""Extract single-segment deletion examples from ProteinGym indel tables."""

import argparse
import csv
import os
from collections import Counter, defaultdict

import pandas as pd


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def is_standard(sequence):
    return isinstance(sequence, str) and bool(sequence) and set(sequence) <= STANDARD_AA


def find_single_deletion(target, mutated):
    """Return 0-based deletion interval if mutated is target with one segment removed."""
    if not isinstance(target, str) or not isinstance(mutated, str):
        return None
    if len(mutated) >= len(target):
        return None
    i = 0
    while i < len(mutated) and target[i] == mutated[i]:
        i += 1
    suffix = 0
    while suffix < (len(mutated) - i) and target[len(target) - 1 - suffix] == mutated[len(mutated) - 1 - suffix]:
        suffix += 1
    if i + suffix != len(mutated):
        return None
    deletion_len = len(target) - len(mutated)
    start = i
    end = i + deletion_len - 1
    recovered = target[:start] + target[end + 1 :]
    if recovered != mutated:
        return None
    return start, end


def parse_dms_rows(path, source_name, max_len):
    df = pd.read_parquet(path)
    rows = []
    stats = Counter()
    assay_counts = Counter()
    for idx, row in df.iterrows():
        target = row.get("target_seq")
        mutated = row.get("mutated_sequence")
        stats["input_rows"] += 1
        if not is_standard(target) or not is_standard(mutated):
            stats["nonstandard_sequence"] += 1
            continue
        if len(target) > max_len:
            stats["target_too_long"] += 1
            continue
        interval = find_single_deletion(target, mutated)
        if interval is None:
            stats["not_single_segment_deletion"] += 1
            continue
        start, end = interval
        deletion_len = end - start + 1
        deletion_seq = target[start : end + 1]
        dms_id = str(row.get("DMS_id", ""))
        assay_counts[dms_id] += 1
        rows.append({
            "example_id": "{}:{}:{}".format(source_name, dms_id, idx),
            "source": source_name,
            "assay_id": dms_id,
            "protein_id": "",
            "target_sequence": target,
            "mutated_sequence": mutated,
            "target_length": len(target),
            "mutated_length": len(mutated),
            "deletion_start": start,
            "deletion_end": end,
            "deletion_len": deletion_len,
            "deletion_sequence": deletion_seq,
            "normalized_start": start / float(len(target)),
            "normalized_end": end / float(len(target)),
            "normalized_midpoint": ((start + end) / 2.0) / float(len(target)),
            "deletion_fraction": deletion_len / float(len(target)),
            "is_terminal_deletion": start == 0 or end == len(target) - 1,
            "DMS_score": row.get("DMS_score", ""),
            "DMS_score_bin": row.get("DMS_score_bin", ""),
            "clinical_annotation": "",
            "raw_mutant": row.get("mutant", ""),
        })
        stats["single_segment_deletion"] += 1
    return rows, stats, assay_counts


def parse_clinical_rows(path, source_name, max_len):
    df = pd.read_parquet(path)
    rows = []
    stats = Counter()
    annotation_counts = Counter()
    for idx, row in df.iterrows():
        target = row.get("target_seq")
        mutated = row.get("mutated_sequence")
        stats["input_rows"] += 1
        if not is_standard(target) or not is_standard(mutated):
            stats["nonstandard_sequence"] += 1
            continue
        if len(target) > max_len:
            stats["target_too_long"] += 1
            continue
        interval = find_single_deletion(target, mutated)
        if interval is None:
            stats["not_single_segment_deletion"] += 1
            continue
        start, end = interval
        deletion_len = end - start + 1
        annotation = str(row.get("annotation", ""))
        annotation_counts[annotation] += 1
        rows.append({
            "example_id": "{}:{}:{}".format(source_name, row.get("protein_id", ""), idx),
            "source": source_name,
            "assay_id": "",
            "protein_id": row.get("protein_id", ""),
            "target_sequence": target,
            "mutated_sequence": mutated,
            "target_length": len(target),
            "mutated_length": len(mutated),
            "deletion_start": start,
            "deletion_end": end,
            "deletion_len": deletion_len,
            "deletion_sequence": target[start : end + 1],
            "normalized_start": start / float(len(target)),
            "normalized_end": end / float(len(target)),
            "normalized_midpoint": ((start + end) / 2.0) / float(len(target)),
            "deletion_fraction": deletion_len / float(len(target)),
            "is_terminal_deletion": start == 0 or end == len(target) - 1,
            "DMS_score": "",
            "DMS_score_bin": "",
            "clinical_annotation": annotation,
            "raw_mutant": row.get("mutant", ""),
        })
        stats["single_segment_deletion"] += 1
    return rows, stats, annotation_counts


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "example_id", "source", "assay_id", "protein_id", "target_sequence", "mutated_sequence",
        "target_length", "mutated_length", "deletion_start", "deletion_end", "deletion_len",
        "deletion_sequence", "normalized_start", "normalized_end", "normalized_midpoint",
        "deletion_fraction", "is_terminal_deletion", "DMS_score", "DMS_score_bin",
        "clinical_annotation", "raw_mutant",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, args, rows, stats_by_source, assay_counts, annotation_counts):
    ensure_parent(path)
    by_source = Counter(row["source"] for row in rows)
    by_len = Counter()
    terminal = Counter()
    for row in rows:
        length = int(row["deletion_len"])
        if length == 1:
            key = "1"
        elif length <= 3:
            key = "2-3"
        elif length <= 10:
            key = "4-10"
        else:
            key = ">10"
        by_len[key] += 1
        terminal[str(row["is_terminal_deletion"])] += 1

    with open(path, "w") as handle:
        handle.write("ProteinGym single-segment deletion benchmark summary\n\n")
        handle.write("dms_parquet: {}\n".format(args.dms_parquet))
        handle.write("clinical_parquet: {}\n".format(args.clinical_parquet))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("max_target_len: {}\n\n".format(args.max_target_len))
        handle.write("total_deletions: {}\n".format(len(rows)))
        handle.write("by_source: {}\n".format(dict(by_source)))
        handle.write("deletion_len_bins: {}\n".format(dict(by_len)))
        handle.write("terminal_deletion_counts: {}\n\n".format(dict(terminal)))
        handle.write("source_stats:\n")
        for source, stats in stats_by_source.items():
            handle.write("- {}: {}\n".format(source, dict(stats)))
        handle.write("\nDMS assays with deletions: {}\n".format(len(assay_counts)))
        handle.write("Top DMS assays:\n")
        for assay, count in assay_counts.most_common(20):
            handle.write("- {}: {}\n".format(assay, count))
        handle.write("\nClinical annotation counts:\n")
        for annotation, count in annotation_counts.most_common():
            handle.write("- {}: {}\n".format(annotation, count))
        handle.write("\nQuality checks:\n")
        handle.write("- all rows are standard amino acids: yes\n")
        handle.write("- all rows recover mutated_sequence after deleting target interval: yes\n")
        handle.write("- coordinates are 0-based closed intervals: yes\n")
        handle.write("\nPROTEINGYM_DELETION_BENCHMARK_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Build ProteinGym single-segment deletion benchmark.")
    parser.add_argument("--dms_parquet", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/external/proteingym/raw/DMS_indels.parquet")
    parser.add_argument("--clinical_parquet", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/external/proteingym/raw/clinical_indels.parquet")
    parser.add_argument("--out_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions.csv")
    parser.add_argument("--summary_txt", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_summary.txt")
    parser.add_argument("--max_target_len", type=int, default=1200)
    args = parser.parse_args()

    all_rows = []
    stats_by_source = {}
    dms_rows, dms_stats, assay_counts = parse_dms_rows(args.dms_parquet, "DMS_indels", args.max_target_len)
    clinical_rows, clinical_stats, annotation_counts = parse_clinical_rows(args.clinical_parquet, "clinical_indels", args.max_target_len)
    stats_by_source["DMS_indels"] = dms_stats
    stats_by_source["clinical_indels"] = clinical_stats
    all_rows.extend(dms_rows)
    all_rows.extend(clinical_rows)
    write_csv(args.out_csv, all_rows)
    write_summary(args.summary_txt, args, all_rows, stats_by_source, assay_counts, annotation_counts)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
