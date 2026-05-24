#!/usr/bin/env python3
"""Evaluate SCISOR-style Stage-1 interval proposals against teacher labels.

This is an Experiment-1 quality check.  It measures whether the proposer
recovers known positive intervals from the Core-1K teacher labels and how much
of the positive mass is covered by the proposed runs.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def overlap(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or a_start > b_end)


def interval_length(start, end):
    return max(0, end - start + 1)


def recall_for_protein(preds, golds):
    if not golds:
        return None
    covered = 0
    for g_start, g_end in golds:
        g_len = interval_length(g_start, g_end)
        if g_len == 0:
            continue
        overlap_len = 0
        for p_start, p_end in preds:
            if not overlap(p_start, p_end, g_start, g_end):
                continue
            overlap_len += interval_length(max(p_start, g_start), min(p_end, g_end))
        covered += min(g_len, overlap_len)
    total = sum(interval_length(start, end) for start, end in golds)
    return covered / float(total) if total else None


def coverage_at_least_one(preds, golds):
    if not golds:
        return None
    hit = 0
    for g_start, g_end in golds:
        if any(overlap(p_start, p_end, g_start, g_end) for p_start, p_end in preds):
            hit += 1
    return hit / float(len(golds))


def read_proposals(path):
    rows, _ = read_csv(path)
    by_accession = defaultdict(list)
    for row in rows:
        accession = row.get("accession") or row.get("protein_id")
        if not accession:
            continue
        by_accession[accession].append(
            (
                safe_int(row.get("seg_start")),
                safe_int(row.get("seg_end")),
                row,
            )
        )
    for accession in by_accession:
        by_accession[accession].sort(key=lambda item: (safe_int(item[2].get("stage1_interval_rank"), 10**9), item[0], item[1]))
    return by_accession


def read_teacher_labels(path):
    rows, _ = read_csv(path)
    by_accession = defaultdict(list)
    for row in rows:
        if str(row.get("selected_by_teacher", "")).lower() not in ("true", "1", "yes"):
            continue
        accession = row.get("accession")
        if not accession:
            continue
        by_accession[accession].append((safe_int(row.get("seg_start")), safe_int(row.get("seg_end"))))
    return by_accession


def evaluate(proposals_path, teacher_path, output_path, min_rank=None, max_rank=None):
    proposals = read_proposals(proposals_path)
    teacher = read_teacher_labels(teacher_path)
    rows = []
    stats = Counter()
    accessions = sorted(set(proposals) | set(teacher))
    for accession in accessions:
        pred_rows = proposals.get(accession, [])
        if min_rank is not None or max_rank is not None:
            filtered = []
            for start, end, row in pred_rows:
                rank = safe_int(row.get("stage1_interval_rank"), 10**9)
                if min_rank is not None and rank < min_rank:
                    continue
                if max_rank is not None and rank > max_rank:
                    continue
                filtered.append((start, end, row))
            pred_rows = filtered
        pred_intervals = [(start, end) for start, end, _ in pred_rows]
        gold_intervals = teacher.get(accession, [])
        if not gold_intervals:
            stats["no_gold"] += 1
            continue
        protein_recall = recall_for_protein(pred_intervals, gold_intervals)
        interval_hit_rate = coverage_at_least_one(pred_intervals, gold_intervals)
        selected_rows = [row for _, _, row in pred_rows]
        covered_gold = 0
        for g_start, g_end in gold_intervals:
            if any(overlap(p_start, p_end, g_start, g_end) for p_start, p_end in pred_intervals):
                covered_gold += 1
        rows.append(
            {
                "accession": accession,
                "n_teacher_positive": len(gold_intervals),
                "n_proposals": len(pred_intervals),
                "teacher_interval_recall": "" if protein_recall is None else "{:.6f}".format(protein_recall),
                "teacher_interval_hit_rate": "" if interval_hit_rate is None else "{:.6f}".format(interval_hit_rate),
                "teacher_positive_covered": covered_gold,
                "teacher_positive_total": len(gold_intervals),
                "mean_stage1_utility": "{:.6f}".format(
                    sum(float(row.get("stage1_utility_score", 0.0) or 0.0) for row in selected_rows) / float(len(selected_rows))
                ) if selected_rows else "",
                "best_stage1_utility": "{:.6f}".format(
                    max(float(row.get("stage1_utility_score", 0.0) or 0.0) for row in selected_rows)
                ) if selected_rows else "",
            }
        )
        if protein_recall is not None:
            stats["proteins_with_gold"] += 1
            stats["total_recall_num"] += protein_recall
            stats["total_hit_num"] += interval_hit_rate if interval_hit_rate is not None else 0.0
            stats["covered_gold"] += covered_gold
            stats["total_gold"] += len(gold_intervals)
    mean_recall = stats["total_recall_num"] / float(stats["proteins_with_gold"]) if stats["proteins_with_gold"] else 0.0
    mean_hit = stats["total_hit_num"] / float(stats["proteins_with_gold"]) if stats["proteins_with_gold"] else 0.0
    overall_cov = stats["covered_gold"] / float(stats["total_gold"]) if stats["total_gold"] else 0.0
    ensure_parent(output_path)
    with open(output_path, "w") as handle:
        handle.write("# Stage-1 Interval Proposal Evaluation\n\n")
        handle.write("proposals_csv: {}\n".format(proposals_path))
        handle.write("teacher_csv: {}\n".format(teacher_path))
        handle.write("proteins_with_teacher_labels: {}\n".format(stats["proteins_with_gold"]))
        handle.write("mean_teacher_interval_recall: {:.6f}\n".format(mean_recall))
        handle.write("mean_teacher_interval_hit_rate: {:.6f}\n".format(mean_hit))
        handle.write("overall_teacher_positive_coverage: {:.6f}\n".format(overall_cov))
        handle.write("proposal_rows: {}\n".format(len(rows)))
        handle.write("\nSTAGE1_INTERVAL_PROPOSAL_EVAL_PASS\n")
    with open(output_path.replace(".md", ".csv"), "w", newline="") as handle:
        fieldnames = [
            "accession",
            "n_teacher_positive",
            "n_proposals",
            "teacher_interval_recall",
            "teacher_interval_hit_rate",
            "teacher_positive_covered",
            "teacher_positive_total",
            "mean_stage1_utility",
            "best_stage1_utility",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 interval proposal quality.")
    parser.add_argument("--proposals_csv", required=True)
    parser.add_argument("--teacher_csv", default="data/train/core_1k_bioprior_teacher_labels.csv")
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--min_rank", type=int, default=None)
    parser.add_argument("--max_rank", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.proposals_csv, args.teacher_csv, args.output_md, args.min_rank, args.max_rank)
    print("Wrote {}".format(args.output_md))
    print("Wrote {}".format(args.output_md.replace(".md", ".csv")))


if __name__ == "__main__":
    main()
