#!/usr/bin/env python3
"""Score ProteinGym deletion rows with a SCISOR-style Stage-1 checkpoint."""

import argparse
import csv
import os
import sys
from collections import Counter

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from propose_stage1_scisor_intervals import load_model, token_scores  # noqa: E402


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def parse_budgets(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def budget_tag(value):
    return "b{:02d}".format(int(round(float(value) * 100)))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def segment_stats(values, start, end):
    vals = values[start : end + 1]
    if not vals:
        return "", "", ""
    return mean(vals), sum(vals), max(vals)


def fmt(value):
    if value == "":
        return ""
    return "{:.6f}".format(float(value))


def sequence_batches(keys, sequences, batch_size):
    batch = []
    for key in keys:
        batch.append((key, sequences[key]))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_summary(path, args, rows, status_counts, values):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("ProteinGym SCISOR-style Stage-1 scoring summary\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("checkpoint: {}\n".format(args.checkpoint))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("budgets: {}\n".format(args.budgets))
        handle.write("default_budget: {}\n".format(args.default_budget))
        handle.write("device: {}\n\n".format(args.device))
        handle.write("input_rows: {}\n".format(len(rows)))
        handle.write("status_counts: {}\n".format(dict(status_counts)))
        handle.write("scored_rows: {}\n".format(status_counts.get("success", 0)))
        handle.write("mean_stage1_utility: {:.6f}\n".format(mean(values)))
        handle.write("min_stage1_utility: {:.6f}\n".format(min(values) if values else 0.0))
        handle.write("max_stage1_utility: {:.6f}\n".format(max(values) if values else 0.0))
        handle.write("\nPROTEINGYM_SCISOR_STYLE_STAGE1_SCORING_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Score ProteinGym deletions with SCISOR-style Stage-1.")
    parser.add_argument("--input_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--p0", default="p0.pt")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--default_budget", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--disable-fa", action="store_true")
    args = parser.parse_args()

    if args.disable_fa:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    rows, fields = read_csv(args.input_csv)
    budgets = sorted(set(parse_budgets(args.budgets) + [args.default_budget]))
    device = torch.device(args.device)
    model, _ = load_model(args, device)

    sequences = {}
    for row in rows:
        seq = row.get("target_sequence", "")
        if seq and len(seq) <= args.max_len:
            sequences[seq] = seq
    score_cache = {seq: {} for seq in sequences}
    for batch in sequence_batches(list(sequences), sequences, max(1, args.batch_size)):
        batch_keys = [item[0] for item in batch]
        batch_sequences = [item[1] for item in batch]
        for budget in budgets:
            batch_scores = token_scores(model, batch_sequences, budget, device)
            for seq, scores in zip(batch_keys, batch_scores):
                score_cache[seq][budget] = scores

    added = []
    for budget in budgets:
        tag = budget_tag(budget)
        added.extend(
            [
                "stage1_mean_prob_{}".format(tag),
                "stage1_sum_prob_{}".format(tag),
                "stage1_max_prob_{}".format(tag),
            ]
        )
    added.extend(["stage1_utility_score", "stage1_utility_budget", "stage1_scoring_status"])
    status_counts = Counter()
    values = []
    default_tag = budget_tag(args.default_budget)
    for row in rows:
        seq = row.get("target_sequence", "")
        try:
            start = int(float(row.get("deletion_start", "")))
            end = int(float(row.get("deletion_end", "")))
        except ValueError:
            start = end = -1
        if len(seq) > args.max_len:
            row["stage1_scoring_status"] = "target_too_long"
            status_counts[row["stage1_scoring_status"]] += 1
        elif seq not in score_cache:
            row["stage1_scoring_status"] = "missing_sequence_score"
            status_counts[row["stage1_scoring_status"]] += 1
        elif start < 0 or end >= len(seq) or start > end:
            row["stage1_scoring_status"] = "segment_oob"
            status_counts[row["stage1_scoring_status"]] += 1
        else:
            for budget in budgets:
                tag = budget_tag(budget)
                pmean, psum, pmax = segment_stats(score_cache[seq][budget], start, end)
                row["stage1_mean_prob_{}".format(tag)] = fmt(pmean)
                row["stage1_sum_prob_{}".format(tag)] = fmt(psum)
                row["stage1_max_prob_{}".format(tag)] = fmt(pmax)
            row["stage1_utility_score"] = row.get("stage1_mean_prob_{}".format(default_tag), "")
            row["stage1_utility_budget"] = args.default_budget
            row["stage1_scoring_status"] = "success"
            if row["stage1_utility_score"]:
                values.append(float(row["stage1_utility_score"]))
            status_counts["success"] += 1
        for field in added:
            row.setdefault(field, "")

    ensure_parent(args.out_csv)
    out_fields = fields + [field for field in added if field not in fields]
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(args.summary_txt, args, rows, status_counts, values)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
