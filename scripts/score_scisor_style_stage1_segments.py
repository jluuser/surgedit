#!/usr/bin/env python3
"""Score contiguous deletion segments with the SCISOR-style Stage-1 proposer.

The earlier BioDel scorer targets the small binary deletion-prior checkpoint
format.  The current Stage-1 model is a fine-tuned SCISOR/ShorteningSCUD
checkpoint whose native output is a token deletion posterior.  This script
bridges that checkpoint into the existing segment CSV schema by aggregating
posterior probabilities over each segment.
"""

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


def parse_budgets(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def budget_tag(value):
    return "b{:02d}".format(int(round(float(value) * 100)))


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def read_core_sequences(path):
    rows, _ = read_csv(path)
    sequences = {}
    for row in rows:
        accession = row.get("accession") or row.get("protein_id")
        sequence = row.get("sequence")
        if accession and sequence:
            sequences[accession] = sequence
    return sequences


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def segment_stats(values, start, end):
    vals = values[start : end + 1]
    if not vals:
        return "", "", ""
    return mean(vals), sum(vals), max(vals)


def fmt(value):
    if value == "":
        return ""
    return "{:.6f}".format(float(value))


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_sequence_batches(accessions, sequences, batch_size):
    batch = []
    for accession in accessions:
        batch.append((accession, sequences[accession]))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_summary(path, args, stats, checkpoint, default_scores):
    ensure_parent(path)
    val_metrics = checkpoint.get("val_metrics", {})
    with open(path, "w") as handle:
        handle.write("SCISOR-style Stage-1 segment scoring summary\n\n")
        handle.write("checkpoint: {}\n".format(args.checkpoint))
        handle.write("base_checkpoint: {}\n".format(args.base_checkpoint or checkpoint.get("args", {}).get("checkpoint", "")))
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("core_csv: {}\n".format(args.core_csv))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("budgets: {}\n".format(args.budgets))
        handle.write("default_budget: {}\n".format(args.default_budget))
        handle.write("device: {}\n".format(args.device))
        handle.write("batch_size: {}\n\n".format(args.batch_size))
        for key in [
            "input_segments",
            "output_segments",
            "unique_accessions_in_segments",
            "scored_accessions",
            "missing_sequence_segments",
            "length_mismatch_segments",
            "oob_segments",
            "too_long_segments",
        ]:
            handle.write("{}: {}\n".format(key, stats[key]))
        handle.write("\nCheckpoint validation metrics:\n")
        for key in sorted(val_metrics):
            handle.write("- {}: {}\n".format(key, val_metrics[key]))
        handle.write("\nDefault-budget segment posterior distribution:\n")
        handle.write("mean: {:.6f}\n".format(mean(default_scores)))
        handle.write("min: {:.6f}\n".format(min(default_scores) if default_scores else 0.0))
        handle.write("max: {:.6f}\n".format(max(default_scores) if default_scores else 0.0))
        passed = stats["output_segments"] > 0 and stats["oob_segments"] == 0
        handle.write("\n{}\n".format("SCISOR_STYLE_STAGE1_SEGMENT_SCORING_PASS" if passed else "SCISOR_STYLE_STAGE1_SEGMENT_SCORING_WARN"))


def score_segments(args):
    if args.disable_fa:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device(args.device)
    model, checkpoint = load_model(args, device)
    budgets = sorted(set(parse_budgets(args.budgets) + [float(args.default_budget)]))
    sequences = read_core_sequences(args.core_csv)
    rows, fields = read_csv(args.segments_csv)
    if args.limit_segments is not None:
        rows = rows[: args.limit_segments]

    accessions = sorted({row.get("accession", "") for row in rows if row.get("accession")})
    if args.limit_proteins is not None:
        allowed = set(accessions[: args.limit_proteins])
        rows = [row for row in rows if row.get("accession") in allowed]
        accessions = sorted(allowed)

    accession_sequences = {}
    for accession in accessions:
        sequence = sequences.get(accession)
        if sequence is not None and len(sequence) <= args.max_len:
            accession_sequences[accession] = sequence

    token_score_cache = {accession: {} for accession in accession_sequences}
    for batch in build_sequence_batches(list(accession_sequences), accession_sequences, max(1, args.batch_size)):
        batch_accessions = [item[0] for item in batch]
        batch_sequences = [item[1] for item in batch]
        for budget in budgets:
            batch_scores = token_scores(model, batch_sequences, budget, device)
            for accession, scores in zip(batch_accessions, batch_scores):
                token_score_cache[accession][budget] = scores

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
    out_fields = fields + [field for field in added if field not in fields]

    default_tag = budget_tag(args.default_budget)
    stats = Counter()
    stats["unique_accessions_in_segments"] = len(accessions)
    stats["scored_accessions"] = len(token_score_cache)
    default_scores = []
    ensure_parent(args.out_csv)
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            stats["input_segments"] += 1
            accession = row.get("accession")
            sequence = sequences.get(accession)
            if sequence is None:
                row["stage1_scoring_status"] = "missing_sequence"
                stats["missing_sequence_segments"] += 1
                writer.writerow(row)
                continue
            if len(sequence) > args.max_len:
                row["stage1_scoring_status"] = "target_too_long"
                stats["too_long_segments"] += 1
                writer.writerow(row)
                continue
            if accession not in token_score_cache:
                row["stage1_scoring_status"] = "missing_sequence_score"
                stats["missing_sequence_segments"] += 1
                writer.writerow(row)
                continue
            start = safe_int(row.get("seg_start"))
            end = safe_int(row.get("seg_end"))
            protein_length = safe_int(row.get("protein_length"), len(sequence))
            if protein_length != len(sequence):
                row["stage1_scoring_status"] = "length_mismatch"
                stats["length_mismatch_segments"] += 1
                writer.writerow(row)
                continue
            if start < 0 or end >= len(sequence) or start > end:
                row["stage1_scoring_status"] = "segment_oob"
                stats["oob_segments"] += 1
                writer.writerow(row)
                continue
            for budget in budgets:
                tag = budget_tag(budget)
                pmean, psum, pmax = segment_stats(token_score_cache[accession][budget], start, end)
                row["stage1_mean_prob_{}".format(tag)] = fmt(pmean)
                row["stage1_sum_prob_{}".format(tag)] = fmt(psum)
                row["stage1_max_prob_{}".format(tag)] = fmt(pmax)
            row["stage1_utility_score"] = row.get("stage1_mean_prob_{}".format(default_tag), "")
            row["stage1_utility_budget"] = args.default_budget
            row["stage1_scoring_status"] = "success"
            if row["stage1_utility_score"] != "":
                default_scores.append(float(row["stage1_utility_score"]))
            stats["output_segments"] += 1
            writer.writerow(row)

    write_summary(args.summary_txt, args, stats, checkpoint, default_scores)
    print("Scored accessions: {}".format(stats["scored_accessions"]))
    print("Output segments: {}".format(stats["output_segments"]))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


def parse_args():
    parser = argparse.ArgumentParser(description="Score deletion segments with a SCISOR-style Stage-1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--p0", default="p0.pt")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--core_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--default_budget", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--limit_proteins", type=int, default=None)
    parser.add_argument("--limit_segments", type=int, default=None)
    parser.add_argument("--disable-fa", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    score_segments(parse_args())
