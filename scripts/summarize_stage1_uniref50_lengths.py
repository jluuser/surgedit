#!/usr/bin/env python3
"""Summarize original and corrupted sequence lengths for Stage-1 streaming data."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_stage1_deletion_pretrain_dataset import load_config  # noqa: E402


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def stream_fasta(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        header = None
        chunks = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def seq_id_from_header(header):
    return header.split(None, 1)[0]


def stable_int(text):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def split_for_sequence(sequence, train_frac, val_frac):
    seq_hash = hashlib.sha1(sequence.encode("ascii")).hexdigest()
    value = int(seq_hash[:12], 16) / float(16 ** 12)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "val"
    return "test"


def percentile_hist(hist, q):
    if not hist:
        return None
    total = sum(hist.values())
    rank = int(math.ceil((q / 100.0) * total))
    rank = min(max(rank, 1), total)
    running = 0
    for value in sorted(hist):
        running += hist[value]
        if running >= rank:
            return value
    return max(hist)


def summarize_hist(hist):
    count = sum(hist.values())
    if not count:
        return {}
    total = sum(value * n for value, n in hist.items())
    return {
        "count": count,
        "mean": total / float(count),
        "min": min(hist),
        "p50": percentile_hist(hist, 50),
        "p75": percentile_hist(hist, 75),
        "p90": percentile_hist(hist, 90),
        "p95": percentile_hist(hist, 95),
        "p99": percentile_hist(hist, 99),
        "max": max(hist),
    }


def cutoff_rows(hist, cutoffs):
    total = float(sum(hist.values()) or 1)
    rows = []
    for cutoff in cutoffs:
        kept = sum(n for value, n in hist.items() if value <= cutoff)
        rows.append({"cutoff": cutoff, "count": kept, "fraction": kept / total})
    return rows


def write_csv(path, rows, fields):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_fasta", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/raw/uniref50/uniref50.fasta.gz")
    parser.add_argument("--config", default="configs/stage1_corruption.yaml")
    parser.add_argument("--out_dir", default="results/stage1_length_stats")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--min_len", type=int, default=80)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--allow_nonstandard_aa", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", default="400,600,800,1000,1200,1500,2000,3000")
    args = parser.parse_args()

    config = load_config(args.config)
    config["seed"] = args.seed
    budgets = [float(value) for value in config["budgets"]]
    cutoffs = [int(value) for value in args.cutoffs.split(",") if value.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    sequence_lengths = {"train": Counter(), "val": Counter(), "test": Counter()}
    sample_original_lengths = {"train": Counter(), "val": Counter(), "test": Counter()}
    sample_corrupted_lengths = {"train": Counter(), "val": Counter(), "test": Counter()}
    sample_inserted_lengths = {"train": Counter(), "val": Counter(), "test": Counter()}
    counts = Counter()
    skipped = Counter()

    start_time = __import__("time").time()
    for idx, (header, raw_sequence) in enumerate(stream_fasta(args.input_fasta), start=1):
        if args.max_records is not None and idx > args.max_records:
            break
        sequence = "".join(raw_sequence.split()).upper()
        counts["scanned_records"] += 1
        if len(sequence) < args.min_len:
            skipped["too_short"] += 1
            continue
        if not args.allow_nonstandard_aa and not set(sequence).issubset(STANDARD_AA):
            skipped["nonstandard_aa"] += 1
            continue
        split = split_for_sequence(sequence, args.train_frac, args.val_frac)
        sequence_lengths[split][len(sequence)] += 1
        counts["usable_sequences"] += 1
        counts["{}_sequences".format(split)] += 1
        for budget in budgets:
            inserted_total_length = max(1, int(math.floor(len(sequence) * budget)))
            sample_original_lengths[split][len(sequence)] += 1
            sample_corrupted_lengths[split][len(sequence) + inserted_total_length] += 1
            sample_inserted_lengths[split][inserted_total_length] += 1
            counts["usable_samples"] += 1
            counts["{}_samples".format(split)] += 1
        if idx % 100000 == 0:
            elapsed = __import__("time").time() - start_time
            print("scanned={} elapsed={:.1f}s".format(idx, elapsed), flush=True)

    summary = {
        "input_fasta": args.input_fasta,
        "config": args.config,
        "budgets": budgets,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "min_len": args.min_len,
        "max_records": args.max_records,
        "sample_length_mode": "corrupted_length = original_length + max(1, floor(original_length * budget))",
        "counts": dict(counts),
        "skipped": dict(skipped),
        "sequence_original_lengths": {split: summarize_hist(values) for split, values in sequence_lengths.items()},
        "sample_original_lengths": {split: summarize_hist(values) for split, values in sample_original_lengths.items()},
        "sample_corrupted_lengths": {split: summarize_hist(values) for split, values in sample_corrupted_lengths.items()},
        "sample_inserted_lengths": {split: summarize_hist(values) for split, values in sample_inserted_lengths.items()},
        "elapsed_sec": __import__("time").time() - start_time,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    rows = []
    for split in ("train", "val", "test"):
        for row in cutoff_rows(sequence_lengths[split], cutoffs):
            rows.append({"split": split, "kind": "sequence_original_length", **row})
        for row in cutoff_rows(sample_corrupted_lengths[split], cutoffs):
            rows.append({"split": split, "kind": "sample_corrupted_length", **row})
    write_csv(os.path.join(args.out_dir, "cutoff_coverage.csv"), rows, ["split", "kind", "cutoff", "count", "fraction"])

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote {}".format(args.out_dir))


if __name__ == "__main__":
    main()
