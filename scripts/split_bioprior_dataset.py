#!/usr/bin/env python3
"""Create protein-level train/val/test splits and leakage report for BioPrior data."""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict


LENGTH_BINS = [
    ("100-200", 100, 200),
    ("201-400", 201, 400),
    ("401-600", 401, 600),
    ("601-800", 601, 800),
]


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def length_bin(length):
    for label, lo, hi in LENGTH_BINS:
        if lo <= length <= hi:
            return label
    return "outside"


def seq_hash(sequence):
    return hashlib.sha1(sequence.encode("ascii")).hexdigest()


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        for row in rows:
            primary = row.get("primary_protected_type", "").replace(" ", "_")
            header = "{}|primary={}|split={}".format(row["accession"], primary, row["split"])
            handle.write(">{}\n".format(header))
            seq = row["sequence"]
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def read_stage1_hashes(path):
    if not path:
        return set()
    hashes = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequence = row.get("sequence", "")
            if sequence:
                hashes.add(seq_hash(sequence))
    return hashes


def make_exact_sequence_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[seq_hash(row["sequence"])].append(row)
    return list(groups.values())


def assign_splits(rows, train_frac, val_frac, seed):
    rng = random.Random(seed)
    groups = make_exact_sequence_groups(rows)
    by_primary = defaultdict(list)
    for group in groups:
        primary = group[0].get("primary_protected_type", "")
        by_primary[primary].append(group)

    split_groups = {"train": [], "val": [], "test": []}
    for primary, primary_groups in by_primary.items():
        rng.shuffle(primary_groups)
        n = len(primary_groups)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        split_groups["train"].extend(primary_groups[:n_train])
        split_groups["val"].extend(primary_groups[n_train : n_train + n_val])
        split_groups["test"].extend(primary_groups[n_train + n_val :])

    assignments = {}
    for split, groups_for_split in split_groups.items():
        for group in groups_for_split:
            for row in group:
                assignments[row["accession"]] = split
    return assignments, split_groups


def summarize_rows(rows):
    primary = Counter(row.get("primary_protected_type", "") for row in rows)
    length_bins = Counter(length_bin(int(row["length"])) for row in rows)
    protected_counts = [int(row["n_protected"]) for row in rows]
    lengths = [int(row["length"]) for row in rows]
    return {
        "n": len(rows),
        "primary": dict(primary),
        "length_bins": dict(length_bins),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": sum(lengths) / float(len(lengths)) if lengths else 0.0,
        "protected_min": min(protected_counts) if protected_counts else 0,
        "protected_max": max(protected_counts) if protected_counts else 0,
        "protected_mean": sum(protected_counts) / float(len(protected_counts)) if protected_counts else 0.0,
    }


def split_overlaps(split_rows):
    accession_sets = {split: set(row["accession"] for row in rows) for split, rows in split_rows.items()}
    sequence_sets = {split: set(seq_hash(row["sequence"]) for row in rows) for split, rows in split_rows.items()}
    checks = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        checks["accession_overlap_{}_{}".format(a, b)] = sorted(accession_sets[a] & accession_sets[b])
        checks["exact_sequence_overlap_{}_{}".format(a, b)] = sorted(sequence_sets[a] & sequence_sets[b])
    return checks


def write_split_json(path, split_rows):
    payload = {
        split: [row["accession"] for row in rows]
        for split, rows in split_rows.items()
    }
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_report(path, args, split_rows, summaries, leakage, stage1_overlap):
    ensure_dir(os.path.dirname(path))
    all_leak_free = all(len(value) == 0 for value in leakage.values())
    with open(path, "w") as handle:
        handle.write("BioPrior protein-level split and leakage report\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("out_dir: {}\n".format(args.out_dir))
        handle.write("seed: {}\n".format(args.seed))
        handle.write("split fractions: train={} val={} test={}\n".format(args.train_frac, args.val_frac, 1.0 - args.train_frac - args.val_frac))
        handle.write("split unit: exact_sequence_group/protein accession\n")
        handle.write("segment-row random split: not used\n")
        handle.write("identity clustering > exact sequence: not applied in this script; use MMseqs2 for final family-level split if available.\n\n")
        for split in ["train", "val", "test"]:
            summary = summaries[split]
            handle.write("{}:\n".format(split))
            handle.write("- proteins: {}\n".format(summary["n"]))
            handle.write("- primary distribution: {}\n".format(summary["primary"]))
            handle.write("- length bins: {}\n".format(summary["length_bins"]))
            handle.write("- length min/max/mean: {}/{}/{:.2f}\n".format(summary["length_min"], summary["length_max"], summary["length_mean"]))
            handle.write("- protected count min/max/mean: {}/{}/{:.2f}\n".format(summary["protected_min"], summary["protected_max"], summary["protected_mean"]))
        handle.write("\nLeakage checks:\n")
        for key, values in sorted(leakage.items()):
            handle.write("- {}: {}\n".format(key, len(values)))
        if stage1_overlap is not None:
            handle.write("\nStage-1 exact sequence overlap with this BioPrior split:\n")
            for split in ["train", "val", "test"]:
                handle.write("- {}: {}\n".format(split, stage1_overlap[split]))
        handle.write("\nCore-1K note:\n")
        handle.write("- Existing Core-1K should be treated as dev-only; do not use it as final test after planner development.\n")
        handle.write("\n{}\n".format("BIOPRIOR_SPLIT_LEAKAGE_PASS" if all_leak_free else "BIOPRIOR_SPLIT_LEAKAGE_WARN"))


def run(args):
    rows, fieldnames = read_csv(args.input_csv)
    assignments, _ = assign_splits(rows, args.train_frac, args.val_frac, args.seed)
    split_rows = {"train": [], "val": [], "test": []}
    output_rows = []
    for row in rows:
        row = dict(row)
        row["split"] = assignments[row["accession"]]
        split_rows[row["split"]].append(row)
        output_rows.append(row)

    ensure_dir(args.out_dir)
    fieldnames_with_split = list(fieldnames)
    if "split" not in fieldnames_with_split:
        fieldnames_with_split.append("split")
    assignments_csv = os.path.join(args.out_dir, "split_assignments.csv")
    write_csv(assignments_csv, output_rows, fieldnames_with_split)
    write_split_json(os.path.join(args.out_dir, "splits.json"), split_rows)

    for split in ["train", "val", "test"]:
        write_csv(os.path.join(args.out_dir, "{}.csv".format(split)), split_rows[split], fieldnames_with_split)
        write_fasta(os.path.join(args.out_dir, "{}.fasta".format(split)), split_rows[split])

    summaries = {split: summarize_rows(split_rows[split]) for split in ["train", "val", "test"]}
    leakage = split_overlaps(split_rows)
    stage1_overlap = None
    if args.stage1_csv:
        stage1_hashes = read_stage1_hashes(args.stage1_csv)
        stage1_overlap = {
            split: sum(1 for row in split_rows[split] if seq_hash(row["sequence"]) in stage1_hashes)
            for split in ["train", "val", "test"]
        }
    write_report(os.path.join(args.out_dir, "split_leakage_report.txt"), args, split_rows, summaries, leakage, stage1_overlap)

    print("Wrote {}".format(assignments_csv))
    print("Wrote {}".format(os.path.join(args.out_dir, "split_leakage_report.txt")))
    for split in ["train", "val", "test"]:
        print("{} proteins: {}".format(split, len(split_rows[split])))


def parse_args():
    parser = argparse.ArgumentParser(description="Create BioPrior protein-level splits and leakage report.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1_csv", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
