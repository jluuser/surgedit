#!/usr/bin/env python3
"""Stream-build UniRef50 Stage-1 pretraining subsets with balanced length bins."""

import argparse
import csv
import gzip
import hashlib
import os
import random
from collections import Counter


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
LENGTH_BINS = [
    ("80-150", 80, 150),
    ("151-300", 151, 300),
    ("301-500", 301, 500),
    ("501-800", 501, 800),
]


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def length_bin(length):
    for label, lo, hi in LENGTH_BINS:
        if lo <= length <= hi:
            return label
    return None


def parse_seq_id(header):
    first = header.split(None, 1)[0]
    if "|" in first:
        return first.split("|")[0]
    return first


def stream_fasta_gz(path, max_records=None):
    header = None
    chunks = []
    scanned = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    scanned += 1
                    yield header, "".join(chunks)
                    if max_records is not None and scanned >= max_records:
                        return
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            scanned += 1
            if max_records is None or scanned <= max_records:
                yield header, "".join(chunks)


def reservoir_add(reservoir, item, seen_count, capacity, rng):
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    j = rng.randint(1, seen_count)
    if j <= capacity:
        reservoir[j - 1] = item


def quotas_for_target(target_size):
    base = target_size // len(LENGTH_BINS)
    quotas = {label: base for label, _, _ in LENGTH_BINS}
    remainder = target_size - base * len(LENGTH_BINS)
    for label, _, _ in LENGTH_BINS[:remainder]:
        quotas[label] += 1
    return quotas


def read_and_sample(args):
    rng = random.Random(args.seed)
    quotas = quotas_for_target(args.target_size)
    reservoirs = {label: [] for label in quotas}
    global_reservoir = []
    valid_seen_by_bin = Counter()
    stats = Counter()
    seen_sequence_hashes = set()
    example_headers = []

    input_size = os.path.getsize(args.input_fasta) if os.path.exists(args.input_fasta) else 0
    if not os.path.exists(args.input_fasta):
        raise FileNotFoundError(args.input_fasta)

    for header, sequence in stream_fasta_gz(args.input_fasta, args.max_records):
        stats["scanned_records"] += 1
        seq = "".join(sequence.split()).upper()
        length = len(seq)

        if length < args.min_len:
            stats["too_short_removed_count"] += 1
            continue
        if length > args.max_len:
            stats["too_long_removed_count"] += 1
            continue
        if args.exclude_nonstandard_aa and not set(seq).issubset(STANDARD_AA):
            stats["nonstandard_removed_count"] += 1
            continue
        seq_hash = hashlib.sha1(seq.encode("ascii")).hexdigest()
        if args.deduplicate_exact and seq_hash in seen_sequence_hashes:
            stats["duplicate_removed_count"] += 1
            continue
        if args.deduplicate_exact:
            seen_sequence_hashes.add(seq_hash)

        bin_label = length_bin(length)
        if bin_label is None:
            continue
        stats["valid_records_seen"] += 1
        valid_seen_by_bin[bin_label] += 1
        if len(example_headers) < 10:
            example_headers.append(header)
        item = {
            "seq_id": parse_seq_id(header),
            "source_header": header,
            "sequence": seq,
            "length": length,
            "length_bin": bin_label,
            "source_file": args.input_fasta,
            "_seq_hash": seq_hash,
        }
        reservoir_add(
            global_reservoir,
            item,
            stats["valid_records_seen"],
            args.target_size,
            rng,
        )
        reservoir_add(
            reservoirs[bin_label],
            item,
            valid_seen_by_bin[bin_label],
            quotas[bin_label],
            rng,
        )

    selected = []
    for label, _, _ in LENGTH_BINS:
        selected.extend(reservoirs[label])

    if len(selected) < args.target_size:
        selected_keys = set(row["_seq_hash"] for row in selected)
        refill_candidates = list(global_reservoir)
        rng.shuffle(refill_candidates)
        for row in refill_candidates:
            if len(selected) >= args.target_size:
                break
            if row["_seq_hash"] in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(row["_seq_hash"])

    rng.shuffle(selected)
    selected = selected[: args.target_size]
    selected.sort(key=lambda row: (row["length_bin"], row["seq_id"]))
    stats["selected_records"] = len(selected)
    return selected, stats, valid_seen_by_bin, quotas, input_size, example_headers


def write_fasta(path, rows):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(">{0}\n".format(row["seq_id"]))
            seq = row["sequence"]
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def write_csv(path, rows):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    fields = ["seq_id", "source_header", "sequence", "length", "length_bin", "source_file"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    return "{:.2f} PB".format(value)


def write_summary(path, args, rows, stats, valid_seen_by_bin, quotas, input_size, example_headers):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    selected_bin_counts = Counter(row["length_bin"] for row in rows)
    output_sizes = {
        args.out_fasta: os.path.getsize(args.out_fasta) if os.path.exists(args.out_fasta) else 0,
        args.out_csv: os.path.getsize(args.out_csv) if os.path.exists(args.out_csv) else 0,
    }
    target_reached = stats["selected_records"] == args.target_size
    all_pass = target_reached and stats["selected_records"] > 0
    with open(path, "w") as handle:
        handle.write("UniRef50 Stage-1 subset summary\n\n")
        handle.write("input_fasta: {}\n".format(args.input_fasta))
        handle.write("input_file_size: {} ({})\n".format(input_size, human_size(input_size)))
        handle.write("out_fasta: {}\n".format(args.out_fasta))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("target_size: {}\n".format(args.target_size))
        handle.write("min_len: {}\n".format(args.min_len))
        handle.write("max_len: {}\n".format(args.max_len))
        handle.write("max_records: {}\n".format(args.max_records if args.max_records is not None else "None"))
        handle.write("random_seed: {}\n".format(args.seed))
        handle.write("deduplicate_exact: {}\n".format(args.deduplicate_exact))
        handle.write("exclude_nonstandard_aa: {}\n\n".format(args.exclude_nonstandard_aa))
        for key in [
            "scanned_records",
            "valid_records_seen",
            "selected_records",
            "duplicate_removed_count",
            "nonstandard_removed_count",
            "too_short_removed_count",
            "too_long_removed_count",
        ]:
            handle.write("{}: {}\n".format(key, stats[key]))
        handle.write("target_reached: {}\n".format(target_reached))
        handle.write("\nLength bin distribution:\n")
        for label, _, _ in LENGTH_BINS:
            handle.write(
                "- {}: selected={} valid_seen={} quota={}\n".format(
                    label, selected_bin_counts[label], valid_seen_by_bin[label], quotas[label]
                )
            )
        handle.write("\nExample headers:\n")
        for header in example_headers:
            handle.write("- {}\n".format(header))
        handle.write("\nOutput file sizes:\n")
        for path_key, size in output_sizes.items():
            handle.write("- {}: {} ({})\n".format(path_key, size, human_size(size)))
        handle.write("\n{}\n".format("ALL_PASS" if all_pass else "WARN_INCOMPLETE"))


def parse_args():
    parser = argparse.ArgumentParser(description="Build UniRef50 Stage-1 streaming subset.")
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--out_fasta", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--target_size", type=int, required=True)
    parser.add_argument("--min_len", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--deduplicate_exact", type=str_to_bool, default=True)
    parser.add_argument("--exclude_nonstandard_aa", type=str_to_bool, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    selected, stats, valid_seen_by_bin, quotas, input_size, example_headers = read_and_sample(args)
    write_fasta(args.out_fasta, selected)
    write_csv(args.out_csv, selected)
    write_summary(args.summary_txt, args, selected, stats, valid_seen_by_bin, quotas, input_size, example_headers)
    print("Scanned records: {}".format(stats["scanned_records"]))
    print("Valid records seen: {}".format(stats["valid_records_seen"]))
    print("Selected records: {}".format(stats["selected_records"]))
    print("Wrote {}".format(args.out_fasta))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
