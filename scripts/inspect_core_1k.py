#!/usr/bin/env python3
"""Inspect Core-1K motif-constrained CSV/FASTA outputs."""

from __future__ import print_function

import argparse
import csv
import os
from collections import Counter


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
FEATURE_TYPES = ["active site", "binding site", "site", "short sequence motif"]


def read_fasta(path):
    records = []
    header = None
    chunks = []
    with open(path, "rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def fasta_accession(header):
    return header.split("|", 1)[0]


def parse_positions(value):
    if value is None or value == "":
        return []
    return [int(token) for token in value.split(";") if token != ""]


def bin_length(length):
    if 100 <= length <= 200:
        return "100-200"
    if 201 <= length <= 400:
        return "201-400"
    if 401 <= length <= 600:
        return "401-600"
    if 601 <= length <= 800:
        return "601-800"
    return "outside"


def bin_protected_count(count):
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    if count <= 30:
        return "21-30"
    return ">30"


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Core-1K motif dataset.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fasta", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        print("FAIL: missing CSV {0}".format(args.csv))
        raise SystemExit(1)
    if not os.path.exists(args.fasta):
        print("FAIL: missing FASTA {0}".format(args.fasta))
        raise SystemExit(1)

    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    fasta_records = read_fasta(args.fasta)

    all_pass = True
    warnings = []
    primary_dist = Counter()
    protected_type_dist = Counter()
    length_dist = Counter()
    protected_count_dist = Counter()
    accessions = set()
    sequences = set()

    if len(rows) != len(fasta_records):
        all_pass = False
        print(
            "FAIL: CSV rows {0} != FASTA records {1}".format(
                len(rows), len(fasta_records)
            )
        )

    for index, row in enumerate(rows):
        reasons = []
        accession = row.get("accession") or row.get("protein_id", "")
        sequence = row.get("sequence", "")
        length = int(row.get("length") or len(sequence))

        if accession in accessions:
            reasons.append("duplicate accession")
        accessions.add(accession)
        if sequence in sequences:
            reasons.append("duplicate sequence")
        sequences.add(sequence)

        if index < len(fasta_records):
            fasta_header, fasta_sequence = fasta_records[index]
            if fasta_accession(fasta_header) != accession:
                reasons.append("FASTA accession mismatch: {0}".format(fasta_header))
            if fasta_sequence != sequence:
                reasons.append("FASTA sequence mismatch")
        else:
            reasons.append("missing FASTA record")

        if length != len(sequence):
            reasons.append("length column mismatch")
        if not (100 <= length <= 800):
            reasons.append("length out of 100-800")
        if not sequence or not set(sequence).issubset(STANDARD_AA):
            reasons.append("non-standard amino acid")

        try:
            positions = parse_positions(row.get("protected_positions", ""))
        except ValueError:
            positions = []
            reasons.append("non-integer protected_positions")

        if not positions:
            reasons.append("no protected positions")
        if positions != sorted(positions):
            reasons.append("protected_positions not sorted")
        if len(set(positions)) != len(positions):
            reasons.append("duplicate protected_positions")
        if any(pos < 0 or pos >= length for pos in positions):
            reasons.append("protected_positions out of 0-based bounds")
        if positions and min(positions) < 0:
            reasons.append("positions are not valid 0-based indexes")

        primary_type = row.get("primary_protected_type", "")
        primary_dist[primary_type] += 1
        protected_type_dist[row.get("protected_types", "")] += 1
        length_dist[bin_length(length)] += 1
        protected_count_dist[bin_protected_count(len(positions))] += 1

        if reasons:
            all_pass = False
            print("FAIL\t{0}\t{1}".format(accession, "; ".join(reasons)))

    print("CSV rows: {0}".format(len(rows)))
    print("FASTA records: {0}".format(len(fasta_records)))
    print("")
    print("Primary protected type distribution:")
    for feature_type in FEATURE_TYPES:
        print("- {0}: {1}".format(feature_type, primary_dist[feature_type]))
    for feature_type in sorted(set(primary_dist) - set(FEATURE_TYPES)):
        print("- {0}: {1}".format(feature_type, primary_dist[feature_type]))

    print("")
    print("Protected type combination distribution:")
    for key, value in protected_type_dist.most_common():
        print("- {0}: {1}".format(key, value))

    print("")
    print("Length distribution:")
    for label in ["100-200", "201-400", "401-600", "601-800", "outside"]:
        print("- {0}: {1}".format(label, length_dist[label]))

    print("")
    print("Protected count distribution:")
    for label in ["1", "2-5", "6-10", "11-20", "21-30", ">30"]:
        print("- {0}: {1}".format(label, protected_count_dist[label]))

    if primary_dist["short sequence motif"] > 250:
        warnings.append("short sequence motif count is higher than expected")
    if primary_dist["active site"] < 200:
        warnings.append("active site count is substantially below requested quota")
    if primary_dist["binding site"] < 200:
        warnings.append("binding site count is substantially below requested quota")
    if length_dist["outside"] > 0:
        warnings.append("some records are outside length range")
    if protected_count_dist[">30"] > 0:
        warnings.append("some records have >30 protected positions")

    if warnings:
        print("")
        for warning in warnings:
            print("WARNING: {0}".format(warning))

    if all_pass:
        print("")
        print("ALL_PASS")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
