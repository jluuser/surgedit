#!/usr/bin/env python3
"""Filter residue-level BioPrior features to a split CSV."""

import argparse
import csv
import os


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_accessions(path):
    accessions = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if accession:
                accessions.add(accession)
    return accessions


def main():
    parser = argparse.ArgumentParser(description="Filter residue BioPrior CSV by split accessions.")
    parser.add_argument("--residue_csv", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    accessions = read_accessions(args.split_csv)
    ensure_parent(args.out_csv)
    kept_rows = 0
    input_rows = 0
    kept_accessions = set()
    with open(args.residue_csv, newline="") as in_handle, open(args.out_csv, "w", newline="") as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            input_rows += 1
            accession = row.get("accession") or row.get("protein_id")
            if accession in accessions:
                writer.writerow(row)
                kept_rows += 1
                kept_accessions.add(accession)
    print("split_accessions: {}".format(len(accessions)))
    print("input_rows: {}".format(input_rows))
    print("kept_rows: {}".format(kept_rows))
    print("kept_accessions: {}".format(len(kept_accessions)))
    print("Wrote {}".format(args.out_csv))


if __name__ == "__main__":
    main()
