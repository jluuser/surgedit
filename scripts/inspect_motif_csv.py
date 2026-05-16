from __future__ import print_function

import argparse
import csv
import os


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


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


def parse_positions(value):
    if value is None or value == "":
        return []
    positions = []
    for token in value.split(";"):
        if token == "":
            continue
        positions.append(int(token))
    return positions


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect motif CSV and FASTA outputs.")
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--fasta", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    all_pass = True

    if not os.path.exists(args.motif_csv):
        print("FAIL: CSV does not exist: {0}".format(args.motif_csv))
        raise SystemExit(1)
    if not os.path.exists(args.fasta):
        print("FAIL: FASTA does not exist: {0}".format(args.fasta))
        raise SystemExit(1)

    with open(args.motif_csv, "rb") as handle:
        rows = list(csv.DictReader(handle))
    fasta_records = read_fasta(args.fasta)

    if len(rows) != len(fasta_records):
        all_pass = False
        print(
            "FAIL: CSV row count {0} != FASTA record count {1}".format(
                len(rows), len(fasta_records)
            )
        )

    for index, row in enumerate(rows):
        reasons = []
        protein_id = row.get("protein_id", "")
        sequence = row.get("sequence", "")

        if index >= len(fasta_records):
            reasons.append("missing FASTA record")
            fasta_header = ""
            fasta_sequence = ""
        else:
            fasta_header, fasta_sequence = fasta_records[index]
            if protein_id != fasta_header:
                reasons.append(
                    "protein_id does not match FASTA header {0}".format(fasta_header)
                )
            if sequence != fasta_sequence:
                reasons.append("CSV sequence does not match FASTA sequence")

        if not sequence or not set(sequence).issubset(STANDARD_AA):
            reasons.append("sequence contains non-standard amino acids")

        try:
            positions = parse_positions(row.get("protected_positions", ""))
        except ValueError:
            positions = []
            reasons.append("protected_positions contains non-integer values")

        if not positions:
            reasons.append("no protected positions")
        if len(set(positions)) != len(positions):
            reasons.append("protected_positions contains duplicates")
        if positions != sorted(positions):
            reasons.append("protected_positions is not sorted ascending")

        out_of_bounds = [pos for pos in positions if pos < 0 or pos >= len(sequence)]
        if out_of_bounds:
            reasons.append(
                "protected_positions out of 0-based bounds: {0}".format(out_of_bounds)
            )

        letters = []
        for pos in positions:
            if 0 <= pos < len(sequence):
                letters.append("{0}{1}".format(sequence[pos], pos))

        status = "PASS" if not reasons else "FAIL"
        if reasons:
            all_pass = False
        print("{0}\t{1}\t{2}".format(protein_id, status, ";".join(letters)))
        if reasons:
            print("  reasons: {0}".format("; ".join(reasons)))

    if all_pass:
        print("ALL_PASS")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
