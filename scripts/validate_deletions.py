import argparse
import json
import os
import re
from pathlib import Path


def read_fasta(path):
    records = []
    header = None
    chunks = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
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


def base_header(header):
    return header.split("|", 1)[0]


def get_positions(entry):
    for key in (
        "deletion_positions_zero_based",
        "deleted_positions",
        "deleted_positions_zero_based",
    ):
        if key in entry:
            return entry[key], key
    return None, None


def parse_deletion_letters(entry):
    field = entry.get("deletions_header_field", "")
    if not field:
        return []

    parsed = []
    for token in field.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"([A-Za-z])(\d+)", token)
        if match is None:
            return None
        parsed.append((match.group(1), int(match.group(2))))
    return parsed


def validate_record(header, original_seq, shrunk_seq, entry):
    reasons = []
    positions, positions_key = get_positions(entry)
    if positions is None:
        reasons.append("missing deletion positions field")
        positions = []
    elif not isinstance(positions, list):
        reasons.append(f"{positions_key} is not a list")
        positions = []
    elif not all(isinstance(pos, int) for pos in positions):
        reasons.append(f"{positions_key} contains non-integer values")
        positions = [pos for pos in positions if isinstance(pos, int)]

    if len(set(positions)) != len(positions):
        reasons.append("deleted positions contain duplicates")

    out_of_bounds = [pos for pos in positions if pos < 0 or pos >= len(original_seq)]
    if out_of_bounds:
        reasons.append(f"deleted positions out of bounds: {out_of_bounds}")

    if positions != sorted(positions):
        reasons.append("deleted positions are not sorted ascending")

    valid_positions = set(pos for pos in positions if 0 <= pos < len(original_seq))
    reconstructed = "".join(
        residue for index, residue in enumerate(original_seq) if index not in valid_positions
    )
    if reconstructed != shrunk_seq:
        reasons.append("sequence reconstructed from deletions does not match shrunk FASTA")

    parsed_letters = parse_deletion_letters(entry)
    if parsed_letters is None:
        if entry.get("deletions_header_field", ""):
            reasons.append("deletions_header_field is not parseable")
        else:
            reasons.append("missing deletions_header_field")
    else:
        parsed_positions = [pos for _, pos in parsed_letters]
        if parsed_positions != positions:
            reasons.append("positions in deletions_header_field do not match JSON positions")

        letter_mismatches = []
        for residue, pos in parsed_letters:
            if 0 <= pos < len(original_seq) and original_seq[pos] != residue:
                letter_mismatches.append(f"{residue}{pos}!=original:{original_seq[pos]}")
        if letter_mismatches:
            reasons.append(
                "deleted residue letters do not match original sequence: "
                + ",".join(letter_mismatches)
            )

    status = "PASS" if not reasons else "FAIL"
    return {
        "header": header,
        "status": status,
        "original_length": len(original_seq),
        "shrunk_length": len(shrunk_seq),
        "deletion_count": len(positions),
        "reasons": reasons,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate SCISOR deletion metadata.")
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--shrunk_fasta", required=True)
    parser.add_argument("--deletions_json", required=True)
    parser.add_argument("--out_report", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    original_records = {base_header(header): seq for header, seq in read_fasta(args.input_fasta)}
    shrunk_records = {base_header(header): seq for header, seq in read_fasta(args.shrunk_fasta)}
    with open(args.deletions_json) as handle:
        deletion_entries = json.load(handle)
    deletion_by_header = {entry.get("header"): entry for entry in deletion_entries}

    all_headers = list(original_records)
    report_rows = []
    for header in all_headers:
        reasons = []
        original_seq = original_records[header]
        shrunk_seq = shrunk_records.get(header)
        entry = deletion_by_header.get(header)
        if shrunk_seq is None:
            reasons.append("missing corresponding shrunk FASTA record")
        if entry is None:
            reasons.append("missing corresponding deletion JSON entry")
        if reasons:
            report_rows.append(
                {
                    "header": header,
                    "status": "FAIL",
                    "original_length": len(original_seq),
                    "shrunk_length": "NA" if shrunk_seq is None else len(shrunk_seq),
                    "deletion_count": "NA",
                    "reasons": reasons,
                }
            )
            continue

        report_rows.append(validate_record(header, original_seq, shrunk_seq, entry))

    extra_shrunk = sorted(set(shrunk_records) - set(original_records))
    extra_json = sorted(set(deletion_by_header) - set(original_records))
    for header in extra_shrunk:
        report_rows.append(
            {
                "header": header,
                "status": "FAIL",
                "original_length": "NA",
                "shrunk_length": len(shrunk_records[header]),
                "deletion_count": "NA",
                "reasons": ["shrunk FASTA record has no original FASTA record"],
            }
        )
    for header in extra_json:
        report_rows.append(
            {
                "header": str(header),
                "status": "FAIL",
                "original_length": "NA",
                "shrunk_length": "NA",
                "deletion_count": "NA",
                "reasons": ["deletion JSON entry has no original FASTA record"],
            }
        )

    out_path = Path(args.out_report)
    if out_path.parent:
        os.makedirs(out_path.parent, exist_ok=True)

    all_pass = all(row["status"] == "PASS" for row in report_rows)
    with open(out_path, "w") as handle:
        handle.write("SCISOR deletion validation report\n")
        handle.write(f"input_fasta: {args.input_fasta}\n")
        handle.write(f"shrunk_fasta: {args.shrunk_fasta}\n")
        handle.write(f"deletions_json: {args.deletions_json}\n\n")
        handle.write(
            "header\tstatus\toriginal_length\tshrunk_length\tdeletion_count\treasons\n"
        )
        for row in report_rows:
            reasons = "OK" if not row["reasons"] else "; ".join(row["reasons"])
            handle.write(
                f"{row['header']}\t{row['status']}\t{row['original_length']}\t"
                f"{row['shrunk_length']}\t{row['deletion_count']}\t{reasons}\n"
            )
        if all_pass:
            handle.write("\nALL_PASS\n")

    print(f"Wrote validation report to {out_path}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
