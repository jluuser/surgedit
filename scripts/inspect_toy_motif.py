import csv
from pathlib import Path


INPUT_FASTA = Path("data/examples/toy_input.fasta")
MOTIF_CSV = Path("data/examples/toy_motif.csv")


def read_fasta(path):
    records = {}
    header = None
    chunks = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records[header] = "".join(chunks)
    return records


def parse_positions(value):
    if not value:
        return []
    return [int(token) for token in value.split(";") if token != ""]


def main():
    fasta_records = read_fasta(INPUT_FASTA)
    all_pass = True

    with open(MOTIF_CSV, newline="") as handle:
        reader = csv.DictReader(handle)
        motif_rows = list(reader)

    seen = set()
    for row in motif_rows:
        protein_id = row["protein_id"]
        sequence = row["sequence"]
        seen.add(protein_id)
        reasons = []

        if protein_id not in fasta_records:
            reasons.append("protein_id not found in FASTA")
            fasta_sequence = ""
        else:
            fasta_sequence = fasta_records[protein_id]
            if sequence != fasta_sequence:
                reasons.append("sequence does not match FASTA")

        try:
            positions = parse_positions(row["protected_positions"])
        except ValueError:
            positions = []
            reasons.append("protected_positions contains non-integer values")

        if not positions:
            reasons.append("no protected positions")

        out_of_bounds = [
            pos for pos in positions if pos < 0 or pos >= len(fasta_sequence)
        ]
        if out_of_bounds:
            reasons.append(f"protected positions out of bounds: {out_of_bounds}")

        letters = [
            f"{fasta_sequence[pos]}{pos}"
            for pos in positions
            if 0 <= pos < len(fasta_sequence)
        ]
        status = "PASS" if not reasons else "FAIL"
        if reasons:
            all_pass = False
        print(f"{protein_id}\t{status}\t{';'.join(letters)}")
        if reasons:
            print(f"  reasons: {'; '.join(reasons)}")

    missing = sorted(set(fasta_records) - seen)
    if missing:
        all_pass = False
        print(f"Missing motif rows for FASTA proteins: {','.join(missing)}")

    if all_pass:
        print("ALL_PASS")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
