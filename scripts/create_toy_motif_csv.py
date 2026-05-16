import csv
from pathlib import Path


INPUT_FASTA = Path("data/examples/toy_input.fasta")
OUTPUT_CSV = Path("data/examples/toy_motif.csv")


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


def choose_mid_positions(length, count=5):
    start = max(1, int(length * 0.2))
    end = min(length - 2, int(length * 0.8))
    if end < start:
        start, end = 0, length - 1

    if count == 1:
        return [length // 2]

    span = end - start
    positions = [round(start + span * i / (count - 1)) for i in range(count)]
    positions = sorted(set(max(0, min(length - 1, pos)) for pos in positions))

    candidate = 0
    while len(positions) < count and candidate < length:
        if candidate not in positions:
            positions.append(candidate)
        candidate += 1
    return sorted(positions[:count])


def main():
    records = read_fasta(INPUT_FASTA)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protein_id",
                "sequence",
                "protected_positions",
                "protected_type",
                "structure_path",
            ],
        )
        writer.writeheader()
        for protein_id, sequence in records:
            positions = choose_mid_positions(len(sequence), count=5)
            writer.writerow(
                {
                    "protein_id": protein_id,
                    "sequence": sequence,
                    "protected_positions": ";".join(str(pos) for pos in positions),
                    "protected_type": "toy_motif",
                    "structure_path": "",
                }
            )

    print(f"Wrote {len(records)} toy motif records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
