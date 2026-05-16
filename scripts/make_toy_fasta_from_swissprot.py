import gzip
from pathlib import Path

in_fasta = Path("data/raw/uniprot/uniprot_sprot.fasta.gz")
out_fasta = Path("data/examples/toy_input.fasta")

standard_aa = set("ACDEFGHIKLMNPQRSTVWY")

def read_fasta_gz(path):
    header = None
    seq_chunks = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)

selected = []
for header, seq in read_fasta_gz(in_fasta):
    if 100 <= len(seq) <= 400 and set(seq).issubset(standard_aa):
        selected.append((header, seq))
    if len(selected) >= 5:
        break

out_fasta.parent.mkdir(parents=True, exist_ok=True)

with open(out_fasta, "w") as out:
    for header, seq in selected:
        parts = header.split("|")
        if len(parts) >= 3:
            protein_id = parts[1]
            name = parts[2].split()[0]
            simple_header = f">{protein_id}_{name}"
        else:
            simple_header = header.split()[0]

        out.write(simple_header + "\n")
        for i in range(0, len(seq), 80):
            out.write(seq[i:i+80] + "\n")

print(f"Saved {len(selected)} sequences to {out_fasta}")
for h, s in selected:
    print(h.split()[0], "length =", len(s))
