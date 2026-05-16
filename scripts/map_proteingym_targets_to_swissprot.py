#!/usr/bin/env python3
"""Map ProteinGym deletion benchmark targets to Swiss-Prot by exact sequence."""

import argparse
import csv
import gzip
import hashlib
import os
from collections import Counter, defaultdict


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def sequence_key(sequence):
    return hashlib.sha1(sequence.encode("ascii")).hexdigest()


def parse_fasta_header(header):
    # Expected Swiss-Prot header:
    # >sp|Q6GZX4|001R_FRG3G Putative transcription factor ...
    raw = header[1:] if header.startswith(">") else header
    parts = raw.split(None, 1)
    ident = parts[0]
    description = parts[1] if len(parts) > 1 else ""
    fields = ident.split("|")
    if len(fields) >= 3:
        return {
            "db": fields[0],
            "accession": fields[1],
            "entry_name": fields[2],
            "description": description,
        }
    return {"db": "", "accession": ident, "entry_name": "", "description": description}


def iter_fasta(path):
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
                header = line
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def load_deletion_rows(path):
    rows = []
    by_sequence = {}
    stats = defaultdict(lambda: {"row_count": 0, "sources": Counter(), "assays": Counter(), "protein_ids": Counter()})
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        for row in reader:
            rows.append(row)
            seq = row["target_sequence"]
            by_sequence[seq] = seq
            item = stats[seq]
            item["row_count"] += 1
            item["sources"][row.get("source", "")] += 1
            if row.get("assay_id"):
                item["assays"][row["assay_id"]] += 1
            if row.get("protein_id"):
                item["protein_ids"][row["protein_id"]] += 1
    return rows, fields, by_sequence, stats


def load_bioprior_index(path):
    index = defaultdict(list)
    if not path or not os.path.exists(path):
        return index
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = row.get("sequence", "")
            if not seq:
                continue
            index[seq].append({
                "accession": row.get("accession") or row.get("protein_id", ""),
                "entry_name": row.get("entry_name", ""),
                "protein_name": row.get("protein_name", ""),
            })
    return index


def load_afdb_paths(mapping_csv):
    paths = {}
    if not mapping_csv or not os.path.exists(mapping_csv):
        return paths
    with open(mapping_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession", "")
            if not accession:
                continue
            ok = str(row.get("found", "")).lower() == "true"
            paths[accession] = {
                "found": ok,
                "structure_path": row.get("structure_path", ""),
            }
    return paths


def map_swissprot(target_sequences, fasta_path):
    target_set = set(target_sequences)
    matches = defaultdict(list)
    scanned = 0
    for header, sequence in iter_fasta(fasta_path):
        scanned += 1
        if sequence not in target_set:
            continue
        parsed = parse_fasta_header(header)
        matches[sequence].append(parsed)
    return matches, scanned


def join_counter(counter, max_items=None):
    items = counter.most_common(max_items)
    return ";".join("{}:{}".format(key, value) for key, value in items if key)


def join_values(values):
    return ";".join(str(value) for value in values if value)


def build_mapping_rows(target_sequences, stats, swiss_matches, bioprior_index, afdb_paths):
    rows = []
    for idx, sequence in enumerate(sorted(target_sequences, key=lambda seq: (len(seq), seq)), start=1):
        seq_stats = stats[sequence]
        swiss = swiss_matches.get(sequence, [])
        bioprior = bioprior_index.get(sequence, [])
        swiss_accessions = [row["accession"] for row in swiss]
        bioprior_accessions = [row["accession"] for row in bioprior]
        afdb_hits = [afdb_paths.get(acc, {}) for acc in bioprior_accessions]
        afdb_found = any(hit.get("found") for hit in afdb_hits)
        afdb_paths_joined = join_values(hit.get("structure_path", "") for hit in afdb_hits if hit.get("found"))
        rows.append({
            "target_id": "proteingym_target_{:05d}".format(idx),
            "target_sha1": sequence_key(sequence),
            "target_length": len(sequence),
            "row_count": seq_stats["row_count"],
            "sources": join_counter(seq_stats["sources"]),
            "top_assays": join_counter(seq_stats["assays"], max_items=10),
            "clinical_protein_ids": join_counter(seq_stats["protein_ids"], max_items=10),
            "swissprot_exact_match": bool(swiss),
            "swissprot_n_matches": len(swiss),
            "swissprot_accessions": join_values(swiss_accessions),
            "swissprot_entry_names": join_values(row["entry_name"] for row in swiss),
            "swissprot_first_description": swiss[0]["description"] if swiss else "",
            "bioprior10k_exact_match": bool(bioprior),
            "bioprior10k_n_matches": len(bioprior),
            "bioprior10k_accessions": join_values(bioprior_accessions),
            "bioprior10k_entry_names": join_values(row["entry_name"] for row in bioprior),
            "bioprior10k_afdb_found": afdb_found,
            "bioprior10k_afdb_paths": afdb_paths_joined,
        })
    return rows


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_annotated_deletions(path, input_rows, input_fields, mapping_by_sequence):
    added_fields = [
        "target_sha1",
        "swissprot_exact_match",
        "swissprot_accessions",
        "swissprot_entry_names",
        "bioprior10k_exact_match",
        "bioprior10k_accessions",
        "bioprior10k_afdb_found",
        "bioprior10k_afdb_paths",
    ]
    fields = input_fields + [field for field in added_fields if field not in input_fields]
    out_rows = []
    for row in input_rows:
        mapped = mapping_by_sequence[row["target_sequence"]]
        out = dict(row)
        for field in added_fields:
            out[field] = mapped[field]
        out_rows.append(out)
    write_csv(path, out_rows, fields)


def write_summary(path, args, target_rows, input_rows, swiss_scanned):
    ensure_parent(path)
    unique_targets = len(target_rows)
    swiss_targets = sum(1 for row in target_rows if row["swissprot_exact_match"])
    bioprior_targets = sum(1 for row in target_rows if row["bioprior10k_exact_match"])
    afdb_targets = sum(1 for row in target_rows if row["bioprior10k_afdb_found"])
    rows_with_swiss = sum(int(row["row_count"]) for row in target_rows if row["swissprot_exact_match"])
    rows_with_bioprior = sum(int(row["row_count"]) for row in target_rows if row["bioprior10k_exact_match"])
    rows_with_afdb = sum(int(row["row_count"]) for row in target_rows if row["bioprior10k_afdb_found"])
    by_source = Counter(row.get("source", "") for row in input_rows)

    with open(path, "w") as handle:
        handle.write("ProteinGym target-to-SwissProt exact mapping summary\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("swissprot_fasta: {}\n".format(args.swissprot_fasta))
        handle.write("bioprior_csv: {}\n".format(args.bioprior_csv))
        handle.write("afdb_mapping_csv: {}\n".format(args.afdb_mapping_csv))
        handle.write("out_targets_csv: {}\n".format(args.out_targets_csv))
        handle.write("out_annotated_csv: {}\n\n".format(args.out_annotated_csv))
        handle.write("input_rows: {}\n".format(len(input_rows)))
        handle.write("input_rows_by_source: {}\n".format(dict(by_source)))
        handle.write("unique_targets: {}\n".format(unique_targets))
        handle.write("swissprot_records_scanned: {}\n\n".format(swiss_scanned))
        handle.write("Exact target matches:\n")
        handle.write("- Swiss-Prot targets: {} / {}\n".format(swiss_targets, unique_targets))
        handle.write("- Swiss-Prot rows: {} / {}\n".format(rows_with_swiss, len(input_rows)))
        handle.write("- BioPrior-10K targets: {} / {}\n".format(bioprior_targets, unique_targets))
        handle.write("- BioPrior-10K rows: {} / {}\n".format(rows_with_bioprior, len(input_rows)))
        handle.write("- BioPrior-10K AFDB targets: {} / {}\n".format(afdb_targets, unique_targets))
        handle.write("- BioPrior-10K AFDB rows: {} / {}\n\n".format(rows_with_afdb, len(input_rows)))
        handle.write("Notes:\n")
        handle.write("- This script only performs exact full-sequence matching.\n")
        handle.write("- Unmatched rows may still be mappable with assay metadata, isoform handling, or sequence alignment.\n")
        handle.write("- BioPrior-10K exact matches are expected to be rare because ProteinGym targets are external benchmark proteins.\n\n")
        handle.write("PROTEINGYM_SWISSPROT_MAPPING_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Map ProteinGym deletion targets to Swiss-Prot by exact sequence.")
    parser.add_argument("--input_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored.csv")
    parser.add_argument("--swissprot_fasta", default="data/raw/uniprot/uniprot_sprot.fasta.gz")
    parser.add_argument("--bioprior_csv", default="data/processed/swissprot_motif_bioprior_10k.csv")
    parser.add_argument("--afdb_mapping_csv", default="data/structures/afdb_bioprior_10k_mapping.csv")
    parser.add_argument("--out_targets_csv", default="results/proteingym_deletion_benchmark/proteingym_target_swissprot_mapping.csv")
    parser.add_argument("--out_annotated_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_mapped.csv")
    parser.add_argument("--summary_txt", default="results/proteingym_deletion_benchmark/proteingym_target_swissprot_mapping_summary.txt")
    args = parser.parse_args()

    input_rows, input_fields, target_sequences, stats = load_deletion_rows(args.input_csv)
    swiss_matches, swiss_scanned = map_swissprot(target_sequences, args.swissprot_fasta)
    bioprior_index = load_bioprior_index(args.bioprior_csv)
    afdb_paths = load_afdb_paths(args.afdb_mapping_csv)
    target_rows = build_mapping_rows(target_sequences, stats, swiss_matches, bioprior_index, afdb_paths)
    target_fields = list(target_rows[0].keys()) if target_rows else []
    write_csv(args.out_targets_csv, target_rows, target_fields)
    mapping_by_sequence = {row["target_sha1"]: row for row in target_rows}
    sequence_to_row = {seq: mapping_by_sequence[sequence_key(seq)] for seq in target_sequences}
    write_annotated_deletions(args.out_annotated_csv, input_rows, input_fields, sequence_to_row)
    write_summary(args.summary_txt, args, target_rows, input_rows, swiss_scanned)
    print("Wrote {}".format(args.out_targets_csv))
    print("Wrote {}".format(args.out_annotated_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
