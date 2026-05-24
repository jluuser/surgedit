#!/usr/bin/env python3
"""Run ESMFold on current-best deleted case-study sequences and summarize them."""

import argparse
import csv
import os
from collections import defaultdict

import torch


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    ensure_parent(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value, default=0.0):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_positions(text):
    if not text:
        return set()
    return {int(float(token)) for token in str(text).split(";") if token.strip() != ""}


def parse_blocks(text):
    blocks = []
    for token in str(text or "").split(";"):
        token = token.strip()
        if not token:
            continue
        start, end = token.split("-", 1)
        blocks.append((int(float(start)), int(float(end))))
    return blocks


def read_fasta(path):
    header = ""
    seq = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
            else:
                seq.append(line)
    return header, "".join(seq)


def read_core_protected(path):
    protected = {}
    for row in read_csv(path):
        accession = row.get("accession") or row.get("protein_id")
        if accession:
            protected[accession] = parse_positions(row.get("protected_positions", ""))
    return protected


def deleted_positions_from_blocks(blocks):
    deleted = set()
    for start, end in blocks:
        deleted.update(range(start, end + 1))
    return deleted


def mapped_index(original_index, deleted_positions):
    if original_index in deleted_positions:
        return None
    return original_index - sum(1 for pos in deleted_positions if pos < original_index)


def parse_pdb_ca(path):
    residues = []
    seen = set()
    with open(path, "rb") as handle:
        for raw in handle:
            if not raw.startswith(b"ATOM"):
                continue
            line = raw.decode("ascii", "ignore")
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            key = (line[21:22], line[22:26].strip(), line[26:27].strip())
            if key in seen:
                continue
            seen.add(key)
            try:
                residues.append(
                    {
                        "aa": AA3_TO_AA1.get(line[17:20].strip().upper(), "X"),
                        "coord": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                        "plddt": float(line[60:66]),
                    }
                )
            except ValueError:
                continue
    return residues


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / float(len(values)) if values else None


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def fmt(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def structure_metrics(pdb_path, original_length, blocks, protected_positions):
    residues = parse_pdb_ca(pdb_path)
    deleted = deleted_positions_from_blocks(blocks)
    protected_mapped = []
    for pos in sorted(protected_positions):
        idx = mapped_index(pos, deleted)
        if idx is not None and 0 <= idx < len(residues):
            protected_mapped.append(idx)

    breakpoint_distances = []
    for start, end in blocks:
        left = start - 1
        right = end + 1
        if left < 0 or right >= original_length:
            continue
        left_idx = mapped_index(left, deleted)
        right_idx = mapped_index(right, deleted)
        if left_idx is None or right_idx is None:
            continue
        if 0 <= left_idx < len(residues) and 0 <= right_idx < len(residues):
            breakpoint_distances.append(dist(residues[left_idx]["coord"], residues[right_idx]["coord"]))

    return {
        "predicted_length": len(residues),
        "mean_plddt": fmt(mean([row["plddt"] for row in residues])),
        "min_plddt": fmt(min([row["plddt"] for row in residues]) if residues else None),
        "protected_positions_mapped": len(protected_mapped),
        "protected_mean_plddt": fmt(mean([residues[idx]["plddt"] for idx in protected_mapped])),
        "breakpoint_pairs": len(breakpoint_distances),
        "breakpoint_mean_ca_distance": fmt(mean(breakpoint_distances)),
        "breakpoint_max_ca_distance": fmt(max(breakpoint_distances) if breakpoint_distances else None),
    }


def load_model(device, chunk_size):
    import esm

    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if hasattr(model, "set_chunk_size"):
        model.set_chunk_size(chunk_size)
    model = model.to(device)
    return model


def run(args):
    ensure_dir(args.prediction_dir)
    cases = read_csv(args.case_csv)
    protected_by_accession = read_core_protected(args.core_csv)
    device = torch.device(args.device)
    model = load_model(device, args.chunk_size)
    rows = []
    for row in cases:
        accession = row["accession"]
        _, sequence = read_fasta(row["deleted_fasta"])
        out_pdb = os.path.join(args.prediction_dir, "{}_deleted_esmfold.pdb".format(accession))
        if not (args.skip_existing and os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0):
            with torch.no_grad():
                pdb = model.infer_pdb(sequence)
            ensure_parent(out_pdb)
            with open(out_pdb, "w") as handle:
                handle.write(pdb)
        metrics = structure_metrics(
            out_pdb,
            int(float(row["original_length"])),
            parse_blocks(row["deletion_blocks_0based"]),
            protected_by_accession.get(accession, set()),
        )
        out = dict(row)
        out.update(metrics)
        out.update(
            {
                "prediction_status": "success",
                "prediction_model": "esmfold_v1",
                "prediction_pdb": out_pdb,
                "device": str(device),
            }
        )
        rows.append(out)
    write_csv(args.out_summary_csv, rows)
    write_report(args.out_report, args, rows)
    print("Wrote {}".format(args.out_summary_csv))
    print("Wrote {}".format(args.out_report))


def write_report(path, args, rows):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("# Current-Best ESMFold Case Studies\n\n")
        handle.write("Deleted current-best case-study sequences were re-predicted with ESMFold v1.\n\n")
        handle.write("- case_csv: `{}`\n".format(args.case_csv))
        handle.write("- prediction_dir: `{}`\n".format(args.prediction_dir))
        handle.write("- device: `{}`\n\n".format(args.device))
        handle.write("| Accession | Deleted Len | Mean pLDDT | Protected pLDDT | Breakpoint Mean CA | Breakpoint Max CA |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                "| {accession} | {new_length} | {mean_plddt} | {protected_mean_plddt} | {breakpoint_mean_ca_distance} | {breakpoint_max_ca_distance} |\n".format(
                    **row
                )
            )
        handle.write("\nCURRENT_BEST_ESMFOLD_CASE_STUDIES_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ESMFold current-best case-study predictions.")
    parser.add_argument("--case_csv", default="results/current_best_experiments/experiment6_case_studies/current_best_case_studies.csv")
    parser.add_argument("--core_csv", default="data/processed/swissprot_motif_bioprior_10k.csv")
    parser.add_argument("--prediction_dir", default="results/current_best_experiments/experiment6_case_studies/esmfold_predictions")
    parser.add_argument("--out_summary_csv", default="results/current_best_experiments/experiment6_case_studies/current_best_esmfold_structure_summary.csv")
    parser.add_argument("--out_report", default="results/current_best_experiments/experiment6_case_studies/CURRENT_BEST_ESMFOLD_CASE_STUDIES_REPORT.md")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
