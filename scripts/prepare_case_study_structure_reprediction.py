#!/usr/bin/env python3
"""Prepare and summarize structure re-prediction case studies.

The script consumes the case-study outputs produced by
generate_case_study_visualizations.py.  It exports FASTA inputs for original and
deleted sequences and, when predicted PDBs are available, computes lightweight
structure-quality summaries around motifs, deleted residues, and deletion
breakpoints.
"""

import argparse
import csv
import os
from collections import Counter


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


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def safe_float(value):
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / float(len(values)) if values else None


def fmt(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def wrap_fasta(sequence, width=80):
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def write_fasta(path, header, sequence):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write(">{}\n{}\n".format(header, wrap_fasta(sequence)))


def parse_deleted_positions(rows, method, budget):
    col = "deleted_{}_{}".format(method, budget)
    return {int(row["position"]) for row in rows if truthy(row.get(col))}


def deleted_sequence(sequence, deleted_positions):
    return "".join(aa for idx, aa in enumerate(sequence) if idx not in deleted_positions)


def deletion_blocks(deleted_positions):
    blocks = []
    ordered = sorted(deleted_positions)
    if not ordered:
        return blocks
    start = prev = ordered[0]
    for pos in ordered[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        blocks.append((start, prev))
        start = prev = pos
    blocks.append((start, prev))
    return blocks


def breakpoint_pairs(deleted_positions, length):
    pairs = []
    for start, end in deletion_blocks(deleted_positions):
        left = start - 1
        right = end + 1
        if left >= 0 and right < length:
            pairs.append((left, right))
    return pairs


def parse_pdb_ca(path):
    residues = []
    if not path or not os.path.exists(path):
        return residues
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
            chain = line[21:22]
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            key = (chain, resseq, icode)
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


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def mapped_index(original_index, deleted_positions):
    if original_index in deleted_positions:
        return None
    return original_index - sum(1 for pos in deleted_positions if pos < original_index)


def map_positions(positions, deleted_positions, predicted_len):
    mapped = []
    for pos in sorted(positions):
        idx = mapped_index(pos, deleted_positions)
        if idx is not None and 0 <= idx < predicted_len:
            mapped.append(idx)
    return mapped


def pdb_summary(pdb_path, motif_positions, deleted_positions, length):
    residues = parse_pdb_ca(pdb_path)
    if not residues:
        return {
            "prediction_status": "missing",
            "predicted_length": "",
            "mean_plddt": "",
            "motif_mean_plddt": "",
            "breakpoint_mean_ca_distance": "",
            "breakpoint_max_ca_distance": "",
        }
    motif_mapped = map_positions(motif_positions, deleted_positions, len(residues))
    breakpoint_distances = []
    for left, right in breakpoint_pairs(deleted_positions, length):
        left_idx = mapped_index(left, deleted_positions)
        right_idx = mapped_index(right, deleted_positions)
        if left_idx is None or right_idx is None:
            continue
        if 0 <= left_idx < len(residues) and 0 <= right_idx < len(residues):
            breakpoint_distances.append(dist(residues[left_idx]["coord"], residues[right_idx]["coord"]))
    return {
        "prediction_status": "success",
        "predicted_length": len(residues),
        "mean_plddt": fmt(safe_mean(row["plddt"] for row in residues)),
        "motif_mean_plddt": fmt(safe_mean(residues[idx]["plddt"] for idx in motif_mapped)),
        "breakpoint_mean_ca_distance": fmt(safe_mean(breakpoint_distances)),
        "breakpoint_max_ca_distance": fmt(max(breakpoint_distances) if breakpoint_distances else None),
    }


def resolve_prediction_path(prediction_dir, protein_id, method, budget):
    candidates = [
        os.path.join(prediction_dir, protein_id, "{}_{}_predicted.pdb".format(method, budget)),
        os.path.join(prediction_dir, protein_id, "{}_{}.pdb".format(method, budget)),
        os.path.join(prediction_dir, "{}_{}_{}.pdb".format(protein_id, method, budget)),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return ""


def build_case(case_row, case_root, args):
    protein_id = case_row["protein_id"]
    case_dir = os.path.join(case_root, protein_id)
    residue_path = os.path.join(case_dir, "residue_sets.csv")
    residue_rows = read_csv(residue_path)
    sequence = "".join(row["aa"] for row in residue_rows)
    motif_positions = {int(row["position"]) for row in residue_rows if truthy(row.get("is_motif"))}
    shadow_positions = {int(row["position"]) for row in residue_rows if truthy(row.get("is_shadow"))}

    out_case_dir = os.path.join(args.out_dir, protein_id)
    ensure_dir(out_case_dir)
    write_fasta(os.path.join(out_case_dir, "{}_original.fasta".format(protein_id)), protein_id + "|original", sequence)

    rows = []
    for method in args.methods.split(","):
        method = method.strip()
        if not method:
            continue
        for budget in [item.strip() for item in args.budgets.split(",") if item.strip()]:
            deleted = parse_deleted_positions(residue_rows, method, budget)
            if not deleted:
                continue
            new_sequence = deleted_sequence(sequence, deleted)
            header = "{}|{}|{}|deleted_len={}".format(protein_id, method, budget, len(deleted))
            fasta_path = os.path.join(out_case_dir, "{}_{}_{}.fasta".format(protein_id, method, budget))
            write_fasta(fasta_path, header, new_sequence)
            prediction_path = resolve_prediction_path(args.prediction_dir, protein_id, method, budget) if args.prediction_dir else ""
            summary = pdb_summary(prediction_path, motif_positions, deleted, len(sequence)) if prediction_path else {
                "prediction_status": "not_run",
                "predicted_length": "",
                "mean_plddt": "",
                "motif_mean_plddt": "",
                "breakpoint_mean_ca_distance": "",
                "breakpoint_max_ca_distance": "",
            }
            deleted_shadow = len(deleted & shadow_positions)
            deleted_motif = len(deleted & motif_positions)
            rows.append(
                {
                    "protein_id": protein_id,
                    "primary_protected_type": case_row.get("primary_protected_type", ""),
                    "method": method,
                    "budget": budget,
                    "original_length": len(sequence),
                    "deleted_length": len(deleted),
                    "deleted_ratio": fmt(len(deleted) / float(len(sequence))),
                    "new_length": len(new_sequence),
                    "n_deletion_blocks": len(deletion_blocks(deleted)),
                    "n_internal_breakpoints": len(breakpoint_pairs(deleted, len(sequence))),
                    "deleted_motif_residues": deleted_motif,
                    "deleted_shadow_residues": deleted_shadow,
                    "motif_shadow_deleted_rate": fmt(deleted_shadow / float(len(deleted) or 1)),
                    "fasta_path": fasta_path,
                    "prediction_pdb_path": prediction_path,
                    **summary,
                }
            )
    return rows


def write_report(path, args, rows):
    ensure_parent(path)
    status_counts = Counter(row["prediction_status"] for row in rows)
    with open(path, "w") as handle:
        handle.write("# Case Study Structure Re-Prediction Prep/Evaluation\n\n")
        handle.write("This report prepares deletion case-study FASTA files and summarizes re-predicted structures when prediction PDBs are present.\n\n")
        handle.write("## Inputs\n\n")
        handle.write("- Case-study dir: `{}`\n".format(args.case_study_dir))
        handle.write("- Prediction dir: `{}`\n".format(args.prediction_dir or "not provided"))
        handle.write("- Output dir: `{}`\n".format(args.out_dir))
        handle.write("- Rows: {}\n".format(len(rows)))
        handle.write("- Prediction status counts: {}\n\n".format(dict(status_counts)))
        handle.write("## Summary Table\n\n")
        handle.write("| Protein | Method | Budget | Deleted | Shadow deleted | Prediction | Mean pLDDT | Breakpoint CA dist |\n")
        handle.write("|---|---|---:|---:|---:|---|---:|---:|\n")
        for row in rows:
            handle.write(
                "| {protein_id} | {method} | {budget} | {deleted_length} | {deleted_shadow_residues} | {prediction_status} | {mean_plddt} | {breakpoint_mean_ca_distance} |\n".format(**row)
            )
        handle.write("\n## Next Step\n\n")
        handle.write("Run AF2/ESMFold/OmegaFold externally on the exported FASTA files, place predicted PDBs under the documented prediction directory naming scheme, and rerun this script to populate structure metrics.\n\n")
        handle.write("CASE_STUDY_STRUCTURE_REPREDICTION_READY\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare and summarize case-study structure re-prediction inputs.")
    parser.add_argument("--case_study_dir", default="results/case_studies")
    parser.add_argument("--selected_cases_csv", default="results/case_studies/selected_cases.csv")
    parser.add_argument("--prediction_dir", default="results/case_studies/repredicted_structures")
    parser.add_argument("--methods", default="shadow02,hardmask")
    parser.add_argument("--budgets", default="10,20,30")
    parser.add_argument("--out_dir", default="results/case_studies/structure_reprediction")
    parser.add_argument("--out_summary_csv", default="results/case_studies/structure_reprediction/case_study_structure_reprediction_summary.csv")
    parser.add_argument("--out_report", default="results/case_studies/structure_reprediction/CASE_STUDY_STRUCTURE_REPREDICTION_REPORT.md")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    cases = read_csv(args.selected_cases_csv)
    rows = []
    for case in cases:
        rows.extend(build_case(case, args.case_study_dir, args))
    fields = [
        "protein_id",
        "primary_protected_type",
        "method",
        "budget",
        "original_length",
        "deleted_length",
        "deleted_ratio",
        "new_length",
        "n_deletion_blocks",
        "n_internal_breakpoints",
        "deleted_motif_residues",
        "deleted_shadow_residues",
        "motif_shadow_deleted_rate",
        "fasta_path",
        "prediction_pdb_path",
        "prediction_status",
        "predicted_length",
        "mean_plddt",
        "motif_mean_plddt",
        "breakpoint_mean_ca_distance",
        "breakpoint_max_ca_distance",
    ]
    write_csv(args.out_summary_csv, rows, fields)
    write_report(args.out_report, args, rows)
    print("Wrote {}".format(args.out_summary_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
