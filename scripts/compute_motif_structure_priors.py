from __future__ import print_function

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


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


def open_csv_read(path):
    if sys.version_info[0] >= 3:
        return open(path, "r", newline="")
    return open(path, "rb")


def open_csv_write(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    if sys.version_info[0] >= 3:
        return open(path, "w", newline="")
    return open(path, "wb")


def parse_positions(value):
    if value is None or value == "":
        return []
    return [int(token) for token in value.split(";") if token != ""]


def read_motif_csv(path):
    rows = []
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            row["length_int"] = int(row["length"])
            row["protected_positions_list"] = parse_positions(row["protected_positions"])
            rows.append(row)
    return rows


def read_structure_index(path):
    rows = {}
    counts = defaultdict(int)
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            rows[row["accession"]] = row
            counts[row.get("status", "")] += 1
    return rows, counts


def parse_pdb_ca(path):
    residues = []
    seen = set()
    with open(path, "rb") as handle:
        for raw_line in handle:
            if not raw_line.startswith(b"ATOM"):
                continue
            line = raw_line.decode("ascii", "ignore")
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            resname = line[17:20].strip()
            chain = line[21:22]
            resseq = line[22:26].strip()
            icode = line[26:27]
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            aa = AA3_TO_AA1.get(resname, "X")
            residues.append(
                {
                    "aa": aa,
                    "coord": (
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ),
                    "plddt": float(line[60:66]),
                }
            )
    return residues


def distance(coord_a, coord_b):
    dx = coord_a[0] - coord_b[0]
    dy = coord_a[1] - coord_b[1]
    dz = coord_a[2] - coord_b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_residue_features(protein_id, sequence, motif_positions, residues, cutoff):
    n = len(residues)
    motif_set = set(motif_positions)
    coords = [residue["coord"] for residue in residues]
    dist_to_motif = [float("inf")] * n
    contact_density = [0] * n
    motif_contact_count = [0] * n

    for pos in motif_set:
        dist_to_motif[pos] = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            d = distance(coords[i], coords[j])
            if d <= cutoff:
                contact_density[i] += 1
                contact_density[j] += 1
                if j in motif_set:
                    motif_contact_count[i] += 1
                if i in motif_set:
                    motif_contact_count[j] += 1
            if j in motif_set and d < dist_to_motif[i]:
                dist_to_motif[i] = d
            if i in motif_set and d < dist_to_motif[j]:
                dist_to_motif[j] = d

    rows = []
    for i, residue in enumerate(residues):
        is_motif = i in motif_set
        min_dist = dist_to_motif[i]
        is_shadow = (not is_motif) and min_dist <= cutoff
        rows.append(
            {
                "protein_id": protein_id,
                "position": i,
                "aa": sequence[i],
                "is_motif": str(is_motif),
                "distance_to_motif": "{0:.4f}".format(min_dist),
                "is_motif_shadow_8A": str(is_shadow),
                "contact_density_8A": contact_density[i],
                "motif_contact_count_8A": motif_contact_count[i],
                "plddt": "{0:.2f}".format(residue["plddt"]),
                "structure_status": "success",
            }
        )
    return rows


def mean(values):
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def summarize_success(protein_id, length, motif_positions, residue_rows):
    nonmotif_distances = [
        float(row["distance_to_motif"])
        for row in residue_rows
        if row["is_motif"] == "False"
    ]
    num_shadow = sum(1 for row in residue_rows if row["is_motif_shadow_8A"] == "True")
    return {
        "protein_id": protein_id,
        "length": length,
        "num_motif": len(motif_positions),
        "num_shadow_8A": num_shadow,
        "shadow_fraction": "{0:.6f}".format(float(num_shadow) / length if length else 0.0),
        "mean_contact_density_8A": "{0:.6f}".format(
            mean([int(row["contact_density_8A"]) for row in residue_rows])
        ),
        "mean_motif_contact_count_8A": "{0:.6f}".format(
            mean([int(row["motif_contact_count_8A"]) for row in residue_rows])
        ),
        "mean_plddt": "{0:.6f}".format(
            mean([float(row["plddt"]) for row in residue_rows])
        ),
        "min_distance_to_motif_nonmotif": "{0:.6f}".format(
            min(nonmotif_distances) if nonmotif_distances else 0.0
        ),
        "status": "success",
        "reason": "",
    }


def summarize_skipped(protein_id, length, motif_positions, status, reason):
    return {
        "protein_id": protein_id,
        "length": length,
        "num_motif": len(motif_positions),
        "num_shadow_8A": "",
        "shadow_fraction": "",
        "mean_contact_density_8A": "",
        "mean_motif_contact_count_8A": "",
        "mean_plddt": "",
        "min_distance_to_motif_nonmotif": "",
        "status": status,
        "reason": reason,
    }


def write_csv(path, fields, rows):
    with open_csv_write(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, motif_count, index_counts, residue_rows, summary_rows):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)

    status_counts = defaultdict(int)
    for row in summary_rows:
        status_counts[row["status"]] += 1
    success_rows = [row for row in summary_rows if row["status"] == "success"]
    success_count = len(success_rows)

    def success_float_values(key):
        return [float(row[key]) for row in success_rows if row[key] != ""]

    with open(path, "w") as handle:
        handle.write("Swiss-Prot diverse structure prior report\n\n")
        handle.write("motif_csv total proteins: {0}\n".format(motif_count))
        handle.write("structure_index success: {0}\n".format(index_counts.get("success", 0)))
        handle.write("structure_index exists: {0}\n".format(index_counts.get("exists", 0)))
        handle.write("structure_index failed: {0}\n".format(index_counts.get("failed", 0)))
        handle.write("successfully computed structure priors: {0}\n".format(success_count))
        handle.write("no_structure skipped: {0}\n".format(status_counts["no_structure"]))
        handle.write(
            "length_or_sequence_mismatch skipped: {0}\n".format(
                status_counts["length_or_sequence_mismatch"]
            )
        )
        handle.write(
            "motif_position_oob skipped: {0}\n".format(status_counts["motif_position_oob"])
        )
        handle.write("residue-level CSV rows: {0}\n".format(len(residue_rows)))
        handle.write(
            "average num_motif per successful protein: {0:.6f}\n".format(
                mean([int(row["num_motif"]) for row in success_rows])
            )
        )
        handle.write(
            "average num_shadow_8A per successful protein: {0:.6f}\n".format(
                mean([int(row["num_shadow_8A"]) for row in success_rows])
            )
        )
        handle.write(
            "average shadow_fraction: {0:.6f}\n".format(
                mean(success_float_values("shadow_fraction"))
            )
        )
        handle.write(
            "average contact_density_8A: {0:.6f}\n".format(
                mean(success_float_values("mean_contact_density_8A"))
            )
        )
        handle.write(
            "average pLDDT: {0:.6f}\n\n".format(mean(success_float_values("mean_plddt")))
        )
        handle.write("First 10 success proteins:\n")
        handle.write(
            "protein_id\tlength\tnum_motif\tnum_shadow_8A\tshadow_fraction\tmean_plddt\n"
        )
        for row in success_rows[:10]:
            handle.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                    row["protein_id"],
                    row["length"],
                    row["num_motif"],
                    row["num_shadow_8A"],
                    row["shadow_fraction"],
                    row["mean_plddt"],
                )
            )
        handle.write("\n")
        if success_count >= 50:
            handle.write("STRUCTURE_PRIOR_COMPUTE_PASS\n")
        else:
            handle.write("STRUCTURE_PRIOR_COMPUTE_WARN\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute first-pass motif structure priors.")
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--structure_index", required=True)
    parser.add_argument("--out_residue_csv", required=True)
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--distance_cutoff", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    motif_rows = read_motif_csv(args.motif_csv)
    structure_index, index_counts = read_structure_index(args.structure_index)

    residue_rows = []
    summary_rows = []
    for motif in motif_rows:
        protein_id = motif["protein_id"]
        accession = motif.get("accession") or protein_id
        sequence = motif["sequence"]
        length = len(sequence)
        motif_positions = motif["protected_positions_list"]

        if any(pos < 0 or pos >= length for pos in motif_positions):
            summary_rows.append(
                summarize_skipped(
                    protein_id,
                    length,
                    motif_positions,
                    "motif_position_oob",
                    "protected position outside sequence length",
                )
            )
            continue

        index_row = structure_index.get(accession)
        if index_row is None or index_row.get("status") not in ("success", "exists"):
            reason = "structure missing or failed"
            if index_row is not None and index_row.get("error"):
                reason = index_row.get("error")
            summary_rows.append(
                summarize_skipped(
                    protein_id, length, motif_positions, "no_structure", reason
                )
            )
            continue

        pdb_path = index_row.get("structure_path", "")
        if not pdb_path or not os.path.exists(pdb_path) or os.path.getsize(pdb_path) == 0:
            summary_rows.append(
                summarize_skipped(
                    protein_id,
                    length,
                    motif_positions,
                    "no_structure",
                    "PDB missing or empty: {0}".format(pdb_path),
                )
            )
            continue

        residues = parse_pdb_ca(pdb_path)
        pdb_sequence = "".join(residue["aa"] for residue in residues)
        if len(residues) != length or pdb_sequence != sequence:
            summary_rows.append(
                summarize_skipped(
                    protein_id,
                    length,
                    motif_positions,
                    "length_or_sequence_mismatch",
                    "PDB CA length {0}; CSV length {1}; sequence_match={2}".format(
                        len(residues), length, pdb_sequence == sequence
                    ),
                )
            )
            continue

        protein_residue_rows = compute_residue_features(
            protein_id, sequence, motif_positions, residues, args.distance_cutoff
        )
        residue_rows.extend(protein_residue_rows)
        summary_rows.append(
            summarize_success(protein_id, length, motif_positions, protein_residue_rows)
        )

    residue_fields = [
        "protein_id",
        "position",
        "aa",
        "is_motif",
        "distance_to_motif",
        "is_motif_shadow_8A",
        "contact_density_8A",
        "motif_contact_count_8A",
        "plddt",
        "structure_status",
    ]
    summary_fields = [
        "protein_id",
        "length",
        "num_motif",
        "num_shadow_8A",
        "shadow_fraction",
        "mean_contact_density_8A",
        "mean_motif_contact_count_8A",
        "mean_plddt",
        "min_distance_to_motif_nonmotif",
        "status",
        "reason",
    ]
    write_csv(args.out_residue_csv, residue_fields, residue_rows)
    write_csv(args.out_summary_csv, summary_fields, summary_rows)

    report_path = os.path.splitext(args.out_summary_csv)[0].replace(
        "_summary", "_report"
    ) + ".txt"
    report_path = os.path.join(
        os.path.dirname(args.out_summary_csv),
        "swissprot_diverse_structure_prior_report.txt",
    )
    write_report(report_path, len(motif_rows), index_counts, residue_rows, summary_rows)

    success_count = sum(1 for row in summary_rows if row["status"] == "success")
    print("Processed proteins: {0}".format(len(motif_rows)))
    print("Successfully computed: {0}".format(success_count))
    print("Residue rows: {0}".format(len(residue_rows)))
    print("Wrote {0}".format(args.out_residue_csv))
    print("Wrote {0}".format(args.out_summary_csv))
    print("Wrote {0}".format(report_path))


if __name__ == "__main__":
    main()
