#!/usr/bin/env python3
"""Compute residue-level BioPrior features for the Swiss-Prot Core-1K dataset."""

from __future__ import print_function

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    np = None


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


OPTIONAL_FEATURE_COLUMNS = [
    "is_domain",
    "is_region",
    "is_repeat",
    "is_transmembrane",
    "is_signal_peptide",
    "is_disulfide_residue",
    "is_modified_residue",
    "is_glycosylation_site",
    "is_lipidation_site",
]


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


def read_core_csv(path):
    rows = []
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            accession = row.get("accession") or row.get("protein_id")
            row["accession"] = accession
            row["length_int"] = int(row["length"])
            row["protected_positions_list"] = parse_positions(row["protected_positions"])
            rows.append(row)
    return rows


def read_mapping_csv(path):
    rows = {}
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            rows[row["accession"]] = row
    return rows


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
            residues.append(
                {
                    "aa": AA3_TO_AA1.get(resname, "X"),
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


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    idx = (len(values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def terminal_region_flag(index, length):
    terminal_window = max(20, int(math.ceil(length * 0.05)))
    return index < terminal_window or index >= length - terminal_window


def compute_residue_rows(accession, sequence, protected_positions, protected_types, residues, shadow_cutoff, contact_cutoff):
    n = len(residues)
    protected_set = set(protected_positions)
    coords = [residue["coord"] for residue in residues]
    if np is not None:
        coord_array = np.asarray(coords, dtype=np.float32)
        diff = coord_array[:, None, :] - coord_array[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        eye = np.eye(n, dtype=bool)
        contact_density = np.sum((dists <= contact_cutoff) & (~eye), axis=1).astype(int).tolist()
        protected_mask = np.zeros(n, dtype=bool)
        for pos in protected_set:
            protected_mask[pos] = True
        if protected_positions:
            shadow_contacts = (dists <= shadow_cutoff)[:, protected_mask]
            motif_contact_count = np.sum(shadow_contacts, axis=1).astype(int)
            for pos in protected_set:
                motif_contact_count[pos] -= 1
            motif_contact_count = motif_contact_count.tolist()
            dist_to_protected = np.min(dists[:, protected_mask], axis=1).astype(float).tolist()
            for pos in protected_set:
                dist_to_protected[pos] = 0.0
        else:
            motif_contact_count = [0] * n
            dist_to_protected = [float("inf")] * n
    else:
        dist_to_protected = [float("inf")] * n
        contact_density = [0] * n
        motif_contact_count = [0] * n

        for pos in protected_set:
            dist_to_protected[pos] = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                d = distance(coords[i], coords[j])
                if d <= contact_cutoff:
                    contact_density[i] += 1
                    contact_density[j] += 1
                if d <= shadow_cutoff:
                    if j in protected_set:
                        motif_contact_count[i] += 1
                    if i in protected_set:
                        motif_contact_count[j] += 1
                if j in protected_set and d < dist_to_protected[i]:
                    dist_to_protected[i] = d
                if i in protected_set and d < dist_to_protected[j]:
                    dist_to_protected[j] = d

    rows = []
    for i, residue in enumerate(residues):
        is_protected = i in protected_set
        dist = dist_to_protected[i]
        is_shadow = (not is_protected) and dist <= shadow_cutoff
        plddt = residue["plddt"]
        row = {
            "accession": accession,
            "residue_index_0based": i,
            "aa": sequence[i],
            "is_protected": str(is_protected),
            "protected_type": protected_types if is_protected else "",
            "distance_to_nearest_protected": "{0:.4f}".format(dist),
            "is_motif_shadow_8A": str(is_shadow),
            "motif_contact_count_8A": motif_contact_count[i],
            "contact_density_8A": contact_density[i],
            "pLDDT": "{0:.2f}".format(plddt),
            "is_low_plddt": str(plddt < 70.0),
            "is_high_plddt": str(plddt >= 90.0),
            "is_terminal_region": str(terminal_region_flag(i, n)),
            "normalized_position": "{0:.6f}".format(float(i) / (n - 1) if n > 1 else 0.0),
            "structure_found": "True",
            "parse_status": "success",
        }
        for column in OPTIONAL_FEATURE_COLUMNS:
            row[column] = "False"
        rows.append(row)
    return rows


def summarize_success(accession, length, protected_positions, residue_rows):
    num_shadow = sum(1 for row in residue_rows if row["is_motif_shadow_8A"] == "True")
    contact_values = [int(row["contact_density_8A"]) for row in residue_rows]
    plddt_values = [float(row["pLDDT"]) for row in residue_rows]
    return {
        "accession": accession,
        "length": length,
        "num_protected": len(protected_positions),
        "num_shadow_8A": num_shadow,
        "shadow_fraction": "{0:.6f}".format(float(num_shadow) / length if length else 0.0),
        "mean_contact_density_8A": "{0:.6f}".format(mean(contact_values)),
        "mean_plddt": "{0:.6f}".format(mean(plddt_values)),
        "plddt_min": "{0:.6f}".format(min(plddt_values) if plddt_values else 0.0),
        "plddt_p25": "{0:.6f}".format(percentile(plddt_values, 0.25)),
        "plddt_median": "{0:.6f}".format(percentile(plddt_values, 0.50)),
        "plddt_p75": "{0:.6f}".format(percentile(plddt_values, 0.75)),
        "plddt_max": "{0:.6f}".format(max(plddt_values) if plddt_values else 0.0),
        "status": "success",
        "reason": "",
    }


def summarize_skipped(accession, length, protected_positions, status, reason):
    return {
        "accession": accession,
        "length": length,
        "num_protected": len(protected_positions),
        "num_shadow_8A": "",
        "shadow_fraction": "",
        "mean_contact_density_8A": "",
        "mean_plddt": "",
        "plddt_min": "",
        "plddt_p25": "",
        "plddt_median": "",
        "plddt_p75": "",
        "plddt_max": "",
        "status": status,
        "reason": reason,
    }


def write_csv(path, fields, rows):
    with open_csv_write(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_txt(path, core_rows, residue_rows, summary_rows):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    status_counts = defaultdict(int)
    for row in summary_rows:
        status_counts[row["status"]] += 1
    success_rows = [row for row in summary_rows if row["status"] == "success"]

    def success_values(key):
        return [float(row[key]) for row in success_rows if row[key] != ""]

    all_plddt = [float(row["pLDDT"]) for row in residue_rows]
    all_contact = [int(row["contact_density_8A"]) for row in residue_rows]

    with open(path, "w") as handle:
        handle.write("Core-1K residue BioPrior feature summary\n\n")
        handle.write("total proteins in Core CSV: {0}\n".format(len(core_rows)))
        handle.write("successfully processed proteins: {0}\n".format(status_counts["success"]))
        handle.write("skipped proteins: {0}\n".format(len(core_rows) - status_counts["success"]))
        for status in sorted(status_counts):
            handle.write("- {0}: {1}\n".format(status, status_counts[status]))
        handle.write("total residue rows: {0}\n".format(len(residue_rows)))
        handle.write(
            "average protected residues per successful protein: {0:.6f}\n".format(
                mean([int(row["num_protected"]) for row in success_rows])
            )
        )
        handle.write(
            "average shadow residues per successful protein: {0:.6f}\n".format(
                mean([int(row["num_shadow_8A"]) for row in success_rows])
            )
        )
        handle.write(
            "average contact density: {0:.6f}\n".format(mean(all_contact))
        )
        handle.write("pLDDT distribution:\n")
        handle.write("- min: {0:.6f}\n".format(min(all_plddt) if all_plddt else 0.0))
        handle.write("- p25: {0:.6f}\n".format(percentile(all_plddt, 0.25)))
        handle.write("- median: {0:.6f}\n".format(percentile(all_plddt, 0.50)))
        handle.write("- p75: {0:.6f}\n".format(percentile(all_plddt, 0.75)))
        handle.write("- max: {0:.6f}\n".format(max(all_plddt) if all_plddt else 0.0))
        handle.write("- mean: {0:.6f}\n".format(mean(all_plddt)))
        handle.write(
            "mean pLDDT per successful protein: {0:.6f}\n".format(
                mean(success_values("mean_plddt"))
            )
        )
        handle.write("\nOptional UniProt feature flags:\n")
        handle.write(
            "domain/region/repeat/transmembrane/signal peptide/disulfide/PTM/glycosylation/lipidation flags are present in the output schema but not enabled from Core-1K CSV annotations yet.\n"
        )
        handle.write("\nSkipped examples:\n")
        skipped = [row for row in summary_rows if row["status"] != "success"]
        if skipped:
            for row in skipped[:50]:
                handle.write("- {0}: {1} ({2})\n".format(row["accession"], row["status"], row["reason"]))
            if len(skipped) > 50:
                handle.write("- ... {0} more\n".format(len(skipped) - 50))
        else:
            handle.write("- none\n")
        handle.write("\nCORE_1K_RESIDUE_BIOPRIORS_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Core-1K residue-level BioPrior features.")
    parser.add_argument("--core_csv", required=True)
    parser.add_argument("--structure_dir", required=True)
    parser.add_argument("--mapping_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--shadow_cutoff", type=float, default=8.0)
    parser.add_argument("--contact_cutoff", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    core_rows = read_core_csv(args.core_csv)
    mapping_rows = read_mapping_csv(args.mapping_csv)
    residue_rows = []
    summary_rows = []

    for row in core_rows:
        accession = row["accession"]
        sequence = row["sequence"]
        length = len(sequence)
        protected_positions = row["protected_positions_list"]
        protected_types = row.get("protected_types", "")

        if any(pos < 0 or pos >= length for pos in protected_positions):
            summary_rows.append(
                summarize_skipped(
                    accession,
                    length,
                    protected_positions,
                    "protected_position_oob",
                    "protected position outside sequence length",
                )
            )
            continue

        mapping = mapping_rows.get(accession)
        if mapping is None or mapping.get("found") != "True":
            reason = "structure missing"
            if mapping is not None:
                reason = mapping.get("error_reason") or reason
            summary_rows.append(
                summarize_skipped(
                    accession, length, protected_positions, "no_structure", reason
                )
            )
            continue

        structure_path = mapping.get("structure_path") or os.path.join(
            args.structure_dir, accession + ".pdb"
        )
        if not structure_path or not os.path.exists(structure_path) or os.path.getsize(structure_path) == 0:
            summary_rows.append(
                summarize_skipped(
                    accession,
                    length,
                    protected_positions,
                    "no_structure",
                    "PDB missing or empty: {0}".format(structure_path),
                )
            )
            continue

        residues = parse_pdb_ca(structure_path)
        pdb_sequence = "".join(residue["aa"] for residue in residues)
        if len(residues) != length or pdb_sequence != sequence:
            summary_rows.append(
                summarize_skipped(
                    accession,
                    length,
                    protected_positions,
                    "length_or_sequence_mismatch",
                    "PDB CA length {0}; CSV length {1}; sequence_match={2}".format(
                        len(residues), length, pdb_sequence == sequence
                    ),
                )
            )
            continue

        protein_rows = compute_residue_rows(
            accession,
            sequence,
            protected_positions,
            protected_types,
            residues,
            args.shadow_cutoff,
            args.contact_cutoff,
        )
        residue_rows.extend(protein_rows)
        summary_rows.append(
            summarize_success(accession, length, protected_positions, protein_rows)
        )

    residue_fields = [
        "accession",
        "residue_index_0based",
        "aa",
        "is_protected",
        "protected_type",
        "distance_to_nearest_protected",
        "is_motif_shadow_8A",
        "motif_contact_count_8A",
        "contact_density_8A",
        "pLDDT",
        "is_low_plddt",
        "is_high_plddt",
        "is_terminal_region",
        "normalized_position",
        "structure_found",
        "parse_status",
    ] + OPTIONAL_FEATURE_COLUMNS
    summary_fields = [
        "accession",
        "length",
        "num_protected",
        "num_shadow_8A",
        "shadow_fraction",
        "mean_contact_density_8A",
        "mean_plddt",
        "plddt_min",
        "plddt_p25",
        "plddt_median",
        "plddt_p75",
        "plddt_max",
        "status",
        "reason",
    ]

    write_csv(args.out_csv, residue_fields, residue_rows)
    write_csv(args.summary_csv, summary_fields, summary_rows)
    write_summary_txt(args.summary_txt, core_rows, residue_rows, summary_rows)

    success_count = sum(1 for row in summary_rows if row["status"] == "success")
    print("Processed proteins: {0}".format(len(core_rows)))
    print("Successfully computed: {0}".format(success_count))
    print("Skipped proteins: {0}".format(len(core_rows) - success_count))
    print("Residue rows: {0}".format(len(residue_rows)))
    print("Average protected residues: {0:.6f}".format(
        mean([int(row["num_protected"]) for row in summary_rows if row["status"] == "success"])
    ))
    print("Average shadow residues: {0:.6f}".format(
        mean([int(row["num_shadow_8A"]) for row in summary_rows if row["status"] == "success"])
    ))
    print("Average contact density: {0:.6f}".format(
        mean([int(row["contact_density_8A"]) for row in residue_rows])
    ))
    plddt_values = [float(row["pLDDT"]) for row in residue_rows]
    print(
        "pLDDT min/p25/median/p75/max/mean: {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}/{5:.2f}".format(
            min(plddt_values) if plddt_values else 0.0,
            percentile(plddt_values, 0.25),
            percentile(plddt_values, 0.50),
            percentile(plddt_values, 0.75),
            max(plddt_values) if plddt_values else 0.0,
            mean(plddt_values),
        )
    )
    print("Wrote {0}".format(args.out_csv))
    print("Wrote {0}".format(args.summary_csv))
    print("Wrote {0}".format(args.summary_txt))


if __name__ == "__main__":
    main()
