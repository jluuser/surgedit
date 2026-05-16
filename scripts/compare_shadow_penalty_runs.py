from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


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
    rows = {}
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            row["length_int"] = int(row["length"])
            row["protected_positions_list"] = parse_positions(row["protected_positions"])
            rows[row["protein_id"]] = row
    return rows


def read_structure_priors(path):
    priors = defaultdict(
        lambda: {
            "motif_positions": set(),
            "shadow_positions": set(),
            "motif_contact_positions": set(),
        }
    )
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            protein_id = row["protein_id"]
            pos = int(row["position"])
            if row.get("is_motif") == "True":
                priors[protein_id]["motif_positions"].add(pos)
            if row.get("is_motif_shadow_8A") == "True":
                priors[protein_id]["shadow_positions"].add(pos)
            if int(row.get("motif_contact_count_8A", 0)) > 0:
                priors[protein_id]["motif_contact_positions"].add(pos)
    return dict(priors)


def read_deletions(path):
    with open(path) as handle:
        return {entry["header"]: entry for entry in json.load(handle)}


def validation_all_pass(path):
    with open(path) as handle:
        return "ALL_PASS" in handle.read()


def ratio(num, den):
    if den == 0:
        return 0.0
    return float(num) / den


def run_stats(deletions, motif_rows, priors, shrink_pct):
    stats = {
        "analyzed": 0,
        "total_deleted": 0,
        "motif_deleted": 0,
        "shadow_deleted": 0,
        "proteins_with_shadow": 0,
        "motif_contact_deleted": 0,
        "compression_success": 0,
        "total_proteins": len(motif_rows),
    }
    per = {}
    for protein_id, motif in motif_rows.items():
        entry = deletions.get(protein_id)
        if entry is None:
            continue
        target_new_len = motif["length_int"] - int(
            math.ceil(motif["length_int"] * shrink_pct / 100.0)
        )
        if int(entry["new_length"]) == target_new_len:
            stats["compression_success"] += 1
        if protein_id not in priors:
            continue

        deleted = set(entry["deletion_positions_zero_based"])
        motif_positions = set(motif["protected_positions_list"])
        shadow_positions = priors[protein_id]["shadow_positions"]
        motif_contact_positions = priors[protein_id]["motif_contact_positions"]
        deleted_motif = deleted & motif_positions
        deleted_shadow = deleted & shadow_positions
        deleted_motif_contact = deleted & motif_contact_positions

        stats["analyzed"] += 1
        stats["total_deleted"] += len(deleted)
        stats["motif_deleted"] += len(deleted_motif)
        stats["shadow_deleted"] += len(deleted_shadow)
        stats["motif_contact_deleted"] += len(deleted_motif_contact)
        if deleted_shadow:
            stats["proteins_with_shadow"] += 1
        per[protein_id] = {
            "protein_id": protein_id,
            "num_deleted": len(deleted),
            "num_deleted_motif": len(deleted_motif),
            "num_deleted_shadow": len(deleted_shadow),
            "num_deleted_motif_contact": len(deleted_motif_contact),
            "shadow_deletion_rate_over_deleted": ratio(len(deleted_shadow), len(deleted)),
            "motif_contact_deletion_rate_over_deleted": ratio(
                len(deleted_motif_contact), len(deleted)
            ),
        }
    return stats, per


def write_csv(path, rows):
    fields = [
        "protein_id",
        "primary_protected_type",
        "protected_type",
        "hardmask_num_deleted_shadow",
        "shadow05_num_deleted_shadow",
        "delta_shadow_deleted",
        "hardmask_num_deleted_motif",
        "shadow05_num_deleted_motif",
        "hardmask_num_deleted_motif_contact",
        "shadow05_num_deleted_motif_contact",
        "hardmask_shadow_deletion_rate_over_deleted",
        "shadow05_shadow_deletion_rate_over_deleted",
    ]
    with open_csv_write(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(count, total):
    return "{0}/{1} ({2:.2%})".format(count, total, ratio(count, total))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare hardmask and shadow-penalty runs.")
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--structure_priors", required=True)
    parser.add_argument("--hardmask_json", required=True)
    parser.add_argument("--shadow_json", required=True)
    parser.add_argument("--hardmask_validation", required=True)
    parser.add_argument("--shadow_validation", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--shrink_pct", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    motif_rows = read_motif_csv(args.motif_csv)
    priors = read_structure_priors(args.structure_priors)
    hardmask_deletions = read_deletions(args.hardmask_json)
    shadow_deletions = read_deletions(args.shadow_json)
    hardmask_validation_pass = validation_all_pass(args.hardmask_validation)
    shadow_validation_pass = validation_all_pass(args.shadow_validation)

    hardmask_stats, hardmask_per = run_stats(
        hardmask_deletions, motif_rows, priors, args.shrink_pct
    )
    shadow_stats, shadow_per = run_stats(
        shadow_deletions, motif_rows, priors, args.shrink_pct
    )

    rows = []
    for protein_id in sorted(set(hardmask_per) | set(shadow_per)):
        hard = hardmask_per.get(protein_id, {})
        shadow = shadow_per.get(protein_id, {})
        motif = motif_rows[protein_id]
        hard_shadow = hard.get("num_deleted_shadow", 0)
        shadow_shadow = shadow.get("num_deleted_shadow", 0)
        rows.append(
            {
                "protein_id": protein_id,
                "primary_protected_type": motif.get("primary_protected_type", ""),
                "protected_type": motif.get("protected_type", ""),
                "hardmask_num_deleted_shadow": hard_shadow,
                "shadow05_num_deleted_shadow": shadow_shadow,
                "delta_shadow_deleted": hard_shadow - shadow_shadow,
                "hardmask_num_deleted_motif": hard.get("num_deleted_motif", 0),
                "shadow05_num_deleted_motif": shadow.get("num_deleted_motif", 0),
                "hardmask_num_deleted_motif_contact": hard.get(
                    "num_deleted_motif_contact", 0
                ),
                "shadow05_num_deleted_motif_contact": shadow.get(
                    "num_deleted_motif_contact", 0
                ),
                "hardmask_shadow_deletion_rate_over_deleted": "{0:.6f}".format(
                    hard.get("shadow_deletion_rate_over_deleted", 0.0)
                ),
                "shadow05_shadow_deletion_rate_over_deleted": "{0:.6f}".format(
                    shadow.get("shadow_deletion_rate_over_deleted", 0.0)
                ),
            }
        )
    write_csv(args.out_csv, rows)

    improved = sorted(rows, key=lambda row: row["delta_shadow_deleted"], reverse=True)[:10]
    still_shadow = sorted(
        rows, key=lambda row: row["shadow05_num_deleted_shadow"], reverse=True
    )[:10]
    compression_rate = ratio(
        shadow_stats["compression_success"], shadow_stats["total_proteins"]
    )

    if not hardmask_validation_pass or not shadow_validation_pass or shadow_stats["motif_deleted"] > 0:
        conclusion = "SHADOW_PENALTY_FAIL"
        reason = "validation failed or motif was deleted"
    elif (
        shadow_stats["shadow_deleted"] < hardmask_stats["shadow_deleted"]
        and compression_rate >= 0.95
    ):
        conclusion = "SHADOW_PENALTY_PASS"
        reason = ""
    else:
        conclusion = "SHADOW_PENALTY_WEAK"
        reason = "shadow05 did not reduce shadow deletion enough"

    parent = os.path.dirname(args.out_report)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.out_report, "w") as handle:
        handle.write("SCISOR hardmask vs hardmask+shadow penalty comparison\n\n")
        handle.write("analyzed proteins with structure prior: {0}\n".format(shadow_stats["analyzed"]))
        handle.write("hardmask motif deletion residues: {0}\n".format(hardmask_stats["motif_deleted"]))
        handle.write("shadow05 motif deletion residues: {0}\n".format(shadow_stats["motif_deleted"]))
        handle.write("hardmask shadow_deleted_residues: {0}\n".format(hardmask_stats["shadow_deleted"]))
        handle.write("shadow05 shadow_deleted_residues: {0}\n".format(shadow_stats["shadow_deleted"]))
        handle.write("hardmask proteins_with_shadow_deletion: {0}\n".format(hardmask_stats["proteins_with_shadow"]))
        handle.write("shadow05 proteins_with_shadow_deletion: {0}\n".format(shadow_stats["proteins_with_shadow"]))
        handle.write(
            "hardmask shadow_deletion_rate_over_deleted: {0:.6f}\n".format(
                ratio(hardmask_stats["shadow_deleted"], hardmask_stats["total_deleted"])
            )
        )
        handle.write(
            "shadow05 shadow_deletion_rate_over_deleted: {0:.6f}\n".format(
                ratio(shadow_stats["shadow_deleted"], shadow_stats["total_deleted"])
            )
        )
        handle.write(
            "hardmask motif_contact_deleted_residues: {0}\n".format(
                hardmask_stats["motif_contact_deleted"]
            )
        )
        handle.write(
            "shadow05 motif_contact_deleted_residues: {0}\n".format(
                shadow_stats["motif_contact_deleted"]
            )
        )
        handle.write(
            "hardmask compression success rate: {0}\n".format(
                pct(hardmask_stats["compression_success"], hardmask_stats["total_proteins"])
            )
        )
        handle.write(
            "shadow05 compression success rate: {0}\n".format(
                pct(shadow_stats["compression_success"], shadow_stats["total_proteins"])
            )
        )
        handle.write("hardmask validation ALL_PASS: {0}\n".format(hardmask_validation_pass))
        handle.write("shadow05 validation ALL_PASS: {0}\n\n".format(shadow_validation_pass))

        handle.write("Top 10 most improved proteins by shadow deletion count:\n")
        handle.write("protein_id\thardmask_num_deleted_shadow\tshadow05_num_deleted_shadow\tdelta\n")
        for row in improved:
            handle.write(
                "{0}\t{1}\t{2}\t{3}\n".format(
                    row["protein_id"],
                    row["hardmask_num_deleted_shadow"],
                    row["shadow05_num_deleted_shadow"],
                    row["delta_shadow_deleted"],
                )
            )
        handle.write("\n")

        handle.write("Top 10 remaining shadow deletions after shadow penalty:\n")
        handle.write("protein_id\tshadow05_num_deleted_shadow\thardmask_num_deleted_shadow\tdelta\n")
        for row in still_shadow:
            handle.write(
                "{0}\t{1}\t{2}\t{3}\n".format(
                    row["protein_id"],
                    row["shadow05_num_deleted_shadow"],
                    row["hardmask_num_deleted_shadow"],
                    row["delta_shadow_deleted"],
                )
            )
        handle.write("\n{0}\n".format(conclusion))
        if reason:
            handle.write("reason: {0}\n".format(reason))

    print("Wrote {0}".format(args.out_csv))
    print("Wrote {0}".format(args.out_report))
    if conclusion == "SHADOW_PENALTY_FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
