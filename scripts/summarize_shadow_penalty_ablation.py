from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


DEFAULT_RUNS = (
    "hardmask=results/scisor_swissprot_diverse/run10_hardmask,"
    "depenalty05=results/scisor_swissprot_diverse/run10_hardmask_shadow05,"
    "depenalty02=results/scisor_swissprot_diverse/run10_hardmask_shadow02,"
    "depenalty00=results/scisor_swissprot_diverse/run10_hardmask_shadow00"
)


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
        lambda: {"shadow_positions": set(), "motif_contact_positions": set()}
    )
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            protein_id = row["protein_id"]
            pos = int(row["position"])
            if row.get("is_motif_shadow_8A") == "True":
                priors[protein_id]["shadow_positions"].add(pos)
            if int(row.get("motif_contact_count_8A", 0)) > 0:
                priors[protein_id]["motif_contact_positions"].add(pos)
    return dict(priors)


def read_deletions(path):
    with open(path) as handle:
        return {entry["header"]: entry for entry in json.load(handle)}


def validation_all_pass(path):
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        return "ALL_PASS" in handle.read()


def parse_runs(value):
    runs = []
    for item in value.split(","):
        if not item:
            continue
        name, path = item.split("=", 1)
        runs.append((name, path))
    return runs


def penalty_for_name(name):
    if name == "hardmask":
        return 1.0
    if name.endswith("05"):
        return 0.5
    if name.endswith("02"):
        return 0.2
    if name.endswith("00"):
        return 0.0
    return ""


def ratio(num, den):
    if den == 0:
        return 0.0
    return float(num) / den


def summarize_run(name, run_dir, motif_rows, priors, shrink_pct):
    notes = []
    deletions_path = os.path.join(run_dir, "deletions.json")
    validation_path = os.path.join(run_dir, "validation_report.txt")
    if not os.path.exists(deletions_path) or os.path.getsize(deletions_path) == 0:
        return {
            "run_name": name,
            "shadow_penalty": penalty_for_name(name),
            "analyzed_proteins_with_structure_prior": 0,
            "total_deleted_residues": 0,
            "motif_deleted_residues": 0,
            "shadow_deleted_residues": 0,
            "proteins_with_shadow_deletion": 0,
            "shadow_deletion_rate_over_deleted": "0",
            "motif_contact_deleted_residues": 0,
            "motif_contact_deletion_rate_over_deleted": "0",
            "compression_success_count": 0,
            "total_proteins": len(motif_rows),
            "compression_success_rate": "0",
            "validation_ALL_PASS": False,
            "notes": "missing deletions.json",
        }

    deletions = read_deletions(deletions_path)
    analyzed = 0
    total_deleted = 0
    motif_deleted = 0
    shadow_deleted = 0
    proteins_with_shadow = 0
    motif_contact_deleted = 0
    compression_success = 0

    for protein_id, motif in motif_rows.items():
        entry = deletions.get(protein_id)
        if entry is None:
            notes.append("missing_deletion:{0}".format(protein_id))
            continue
        target_new_len = motif["length_int"] - int(
            math.ceil(motif["length_int"] * shrink_pct / 100.0)
        )
        if int(entry["new_length"]) == target_new_len:
            compression_success += 1
        deleted = set(entry["deletion_positions_zero_based"])
        motif_positions = set(motif["protected_positions_list"])
        motif_deleted += len(deleted & motif_positions)

        if protein_id not in priors:
            continue
        analyzed += 1
        shadow = deleted & priors[protein_id]["shadow_positions"]
        motif_contact = deleted & priors[protein_id]["motif_contact_positions"]
        total_deleted += len(deleted)
        shadow_deleted += len(shadow)
        motif_contact_deleted += len(motif_contact)
        if shadow:
            proteins_with_shadow += 1

    if analyzed < len(priors):
        notes.append("some_structure_prior_proteins_not_analyzed")
    validation_pass = validation_all_pass(validation_path)
    if not validation_pass:
        notes.append("validation_not_ALL_PASS")
    if motif_deleted:
        notes.append("motif_deleted")

    return {
        "run_name": name,
        "shadow_penalty": penalty_for_name(name),
        "analyzed_proteins_with_structure_prior": analyzed,
        "total_deleted_residues": total_deleted,
        "motif_deleted_residues": motif_deleted,
        "shadow_deleted_residues": shadow_deleted,
        "proteins_with_shadow_deletion": proteins_with_shadow,
        "shadow_deletion_rate_over_deleted": "{0:.6f}".format(
            ratio(shadow_deleted, total_deleted)
        ),
        "motif_contact_deleted_residues": motif_contact_deleted,
        "motif_contact_deletion_rate_over_deleted": "{0:.6f}".format(
            ratio(motif_contact_deleted, total_deleted)
        ),
        "compression_success_count": compression_success,
        "total_proteins": len(motif_rows),
        "compression_success_rate": "{0:.6f}".format(
            ratio(compression_success, len(motif_rows))
        ),
        "validation_ALL_PASS": validation_pass,
        "notes": ";".join(sorted(set(notes))),
    }


def write_csv(path, rows):
    fields = [
        "run_name",
        "shadow_penalty",
        "analyzed_proteins_with_structure_prior",
        "total_deleted_residues",
        "motif_deleted_residues",
        "shadow_deleted_residues",
        "proteins_with_shadow_deletion",
        "shadow_deletion_rate_over_deleted",
        "motif_contact_deleted_residues",
        "motif_contact_deletion_rate_over_deleted",
        "compression_success_count",
        "total_proteins",
        "compression_success_rate",
        "validation_ALL_PASS",
        "notes",
    ]
    with open_csv_write(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_monotonic_nonincreasing(rows):
    ordered = sorted(rows, key=lambda row: float(row["shadow_penalty"]), reverse=True)
    values = [int(row["shadow_deleted_residues"]) for row in ordered]
    return all(values[i] >= values[i + 1] for i in range(len(values) - 1)), ordered


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize shadow penalty ablation runs.")
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--structure_priors", required=True)
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--shrink_pct", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    motif_rows = read_motif_csv(args.motif_csv)
    priors = read_structure_priors(args.structure_priors)
    runs = parse_runs(args.runs)
    rows = [
        summarize_run(name, path, motif_rows, priors, args.shrink_pct)
        for name, path in runs
    ]
    write_csv(args.out_csv, rows)

    monotonic, ordered = is_monotonic_nonincreasing(rows)
    by_name = dict((row["run_name"], row) for row in rows)
    pass_conditions = (
        all(row["validation_ALL_PASS"] for row in rows)
        and all(int(row["motif_deleted_residues"]) == 0 for row in rows)
        and monotonic
    )
    shadow05 = by_name.get("depenalty05")
    shadow02 = by_name.get("depenalty02")
    shadow00 = by_name.get("depenalty00")
    recommendation = ""
    if shadow02 and shadow05:
        if (
            int(shadow02["shadow_deleted_residues"])
            < int(shadow05["shadow_deleted_residues"])
            and int(shadow02["compression_success_count"]) == int(shadow02["total_proteins"])
        ):
            recommendation = "Recommend shadow_penalty=0.2 as default: it reduces shadow deletion vs 0.5 while preserving 80/80 compression success."
    if shadow00 and int(shadow00["compression_success_count"]) == int(shadow00["total_proteins"]):
        recommendation += (
            " shadow_penalty=0.0 also succeeds and can be treated as a strong-protection setting."
        )
    elif shadow00:
        recommendation += (
            " shadow_penalty=0.0 hurts compression; soft penalty is preferable to hard shadow mask."
        )

    conclusion = "SHADOW_ABLATION_PASS" if pass_conditions else "SHADOW_ABLATION_WARN"

    parent = os.path.dirname(args.out_report)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.out_report, "w") as handle:
        handle.write("Shadow penalty strength ablation\n\n")
        handle.write("Runs:\n")
        for name, path in runs:
            handle.write("- {0}: {1}\n".format(name, path))
        handle.write("\nCore metrics:\n")
        handle.write(
            "run_name\tshadow_penalty\tanalyzed\ttotal_deleted\tmotif_deleted\tshadow_deleted\tproteins_with_shadow\tshadow_rate\tmotif_contact_deleted\tmotif_contact_rate\tcompression\tvalidation\tnotes\n"
        )
        for row in rows:
            handle.write(
                "{run_name}\t{shadow_penalty}\t{analyzed_proteins_with_structure_prior}\t{total_deleted_residues}\t{motif_deleted_residues}\t{shadow_deleted_residues}\t{proteins_with_shadow_deletion}\t{shadow_deletion_rate_over_deleted}\t{motif_contact_deleted_residues}\t{motif_contact_deletion_rate_over_deleted}\t{compression_success_count}/{total_proteins}\t{validation_ALL_PASS}\t{notes}\n".format(
                    **row
                )
            )
        handle.write("\nPenalty order by strength:\n")
        for row in ordered:
            handle.write(
                "- penalty={0}: shadow_deleted={1}, compression={2}/{3}\n".format(
                    row["shadow_penalty"],
                    row["shadow_deleted_residues"],
                    row["compression_success_count"],
                    row["total_proteins"],
                )
            )
        handle.write(
            "\nShadow_deleted_residues monotonic non-increasing as penalty strengthens: {0}\n".format(
                monotonic
            )
        )
        handle.write(
            "Compression success affected: {0}\n".format(
                any(
                    int(row["compression_success_count"]) < int(row["total_proteins"])
                    for row in rows
                )
            )
        )
        handle.write(
            "Motif deletion always zero: {0}\n".format(
                all(int(row["motif_deleted_residues"]) == 0 for row in rows)
            )
        )
        handle.write("Recommendation: {0}\n\n".format(recommendation.strip()))
        handle.write("{0}\n".format(conclusion))

    print("Wrote {0}".format(args.out_csv))
    print("Wrote {0}".format(args.out_report))


if __name__ == "__main__":
    main()
