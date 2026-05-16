from __future__ import print_function

import argparse
import csv
import json
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


def read_motif_csv(path):
    rows = {}
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            rows[row["protein_id"]] = row
    return rows


def read_structure_summary(path):
    rows = {}
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            rows[row["protein_id"]] = row
    return rows


def read_structure_priors(path):
    by_protein = defaultdict(dict)
    with open_csv_read(path) as handle:
        for row in csv.DictReader(handle):
            protein_id = row["protein_id"]
            position = int(row["position"])
            row["position_int"] = position
            row["is_motif_bool"] = row["is_motif"] == "True"
            row["is_shadow_bool"] = row["is_motif_shadow_8A"] == "True"
            row["distance_float"] = float(row["distance_to_motif"])
            row["contact_density_int"] = int(row["contact_density_8A"])
            row["motif_contact_int"] = int(row["motif_contact_count_8A"])
            row["plddt_float"] = float(row["plddt"])
            by_protein[protein_id][position] = row
    return by_protein


def read_deletions(path):
    with open(path) as handle:
        data = json.load(handle)
    return {entry["header"]: entry for entry in data}


def mean(values):
    if not values:
        return ""
    return "{0:.6f}".format(float(sum(values)) / len(values))


def percentile_75(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.75 * (len(ordered) - 1))
    return ordered[idx]


def ratio(num, den):
    if den == 0:
        return "0"
    return "{0:.6f}".format(float(num) / den)


def write_csv(path, fields, rows):
    with open_csv_write(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_run(run_name, deletions, structure_priors, structure_summary):
    per_protein = []
    high_risk = []
    skipped = []

    total_deleted = 0
    motif_deleted = 0
    total_motif = 0
    shadow_deleted = 0
    proteins_with_shadow = 0
    motif_contact_deleted = 0
    deleted_distances = []
    all_nonmotif_distances = []
    deleted_contact_density = []
    not_deleted_contact_density = []
    deleted_plddt = []
    not_deleted_plddt = []
    analyzed = 0

    for protein_id, deletion_entry in deletions.items():
        if protein_id not in structure_priors:
            skipped.append(protein_id)
            continue
        residues_by_pos = structure_priors[protein_id]
        residues = [residues_by_pos[pos] for pos in sorted(residues_by_pos)]
        if not residues:
            skipped.append(protein_id)
            continue

        analyzed += 1
        deleted_positions = set(deletion_entry["deletion_positions_zero_based"])
        contact_threshold = percentile_75([row["contact_density_int"] for row in residues])
        deleted_rows = [row for row in residues if row["position_int"] in deleted_positions]
        not_deleted_rows = [
            row for row in residues if row["position_int"] not in deleted_positions
        ]
        motif_rows = [row for row in residues if row["is_motif_bool"]]
        shadow_rows = [row for row in deleted_rows if row["is_shadow_bool"]]
        motif_deleted_rows = [row for row in deleted_rows if row["is_motif_bool"]]
        motif_contact_rows = [
            row for row in deleted_rows if row["motif_contact_int"] > 0
        ]

        total_deleted += len(deleted_rows)
        motif_deleted += len(motif_deleted_rows)
        total_motif += len(motif_rows)
        shadow_deleted += len(shadow_rows)
        if shadow_rows:
            proteins_with_shadow += 1
        motif_contact_deleted += len(motif_contact_rows)

        deleted_distances.extend([row["distance_float"] for row in deleted_rows])
        all_nonmotif_distances.extend(
            [row["distance_float"] for row in residues if not row["is_motif_bool"]]
        )
        deleted_contact_density.extend(
            [row["contact_density_int"] for row in deleted_rows]
        )
        not_deleted_contact_density.extend(
            [row["contact_density_int"] for row in not_deleted_rows]
        )
        deleted_plddt.extend([row["plddt_float"] for row in deleted_rows])
        not_deleted_plddt.extend([row["plddt_float"] for row in not_deleted_rows])

        summary = structure_summary.get(protein_id, {})
        per_protein.append(
            {
                "protein_id": protein_id,
                "run_name": run_name,
                "length": len(residues),
                "num_motif": len(motif_rows),
                "num_shadow_8A": summary.get("num_shadow_8A", ""),
                "num_deleted": len(deleted_rows),
                "num_deleted_motif": len(motif_deleted_rows),
                "num_deleted_shadow": len(shadow_rows),
                "num_deleted_motif_contact": len(motif_contact_rows),
                "shadow_deletion_rate_over_deleted": ratio(
                    len(shadow_rows), len(deleted_rows)
                ),
                "mean_distance_to_motif_deleted": mean(
                    [row["distance_float"] for row in deleted_rows]
                ),
                "mean_contact_density_deleted": mean(
                    [row["contact_density_int"] for row in deleted_rows]
                ),
                "mean_contact_density_not_deleted": mean(
                    [row["contact_density_int"] for row in not_deleted_rows]
                ),
                "mean_plddt_deleted": mean([row["plddt_float"] for row in deleted_rows]),
                "mean_plddt_not_deleted": mean(
                    [row["plddt_float"] for row in not_deleted_rows]
                ),
            }
        )

        for row in deleted_rows:
            risk_labels = []
            if row["is_motif_bool"]:
                risk_labels.append("motif_deleted")
            if row["is_shadow_bool"]:
                risk_labels.append("motif_shadow_deleted")
            if row["motif_contact_int"] > 0:
                risk_labels.append("motif_contact_deleted")
            if row["contact_density_int"] > contact_threshold:
                risk_labels.append("high_contact_deleted")
            if not risk_labels:
                continue
            high_risk.append(
                {
                    "protein_id": protein_id,
                    "run_name": run_name,
                    "position": row["position"],
                    "aa": row["aa"],
                    "is_motif": row["is_motif"],
                    "is_motif_shadow_8A": row["is_motif_shadow_8A"],
                    "distance_to_motif": row["distance_to_motif"],
                    "contact_density_8A": row["contact_density_8A"],
                    "motif_contact_count_8A": row["motif_contact_count_8A"],
                    "plddt": row["plddt"],
                    "risk_type": ";".join(risk_labels),
                }
            )

    summary_row = {
        "run_name": run_name,
        "analyzed_proteins": analyzed,
        "total_deleted_residues": total_deleted,
        "motif_deleted_residues": motif_deleted,
        "motif_deletion_rate": ratio(motif_deleted, total_motif),
        "shadow_deleted_residues": shadow_deleted,
        "shadow_deletion_rate_over_deleted": ratio(shadow_deleted, total_deleted),
        "proteins_with_shadow_deletion": proteins_with_shadow,
        "motif_contact_deleted_residues": motif_contact_deleted,
        "motif_contact_deletion_rate_over_deleted": ratio(
            motif_contact_deleted, total_deleted
        ),
        "mean_distance_to_motif_deleted": mean(deleted_distances),
        "mean_distance_to_motif_all_nonmotif": mean(all_nonmotif_distances),
        "mean_contact_density_deleted": mean(deleted_contact_density),
        "mean_contact_density_not_deleted": mean(not_deleted_contact_density),
        "mean_plddt_deleted": mean(deleted_plddt),
        "mean_plddt_not_deleted": mean(not_deleted_plddt),
    }
    return summary_row, per_protein, high_risk, skipped


def report_value(row, key):
    value = row.get(key, "")
    return value if value != "" else "NA"


def write_report(path, motif_count, analyzable_count, summaries, high_risk, skipped_by_run):
    nomask = summaries["nomask"]
    hardmask = summaries["hardmask"]
    hardmask_shadow_cases = [
        row
        for row in high_risk
        if row["run_name"] == "hardmask"
        and "motif_shadow_deleted" in row["risk_type"].split(";")
    ]
    nomask_shadow_cases = [
        row
        for row in high_risk
        if row["run_name"] == "nomask"
        and "motif_shadow_deleted" in row["risk_type"].split(";")
    ]

    if int(hardmask["analyzed_proteins"]) == 0:
        conclusion = "STRUCTURE_PRIOR_SIGNAL_FAIL"
    elif (
        int(hardmask["shadow_deleted_residues"]) > 0
        and int(hardmask["proteins_with_shadow_deletion"]) >= 5
    ):
        conclusion = "MOTIF_SHADOW_RISK_FOUND"
    else:
        conclusion = "MOTIF_SHADOW_RISK_WEAK"

    with open(path, "w") as handle:
        handle.write("Swiss-Prot diverse structure prior signal report\n\n")
        handle.write("motif_csv proteins: {0}\n".format(motif_count))
        handle.write("proteins with structure prior available: {0}\n".format(analyzable_count))
        handle.write(
            "skipped proteins nomask: {0}\n".format(";".join(skipped_by_run["nomask"]) or "none")
        )
        handle.write(
            "skipped proteins hardmask: {0}\n\n".format(
                ";".join(skipped_by_run["hardmask"]) or "none"
            )
        )

        handle.write("Motif deletion:\n")
        handle.write(
            "- no-mask: {0} residues, motif_deletion_rate={1}\n".format(
                nomask["motif_deleted_residues"], nomask["motif_deletion_rate"]
            )
        )
        handle.write(
            "- hardmask: {0} residues, motif_deletion_rate={1}\n\n".format(
                hardmask["motif_deleted_residues"], hardmask["motif_deletion_rate"]
            )
        )

        handle.write("Motif-shadow deletion:\n")
        handle.write(
            "- no-mask: {0} residues, over_deleted={1}, proteins={2}\n".format(
                nomask["shadow_deleted_residues"],
                nomask["shadow_deletion_rate_over_deleted"],
                nomask["proteins_with_shadow_deletion"],
            )
        )
        handle.write(
            "- hardmask: {0} residues, over_deleted={1}, proteins={2}\n\n".format(
                hardmask["shadow_deleted_residues"],
                hardmask["shadow_deletion_rate_over_deleted"],
                hardmask["proteins_with_shadow_deletion"],
            )
        )

        handle.write("Motif-contact deletion:\n")
        handle.write(
            "- no-mask: {0} residues, over_deleted={1}\n".format(
                nomask["motif_contact_deleted_residues"],
                nomask["motif_contact_deletion_rate_over_deleted"],
            )
        )
        handle.write(
            "- hardmask: {0} residues, over_deleted={1}\n\n".format(
                hardmask["motif_contact_deleted_residues"],
                hardmask["motif_contact_deletion_rate_over_deleted"],
            )
        )

        handle.write(
            "Hardmask protects direct motif residues but still deletes motif-shadow residues: {0}\n".format(
                int(hardmask["shadow_deleted_residues"]) > 0
            )
        )
        handle.write(
            "No-mask contact density deleted/not_deleted: {0}/{1}\n".format(
                report_value(nomask, "mean_contact_density_deleted"),
                report_value(nomask, "mean_contact_density_not_deleted"),
            )
        )
        handle.write(
            "Hardmask contact density deleted/not_deleted: {0}/{1}\n".format(
                report_value(hardmask, "mean_contact_density_deleted"),
                report_value(hardmask, "mean_contact_density_not_deleted"),
            )
        )
        handle.write(
            "No-mask pLDDT deleted/not_deleted: {0}/{1}\n".format(
                report_value(nomask, "mean_plddt_deleted"),
                report_value(nomask, "mean_plddt_not_deleted"),
            )
        )
        handle.write(
            "Hardmask pLDDT deleted/not_deleted: {0}/{1}\n".format(
                report_value(hardmask, "mean_plddt_deleted"),
                report_value(hardmask, "mean_plddt_not_deleted"),
            )
        )
        handle.write("High-risk deletion cases: {0}\n\n".format(len(high_risk)))

        handle.write("Top 10 hardmask motif-shadow deletion cases:\n")
        handle.write(
            "protein_id\tposition\taa\tdistance_to_motif\tcontact_density_8A\tmotif_contact_count_8A\tplddt\n"
        )
        for row in sorted(
            hardmask_shadow_cases,
            key=lambda item: (
                float(item["distance_to_motif"]),
                -int(item["contact_density_8A"]),
            ),
        )[:10]:
            handle.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(
                    row["protein_id"],
                    row["position"],
                    row["aa"],
                    row["distance_to_motif"],
                    row["contact_density_8A"],
                    row["motif_contact_count_8A"],
                    row["plddt"],
                )
            )
        if not hardmask_shadow_cases:
            handle.write("none\n")
        handle.write("\n")

        handle.write("Top 10 no-mask motif-shadow deletion cases:\n")
        handle.write(
            "protein_id\tposition\taa\tdistance_to_motif\tcontact_density_8A\tmotif_contact_count_8A\tplddt\n"
        )
        for row in sorted(
            nomask_shadow_cases,
            key=lambda item: (
                float(item["distance_to_motif"]),
                -int(item["contact_density_8A"]),
            ),
        )[:10]:
            handle.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(
                    row["protein_id"],
                    row["position"],
                    row["aa"],
                    row["distance_to_motif"],
                    row["contact_density_8A"],
                    row["motif_contact_count_8A"],
                    row["plddt"],
                )
            )
        if not nomask_shadow_cases:
            handle.write("none\n")
        handle.write("\n{0}\n".format(conclusion))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deletion risk on structure priors.")
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--structure_priors", required=True)
    parser.add_argument("--structure_summary", required=True)
    parser.add_argument("--nomask_json", required=True)
    parser.add_argument("--hardmask_json", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    motif_rows = read_motif_csv(args.motif_csv)
    structure_summary = read_structure_summary(args.structure_summary)
    structure_priors = read_structure_priors(args.structure_priors)
    nomask_deletions = read_deletions(args.nomask_json)
    hardmask_deletions = read_deletions(args.hardmask_json)

    nomask_summary, nomask_per, nomask_risk, nomask_skipped = analyze_run(
        "nomask", nomask_deletions, structure_priors, structure_summary
    )
    hardmask_summary, hardmask_per, hardmask_risk, hardmask_skipped = analyze_run(
        "hardmask", hardmask_deletions, structure_priors, structure_summary
    )

    per_rows = nomask_per + hardmask_per
    risk_rows = nomask_risk + hardmask_risk
    summary_rows = [nomask_summary, hardmask_summary]

    write_csv(
        os.path.join(args.out_dir, "per_protein_structure_risk.csv"),
        [
            "protein_id",
            "run_name",
            "length",
            "num_motif",
            "num_shadow_8A",
            "num_deleted",
            "num_deleted_motif",
            "num_deleted_shadow",
            "num_deleted_motif_contact",
            "shadow_deletion_rate_over_deleted",
            "mean_distance_to_motif_deleted",
            "mean_contact_density_deleted",
            "mean_contact_density_not_deleted",
            "mean_plddt_deleted",
            "mean_plddt_not_deleted",
        ],
        per_rows,
    )
    write_csv(
        os.path.join(args.out_dir, "high_risk_deletions.csv"),
        [
            "protein_id",
            "run_name",
            "position",
            "aa",
            "is_motif",
            "is_motif_shadow_8A",
            "distance_to_motif",
            "contact_density_8A",
            "motif_contact_count_8A",
            "plddt",
            "risk_type",
        ],
        risk_rows,
    )
    write_csv(
        os.path.join(args.out_dir, "summary.csv"),
        [
            "run_name",
            "analyzed_proteins",
            "total_deleted_residues",
            "motif_deleted_residues",
            "motif_deletion_rate",
            "shadow_deleted_residues",
            "shadow_deletion_rate_over_deleted",
            "proteins_with_shadow_deletion",
            "motif_contact_deleted_residues",
            "motif_contact_deletion_rate_over_deleted",
            "mean_distance_to_motif_deleted",
            "mean_distance_to_motif_all_nonmotif",
            "mean_contact_density_deleted",
            "mean_contact_density_not_deleted",
            "mean_plddt_deleted",
            "mean_plddt_not_deleted",
        ],
        summary_rows,
    )
    write_report(
        os.path.join(args.out_dir, "structure_prior_signal_report.txt"),
        len(motif_rows),
        len(structure_priors),
        {"nomask": nomask_summary, "hardmask": hardmask_summary},
        risk_rows,
        {"nomask": nomask_skipped, "hardmask": hardmask_skipped},
    )

    print("Wrote structure prior signal outputs to {0}".format(args.out_dir))


if __name__ == "__main__":
    main()
