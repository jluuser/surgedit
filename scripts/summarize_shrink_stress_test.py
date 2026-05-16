#!/usr/bin/env python3
"""Summarize 10/20/30% SCISOR motif/shadow stress-test runs."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict


RUNS = [
    (10, "nomask", "results/scisor_swissprot_diverse/run10_nomask"),
    (10, "hardmask", "results/scisor_swissprot_diverse/run10_hardmask"),
    (10, "hardmask_shadow02", "results/scisor_swissprot_diverse/run10_hardmask_shadow02"),
    (20, "nomask", "results/scisor_swissprot_diverse/run20_nomask"),
    (20, "hardmask", "results/scisor_swissprot_diverse/run20_hardmask"),
    (20, "hardmask_shadow02", "results/scisor_swissprot_diverse/run20_hardmask_shadow02"),
    (30, "nomask", "results/scisor_swissprot_diverse/run30_nomask"),
    (30, "hardmask", "results/scisor_swissprot_diverse/run30_hardmask"),
    (30, "hardmask_shadow02", "results/scisor_swissprot_diverse/run30_hardmask_shadow02"),
]


def parse_positions(text):
    if text is None or text == "":
        return []
    return [int(x) for x in text.split(";") if x != ""]


def truthy(text):
    return str(text).strip().lower() in ("true", "1", "yes", "y")


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def safe_rate(num, den):
    if not den:
        return 0.0
    return num / float(den)


def read_motif_csv(path):
    proteins = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_id = row["protein_id"]
            sequence = row["sequence"]
            positions = sorted(set(parse_positions(row.get("protected_positions", ""))))
            proteins[protein_id] = {
                "length": int(row.get("length") or len(sequence)),
                "sequence": sequence,
                "protected_positions": set(positions),
            }
    return proteins


def read_structure_priors(path):
    priors = defaultdict(dict)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("structure_status") != "success":
                continue
            protein_id = row["protein_id"]
            position = int(row["position"])
            priors[protein_id][position] = {
                "is_motif": truthy(row.get("is_motif", "")),
                "is_shadow": truthy(row.get("is_motif_shadow_8A", "")),
                "motif_contact_count": int(float(row.get("motif_contact_count_8A") or 0)),
                "contact_density": float(row.get("contact_density_8A") or 0.0),
                "plddt": float(row.get("plddt") or 0.0),
            }
    return dict(priors)


def read_deletions(path):
    with open(path) as handle:
        rows = json.load(handle)
    deletions = {}
    for row in rows:
        protein_id = row.get("header") or row.get("protein_id")
        positions = row.get("deletion_positions_zero_based")
        if positions is None:
            positions = row.get("deleted_positions", [])
        deletions[protein_id] = {
            "original_length": int(row.get("original_length", 0)),
            "new_length": int(row.get("new_length", 0)),
            "deleted_positions": set(int(x) for x in positions),
        }
    return deletions


def validation_all_pass(path):
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        text = handle.read()
    return "ALL_PASS" in text and "\tFAIL\t" not in text


def target_new_length(length, shrink_pct):
    return int(length - math.ceil(length * shrink_pct / 100.0))


def summarize_run(shrink_pct, method_name, run_dir, proteins, priors):
    deletion_path = os.path.join(run_dir, "deletions.json")
    validation_path = os.path.join(run_dir, "validation_report.txt")
    if not os.path.exists(deletion_path):
        raise FileNotFoundError(deletion_path)

    deletions = read_deletions(deletion_path)
    validation_pass = validation_all_pass(validation_path)

    total_proteins = len(proteins)
    total_motif_residues = sum(
        len(v["protected_positions"]) for protein_id, v in proteins.items() if protein_id in priors
    )
    motif_deleted_residues = 0
    proteins_with_motif_deletion = 0
    compression_success_count = 0

    analyzed_proteins = 0
    total_deleted_residues = 0
    shadow_deleted_residues = 0
    proteins_with_shadow_deletion = 0
    motif_contact_deleted_residues = 0
    contact_density_deleted = []
    plddt_deleted = []
    shadow_counts_by_protein = {}

    for protein_id, info in proteins.items():
        deleted = deletions.get(protein_id, {}).get("deleted_positions", set())
        new_length = deletions.get(protein_id, {}).get("new_length", info["length"] - len(deleted))
        if new_length == target_new_length(info["length"], shrink_pct):
            compression_success_count += 1

        if protein_id not in priors:
            continue

        analyzed_proteins += 1
        motif_deleted = deleted & info["protected_positions"]
        motif_deleted_residues += len(motif_deleted)
        if motif_deleted:
            proteins_with_motif_deletion += 1

        prior = priors[protein_id]
        deleted_with_prior = [pos for pos in deleted if pos in prior]
        total_deleted_residues += len(deleted_with_prior)

        protein_shadow_deleted = 0
        for pos in deleted_with_prior:
            residue_prior = prior[pos]
            if residue_prior["is_shadow"]:
                shadow_deleted_residues += 1
                protein_shadow_deleted += 1
            if residue_prior["motif_contact_count"] > 0:
                motif_contact_deleted_residues += 1
            contact_density_deleted.append(residue_prior["contact_density"])
            plddt_deleted.append(residue_prior["plddt"])

        if protein_shadow_deleted:
            proteins_with_shadow_deletion += 1
        shadow_counts_by_protein[protein_id] = protein_shadow_deleted

    row = {
        "shrink_pct": shrink_pct,
        "method_name": method_name,
        "total_proteins": total_proteins,
        "analyzed_proteins_with_structure_prior": analyzed_proteins,
        "validation_ALL_PASS": validation_pass,
        "compression_success_count": compression_success_count,
        "compression_success_rate": safe_rate(compression_success_count, total_proteins),
        "total_deleted_residues": total_deleted_residues,
        "motif_deleted_residues": motif_deleted_residues,
        "motif_deletion_rate": safe_rate(motif_deleted_residues, total_motif_residues),
        "proteins_with_motif_deletion": proteins_with_motif_deletion,
        "shadow_deleted_residues": shadow_deleted_residues,
        "shadow_deletion_rate_over_deleted": safe_rate(shadow_deleted_residues, total_deleted_residues),
        "proteins_with_shadow_deletion": proteins_with_shadow_deletion,
        "motif_contact_deleted_residues": motif_contact_deleted_residues,
        "motif_contact_deletion_rate_over_deleted": safe_rate(
            motif_contact_deleted_residues, total_deleted_residues
        ),
        "mean_contact_density_deleted": mean(contact_density_deleted),
        "mean_plddt_deleted": mean(plddt_deleted),
    }
    return row, shadow_counts_by_protein


def format_rate(value):
    return "{:.4f}".format(value)


def write_csv(path, rows):
    fieldnames = [
        "shrink_pct",
        "method_name",
        "total_proteins",
        "analyzed_proteins_with_structure_prior",
        "validation_ALL_PASS",
        "compression_success_count",
        "compression_success_rate",
        "total_deleted_residues",
        "motif_deleted_residues",
        "motif_deletion_rate",
        "proteins_with_motif_deletion",
        "shadow_deleted_residues",
        "shadow_deletion_rate_over_deleted",
        "proteins_with_shadow_deletion",
        "motif_contact_deleted_residues",
        "motif_contact_deletion_rate_over_deleted",
        "mean_contact_density_deleted",
        "mean_plddt_deleted",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_report(rows, shadow_top_by_shrink, out_csv):
    by_key = {(row["shrink_pct"], row["method_name"]): row for row in rows}
    lines = []
    lines.append("Shrink Stress Test Summary")
    lines.append("")
    lines.append("summary_csv: {}".format(out_csv))
    lines.append("")
    lines.append("Core table:")
    lines.append("shrink_pct\tmethod\tmotif_deleted_residues\tshadow_deleted_residues\tcompression_success")
    for row in rows:
        lines.append(
            "{}\t{}\t{}\t{}\t{}/{} ({:.1f}%)".format(
                row["shrink_pct"],
                row["method_name"],
                row["motif_deleted_residues"],
                row["shadow_deleted_residues"],
                row["compression_success_count"],
                row["total_proteins"],
                100.0 * row["compression_success_rate"],
            )
        )

    lines.append("")
    lines.append("By shrink_pct analysis:")
    no_mask_motif = []
    any_validation_fail = False
    hardmask_stress_pass = True
    shadow_stress_pass = True
    shadow_tradeoff = False

    for shrink_pct in (10, 20, 30):
        nomask = by_key[(shrink_pct, "nomask")]
        hardmask = by_key[(shrink_pct, "hardmask")]
        shadow02 = by_key[(shrink_pct, "hardmask_shadow02")]
        no_mask_motif.append((shrink_pct, nomask["motif_deleted_residues"]))
        if not nomask["validation_ALL_PASS"] or not hardmask["validation_ALL_PASS"] or not shadow02["validation_ALL_PASS"]:
            any_validation_fail = True

        hardmask_reduction_text = "NA"
        if hardmask["shadow_deleted_residues"]:
            reduction = (
                hardmask["shadow_deleted_residues"] - shadow02["shadow_deleted_residues"]
            ) / float(hardmask["shadow_deleted_residues"])
            hardmask_reduction_text = "{:.2%}".format(reduction)
        else:
            reduction = 0.0

        if shrink_pct in (20, 30):
            if hardmask["motif_deleted_residues"] != 0:
                hardmask_stress_pass = False
            if not (shadow02["shadow_deleted_residues"] < hardmask["shadow_deleted_residues"] and shadow02["compression_success_rate"] >= 0.95):
                shadow_stress_pass = False
            if shadow02["compression_success_rate"] < 0.95:
                shadow_tradeoff = True

        lines.append(
            "{}%: no-mask motif_deleted={}, hardmask motif_deleted={}, "
            "hardmask shadow_deleted={}, shadow02 shadow_deleted={}, "
            "shadow reduction={}, shadow02 compression={}/{} ({:.1f}%).".format(
                shrink_pct,
                nomask["motif_deleted_residues"],
                hardmask["motif_deleted_residues"],
                hardmask["shadow_deleted_residues"],
                shadow02["shadow_deleted_residues"],
                hardmask_reduction_text,
                shadow02["compression_success_count"],
                shadow02["total_proteins"],
                100.0 * shadow02["compression_success_rate"],
            )
        )

    motif_trend = "non-decreasing"
    for i in range(1, len(no_mask_motif)):
        if no_mask_motif[i][1] < no_mask_motif[i - 1][1]:
            motif_trend = "not monotonic"
            break
    lines.append("")
    lines.append(
        "No-mask motif deletion trend across 10/20/30%: {} ({})".format(
            motif_trend,
            ", ".join("{}%={}".format(pct, count) for pct, count in no_mask_motif),
        )
    )

    lines.append("")
    lines.append("Hardmask vs shadow02 shadow deletion reduction:")
    for shrink_pct in (20, 30):
        hardmask = by_key[(shrink_pct, "hardmask")]
        shadow02 = by_key[(shrink_pct, "hardmask_shadow02")]
        if hardmask["shadow_deleted_residues"]:
            reduction = (
                hardmask["shadow_deleted_residues"] - shadow02["shadow_deleted_residues"]
            ) / float(hardmask["shadow_deleted_residues"])
            reduction_text = "{:.2%}".format(reduction)
        else:
            reduction_text = "NA"
        lines.append("{}% reduction: {}".format(shrink_pct, reduction_text))

    lines.append("")
    lines.append("Top shadow02 motif-shadow deletions by shrink_pct:")
    for shrink_pct in (10, 20, 30):
        lines.append("{}%:".format(shrink_pct))
        top = shadow_top_by_shrink.get(shrink_pct, [])
        if not top:
            lines.append("  none")
            continue
        for protein_id, count in top[:10]:
            lines.append("  {}\t{}".format(protein_id, count))

    lines.append("")
    if hardmask_stress_pass:
        lines.append("HARD_MASK_STRESS_PASS")
    else:
        lines.append("HARD_MASK_STRESS_FAIL: hardmask motif deletion was nonzero at 20% or 30%.")

    if shadow_stress_pass:
        lines.append("SHADOW_STRESS_PASS")
    else:
        lines.append("SHADOW_STRESS_WEAK: shadow02 did not reduce shadow deletion or compression success fell below 95%.")

    if shadow_tradeoff:
        lines.append("SHADOW_STRESS_TRADEOFF: 30% shadow02 compression success fell below 95%.")

    if any_validation_fail:
        lines.append("VALIDATION_WARN: at least one run did not report ALL_PASS.")
    else:
        lines.append("VALIDATION_ALL_PASS")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--structure_priors", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()

    proteins = read_motif_csv(args.motif_csv)
    priors = read_structure_priors(args.structure_priors)

    rows = []
    shadow_top_by_shrink = {}
    for shrink_pct, method_name, run_dir in RUNS:
        row, shadow_counts = summarize_run(shrink_pct, method_name, run_dir, proteins, priors)
        rows.append(row)
        if method_name == "hardmask_shadow02":
            shadow_top_by_shrink[shrink_pct] = sorted(
                shadow_counts.items(), key=lambda item: (-item[1], item[0])
            )[:10]

    write_csv(args.out_csv, rows)
    report = build_report(rows, shadow_top_by_shrink, args.out_csv)
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    with open(args.out_report, "w") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
