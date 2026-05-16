from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


RUN_NOMASK_COMMAND = (
    "timeout 7200s /public/home/zhangyangroup/chengshiz/anaconda3/envs/surgedit/bin/python "
    "scripts/run_scisor_shrink.py --input data/processed/swissprot_motif_mini_diverse.fasta "
    "--output-dir results/scisor_swissprot_diverse/run10_nomask --shrink-pct 10 "
    "--temperature 0 --disable-fa"
)
RUN_HARDMASK_COMMAND = (
    "timeout 7200s /public/home/zhangyangroup/chengshiz/anaconda3/envs/surgedit/bin/python "
    "scripts/run_scisor_shrink.py --input data/processed/swissprot_motif_mini_diverse.fasta "
    "--output-dir results/scisor_swissprot_diverse/run10_hardmask --shrink-pct 10 "
    "--temperature 0 --disable-fa --motif_csv data/processed/swissprot_motif_mini_diverse.csv "
    "--protect_motif"
)


def parse_positions(value):
    if value is None or value == "":
        return []
    return [int(token) for token in value.split(";") if token != ""]


def read_motif_csv(path):
    rows = []
    if sys.version_info[0] >= 3:
        handle = open(path, "r", newline="")
    else:
        handle = open(path, "rb")
    with handle:
        for row in csv.DictReader(handle):
            positions = parse_positions(row.get("protected_positions", ""))
            row["protected_positions_list"] = positions
            row["length_int"] = int(row["length"])
            rows.append(row)
    return rows


def read_deletions(path):
    with open(path) as handle:
        data = json.load(handle)
    return {entry["header"]: entry for entry in data}


def all_pass(path):
    with open(path) as handle:
        return "ALL_PASS" in handle.read()


def mean(values):
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def write_csv(path, rows):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    fields = [
        "protein_id",
        "primary_protected_type",
        "protected_type",
        "orig_len",
        "nomask_new_len",
        "hardmask_new_len",
        "target_new_len",
        "num_protected",
        "nomask_num_deleted_protected",
        "nomask_motif_deletion_rate",
        "hardmask_num_deleted_protected",
        "hardmask_motif_deletion_rate",
        "nomask_compression_success",
        "hardmask_compression_success",
        "nomask_deleted_protected_positions",
        "hardmask_deleted_protected_positions",
    ]
    if sys.version_info[0] >= 3:
        handle = open(path, "w", newline="")
    else:
        handle = open(path, "wb")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SCISOR no-mask and hardmask runs.")
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--nomask_json", required=True)
    parser.add_argument("--hardmask_json", required=True)
    parser.add_argument("--nomask_validation", required=True)
    parser.add_argument("--hardmask_validation", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--shrink_pct", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    motif_rows = read_motif_csv(args.motif_csv)
    nomask = read_deletions(args.nomask_json)
    hardmask = read_deletions(args.hardmask_json)
    nomask_all_pass = all_pass(args.nomask_validation)
    hardmask_all_pass = all_pass(args.hardmask_validation)

    comparison_rows = []
    missing = []
    for motif in motif_rows:
        protein_id = motif["protein_id"]
        if protein_id not in nomask or protein_id not in hardmask:
            missing.append(protein_id)
            continue

        protected = set(motif["protected_positions_list"])
        num_protected = len(protected)
        nomask_deleted = set(nomask[protein_id]["deletion_positions_zero_based"])
        hardmask_deleted = set(hardmask[protein_id]["deletion_positions_zero_based"])
        nomask_deleted_protected = sorted(protected & nomask_deleted)
        hardmask_deleted_protected = sorted(protected & hardmask_deleted)

        orig_len = motif["length_int"]
        target_new_len = orig_len - int(math.ceil(orig_len * args.shrink_pct / 100.0))
        nomask_new_len = int(nomask[protein_id]["new_length"])
        hardmask_new_len = int(hardmask[protein_id]["new_length"])
        nomask_rate = float(len(nomask_deleted_protected)) / num_protected
        hardmask_rate = float(len(hardmask_deleted_protected)) / num_protected

        comparison_rows.append(
            {
                "protein_id": protein_id,
                "primary_protected_type": motif.get("primary_protected_type", ""),
                "protected_type": motif.get("protected_type", ""),
                "orig_len": orig_len,
                "nomask_new_len": nomask_new_len,
                "hardmask_new_len": hardmask_new_len,
                "target_new_len": target_new_len,
                "num_protected": num_protected,
                "nomask_num_deleted_protected": len(nomask_deleted_protected),
                "nomask_motif_deletion_rate": "{0:.6g}".format(nomask_rate),
                "hardmask_num_deleted_protected": len(hardmask_deleted_protected),
                "hardmask_motif_deletion_rate": "{0:.6g}".format(hardmask_rate),
                "nomask_compression_success": str(nomask_new_len == target_new_len),
                "hardmask_compression_success": str(hardmask_new_len == target_new_len),
                "nomask_deleted_protected_positions": ";".join(
                    str(pos) for pos in nomask_deleted_protected
                ),
                "hardmask_deleted_protected_positions": ";".join(
                    str(pos) for pos in hardmask_deleted_protected
                ),
                "_protected_positions": motif.get("protected_positions", ""),
                "_nomask_rate_float": nomask_rate,
                "_hardmask_rate_float": hardmask_rate,
            }
        )

    write_csv(args.out_csv, comparison_rows)

    total = len(comparison_rows)
    nomask_deleted_count = sum(
        1 for row in comparison_rows if row["nomask_num_deleted_protected"] > 0
    )
    hardmask_deleted_count = sum(
        1 for row in comparison_rows if row["hardmask_num_deleted_protected"] > 0
    )
    nomask_avg_rate = mean([row["_nomask_rate_float"] for row in comparison_rows])
    hardmask_avg_rate = mean([row["_hardmask_rate_float"] for row in comparison_rows])
    nomask_success_count = sum(
        1 for row in comparison_rows if row["nomask_compression_success"] == "True"
    )
    hardmask_success_count = sum(
        1 for row in comparison_rows if row["hardmask_compression_success"] == "True"
    )
    hardmask_zero_deleted = hardmask_deleted_count == 0

    grouped_rates = defaultdict(list)
    for row in comparison_rows:
        grouped_rates[row["primary_protected_type"]].append(row["_nomask_rate_float"])

    top_worst = sorted(
        comparison_rows,
        key=lambda row: (
            row["_nomask_rate_float"],
            row["nomask_num_deleted_protected"],
            row["num_protected"],
        ),
        reverse=True,
    )[:10]

    parent = os.path.dirname(args.out_report)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.out_report, "wb") as handle:
        lines = []
        lines.append("Swiss-Prot diverse hardmask comparison")
        lines.append("")
        lines.append("Input paths:")
        lines.append("- motif_csv: {0}".format(args.motif_csv))
        lines.append("- nomask_json: {0}".format(args.nomask_json))
        lines.append("- hardmask_json: {0}".format(args.hardmask_json))
        lines.append("- nomask_validation: {0}".format(args.nomask_validation))
        lines.append("- hardmask_validation: {0}".format(args.hardmask_validation))
        lines.append("- out_csv: {0}".format(args.out_csv))
        lines.append("")
        lines.append("Run commands:")
        lines.append("No mask:")
        lines.append(RUN_NOMASK_COMMAND)
        lines.append("Hard mask:")
        lines.append(RUN_HARDMASK_COMMAND)
        lines.append("")
        lines.append("Mini-set total proteins: {0}".format(total))
        if missing:
            lines.append("Missing proteins in run outputs: {0}".format(";".join(missing)))
        lines.append("No-mask validation ALL_PASS: {0}".format(nomask_all_pass))
        lines.append("Hardmask validation ALL_PASS: {0}".format(hardmask_all_pass))
        lines.append("")
        lines.append(
            "No-mask proteins deleting protected positions: {0}".format(
                nomask_deleted_count
            )
        )
        lines.append("No-mask average motif_deletion_rate: {0:.6f}".format(nomask_avg_rate))
        lines.append("No-mask motif_deletion_rate by primary_protected_type:")
        for feature_type in ["active site", "binding site", "site", "short sequence motif"]:
            lines.append(
                "- {0}: {1:.6f}".format(feature_type, mean(grouped_rates[feature_type]))
            )
        lines.append("")
        lines.append(
            "Hardmask proteins deleting protected positions: {0}".format(
                hardmask_deleted_count
            )
        )
        lines.append(
            "Hardmask average motif_deletion_rate: {0:.6f}".format(hardmask_avg_rate)
        )
        lines.append(
            "Hardmask all num_deleted_protected = 0: {0}".format(
                hardmask_zero_deleted
            )
        )
        lines.append(
            "No-mask compression success rate: {0}/{1} ({2:.2%})".format(
                nomask_success_count,
                total,
                float(nomask_success_count) / total if total else 0.0,
            )
        )
        lines.append(
            "Hardmask compression success rate: {0}/{1} ({2:.2%})".format(
                hardmask_success_count,
                total,
                float(hardmask_success_count) / total if total else 0.0,
            )
        )
        lines.append("")
        lines.append("Top 10 no-mask protected-position deletions:")
        lines.append(
            "protein_id\tprimary_protected_type\tprotected_positions\tdeleted_protected_positions\tmotif_deletion_rate"
        )
        for row in top_worst:
            lines.append(
                "{0}\t{1}\t{2}\t{3}\t{4}".format(
                    row["protein_id"],
                    row["primary_protected_type"],
                    row["_protected_positions"],
                    row["nomask_deleted_protected_positions"],
                    row["nomask_motif_deletion_rate"],
                )
            )
        lines.append("")

        failure_reasons = []
        if not hardmask_all_pass:
            failure_reasons.append("hardmask validation did not ALL_PASS")
        if not hardmask_zero_deleted:
            failure_reasons.append("hardmask deleted protected positions")
        if failure_reasons:
            lines.append("Conclusion:")
            lines.append("HARD_MASK_REAL_DIVERSE_FAIL")
            lines.append("Failure reasons: {0}".format("; ".join(failure_reasons)))
        else:
            lines.append("Conclusion:")
            lines.append("HARD_MASK_REAL_DIVERSE_PASS")

        handle.write(("\n".join(lines) + "\n").encode("utf-8"))

    print("Wrote {0}".format(args.out_csv))
    print("Wrote {0}".format(args.out_report))
    if failure_reasons:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
