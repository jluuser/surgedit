#!/usr/bin/env python3
"""Summarize exported BioDel planner deletion outputs and validation reports."""

import argparse
import csv
import os


def read_key_values(path):
    values = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def validation_all_pass(path):
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        return any(line.strip() == "ALL_PASS" for line in handle)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def summarize(args):
    rows = []
    with open(args.manifest_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out_dir = row["out_dir"]
            report = read_key_values(row["report"])
            validation_path = os.path.join(out_dir, "validation_report.txt")
            rows.append(
                {
                    "setting": row["setting"],
                    "budget_ratio": row["budget_ratio"],
                    "out_dir": out_dir,
                    "total_proteins": report.get("total_proteins", ""),
                    "proteins_with_deletion": report.get("proteins_with_deletion", ""),
                    "total_original_length": report.get("total_original_length", ""),
                    "total_deleted_residues": report.get("total_deleted_residues", ""),
                    "global_fill_ratio_vs_nominal_budget": report.get("global_fill_ratio_vs_nominal_budget", ""),
                    "validation_ALL_PASS": validation_all_pass(validation_path),
                }
            )
    ensure_parent(args.out_csv)
    with open(args.out_csv, "w", newline="") as handle:
        fields = [
            "setting",
            "budget_ratio",
            "out_dir",
            "total_proteins",
            "proteins_with_deletion",
            "total_original_length",
            "total_deleted_residues",
            "global_fill_ratio_vs_nominal_budget",
            "validation_ALL_PASS",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    ensure_parent(args.out_report)
    all_pass = all(row["validation_ALL_PASS"] for row in rows)
    with open(args.out_report, "w") as handle:
        handle.write("BioDel planner deletion output summary\n\n")
        handle.write("manifest_csv: {}\n".format(args.manifest_csv))
        handle.write("num_runs: {}\n".format(len(rows)))
        handle.write("all_validation_ALL_PASS: {}\n\n".format(all_pass))
        handle.write("setting,budget,total_deleted,fill,validation\n")
        for row in rows:
            handle.write(
                "{setting},{budget_ratio},{total_deleted_residues},{global_fill_ratio_vs_nominal_budget},{validation_ALL_PASS}\n".format(
                    **row
                )
            )
        handle.write("\n{}\n".format("BIODEL_PLANNER_DELETION_OUTPUTS_PASS" if all_pass else "BIODEL_PLANNER_DELETION_OUTPUTS_WARN"))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize planner deletion output validation.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
