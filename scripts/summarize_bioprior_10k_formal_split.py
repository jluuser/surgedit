#!/usr/bin/env python3
"""Summarize the BioPrior-10K formal split planner pipeline."""

import argparse
import csv
import os


def read_text(path):
    with open(path) as handle:
        return handle.read()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def key_values_from_text(path):
    values = {}
    with open(path) as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            values[key.strip()] = value.strip()
    return values


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def format_planner_rows(rows):
    lines = ["setting,budget,fill,protected,shadow,closure_unfriendly_len"]
    for row in rows:
        lines.append(
            "{setting},{budget_ratio},{mean_fill_ratio:.4f},{protected_overlap_residues},{shadow_overlap_residues},{closure_unfriendly_len}".format(
                setting=row["setting"],
                budget_ratio=row["budget_ratio"],
                mean_fill_ratio=float(row["mean_fill_ratio"]),
                protected_overlap_residues=row["protected_overlap_residues"],
                shadow_overlap_residues=row["shadow_overlap_residues"],
                closure_unfriendly_len=row["closure_unfriendly_len"],
            )
        )
    return "\n".join(lines)


def write_matching_lines(handle, text, prefixes):
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in prefixes):
            handle.write("{}\n".format(line))


def main():
    parser = argparse.ArgumentParser(description="Summarize BioPrior-10K formal split outputs.")
    parser.add_argument("--split_report", required=True)
    parser.add_argument("--structure_summary", required=True)
    parser.add_argument("--residue_summary", required=True)
    parser.add_argument("--segment_summary", required=True)
    parser.add_argument("--utility_summary", required=True)
    parser.add_argument("--val_summary_csv", required=True)
    parser.add_argument("--test_summary_csv", required=True)
    parser.add_argument("--test_deletion_summary", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()

    split_text = read_text(args.split_report)
    structure_text = read_text(args.structure_summary)
    residue_text = read_text(args.residue_summary)
    utility_text = read_text(args.utility_summary)
    segment_text = read_text(args.segment_summary)
    val_rows = read_csv(args.val_summary_csv)
    test_rows = read_csv(args.test_summary_csv)
    deletion_text = read_text(args.test_deletion_summary)

    pass_checks = [
        "BIOPRIOR_SPLIT_LEAKAGE_PASS" in split_text,
        "STAGE1_SEGMENT_UTILITY_PASS" in utility_text,
        "CORE_1K_BIOPRIOR_SEGMENTS_PASS" in segment_text,
        "BIODEL_PLANNER_DELETION_OUTPUTS_PASS" in deletion_text,
    ]

    ensure_parent(args.out_report)
    with open(args.out_report, "w") as handle:
        handle.write("BioPrior-10K formal split pipeline report\n\n")
        handle.write("Protocol:\n")
        handle.write("- Core-1K remains a dev-only set for method development.\n")
        handle.write("- BioPrior-10K is split by accession/exact-sequence groups into train/val/test.\n")
        handle.write("- Planner settings are inspected on validation and evaluated on held-out test.\n\n")

        handle.write("Inputs:\n")
        handle.write("- split_report: {}\n".format(args.split_report))
        handle.write("- structure_summary: {}\n".format(args.structure_summary))
        handle.write("- residue_summary: {}\n".format(args.residue_summary))
        handle.write("- segment_summary: {}\n".format(args.segment_summary))
        handle.write("- utility_summary: {}\n".format(args.utility_summary))
        handle.write("- val_summary_csv: {}\n".format(args.val_summary_csv))
        handle.write("- test_summary_csv: {}\n".format(args.test_summary_csv))
        handle.write("- test_deletion_summary: {}\n\n".format(args.test_deletion_summary))

        handle.write("Leakage check:\n")
        write_matching_lines(
            handle,
            split_text,
            [
                "- accession_overlap",
                "- exact_sequence_overlap",
                "- train:",
                "- val:",
                "- test:",
                "BIOPRIOR_SPLIT_LEAKAGE_PASS",
            ],
        )
        handle.write("\n")

        handle.write("Structure and residue feature coverage:\n")
        write_matching_lines(
            handle,
            structure_text,
            [
                "targets:",
                "found:",
                "failed:",
                "success_rate:",
                "pLDDT readable",
                "length mismatch count:",
            ],
        )
        write_matching_lines(
            handle,
            residue_text,
            [
                "total proteins",
                "successfully processed proteins:",
                "skipped proteins:",
                "total residue rows:",
                "average protected residues",
                "average shadow residues",
                "average contact density:",
                "- mean:",
                "CORE_1K_RESIDUE_BIOPRIORS_PASS",
            ],
        )
        handle.write("\n")

        handle.write("Segment and Stage-1 utility:\n")
        for line in segment_text.splitlines():
            if line.startswith("processed_proteins") or line.startswith("total_segments") or line.startswith("avg_segments"):
                handle.write("{}\n".format(line))
        for line in utility_text.splitlines():
            if line.startswith("input_segments") or line.startswith("output_segments") or line.startswith("scored_accessions") or line.startswith("mean:") or line.endswith("_PASS"):
                handle.write("{}\n".format(line))
        handle.write("\n")

        handle.write("Validation split planner table:\n")
        handle.write(format_planner_rows(val_rows))
        handle.write("\n\nHeld-out test planner table:\n")
        handle.write(format_planner_rows(test_rows))
        handle.write("\n\nHeld-out test deletion validation:\n")
        for line in deletion_text.splitlines():
            if line.startswith("num_runs") or line.startswith("all_validation") or line.endswith("_PASS"):
                handle.write("{}\n".format(line))
        handle.write("\n")

        handle.write("Conclusion:\n")
        if all(pass_checks):
            handle.write("BIOPRIOR_10K_FORMAL_SPLIT_PIPELINE_PASS\n")
        else:
            handle.write("BIOPRIOR_10K_FORMAL_SPLIT_PIPELINE_WARN\n")

    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
