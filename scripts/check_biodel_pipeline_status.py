#!/usr/bin/env python3
"""Check BioDel/BioPrior-10K pipeline outputs and write a reproducibility report."""

import argparse
import csv
import json
import os
from collections import Counter


def file_status(path, min_size=1):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    ok = exists and size >= min_size
    return {"path": path, "exists": exists, "size": size, "ok": ok}


def contains(path, token):
    if not os.path.exists(path):
        return False
    with open(path, errors="ignore") as handle:
        return token in handle.read()


def csv_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def check_files(paths):
    rows = []
    for label, path in paths:
        st = file_status(path)
        rows.append({"check": label, "path": path, "status": "PASS" if st["ok"] else "FAIL", "detail": "size={}".format(st["size"])})
    return rows


def validation_pass(path):
    return contains(path, "ALL_PASS")


def check_scisor_runs(root):
    rows = []
    for budget in (10, 20, 30):
        for method in ("nomask", "hardmask", "hardmask_shadow02"):
            run = "run{:02d}_{}".format(budget, method)
            run_dir = os.path.join(root, run)
            for filename in ("shrunk_sequences.fasta", "deletions.json", "validation_report.txt"):
                path = os.path.join(run_dir, filename)
                st = file_status(path)
                status = "PASS" if st["ok"] else "FAIL"
                detail = "size={}".format(st["size"])
                if filename == "validation_report.txt":
                    status = "PASS" if validation_pass(path) else "FAIL"
                    detail = "ALL_PASS={}".format(validation_pass(path))
                rows.append({"check": "scisor_{}_{}".format(run, filename), "path": path, "status": status, "detail": detail})
    return rows


def check_final_metrics(table_path):
    rows = []
    if not os.path.exists(table_path):
        return [{"check": "final_metric_table", "path": table_path, "status": "FAIL", "detail": "missing"}]
    table = read_csv(table_path)
    methods = {row["method"] for row in table}
    budgets = {row["budget"] for row in table}
    required_methods = {
        "utility_greedy_protected_only",
        "strict_safe_greedy",
        "v2_safe",
        "v2_balanced",
        "validation_selected_v2",
        "SCISOR no-mask",
        "SCISOR hardmask",
        "SCISOR hardmask+shadow02",
    }
    rows.append({
        "check": "final_methods_present",
        "path": table_path,
        "status": "PASS" if required_methods <= methods else "FAIL",
        "detail": "missing={}".format(",".join(sorted(required_methods - methods))),
    })
    rows.append({
        "check": "final_budgets_present",
        "path": table_path,
        "status": "PASS" if {"0.1", "0.2", "0.3"} <= budgets else "FAIL",
        "detail": "budgets={}".format(",".join(sorted(budgets))),
    })
    selected = [row for row in table if row["method"] == "validation_selected_v2"]
    protected_ok = all(float(row["protected_overlap_residues"]) == 0 for row in selected)
    rows.append({
        "check": "selected_v2_protected_zero",
        "path": table_path,
        "status": "PASS" if protected_ok and len(selected) == 3 else "FAIL",
        "detail": "selected_rows={}".format(len(selected)),
    })
    return rows


def write_csv(path, rows):
    ensure_parent(path)
    fields = ["check", "path", "status", "detail"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, summary):
    ensure_parent(path)
    counts = Counter(row["status"] for row in rows)
    with open(path, "w") as handle:
        handle.write("BioDel pipeline status report\n\n")
        handle.write("PASS: {}\n".format(counts["PASS"]))
        handle.write("FAIL: {}\n\n".format(counts["FAIL"]))
        handle.write("Summary metrics:\n")
        for key, value in summary.items():
            handle.write("- {}: {}\n".format(key, value))
        handle.write("\nFailed checks:\n")
        failed = [row for row in rows if row["status"] != "PASS"]
        if not failed:
            handle.write("- none\n")
        else:
            for row in failed:
                handle.write("- {}: {} ({})\n".format(row["check"], row["path"], row["detail"]))
        handle.write("\nKey artifacts:\n")
        for row in rows:
            if row["status"] == "PASS" and row["check"] in (
                "final_comparison_report",
                "budget_risk_frontier_report",
                "scisor_baseline_summary",
                "formal_split_report",
            ):
                handle.write("- {}: {}\n".format(row["check"], row["path"]))
        handle.write("\n{}\n".format("BIODEL_PIPELINE_STATUS_PASS" if counts["FAIL"] == 0 else "BIODEL_PIPELINE_STATUS_WARN"))


def main():
    parser = argparse.ArgumentParser(description="Check BioDel pipeline status.")
    parser.add_argument("--out_csv", default="results/biodel_planner/pipeline_status.csv")
    parser.add_argument("--out_report", default="results/biodel_planner/pipeline_status_report.txt")
    args = parser.parse_args()

    required_files = [
        ("experiment_config", "configs/bioprior_10k_experiment.yaml"),
        ("planner_config", "configs/biodel_planner_v2.yaml"),
        ("core_csv", "data/processed/swissprot_motif_bioprior_10k.csv"),
        ("train_split", "data/processed/bioprior_10k_splits/train.csv"),
        ("val_split", "data/processed/bioprior_10k_splits/val.csv"),
        ("test_split", "data/processed/bioprior_10k_splits/test.csv"),
        ("formal_split_report", "data/processed/bioprior_10k_splits/split_leakage_report.txt"),
        ("afdb_mapping", "data/structures/afdb_bioprior_10k_mapping.csv"),
        ("residue_biopriors", "data/features/bioprior_10k_residue_biopriors.csv"),
        ("test_residue_biopriors", "data/features/bioprior_10k_test_residue_biopriors.csv"),
        ("segments", "data/features/bioprior_10k_bioprior_segments.csv"),
        ("segments_with_utility", "data/features/bioprior_10k_bioprior_segments_with_stage1_utility.csv"),
        ("stage1_checkpoint", "checkpoints/stage1_deletion_prior_100k/best_model.pt"),
        ("val_v2_summary", "results/biodel_planner/bioprior_10k_val_v2_summary.csv"),
        ("test_v2_summary", "results/biodel_planner/bioprior_10k_test_v2_summary.csv"),
        ("selected_operating_points", "results/biodel_planner/bioprior_10k_selected_operating_points.json"),
        ("budget_risk_frontier_report", "results/biodel_planner/bioprior_10k_budget_risk_frontier_report.txt"),
        ("scisor_baseline_summary", "results/scisor_bioprior_10k_test/scisor_bioprior10k_test_baseline_summary.txt"),
        ("final_comparison_report", "results/biodel_planner/final_test_comparison/final_test_comparison_report.md"),
        ("final_comparison_table", "results/biodel_planner/final_test_comparison/final_test_comparison_table.csv"),
        ("ablation_summary", "results/biodel_planner/bioprior_10k_ablation_summary.csv"),
        ("ablation_selected_segments", "results/biodel_planner/bioprior_10k_ablation_selected_segments.csv"),
        ("ablation_report", "results/biodel_planner/bioprior_10k_ablation_report.txt"),
        ("underfill_summary", "results/biodel_planner/underfill_analysis/underfill_failure_modes_summary.csv"),
        ("underfill_by_protein", "results/biodel_planner/underfill_analysis/underfill_failure_modes_by_protein.csv"),
        ("underfill_report", "results/biodel_planner/underfill_analysis/underfill_failure_modes_report.md"),
        ("dataset_readiness_report", "results/biodel_planner/dataset_readiness_report.txt"),
        ("proteingym_deletion_benchmark", "results/proteingym_deletion_benchmark/proteingym_single_segment_deletions.csv"),
        ("proteingym_deletion_benchmark_summary", "results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_summary.txt"),
        ("proteingym_stage1_scored", "results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored.csv"),
        ("proteingym_stage1_scored_summary", "results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_summary.txt"),
        ("proteingym_external_validation_metrics", "results/proteingym_deletion_benchmark/proteingym_external_validation_metrics.csv"),
        ("proteingym_external_validation_report", "results/proteingym_deletion_benchmark/proteingym_external_validation_report.md"),
        ("internal_benchmark_report", "results/biodel_planner/internal_benchmark_suite_report.md"),
    ]
    rows = check_files(required_files)
    rows.extend(check_scisor_runs("results/scisor_bioprior_10k_test"))
    rows.extend(check_final_metrics("results/biodel_planner/final_test_comparison/final_test_comparison_table.csv"))

    token_checks = [
        ("split_leakage_pass", "data/processed/bioprior_10k_splits/split_leakage_report.txt", "BIOPRIOR_SPLIT_LEAKAGE_PASS"),
        ("stage1_utility_pass", "data/features/bioprior_10k_bioprior_segments_with_stage1_utility_summary.txt", "STAGE1_SEGMENT_UTILITY_PASS"),
        ("scisor_summary_pass", "results/scisor_bioprior_10k_test/scisor_bioprior10k_test_baseline_summary.txt", "SCISOR_BIOPRIOR10K_BASELINES_PASS"),
        ("final_comparison_pass", "results/biodel_planner/final_test_comparison/final_test_comparison_report.md", "BIODEL_FINAL_COMPARISON_PASS"),
        ("ablation_pass", "results/biodel_planner/bioprior_10k_ablation_report.txt", "BIODEL_ABLATION_SUITE_PASS"),
        ("underfill_analysis_pass", "results/biodel_planner/underfill_analysis/underfill_failure_modes_report.md", "BIODEL_UNDERFILL_ANALYSIS_PASS"),
        ("dataset_readiness_pass", "results/biodel_planner/dataset_readiness_report.txt", "DATASET_READINESS_PASS"),
        ("proteingym_benchmark_pass", "results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_summary.txt", "PROTEINGYM_DELETION_BENCHMARK_PASS"),
        ("proteingym_stage1_scoring_pass", "results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_summary.txt", "PROTEINGYM_STAGE1_SCORING_PASS"),
        ("proteingym_external_validation_pass", "results/proteingym_deletion_benchmark/proteingym_external_validation_report.md", "PROTEINGYM_EXTERNAL_VALIDATION_PASS"),
        ("internal_benchmark_pass", "results/biodel_planner/internal_benchmark_suite_report.md", "BIODEL_INTERNAL_BENCHMARK_PASS"),
    ]
    for check, path, token in token_checks:
        rows.append({"check": check, "path": path, "status": "PASS" if contains(path, token) else "FAIL", "detail": "token={}".format(token)})

    summary = {
        "core_rows": csv_rows("data/processed/swissprot_motif_bioprior_10k.csv"),
        "train_rows": csv_rows("data/processed/bioprior_10k_splits/train.csv"),
        "val_rows": csv_rows("data/processed/bioprior_10k_splits/val.csv"),
        "test_rows": csv_rows("data/processed/bioprior_10k_splits/test.csv"),
        "segment_rows": csv_rows("data/features/bioprior_10k_bioprior_segments.csv"),
        "segment_utility_rows": csv_rows("data/features/bioprior_10k_bioprior_segments_with_stage1_utility.csv"),
        "final_comparison_rows": csv_rows("results/biodel_planner/final_test_comparison/final_test_comparison_table.csv"),
        "ablation_rows": csv_rows("results/biodel_planner/bioprior_10k_ablation_summary.csv"),
        "underfill_rows": csv_rows("results/biodel_planner/underfill_analysis/underfill_failure_modes_summary.csv"),
        "proteingym_deletion_rows": csv_rows("results/proteingym_deletion_benchmark/proteingym_single_segment_deletions.csv"),
        "proteingym_metric_rows": csv_rows("results/proteingym_deletion_benchmark/proteingym_external_validation_metrics.csv"),
        "internal_benchmark_report": "results/biodel_planner/internal_benchmark_suite_report.md",
    }
    write_csv(args.out_csv, rows)
    write_report(args.out_report, rows, summary)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
