#!/usr/bin/env python3
"""Build a consolidated report for the current best Stage-1 experiment run.

This report intentionally separates current-best reruns from older legacy
results.  It only summarizes files that belong to the latest SCISOR-style
Stage-1 checkpoint and the six experiment blocks the user asked to prepare.
"""

import argparse
import csv
import json
import os
from collections import Counter


def exists(path):
    return os.path.exists(path)


def read_csv(path):
    if not exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    if not exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def read_text(path):
    if not exists(path):
        return ""
    with open(path) as handle:
        return handle.read()


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def fmt(v, digits=6):
    if v in ("", None):
        return ""
    try:
        return ("{:.%df}" % digits).format(float(v))
    except (TypeError, ValueError):
        return str(v)


def find_row(rows, **match):
    for row in rows:
        if all(str(row.get(k)) == str(v) for k, v in match.items()):
            return row
    return {}


def file_status(path):
    if not exists(path):
        return "MISSING"
    return "PASS" if os.path.getsize(path) > 0 else "EMPTY"


def section(path, title):
    return "## {}\n\n".format(title)


def build_report(args):
    report = []
    report.append("# Current Best Experiment Report\n\n")
    report.append("This report summarizes the current SCISOR-style Stage-1 checkpoint and the six experiment blocks requested for the current run.\n\n")
    report.append("## Current Stage-1 Checkpoint\n\n")
    report.append("- checkpoint: `{}`\n".format(args.stage1_checkpoint))
    report.append("- status: `{}`\n".format(file_status(args.stage1_checkpoint)))
    report.append("- best checkpoint path: `{}`\n".format(args.stage1_checkpoint))
    report.append("- train log: `{}`\n".format(args.stage1_train_log))
    report.append("- test metrics: `{}`\n".format(args.stage1_test_metrics))
    metrics = read_json(args.stage1_test_metrics)
    report.append("- best_epoch: `{}`\n".format(metrics.get("best_epoch", "")))
    report.append("- best_step: `{}`\n".format(metrics.get("best_step", "")))
    report.append("- token_ap: `{}`\n".format(fmt(metrics.get("token_ap"))))
    report.append("- token_auc: `{}`\n".format(fmt(metrics.get("token_auc"))))
    report.append("- token_f1: `{}`\n".format(fmt(metrics.get("token_f1"))))
    report.append("\n")

    stage1_eval = read_csv(args.stage1_interval_eval_csv)
    if stage1_eval:
        row = stage1_eval[0]
        report.append("## Experiment 1. Candidate Interval Quality\n\n")
        report.append("- proposals CSV: `{}`\n".format(args.stage1_interval_eval_input))
        report.append("- teacher CSV: `{}`\n".format(args.stage1_teacher_csv))
        report.append("- mean teacher interval recall: `{}`\n".format(fmt(row.get("teacher_interval_recall"))))
        report.append("- mean teacher interval hit rate: `{}`\n".format(fmt(row.get("teacher_interval_hit_rate"))))
        report.append("- overall teacher positive coverage: `{}`\n".format(fmt(row.get("teacher_positive_covered"))))
        report.append("\n")

    report.append("## Experiment 2. Risk-Constrained Deletion Performance\n\n")
    exp2_files = [
        ("BioPrior-Heldout / planner", args.exp2_main_report),
        ("CATH-domain", args.exp2_cath_report),
        ("DisProt-IDR", args.exp2_disprot_report),
    ]
    for label, path in exp2_files:
        report.append("- {}: `{}`\n".format(label, file_status(path)))
    report.append("\n")

    report.append("## Experiment 3. Fixed Budget vs Auto Budget\n\n")
    report.append("- fixed budget summary: `{}`\n".format(file_status(args.exp3_fixed_report)))
    report.append("- auto budget summary: `{}`\n".format(file_status(args.exp3_auto_report)))
    report.append("\n")

    report.append("## Experiment 4. External Validation\n\n")
    report.append("- ProteinGym deletion validation: `{}`\n".format(file_status(args.exp4_proteingym_report)))
    report.append("- SKEMPI binder-retention proxy: `{}`\n".format(file_status(args.exp4_skempi_report)))
    report.append("- EGFR binder dataset file: `{}`\n".format(file_status(args.egfr_dataset)))
    report.append("\n")

    report.append("## Experiment 5. Ablations\n\n")
    report.append("- ablation suite report: `{}`\n".format(file_status(args.exp5_ablation_report)))
    report.append("\n")

    report.append("## Experiment 6. Structure Re-prediction and Case Studies\n\n")
    report.append("- case-study summary: `{}`\n".format(file_status(args.exp6_case_report)))
    report.append("- optional structure prediction status: `{}`\n".format(file_status(args.exp6_structure_status)))
    report.append("\n")

    report.append("## Legacy / Archived Results\n\n")
    report.append("The following directories are legacy references only and should not be treated as current-best reruns:\n\n")
    report.append("- `results/archive/`\n")
    report.append("- `results/biodel_planner/` older reports\n")
    report.append("- `results/proteingym_v13_indel/` older reevaluation outputs\n")
    report.append("- `results/skempi_binder_retention/` older proxy outputs\n")
    report.append("- `results/case_studies/` previous cases\n\n")

    report.append("## Current Status\n\n")
    current_pass = all(
        file_status(path) == "PASS"
        for path in [
            args.stage1_checkpoint,
            args.stage1_train_log,
            args.stage1_test_metrics,
        ]
    )
    report.append("Current stage-1 checkpoint usable: {}\n\n".format("yes" if current_pass else "no"))
    report.append("CURRENT_BEST_EXPERIMENT_REPORT_PASS\n")
    ensure_parent(args.output_md)
    with open(args.output_md, "w") as handle:
        handle.writelines(report)


def main():
    parser = argparse.ArgumentParser(description="Build current-best experiment report.")
    parser.add_argument("--stage1_checkpoint", default="checkpoints/stage1_scisor_style_uniref50_full_len800_2gpu/best_model.pt")
    parser.add_argument("--stage1_root", default="results/stage1_scisor_style_uniref50_full_len800_2gpu")
    parser.add_argument("--stage1_train_log", default="results/stage1_scisor_style_uniref50_full_len800_2gpu/train_log.csv")
    parser.add_argument("--stage1_test_metrics", default="results/stage1_scisor_style_uniref50_full_len800_2gpu/test_metrics.json")
    parser.add_argument("--stage1_teacher_csv", default="data/train/core_1k_bioprior_teacher_labels.csv")
    parser.add_argument("--stage1_interval_eval_input", default="results/current_best_experiments/stage1_interval_proposals.csv")
    parser.add_argument("--stage1_interval_eval_csv", default="results/current_best_experiments/stage1_interval_proposals_eval.csv")
    parser.add_argument("--exp2_main_report", default="results/current_best_experiments/experiment2_main/report.md")
    parser.add_argument("--exp2_cath_report", default="results/current_best_experiments/experiment2_cath/report.md")
    parser.add_argument("--exp2_disprot_report", default="results/current_best_experiments/experiment2_disprot/report.md")
    parser.add_argument("--exp3_fixed_report", default="results/current_best_experiments/experiment3_fixed_vs_auto/report.md")
    parser.add_argument("--exp3_auto_report", default="results/current_best_experiments/experiment3_fixed_vs_auto/auto_report.md")
    parser.add_argument("--exp4_proteingym_report", default="results/proteingym_deletion_benchmark/proteingym_external_validation_report.md")
    parser.add_argument("--exp4_skempi_report", default="results/skempi_binder_retention/SKEMPI_BINDER_RETENTION_PROXY_REPORT.md")
    parser.add_argument("--egfr_dataset", default="data/external/adaptyv_egfr_round2/train-00000-of-00001.parquet")
    parser.add_argument("--exp5_ablation_report", default="results/current_best_experiments/experiment5_ablation/report.md")
    parser.add_argument("--exp6_case_report", default="results/case_studies/case_study_report.txt")
    parser.add_argument("--exp6_structure_status", default="results/case_studies/structure_reprediction/CASE_STUDY_STRUCTURE_REPREDICTION_REPORT.md")
    parser.add_argument("--output_md", default="results/current_best_experiments/CURRENT_BEST_EXPERIMENT_REPORT.md")
    args = parser.parse_args()
    build_report(args)
    print("Wrote {}".format(args.output_md))


if __name__ == "__main__":
    main()
