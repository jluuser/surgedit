#!/usr/bin/env python3
"""Run BioDel planners on the family-held-out BioPrior-10K split."""

import argparse
import os
import subprocess
import sys


PYTHON = "/public/home/zhangyangroup/chengshiz/anaconda3/envs/surgedit/bin/python"


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def run(cmd, execute, env=None):
    print(" ".join(cmd))
    if execute:
        subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run family-split BioDel planners.")
    parser.add_argument("--segments_csv", default="data/features/bioprior_10k_bioprior_segments_with_stage1_utility_certified.csv")
    parser.add_argument("--val_csv", default="data/processed/bioprior_10k_family_splits/val.csv")
    parser.add_argument("--test_csv", default="data/processed/bioprior_10k_family_splits/test.csv")
    parser.add_argument("--residue_priors", default="data/features/bioprior_10k_family_test_residue_biopriors.csv")
    parser.add_argument("--out_root", default="results/biodel_planner/family_split")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow_protected_overlap", action="store_true")
    parser.add_argument("--include_post_selection_calibration", action="store_true")
    parser.add_argument("--skip_scisor", action="store_true")
    parser.add_argument("--scisor_out_root", default="results/scisor_bioprior_10k_family_test")
    parser.add_argument("--scisor_budgets", default="10,20,30")
    parser.add_argument("--scisor_max_sequences", type=int, default=1000)
    args = parser.parse_args()

    ensure_dir(args.out_root)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"

    certified_cmd = [
        PYTHON,
        "scripts/run_certified_frontier_planner.py",
        "--segments_csv",
        args.segments_csv,
        "--calibration_filter_csv",
        args.val_csv,
        "--planning_filter_csv",
        args.test_csv,
        "--out_frontier_csv",
        os.path.join(args.out_root, "certified_frontier_test_frontier.csv"),
        "--out_protein_selection_csv",
        os.path.join(args.out_root, "certified_frontier_test_protein_selection.csv"),
        "--out_selected_segments_csv",
        os.path.join(args.out_root, "certified_frontier_test_selected_segments.csv"),
        "--out_summary_csv",
        os.path.join(args.out_root, "certified_frontier_test_summary.csv"),
        "--out_report",
        os.path.join(args.out_root, "CERTIFIED_FRONTIER_PLANNER_REPORT.md"),
    ]
    if args.allow_protected_overlap:
        certified_cmd.append("--allow_protected_overlap")
    run(certified_cmd, args.execute, env=env)

    auto_cmd = [
        PYTHON,
        "scripts/run_biodel_auto_budget_planner.py",
        "--segments_csv",
        args.segments_csv,
        "--accession_filter_csv",
        args.test_csv,
        "--out_frontier_csv",
        os.path.join(args.out_root, "bioprior_10k_test_auto_budget_frontier.csv"),
        "--out_selected_csv",
        os.path.join(args.out_root, "bioprior_10k_test_auto_budget_certified_selected_segments.csv"),
        "--out_summary_csv",
        os.path.join(args.out_root, "bioprior_10k_test_auto_budget_certified_summary.csv"),
        "--out_report",
        os.path.join(args.out_root, "bioprior_10k_test_auto_budget_certified_report.txt"),
    ]
    run(auto_cmd, args.execute, env=env)

    v2_cmd = [
        PYTHON,
        "scripts/run_biodel_planner_v2.py",
        "--segments_csv",
        args.segments_csv,
        "--accession_filter_csv",
        args.test_csv,
        "--out_selected_csv",
        os.path.join(args.out_root, "bioprior_10k_test_v2_selected_segments.csv"),
        "--out_summary_csv",
        os.path.join(args.out_root, "bioprior_10k_test_v2_summary.csv"),
        "--out_report",
        os.path.join(args.out_root, "bioprior_10k_test_v2_report.txt"),
    ]
    run(v2_cmd, args.execute, env=env)

    val_v2_cmd = [
        PYTHON,
        "scripts/run_biodel_planner_v2.py",
        "--segments_csv",
        args.segments_csv,
        "--accession_filter_csv",
        args.val_csv,
        "--out_selected_csv",
        os.path.join(args.out_root, "bioprior_10k_val_v2_selected_segments.csv"),
        "--out_summary_csv",
        os.path.join(args.out_root, "bioprior_10k_val_v2_summary.csv"),
        "--out_report",
        os.path.join(args.out_root, "bioprior_10k_val_v2_report.txt"),
    ]
    run(val_v2_cmd, args.execute, env=env)

    for split_name, split_csv, selected_csv in [
        ("family_val", args.val_csv, os.path.join(args.out_root, "bioprior_10k_val_v2_selected_segments.csv")),
        ("family_test", args.test_csv, os.path.join(args.out_root, "bioprior_10k_test_v2_selected_segments.csv")),
    ]:
        compare_cmd = [
            PYTHON,
            "scripts/compare_biodel_planner_baselines.py",
            "--segments_csv",
            args.segments_csv,
            "--split_csv",
            split_csv,
            "--split_name",
            split_name,
            "--v2_selected_csv",
            selected_csv,
            "--out_csv",
            os.path.join(args.out_root, "bioprior_10k_{}_planner_baseline_comparison.csv".format("val" if split_name.endswith("val") else "test")),
            "--out_report",
            os.path.join(args.out_root, "bioprior_10k_{}_planner_baseline_comparison.txt".format("val" if split_name.endswith("val") else "test")),
        ]
        run(compare_cmd, args.execute, env=env)

    select_cmd = [
        PYTHON,
        "scripts/select_biodel_operating_point.py",
        "--val_comparison_csv",
        os.path.join(args.out_root, "bioprior_10k_val_planner_baseline_comparison.csv"),
        "--test_comparison_csv",
        os.path.join(args.out_root, "bioprior_10k_test_planner_baseline_comparison.csv"),
        "--out_json",
        os.path.join(args.out_root, "bioprior_10k_selected_operating_points.json"),
        "--out_report",
        os.path.join(args.out_root, "bioprior_10k_selected_operating_points.txt"),
    ]
    run(select_cmd, args.execute, env=env)

    frontier_cmd = [
        PYTHON,
        "scripts/build_biodel_budget_risk_frontier.py",
        "--val_comparison_csv",
        os.path.join(args.out_root, "bioprior_10k_val_planner_baseline_comparison.csv"),
        "--test_comparison_csv",
        os.path.join(args.out_root, "bioprior_10k_test_planner_baseline_comparison.csv"),
        "--selected_operating_points_json",
        os.path.join(args.out_root, "bioprior_10k_selected_operating_points.json"),
        "--out_csv",
        os.path.join(args.out_root, "bioprior_10k_budget_risk_frontier.csv"),
        "--out_report",
        os.path.join(args.out_root, "bioprior_10k_budget_risk_frontier_report.txt"),
    ]
    run(frontier_cmd, args.execute, env=env)

    if args.include_post_selection_calibration:
        post_selection_cmd = [
            PYTHON,
            "scripts/run_post_selection_calibration_experiment.py",
            "--segments_csv",
            args.segments_csv,
            "--calibration_filter_csv",
            args.val_csv,
            "--planning_filter_csv",
            args.test_csv,
            "--risk_field",
            "risk_upper",
            "--frontier_max_plan_risk",
            "0.35",
            "--selection_risk_caps",
            "0.35,0.50,0.70,0.90",
            "--out_csv",
            os.path.join(args.out_root, "post_selection_calibration_riskupper_plannercap_sweep.csv"),
            "--out_report",
            os.path.join(args.out_root, "POST_SELECTION_CALIBRATION_RISKUPPER_PLANNERCAP_SWEEP_REPORT.md"),
        ]
        run(post_selection_cmd, args.execute, env=env)

    scisor_summary_csv = os.path.join(
        args.scisor_out_root,
        "scisor_bioprior10k_family_test_baseline_summary.csv",
    )
    if not args.skip_scisor:
        scisor_cmd = [
            PYTHON,
            "scripts/run_scisor_bioprior10k_test_baselines.py",
            "--input_fasta",
            args.test_csv.replace(".csv", ".fasta"),
            "--split_csv",
            args.test_csv,
            "--residue_priors",
            args.residue_priors,
            "--out_root",
            args.scisor_out_root,
            "--budgets",
            args.scisor_budgets,
            "--max_sequences",
            str(args.scisor_max_sequences),
            "--validate",
        ]
        if args.execute:
            scisor_cmd.append("--execute")
        run(scisor_cmd, args.execute, env=env)

        scisor_summary_cmd = [
            PYTHON,
            "scripts/summarize_scisor_bioprior10k_test_baselines.py",
            "--test_csv",
            args.test_csv,
            "--residue_priors",
            args.residue_priors,
            "--scisor_root",
            args.scisor_out_root,
            "--budgets",
            args.scisor_budgets,
            "--out_csv",
            scisor_summary_csv,
            "--out_report",
            os.path.join(args.scisor_out_root, "scisor_bioprior10k_family_test_baseline_summary.txt"),
        ]
        run(scisor_summary_cmd, args.execute, env=env)

    final_cmd = [
        PYTHON,
        "scripts/build_biodel_final_comparison_report.py",
        "--biodel_dir",
        args.out_root,
        "--split_csv",
        args.test_csv,
        "--split_name",
        "family_test",
        "--segments_csv",
        args.segments_csv,
        "--selected_segments_csv",
        os.path.join(args.out_root, "bioprior_10k_test_v2_selected_segments.csv"),
        "--scisor_summary_csv",
        scisor_summary_csv,
        "--out_dir",
        os.path.join(args.out_root, "final_test_family_comparison"),
    ]
    run(final_cmd, args.execute, env=env)

    print("Done. execute={}".format(args.execute))


if __name__ == "__main__":
    main()
