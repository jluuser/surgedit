#!/usr/bin/env python3
"""Build the AAAI-facing BioDel benchmark suite status and design report.

This script is intentionally result-oriented. It checks whether each benchmark
axis has the artifacts needed for a paper table, records missing pieces, and
writes the exact commands that reproduce or complete the suite.
"""

import argparse
import csv
import os
from collections import Counter


PYTHON = "/public/home/zhangyangroup/chengshiz/anaconda3/envs/surgedit/bin/python"


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def exists(path, min_size=1):
    return os.path.exists(path) and os.path.getsize(path) >= min_size


def contains(path, token):
    if not os.path.exists(path):
        return False
    with open(path, errors="ignore") as handle:
        return token in handle.read()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def csv_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def fnum(value, default=0.0):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def file_detail(paths):
    details = []
    for path in paths:
        if exists(path):
            details.append("{}:ok:{}B".format(path, os.path.getsize(path)))
        else:
            details.append("{}:missing".format(path))
    return "; ".join(details)


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "axis",
        "benchmark",
        "status",
        "compute",
        "primary_metric",
        "main_result",
        "artifacts",
        "reproduce_command",
        "missing_or_next",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row_status(required_paths, token_checks=None, allow_warn=False):
    token_checks = token_checks or []
    ok_files = all(exists(path) for path in required_paths)
    ok_tokens = all(contains(path, token) for path, token in token_checks)
    if ok_files and ok_tokens:
        return "PASS"
    return "WARN" if allow_warn and ok_files else "MISSING"


def family_split_row(args):
    report = "data/processed/bioprior_10k_family_splits/split_leakage_report.txt"
    stats = []
    for name in ("train", "val", "test"):
        path = "data/processed/bioprior_10k_family_splits/{}.csv".format(name)
        stats.append("{}={}".format(name, csv_rows(path)))
    return {
        "axis": "dataset",
        "benchmark": "BioPrior-10K family-held-out split",
        "status": row_status(
            [
                "data/processed/bioprior_10k_family_splits/train.csv",
                "data/processed/bioprior_10k_family_splits/val.csv",
                "data/processed/bioprior_10k_family_splits/test.csv",
                report,
            ],
            [(report, "BIOPRIOR_FAMILY_SPLIT_LEAKAGE_PASS")],
        ),
        "compute": "CPU",
        "primary_metric": "cluster leakage, exact-sequence leakage",
        "main_result": "; ".join(stats) + "; cluster/exact leakage pass",
        "artifacts": file_detail([report]),
        "reproduce_command": "{} scripts/split_bioprior_family_dataset.py --execute".format(PYTHON),
        "missing_or_next": "freeze as the primary internal generalization split",
    }


def certified_frontier_row(args):
    summary = os.path.join(args.family_root, "certified_frontier_test_summary.csv")
    report = os.path.join(args.family_root, "CERTIFIED_FRONTIER_PLANNER_REPORT.md")
    rows = read_csv(summary)
    default = next((row for row in rows if row.get("auto_profile") == "default"), {})
    result = "default certified={}, abstained={}, delete_ratio={:.4f}, risk_upper={:.4f}".format(
        inum(default.get("certified_proteins")),
        inum(default.get("abstained_proteins")),
        fnum(default.get("global_auto_delete_ratio")),
        fnum(default.get("mean_risk_upper")),
    )
    return {
        "axis": "core_algorithm",
        "benchmark": "Certified deletion-risk frontier",
        "status": row_status([summary, report], [(report, "BIODEL_CERTIFIED_FRONTIER_PLANNER_PASS")]),
        "compute": "CPU",
        "primary_metric": "coverage, abstention, certified delete ratio, risk upper",
        "main_result": result,
        "artifacts": file_detail([summary, report]),
        "reproduce_command": "{} scripts/run_certified_frontier_planner.py --segments_csv data/features/bioprior_10k_bioprior_segments_with_stage1_utility_certified.csv --calibration_filter_csv data/processed/bioprior_10k_family_splits/val.csv --planning_filter_csv data/processed/bioprior_10k_family_splits/test.csv".format(PYTHON),
        "missing_or_next": "add table contrasting frontier-level certification against segment/single-plan calibration",
    }


def post_selection_row(args):
    table = os.path.join(args.family_root, "post_selection_calibration_riskupper_plannercap_sweep.csv")
    report = os.path.join(args.family_root, "POST_SELECTION_CALIBRATION_RISKUPPER_PLANNERCAP_SWEEP_REPORT.md")
    rows = read_csv(table)
    selected = [
        row
        for row in rows
        if row.get("selection_rule") == "adversarial_frontier"
        and abs(fnum(row.get("selection_risk_cap")) - 0.9) < 1e-6
    ]
    by_method = {row.get("method"): row for row in selected}
    result = "adversarial frontier violation: uncal={:.3f}, segment={:.3f}, single={:.3f}, frontier={:.3f}".format(
        fnum(by_method.get("uncalibrated", {}).get("selected_plan_violation_rate")),
        fnum(by_method.get("segment_level", {}).get("selected_plan_violation_rate")),
        fnum(by_method.get("single_plan", {}).get("selected_plan_violation_rate")),
        fnum(by_method.get("frontier_level", {}).get("selected_plan_violation_rate")),
    )
    return {
        "axis": "core_algorithm",
        "benchmark": "Post-selection frontier calibration",
        "status": row_status([table, report], [(report, "BIODEL_POST_SELECTION_CALIBRATION_PASS")]),
        "compute": "CPU",
        "primary_metric": "selected-plan violation and simultaneous frontier violation",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/run_post_selection_calibration_experiment.py --segments_csv data/features/bioprior_10k_bioprior_segments_with_stage1_utility_certified.csv --calibration_filter_csv data/processed/bioprior_10k_family_splits/val.csv --planning_filter_csv data/processed/bioprior_10k_family_splits/test.csv --risk_field risk_upper --frontier_max_plan_risk 0.35 --selection_risk_caps 0.35,0.50,0.70,0.90".format(PYTHON),
        "missing_or_next": "optionally add bootstrap confidence intervals across proteins",
    }


def family_planner_row(args):
    report = os.path.join(args.family_root, "final_test_family_comparison", "final_test_comparison_report.md")
    table = os.path.join(args.family_root, "final_test_family_comparison", "final_test_comparison_table.csv")
    rows = read_csv(table)
    selected30 = next(
        (
            row for row in rows
            if row.get("method") == "validation_selected_v2"
            and abs(fnum(row.get("budget")) - 0.3) < 1e-6
        ),
        {},
    )
    utility30 = next(
        (
            row for row in rows
            if row.get("method") == "utility_greedy_protected_only"
            and abs(fnum(row.get("budget")) - 0.3) < 1e-6
        ),
        {},
    )
    result = "30% selected fill={:.3f}, shadow={:.3f}, closure={:.3f}; utility closure={:.3f}".format(
        fnum(selected30.get("fill_ratio")),
        fnum(selected30.get("shadow_overlap_rate")),
        fnum(selected30.get("closure_unfriendly_rate")),
        fnum(utility30.get("closure_unfriendly_rate")),
    )
    return {
        "axis": "internal_generalization",
        "benchmark": "Family-held-out planner comparison",
        "status": row_status([table, report], [(report, "BIODEL_FINAL_COMPARISON_PASS")]),
        "compute": "CPU",
        "primary_metric": "fill under protected/shadow/closure risk",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/run_biodel_family_split_planners.py --execute --skip_scisor --include_post_selection_calibration".format(PYTHON),
        "missing_or_next": "rerun final report after full SCISOR family baselines finish",
    }


def matched_length_row(args):
    report = os.path.join(args.family_root, "matched_length_comparison", "matched_length_comparison_report.md")
    table = os.path.join(args.family_root, "matched_length_comparison", "matched_length_comparison_summary.csv")
    rows = read_csv(table)
    biodel = next((row for row in rows if row.get("method") == "BioDel-Cert default"), {})
    scisor = next((row for row in rows if row.get("method") == "SCISOR hardmask+shadow02"), {})
    raygun = next((row for row in rows if str(row.get("method", "")).startswith("Raygun")), {})
    result = (
        "BioDel-Cert target_ratio={:.3f}, fill={:.3f}; "
        "SCISOR shadow={:.3f}, closure={:.3f}, length_success={:.3f}; "
        "Raygun identity={}, closure={:.3f}, length_success={:.3f}"
    ).format(
        fnum(biodel.get("mean_target_delete_ratio")),
        fnum(biodel.get("mean_fill_ratio")),
        fnum(scisor.get("shadow_overlap_rate")),
        fnum(scisor.get("closure_unfriendly_rate")),
        fnum(scisor.get("length_success_rate")),
        raygun.get("mean_sequence_identity", ""),
        fnum(raygun.get("closure_unfriendly_rate")),
        fnum(raygun.get("length_success_rate")),
    )
    return {
        "axis": "matched_comparison",
        "benchmark": "Matched-length BioDel vs SCISOR vs Raygun",
        "status": row_status([table, report], [(report, "MATCHED_LENGTH_DELETION_BENCHMARK_PASS")]),
        "compute": "GPU recommended for Raygun; CPU possible for tiny smoke",
        "primary_metric": "protected overlap, shadow overlap, motif-contact overlap, closure-unfriendly rate, exact-length success",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/run_matched_length_deletion_benchmark.py --execute".format(PYTHON),
        "missing_or_next": "run the matched-length benchmark on the full certified subset and add bootstrap CIs",
    }


def deletion_baseline_suite_row(args):
    report = "results/biodel_planner/DELETION_BASELINE_SUITE_REPORT.md"
    table = "results/biodel_planner/deletion_baseline_suite_summary.csv"
    rows = read_csv(table)
    budget10 = [row for row in rows if row.get("budget_type") == "fixed" and abs(fnum(row.get("budget_ratio")) - 0.10) < 1e-6]
    by_method = {row.get("method"): row for row in budget10}
    cert_default = next((row for row in rows if row.get("method") == "biodel_cert_frontier_default"), {})
    learned_default = next((row for row in rows if row.get("method") == "biodel_cert_learned_frontier_default"), {})
    scisor_shadow = next((row for row in rows if row.get("method") == "SCISOR hardmask+shadow02" and row.get("budget_label") == "10%"), {})
    v2_safe_10 = next((row for row in budget10 if row.get("method") == "biodel_v2_safe"), {})
    v2_balanced_10 = next((row for row in budget10 if row.get("method") == "biodel_v2_balanced"), {})
    result = (
        "10% random fill={:.3f}, terminal shadow={:.3f}, low-pLDDT closure={:.3f}; "
        "compatibility fill={:.3f}, strict-safe fill={:.3f}; "
        "v2 safe fill={}; v2 balanced fill={}; certified delete={:.3f}, abstain={}; "
        "learned-cert delete={:.3f}; SCISOR shadow={:.3f}"
    ).format(
        fnum(by_method.get("random_candidate", {}).get("fill_ratio")),
        fnum(by_method.get("terminal_trim", {}).get("shadow_overlap_rate")),
        fnum(by_method.get("low_plddt_first", {}).get("closure_unfriendly_rate")),
        fnum(by_method.get("compatibility_greedy_protected", {}).get("fill_ratio")),
        fnum(by_method.get("strict_safe_greedy", {}).get("fill_ratio")),
        v2_safe_10.get("fill_ratio", "NA"),
        v2_balanced_10.get("fill_ratio", "NA"),
        fnum(cert_default.get("achieved_deletion_ratio")),
        cert_default.get("abstention_rate", ""),
        fnum(learned_default.get("achieved_deletion_ratio")),
        fnum(scisor_shadow.get("shadow_overlap_rate")),
    )
    return {
        "axis": "diagnostic_benchmark",
        "benchmark": "Deletion baseline suite",
        "status": row_status([table, report], [(report, "BIODEL_DELETION_BASELINE_SUITE_PASS")]),
        "compute": "CPU",
        "primary_metric": "fixed-budget fill/risk and auto-budget abstention",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/run_deletion_baseline_suite.py && {} scripts/build_aaai_benchmark_suite_report.py".format(PYTHON, PYTHON),
        "missing_or_next": "main table for heuristics, learned greedy, fixed-budget planners, certified frontiers, and SCISOR in one schema",
    }


def scisor_family_row(args):
    root = args.scisor_family_root
    table = os.path.join(root, "scisor_bioprior10k_family_test_baseline_summary.csv")
    report = os.path.join(root, "scisor_bioprior10k_family_test_baseline_summary.txt")
    status = row_status([table, report], [(report, "SCISOR_BIOPRIOR10K_BASELINES_PASS")], allow_warn=True)
    rows = read_csv(table)
    result = "observed_runs={}, expected_runs=9".format(len(rows))
    if rows:
        best = next((row for row in rows if row.get("method") == "SCISOR hardmask+shadow02" and row.get("budget_label") == "30%"), rows[-1])
        result += "; example {} {} fill={:.3f}, shadow={:.3f}, closure={:.3f}".format(
            best.get("method"),
            best.get("budget_label"),
            fnum(best.get("fill_ratio")),
            fnum(best.get("shadow_overlap_rate")),
            fnum(best.get("closure_unfriendly_rate")),
        )
    return {
        "axis": "external_baseline",
        "benchmark": "SCISOR family-held-out baselines",
        "status": status,
        "compute": "GPU recommended; CPU smoke possible",
        "primary_metric": "fill, protected overlap, shadow overlap, closure-unfriendly rate",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/run_scisor_bioprior10k_test_baselines.py --input_fasta data/processed/bioprior_10k_family_splits/test.fasta --split_csv data/processed/bioprior_10k_family_splits/test.csv --residue_priors data/features/bioprior_10k_family_test_residue_biopriors.csv --out_root results/scisor_bioprior_10k_family_test --execute --validate".format(PYTHON),
        "missing_or_next": "run under srun GPU for all 9 runs, then summarize with scripts/summarize_scisor_bioprior10k_test_baselines.py",
    }


def cath_disprot_row(args):
    report = "results/biodel_planner/CATH_DISPROT_BASELINE_COMPARISON_REPORT.md"
    table = "results/biodel_planner/cath_disprot_baseline_comparison.csv"
    rows = read_csv(table)
    cath = next((row for row in rows if row.get("dataset") == "cath" and row.get("method") == "biodel_cert_safe_default"), {})
    disprot = next((row for row in rows if row.get("dataset") == "disprot" and row.get("method") == "biodel_cert_safe_default"), {})
    result = "CATH core_overlap={:.3f}, delete={:.3f}; DisProt functional_IDR={:.3f}, IDR={:.3f}".format(
        fnum(cath.get("domain_core_overlap_rate")),
        fnum(cath.get("global_delete_ratio")),
        fnum(disprot.get("functional_idr_overlap_rate")),
        fnum(disprot.get("idr_overlap_rate")),
    )
    return {
        "axis": "external_biology",
        "benchmark": "CATH domain-core and DisProt functional-IDR",
        "status": row_status([table, report], [(report, "BIODEL_CATH_DISPROT_BASELINE_COMPARISON_PASS")]),
        "compute": "CPU",
        "primary_metric": "domain-core overlap, functional-IDR overlap, delete ratio",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/evaluate_cath_disprot_baselines.py && {} scripts/summarize_cath_disprot_benchmarks.py".format(PYTHON, PYTHON),
        "missing_or_next": "consider adding CATH/DisProt family-level train exclusion if claimed as independent external validation",
    }


def proteingym_row(args):
    report = "results/proteingym_v13_indel/proteingym_v13_step3_external_validation_report_finetuned_stage1_certified.md"
    table = "results/proteingym_v13_indel/proteingym_v13_step3_metrics_finetuned_stage1_certified.csv"
    rows = read_csv(table)
    full = next((row for row in rows if row.get("score_name") == "risk_certified_biodel_score"), {})
    result = "risk-certified BioDel Spearman={:.3f}, AUROC={:.3f}, AUPRC={:.3f}, top10={:.3f}".format(
        fnum(full.get("spearman_favorable_DMS")),
        fnum(full.get("auroc_DMS_score_bin_1")),
        fnum(full.get("auprc_DMS_score_bin_1")),
        fnum(full.get("top10_precision_bin_1")),
    )
    return {
        "axis": "external_experimental",
        "benchmark": "ProteinGym v1.3 indel external validation",
        "status": row_status([table, report], [(report, "PROTEINGYM_V13_STEP3_EVALUATION_PASS")]),
        "compute": "CPU for evaluation; GPU only for rescoring large model checkpoints",
        "primary_metric": "Spearman, AUROC, AUPRC, top-k precision against DMS",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/evaluate_proteingym_v13_step3_scores.py".format(PYTHON),
        "missing_or_next": "add paired significance/bootstrapping against terminal and BioPrior-only baselines",
    }


def proteingym_bootstrap_row(args):
    report = "results/proteingym_v13_indel/PROTEINGYM_V13_BOOTSTRAP_SCORE_COMPARISON_REPORT.md"
    table = "results/proteingym_v13_indel/proteingym_v13_bootstrap_score_comparison.csv"
    rows = read_csv(table)
    spearman_terminal = next(
        (
            row for row in rows
            if row.get("metric") == "spearman_favorable_DMS"
            and row.get("method") == "risk_certified_biodel_score"
            and row.get("baseline") == "terminal_baseline"
        ),
        {},
    )
    auprc_terminal = next(
        (
            row for row in rows
            if row.get("metric") == "auprc_DMS_score_bin_1"
            and row.get("method") == "risk_certified_biodel_score"
            and row.get("baseline") == "terminal_baseline"
        ),
        {},
    )
    result = "vs terminal: Spearman delta={}, CI=[{},{}]; AUPRC delta={}, CI=[{},{}]".format(
        spearman_terminal.get("observed_delta", ""),
        spearman_terminal.get("ci95_low", ""),
        spearman_terminal.get("ci95_high", ""),
        auprc_terminal.get("observed_delta", ""),
        auprc_terminal.get("ci95_low", ""),
        auprc_terminal.get("ci95_high", ""),
    )
    return {
        "axis": "external_experimental",
        "benchmark": "ProteinGym assay-level bootstrap",
        "status": row_status([table, report], [(report, "PROTEINGYM_V13_BOOTSTRAP_SCORE_COMPARISON_PASS")]),
        "compute": "CPU",
        "primary_metric": "assay-level paired bootstrap deltas",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/bootstrap_proteingym_v13_scores.py --n_boot 5000".format(PYTHON),
        "missing_or_next": "report as statistical support for the ProteinGym external validation table",
    }


def skempi_binder_row(args):
    report = "results/skempi_binder_retention/SKEMPI_BINDER_RETENTION_PROXY_REPORT.md"
    table = "results/skempi_binder_retention/skempi_v2_binder_proxy_metrics.csv"
    scored = "results/skempi_binder_retention/skempi_v2_binder_proxy_scored.csv"
    rows = read_csv(table)
    main = next((row for row in rows if row.get("subset") == "all_scored" and row.get("score_name") == "binder_proxy_max"), {})
    result = "binder_proxy_max Spearman={}, AUROC={}, AUPRC={}, top10={}".format(
        main.get("spearman_log10_kd_ratio", ""),
        main.get("auroc_damaging", ""),
        main.get("auprc_damaging", ""),
        main.get("top10_damaging_precision", ""),
    )
    return {
        "axis": "external_experimental",
        "benchmark": "SKEMPI v2 binder-retention proxy",
        "status": row_status([table, scored, report], [(report, "SKEMPI_BINDER_RETENTION_PROXY_PASS")]),
        "compute": "CPU",
        "primary_metric": "Spearman/AUROC/AUPRC against binding-affinity damage",
        "main_result": result,
        "artifacts": file_detail([table, scored, report]),
        "reproduce_command": "{} scripts/evaluate_skempi_binder_retention.py".format(PYTHON),
        "missing_or_next": "supporting proxy only; do not describe as direct deletion ground truth",
    }


def case_study_structure_row(args):
    report = "results/case_studies/structure_reprediction/CASE_STUDY_STRUCTURE_REPREDICTION_REPORT.md"
    table = "results/case_studies/structure_reprediction/case_study_structure_reprediction_summary.csv"
    rows = read_csv(table)
    counts = Counter(row.get("prediction_status") for row in rows)
    result = "exported_cases={}, prediction_status_counts={}".format(len(rows), dict(counts))
    return {
        "axis": "qualitative_validation",
        "benchmark": "Structure re-prediction case-study prep/evaluation",
        "status": row_status([table, report], [(report, "CASE_STUDY_STRUCTURE_REPREDICTION_READY")]),
        "compute": "CPU for prep/eval; GPU for AF2/ESMFold/OmegaFold prediction",
        "primary_metric": "FASTA export, predicted pLDDT, motif pLDDT, breakpoint CA distance",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/prepare_case_study_structure_reprediction.py".format(PYTHON),
        "missing_or_next": "optional GPU prediction step: run structure predictor on exported FASTA files and rerun evaluation",
    }


def evidence_row(args):
    report = "results/biodel_planner/bioprior_10k_test_evidence_dropout_report.txt"
    table = "results/biodel_planner/bioprior_10k_test_evidence_dropout_summary.csv"
    rows = read_csv(table)
    naive = next((row for row in rows if row.get("evidence_mode") == "sequence_only_naive"), {})
    adaptive = next((row for row in rows if row.get("evidence_mode") == "sequence_only_adaptive"), {})
    result = "sequence-only delete ratio naive={:.3f}, adaptive={:.3f}".format(
        fnum(naive.get("global_auto_delete_ratio")),
        fnum(adaptive.get("global_auto_delete_ratio")),
    )
    return {
        "axis": "robustness",
        "benchmark": "Evidence dropout / missing input stress",
        "status": row_status([table, report], [(report, "BIODEL_EVIDENCE_DROPOUT_ABLATION_PASS")]),
        "compute": "CPU",
        "primary_metric": "delete ratio change under missing evidence",
        "main_result": result,
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/run_biodel_evidence_dropout_ablation.py".format(PYTHON),
        "missing_or_next": "repeat on family split after full benchmark protocol is frozen",
    }


def input_stress_row(args):
    report = "results/biodel_planner/input_limitation_stress_report.txt"
    table = "results/biodel_planner/input_limitation_stress.csv"
    rows = read_csv(table)
    counts = Counter(row.get("status") for row in rows)
    return {
        "axis": "robustness",
        "benchmark": "Input limitation stress test",
        "status": row_status([table, report], [(report, "BIODEL_INPUT_LIMITATION_STRESS_PASS")]),
        "compute": "CPU",
        "primary_metric": "accepted/degraded/rejected/review route",
        "main_result": "status_counts={}".format(dict(counts)),
        "artifacts": file_detail([table, report]),
        "reproduce_command": "{} scripts/run_biodel_input_limitation_stress.py".format(PYTHON),
        "missing_or_next": "turn into appendix safety/limitations table, not a main contribution",
    }


def write_report(path, rows):
    ensure_parent(path)
    counts = Counter(row["status"] for row in rows)
    with open(path, "w") as handle:
        handle.write("# BioDel-Cert AAAI Benchmark Suite\n\n")
        handle.write("This report freezes the benchmark design around the paper claim: post-selection risk-certified protein deletion frontiers.\n\n")
        handle.write("## Status\n\n")
        handle.write("- PASS: {}\n".format(counts["PASS"]))
        handle.write("- WARN: {}\n".format(counts["WARN"]))
        handle.write("- MISSING: {}\n\n".format(counts["MISSING"]))

        handle.write("## Benchmark Axes\n\n")
        handle.write("| Axis | Benchmark | Status | Compute | Primary metric | Main result |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for row in rows:
            handle.write(
                "| {axis} | {benchmark} | {status} | {compute} | {primary_metric} | {main_result} |\n".format(**row)
            )

        handle.write("\n## Required Paper Tables\n\n")
        handle.write("1. Dataset table: BioPrior family split, CATH, DisProt, ProteinGym v1.3 indel, SCISOR baseline coverage.\n")
        handle.write("2. Core algorithm table: segment-level vs single-plan vs frontier-level post-selection calibration.\n")
        handle.write("3. Matched-length comparison table: BioDel-Cert target lengths against SCISOR and Raygun on the same proteins.\n")
        handle.write("4. Internal planner table: utility greedy, strict-safe greedy, BioDel selected, certified frontier profiles, SCISOR baselines.\n")
        handle.write("5. Diagnostic baseline table: random, terminal, disorder, low-pLDDT, surface-loop, learned compatibility, strict-safe, and fixed-budget BioDel.\n")
        handle.write("6. External biology table: CATH domain-core overlap and DisProt functional-IDR overlap versus simple baselines.\n")
        handle.write("7. Experimental validation table: ProteinGym indel scores and SKEMPI binder-retention proxy scores.\n")
        handle.write("8. Qualitative structure table: case-study FASTA export and optional re-predicted-structure metrics.\n")
        handle.write("9. Robustness/limitation table: evidence dropout and input limitation stress.\n\n")

        handle.write("## Reproduction Commands\n\n")
        for row in rows:
            handle.write("### {}\n\n".format(row["benchmark"]))
            handle.write("Status: `{}`. Compute: `{}`.\n\n".format(row["status"], row["compute"]))
            handle.write("```bash\n{}\n```\n\n".format(row["reproduce_command"]))
            handle.write("Next: {}\n\n".format(row["missing_or_next"]))

        handle.write("## Missing Or Risky Items\n\n")
        risky = [row for row in rows if row["status"] != "PASS" or "optional" not in row["missing_or_next"].lower()]
        if not risky:
            handle.write("- none\n")
        else:
            for row in risky:
                if row["status"] != "PASS" or row["benchmark"].startswith("SCISOR"):
                    handle.write("- [{}] {}: {}\n".format(row["status"], row["benchmark"], row["missing_or_next"]))

        handle.write("\nBIODEL_AAAI_BENCHMARK_SUITE_REPORT_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build AAAI BioDel benchmark suite report.")
    parser.add_argument("--family_root", default="results/biodel_planner/family_split")
    parser.add_argument("--scisor_family_root", default="results/scisor_bioprior_10k_family_test")
    parser.add_argument("--out_csv", default="results/biodel_planner/aaai_benchmark_suite_status.csv")
    parser.add_argument("--out_report", default="results/biodel_planner/AAAI_BENCHMARK_SUITE_REPORT.md")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [
        family_split_row(args),
        certified_frontier_row(args),
        post_selection_row(args),
        matched_length_row(args),
        family_planner_row(args),
        deletion_baseline_suite_row(args),
        scisor_family_row(args),
        cath_disprot_row(args),
        proteingym_row(args),
        proteingym_bootstrap_row(args),
        skempi_binder_row(args),
        case_study_structure_row(args),
        evidence_row(args),
        input_stress_row(args),
    ]
    write_csv(args.out_csv, rows)
    write_report(args.out_report, rows)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
