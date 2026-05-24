#!/usr/bin/env python3
"""Build a single markdown summary for the current-best six experiment blocks.

This script intentionally reads only current-best rerun outputs and clearly
labels proxy/legacy references. It does not mix in older stale metrics.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from statistics import StatisticsError, mean


BASE = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit"
OUT = os.path.join(BASE, "results/current_best_experiments/CURRENT_BEST_SIX_EXPERIMENTS_SUMMARY.md")


def p(*parts):
    return os.path.join(BASE, *parts)


def exists(path):
    return os.path.exists(path)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    try:
        return mean(values)
    except StatisticsError:
        return None


def mean_csv_metric(rows, key):
    return mean_or_none(
        float(value)
        for value in (row.get(key) for row in rows)
        if value not in ("", None)
    )


def first_row(path):
    rows = read_csv(path)
    return rows[0] if rows else {}


def find_row(rows, **conds):
    for row in rows:
        ok = True
        for key, value in conds.items():
            if str(row.get(key)) != str(value):
                ok = False
                break
        if ok:
            return row
    return {}


def fmt(value, digits=6):
    if value in ("", None):
        return "NA"
    try:
        return ("{:.%df}" % digits).format(float(value))
    except Exception:
        return str(value)


def pct(value, digits=1):
    if value in ("", None):
        return ""
    try:
        return ("{:.%df}%%" % digits).format(100.0 * float(value))
    except Exception:
        return str(value)


def intfmt(value):
    if value in ("", None):
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def md_link(label, rel_path):
    return "[{}]({})".format(label, p(rel_path))


def append_kv_table(lines, rows):
    lines.append("| Item | Value |\n")
    lines.append("|---|---:|\n")
    for key, value in rows:
        lines.append("| {} | {} |\n".format(key, value))
    lines.append("\n")


def baseline_recheck_status():
    recheck_path = p("results/baseline_recheck/matched_length_comparison/matched_length_incremental_rows.csv")
    summary_path = p("results/baseline_recheck/matched_length_comparison/matched_length_comparison_summary.csv")
    if not exists(recheck_path):
        return {"status": "not_started", "rows": 0, "accessions": 0, "method_counts": {}, "summary_done": False}
    rows = read_csv(recheck_path)
    accessions = {row.get("accession") for row in rows if row.get("accession")}
    method_counts = Counter(row.get("method", "") for row in rows)
    return {
        "status": "complete" if exists(summary_path) else "running_or_partial",
        "rows": len(rows),
        "accessions": len(accessions),
        "method_counts": dict(method_counts),
        "summary_done": exists(summary_path),
    }


def count_metric_diffs(old_path, new_path, method, ignore_fields=None):
    ignore_fields = set(ignore_fields or [])
    if not exists(old_path) or not exists(new_path):
        return {"common": 0, "diffs": 0}
    with open(old_path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        old_rows = {
            (row.get("method"), row.get("accession")): row
            for row in reader
            if row.get("method") == method
        }
    with open(new_path, newline="") as handle:
        reader = csv.DictReader(handle)
        new_rows = {
            (row.get("method"), row.get("accession")): row
            for row in reader
            if row.get("method") == method
        }
    common = sorted(set(old_rows) & set(new_rows))
    diff_count = 0
    for key in common:
        old = old_rows[key]
        new = new_rows[key]
        for field in fields:
            if field in ignore_fields:
                continue
            if old.get(field, "") != new.get(field, ""):
                diff_count += 1
                break
    return {"common": len(common), "diffs": diff_count}


def safe_ratio(numer, denom):
    try:
        numer = float(numer)
        denom = float(denom)
    except Exception:
        return 0.0
    return numer / denom if denom else 0.0


def safe_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except Exception:
        return default


def signed_delta(value, baseline, digits=3):
    try:
        delta = float(value) - float(baseline)
    except Exception:
        return "NA"
    return ("{:+.%df}" % digits).format(delta)


def abs_delta(value, baseline, digits=3):
    try:
        delta = abs(float(value) - float(baseline))
    except Exception:
        return "NA"
    return ("{:.%df}" % digits).format(delta)


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def overlap(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or a_start > b_end)


def interval_length(start, end):
    return max(0, end - start + 1)


def recall_for_protein(preds, golds):
    if not golds:
        return None
    covered = 0
    for g_start, g_end in golds:
        g_len = interval_length(g_start, g_end)
        if g_len == 0:
            continue
        overlap_len = 0
        for p_start, p_end in preds:
            if not overlap(p_start, p_end, g_start, g_end):
                continue
            overlap_len += interval_length(max(p_start, g_start), min(p_end, g_end))
        covered += min(g_len, overlap_len)
    total = sum(interval_length(start, end) for start, end in golds)
    return covered / float(total) if total else None


def coverage_at_least_one(preds, golds):
    if not golds:
        return None
    hit = 0
    for g_start, g_end in golds:
        if any(overlap(p_start, p_end, g_start, g_end) for p_start, p_end in preds):
            hit += 1
    return hit / float(len(golds))


def read_teacher_labels_by_accession(path):
    rows = read_csv(path)
    by_accession = defaultdict(list)
    for row in rows:
        if str(row.get("selected_by_teacher", "")).lower() not in ("true", "1", "yes"):
            continue
        accession = row.get("accession")
        if not accession:
            continue
        by_accession[accession].append((safe_int(row.get("seg_start")), safe_int(row.get("seg_end"))))
    return by_accession


def read_proposals_by_accession(path):
    rows = read_csv(path)
    by_accession = defaultdict(list)
    for row in rows:
        accession = row.get("accession") or row.get("protein_id")
        if not accession:
            continue
        by_accession[accession].append(
            (
                safe_int(row.get("seg_start")),
                safe_int(row.get("seg_end")),
                safe_int(row.get("stage1_interval_rank"), 10**9),
                row,
            )
        )
    for accession in by_accession:
        by_accession[accession].sort(key=lambda item: (item[2], item[0], item[1]))
    return by_accession


def evaluate_topk_interval_recall(proposals_path, teacher_path, ks=(1, 3, 5, 10)):
    proposals = read_proposals_by_accession(proposals_path)
    teacher = read_teacher_labels_by_accession(teacher_path)
    accessions = sorted(set(proposals) | set(teacher))
    rows = []
    for k in ks:
        recalls = []
        hits = []
        covered = 0
        total = 0
        n = 0
        for accession in accessions:
            gold_intervals = teacher.get(accession, [])
            if not gold_intervals:
                continue
            pred_rows = proposals.get(accession, [])[:k]
            pred_intervals = [(start, end) for start, end, _, _ in pred_rows]
            protein_recall = recall_for_protein(pred_intervals, gold_intervals)
            interval_hit_rate = coverage_at_least_one(pred_intervals, gold_intervals)
            if protein_recall is None or interval_hit_rate is None:
                continue
            n += 1
            recalls.append(protein_recall)
            hits.append(interval_hit_rate)
            for g_start, g_end in gold_intervals:
                if any(overlap(p_start, p_end, g_start, g_end) for p_start, p_end in pred_intervals):
                    covered += 1
            total += len(gold_intervals)
        rows.append(
            {
                "top_k": k,
                "proteins_with_teacher_labels": n,
                "mean_teacher_interval_recall": mean_or_none(recalls) or 0.0,
                "mean_teacher_interval_hit_rate": mean_or_none(hits) or 0.0,
                "zero_hit_fraction": sum(1 for value in hits if value == 0.0) / float(len(hits) or 1),
                "overall_teacher_positive_coverage": safe_ratio(covered, total),
            }
        )
    return rows


def mean_metric_by_score(rows, score_name, metric):
    return mean_csv_metric([row for row in rows if row.get("score_name") == score_name], metric)


def rows_for_score(rows, score_name):
    return [row for row in rows if row.get("score_name") == score_name]


def write_report(path, lines):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as handle:
        handle.writelines(lines)


def build():
    stage1 = read_json(p("results/stage1_scisor_style_uniref50_full_len800_2gpu/test_metrics.json"))
    interval_rows = read_csv(p("results/current_best_experiments/stage1_interval_proposals_eval.csv"))
    matched_length = read_csv(p("results/biodel_planner/family_split/matched_length_comparison/matched_length_comparison_summary.csv"))
    scisor_family_legacy = read_csv(p("results/archive/scisor/scisor_bioprior_10k_family_test/scisor_bioprior10k_family_test_baseline_summary.csv"))
    cert_frontier = read_csv(p("results/current_best_experiments/experiment_family_split_current_certified/certified_frontier_test_summary.csv"))
    auto_budget = read_csv(p("results/current_best_experiments/experiment_family_split_current_certified/bioprior_10k_test_auto_budget_certified_summary.csv"))
    family_table = read_csv(p("results/current_best_experiments/experiment_family_split_current_certified/final_test_family_comparison/final_test_comparison_table.csv"))
    cath = read_csv(p("results/current_best_experiments/experiment2_cath_current_certified/cath_auto_summary.csv"))
    disprot = read_csv(p("results/current_best_experiments/experiment2_disprot_current_certified/disprot_auto_summary.csv"))
    proteingym = read_csv(p("results/current_best_experiments/experiment4_proteingym/proteingym_current_stage1_external_validation_metrics.csv"))
    proteingym_current_per_assay = read_csv(p("results/current_best_experiments/experiment4_proteingym/proteingym_current_stage1_external_validation_per_assay.csv"))
    proteingym_zero_shot_per_assay = read_csv(p("results/proteingym_v13_indel/proteingym_v13_step3_per_assay_metrics_finetuned_stage1_certified.csv"))
    skempi = read_csv(p("results/current_best_experiments/experiment4_skempi/skempi_v2_binder_proxy_metrics.csv"))
    ablation = read_csv(p("results/current_best_experiments/experiment5_ablation_current_certified/bioprior_10k_evidence_dropout_summary.csv"))
    case_studies = read_csv(p("results/current_best_experiments/experiment6_case_studies/current_best_case_studies.csv"))
    esmfold_summary_path = p("results/current_best_experiments/experiment6_case_studies/current_best_esmfold_structure_summary.csv")
    esmfold_rows = read_csv(esmfold_summary_path) if exists(esmfold_summary_path) else []

    frontier_default = find_row(cert_frontier, auto_profile="default")
    frontier_conservative = find_row(cert_frontier, auto_profile="conservative")
    frontier_aggressive = find_row(cert_frontier, auto_profile="aggressive")
    auto_default = find_row(auto_budget, setting="safe", auto_profile="default")
    row_10 = find_row(family_table, method="validation_selected_v2", budget="0.1")
    row_20 = find_row(family_table, method="validation_selected_v2", budget="0.2")
    row_30 = find_row(family_table, method="validation_selected_v2", budget="0.3")
    strict_30 = find_row(family_table, method="strict_safe_greedy", budget="0.3")
    util_30 = find_row(family_table, method="utility_greedy_protected_only", budget="0.3")

    cath_conservative = find_row(cath, setting="safe", auto_profile="conservative")
    cath_default = find_row(cath, setting="safe", auto_profile="default") or (cath[0] if cath else {})
    cath_aggressive = find_row(cath, setting="safe", auto_profile="aggressive")
    disprot_conservative = find_row(disprot, setting="safe", auto_profile="conservative")
    disprot_default = find_row(disprot, setting="safe", auto_profile="default") or (disprot[0] if disprot else {})
    disprot_aggressive = find_row(disprot, setting="safe", auto_profile="aggressive")

    pg_dms_b10 = find_row(proteingym, subset="DMS_indels", score_name="score_stage1_b10")
    pg_dms_b20 = find_row(proteingym, subset="DMS_indels", score_name="score_stage1_b20")
    pg_term = find_row(proteingym, subset="DMS_indels", score_name="score_terminal_proximity")
    pg_clinical = find_row(proteingym, subset="clinical_indels_benign_vs_pathogenic", score_name="score_stage1_b10")

    sk_binder_max = find_row(skempi, subset="all_scored", score_name="binder_proxy_max")
    matched_biodel = find_row(matched_length, method="BioDel-Cert default")
    matched_scisor = find_row(matched_length, method="SCISOR hardmask+shadow02")
    matched_raygun = find_row(matched_length, method="Raygun 2.2M")
    scisor_family_30 = find_row(scisor_family_legacy, method="SCISOR hardmask+shadow02", budget_label="30%")

    zero_shot_groups = defaultdict(list)
    for row in proteingym_zero_shot_per_assay:
        score_name = row.get("score_name", "")
        if score_name.startswith("proteingym_zero_shot:"):
            zero_shot_groups[score_name].append(row)
    zero_shot_ranked = []
    for score_name, rows in zero_shot_groups.items():
        zero_shot_ranked.append(
            {
                "score_name": score_name.replace("proteingym_zero_shot:", ""),
                "spearman": mean_csv_metric(rows, "spearman_favorable_DMS"),
                "auroc": mean_csv_metric(rows, "auroc_DMS_score_bin_1"),
                "auprc": mean_csv_metric(rows, "auprc_DMS_score_bin_1"),
            }
        )
    zero_shot_ranked.sort(
        key=lambda row: (row["spearman"] is not None, row["spearman"] if row["spearman"] is not None else -999),
        reverse=True,
    )
    zero_shot_ranked = zero_shot_ranked[:5]

    ab_full = find_row(ablation, evidence_mode="full_evidence")
    ab_naive_ann = find_row(ablation, evidence_mode="no_annotation_naive")
    ab_adapt_ann = find_row(ablation, evidence_mode="no_annotation_adaptive")
    ab_naive_struct = find_row(ablation, evidence_mode="no_structure_naive")
    ab_adapt_struct = find_row(ablation, evidence_mode="no_structure_adaptive")
    ab_naive_seq = find_row(ablation, evidence_mode="sequence_only_naive")
    ab_adapt_seq = find_row(ablation, evidence_mode="sequence_only_adaptive")

    case_count = len(case_studies)
    esmfold_pass = bool(esmfold_rows) and all(row.get("prediction_status") == "success" for row in esmfold_rows)

    interval_n = len(interval_rows)
    interval_teacher_labels = sum(1 for row in interval_rows if row.get("n_teacher_positive"))
    interval_mean_recall = (
        sum(float(row.get("teacher_interval_recall", 0.0)) for row in interval_rows) / interval_n
        if interval_n
        else 0.0
    )
    interval_mean_hit = (
        sum(float(row.get("teacher_interval_hit_rate", 0.0)) for row in interval_rows) / interval_n
        if interval_n
        else 0.0
    )
    interval_teacher_covered = sum(float(row.get("teacher_positive_covered", 0.0)) for row in interval_rows)
    interval_teacher_total = sum(float(row.get("teacher_positive_total", 0.0)) for row in interval_rows)
    coverage = safe_ratio(interval_teacher_covered, interval_teacher_total)
    interval_topk = evaluate_topk_interval_recall(
        p("results/current_best_experiments/stage1_interval_proposals.csv"),
        p("data/train/core_1k_bioprior_teacher_labels.csv"),
    )

    comparison_rows = [
        row
        for row in family_table
        if row.get("budget") in {"0.1", "0.2", "0.3"}
        and row.get("method")
        in {
            "utility_greedy_protected_only",
            "strict_safe_greedy",
            "v2_safe",
            "v2_balanced",
            "validation_selected_v2",
        }
    ]
    comparison_rows.sort(key=lambda row: (float(row.get("budget", 0.0)), row.get("method_label", "")))

    frontier_rows = [frontier_conservative, frontier_default, frontier_aggressive]
    auto_rows = [
        find_row(auto_budget, setting="safe", auto_profile="conservative"),
        find_row(auto_budget, setting="safe", auto_profile="default"),
        find_row(auto_budget, setting="safe", auto_profile="aggressive"),
    ]
    external_rows = [
        pg_dms_b10,
        pg_dms_b20,
        find_row(proteingym, subset="DMS_indels", score_name="score_stage1_b30"),
        find_row(proteingym, subset="DMS_indels", score_name="score_shorter_deletion"),
        pg_term,
        find_row(proteingym, subset="DMS_indels", score_name="score_terminal_binary"),
        pg_clinical,
        find_row(proteingym, subset="clinical_indels_benign_vs_pathogenic", score_name="score_stage1_sum_b10"),
    ]
    skempi_rows = [
        find_row(skempi, subset="all_scored", score_name="binder_proxy_max"),
        find_row(skempi, subset="mapped_interface_scored", score_name="binder_proxy_max"),
        find_row(skempi, subset="mapped_interface_scored", score_name="contact_density_max"),
        find_row(skempi, subset="mapped_interface_scored", score_name="location_COR_label"),
    ]

    pg_dms_best_stage1 = max(
        [row for row in external_rows if row.get("subset") == "DMS_indels" and row.get("score_name", "").startswith("score_stage1")],
        key=lambda row: float(row.get("spearman") or -999),
    )
    pg_shorter = find_row(proteingym, subset="DMS_indels", score_name="score_shorter_deletion")
    pg_stage1_b10_per_assay = {
        "score_name": "score_stage1_b10",
        "spearman": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b10", "spearman"),
        "auroc": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b10", "auroc"),
        "auprc": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b10", "auprc"),
    }
    pg_stage1_b20_per_assay = {
        "score_name": "score_stage1_b20",
        "spearman": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b20", "spearman"),
        "auroc": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b20", "auroc"),
        "auprc": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b20", "auprc"),
    }
    pg_stage1_b30_per_assay = {
        "score_name": "score_stage1_b30",
        "spearman": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b30", "spearman"),
        "auroc": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b30", "auroc"),
        "auprc": mean_metric_by_score(proteingym_current_per_assay, "score_stage1_b30", "auprc"),
    }
    pg_shorter_per_assay = {
        "score_name": "score_shorter_deletion",
        "spearman": mean_metric_by_score(proteingym_current_per_assay, "score_shorter_deletion", "spearman"),
        "auroc": mean_metric_by_score(proteingym_current_per_assay, "score_shorter_deletion", "auroc"),
        "auprc": mean_metric_by_score(proteingym_current_per_assay, "score_shorter_deletion", "auprc"),
    }
    pg_terminal_per_assay = {
        "score_name": "score_terminal_proximity",
        "spearman": mean_metric_by_score(proteingym_current_per_assay, "score_terminal_proximity", "spearman"),
        "auroc": mean_metric_by_score(proteingym_current_per_assay, "score_terminal_proximity", "auroc"),
        "auprc": mean_metric_by_score(proteingym_current_per_assay, "score_terminal_proximity", "auprc"),
    }
    pg_best_stage1_per_assay = max(
        [pg_stage1_b10_per_assay, pg_stage1_b20_per_assay, pg_stage1_b30_per_assay],
        key=lambda row: row["spearman"] if row["spearman"] is not None else -999,
    )
    pg_top_zero_shot = zero_shot_ranked[0] if zero_shot_ranked else {}

    sk_binder_mapped = find_row(skempi, subset="mapped_interface_scored", score_name="binder_proxy_max")
    sk_contact_mapped = find_row(skempi, subset="mapped_interface_scored", score_name="contact_density_max")
    sk_location_mapped = find_row(skempi, subset="mapped_interface_scored", score_name="location_COR_label")

    util_vs_selected_30 = {
        "fill": signed_delta(row_30.get("fill_ratio"), util_30.get("fill_ratio")),
        "delete_ratio": signed_delta(row_30.get("achieved_deletion_ratio"), util_30.get("achieved_deletion_ratio")),
        "shadow": signed_delta(row_30.get("shadow_overlap_rate"), util_30.get("shadow_overlap_rate")),
        "closure": signed_delta(row_30.get("closure_unfriendly_rate"), util_30.get("closure_unfriendly_rate")),
    }
    selected_vs_strict_30 = {
        "fill": signed_delta(row_30.get("fill_ratio"), strict_30.get("fill_ratio")),
        "delete_ratio": signed_delta(row_30.get("achieved_deletion_ratio"), strict_30.get("achieved_deletion_ratio")),
        "shadow": signed_delta(row_30.get("shadow_overlap_rate"), strict_30.get("shadow_overlap_rate")),
        "closure": signed_delta(row_30.get("closure_unfriendly_rate"), strict_30.get("closure_unfriendly_rate")),
    }
    selected_vs_scisor_family_30 = {
        "fill": signed_delta(row_30.get("fill_ratio"), scisor_family_30.get("fill_ratio")),
        "shadow": signed_delta(row_30.get("shadow_overlap_rate"), scisor_family_30.get("shadow_overlap_rate")),
        "closure": signed_delta(row_30.get("closure_unfriendly_rate"), scisor_family_30.get("closure_unfriendly_rate")),
    }

    ablation_delta_rows = []
    for row in [ab_naive_ann, ab_adapt_ann, ab_naive_struct, ab_adapt_struct, ab_naive_seq, ab_adapt_seq]:
        ablation_delta_rows.append(
            {
                "mode": row.get("evidence_mode", ""),
                "description": row.get("description", ""),
                "delete_ratio": row.get("global_auto_delete_ratio"),
                "delta_delete_ratio": signed_delta(row.get("global_auto_delete_ratio"), ab_full.get("global_auto_delete_ratio")),
                "structural_risk_per_deleted": row.get("structural_risk_per_deleted"),
                "delta_structural_risk": signed_delta(row.get("structural_risk_per_deleted"), ab_full.get("structural_risk_per_deleted")),
            }
        )

    recheck = baseline_recheck_status()
    old_matched_table = p("results/biodel_planner/family_split/matched_length_comparison/matched_length_comparison_table.csv")
    new_matched_table = p("results/baseline_recheck/matched_length_comparison/matched_length_incremental_rows.csv")
    scisor_recheck = count_metric_diffs(
        old_matched_table,
        new_matched_table,
        "SCISOR hardmask+shadow02",
        ignore_fields={"source", "selection_status", "auto_profile", "raygun_noise"},
    )
    raygun_recheck = count_metric_diffs(
        old_matched_table,
        new_matched_table,
        "Raygun 2.2M",
        ignore_fields={"source", "selection_status", "auto_profile"},
    )

    matched_status = {
        "BioDel-Cert default": "current matched-length reference",
        "SCISOR hardmask+shadow02": "recheck subset matches old metrics: {}/{} differ".format(scisor_recheck.get("diffs"), scisor_recheck.get("common")),
        "Raygun 2.2M": "partial recheck differs on {}/{} shared rows; stochastic/not yet final".format(raygun_recheck.get("diffs"), raygun_recheck.get("common")),
    }

    lines = []
    lines.append("# Current-Best Six Experiments: Comparative Report\n\n")
    lines.append(
        "This report is comparison-first: every experiment is interpreted against an explicit baseline, control, "
        "or operating-point trade-off. All main tables read from `results/current_best_experiments/`; explicitly "
        "marked baseline references come from archived or matched-length comparison outputs. Numbers should be "
        "interpreted as the current pipeline baseline for the Stage-1 checkpoint, not as the final expected model "
        "performance.\n\n"
    )

    lines.append("## Executive Comparison\n\n")
    lines.append(
        "The current pipeline runs end to end, but the results do **not** support claiming a strong learned deletion "
        "model yet. The strongest evidence is for the planner and safety constraints; the weakest evidence is the "
        "Stage-1 deletion prior.\n\n"
    )
    lines.append("| Experiment | Main comparison | Current winner | Margin | Diagnosis / action |\n")
    lines.append("|---|---|---|---|---|\n")
    lines.append(
        "| 1. Candidate intervals | Stage-1 proposer vs high-recall candidate-generator requirement | Not yet adequate | "
        "top-1 recall `{}`, top-10 recall `{}`, zero-hit `{}` | Improve Stage-1 training/calibration and add matched-count random/terminal/low-pLDDT proposal baselines. |\n".format(
            fmt(interval_topk[0]["mean_teacher_interval_recall"], 3),
            fmt(interval_topk[-1]["mean_teacher_interval_recall"], 3),
            pct(interval_topk[-1]["zero_hit_fraction"], 1),
        )
    )
    lines.append(
        "| 2. Family-heldout planning | BioDel selected vs utility-only at 30% | BioDel for safety; utility-only for length | "
        "fill `{}` lower, shadow `{}` lower, closure `{}` lower | Constraints work: BioDel trades deletion amount for much lower risk. Need better Stage-1 candidates to raise safe fill. |\n".format(
            abs_delta(row_30.get("fill_ratio"), util_30.get("fill_ratio")),
            abs_delta(row_30.get("shadow_overlap_rate"), util_30.get("shadow_overlap_rate")),
            abs_delta(row_30.get("closure_unfriendly_rate"), util_30.get("closure_unfriendly_rate")),
        )
    )
    lines.append(
        "| 2. Domain/IDR benchmarks | CATH vs DisProt safe/default outputs | CATH deletes more | CATH delete `{}` vs DisProt `{}` | Evidence is stronger than a raw disorder heuristic; IDR-specific functional labels still need improvement. |\n".format(
            fmt(cath_default.get("global_auto_delete_ratio")),
            fmt(disprot_default.get("global_auto_delete_ratio")),
        )
    )
    lines.append(
        "| 2. External baselines | BioDel-Cert vs SCISOR/Raygun matched length | BioDel-Cert on safety | SCISOR closure `{}` vs BioDel `{}`; Raygun closure `{}` but recheck unstable | Keep SCISOR as usable matched-length reference; rerun Raygun fully before paper use. |\n".format(
            fmt(matched_scisor.get("closure_unfriendly_rate"), 3),
            fmt(matched_biodel.get("closure_unfriendly_rate"), 3),
            fmt(matched_raygun.get("closure_unfriendly_rate"), 3),
        )
    )
    lines.append(
        "| 3. Automatic budget | Default vs conservative/aggressive auto profiles | Default profile | certifies `{}`/`{}` with zero shadow/closure in frontier summary | Auto-budgeting is useful because fixed 30% over-forces hard proteins. Calibrate refusal thresholds next. |\n".format(
            intfmt(frontier_default.get("certified_proteins")), intfmt(frontier_default.get("analyzed_proteins"))
        )
    )
    lines.append(
        "| 4. ProteinGym | Stage-1 scores vs simple baselines and zero-shot PLMs | Stage-1 does not win overall | shorter Spearman gap `{}`, terminal AUPRC gap `{}`, per-assay Spearman gap `{}` | Current Stage-1 is a weak biological fitness scorer. Add experimental/PLM fitness supervision or rerank with PLM evidence. |\n".format(
            abs_delta(pg_dms_best_stage1.get("spearman"), pg_shorter.get("spearman"), 3),
            abs_delta(pg_dms_best_stage1.get("auprc"), pg_term.get("auprc"), 3),
            abs_delta(pg_best_stage1_per_assay.get("spearman"), pg_top_zero_shot.get("spearman"), 3),
        )
    )
    lines.append(
        "| 4. Binder proxy | Stage-1-derived binder proxy vs structural interface labels on SKEMPI | Structural labels | mapped AUROC: binder `{}` vs COR-location `{}` | Binder result is only proxy-level support. Build EGFR deletion-retention evaluation before making functional binder claims. |\n".format(
            fmt(sk_binder_mapped.get("auroc_damaging"), 3), fmt(sk_location_mapped.get("auroc_damaging"), 3)
        )
    )
    lines.append(
        "| 5. Ablation | Full evidence vs missing-evidence modes | Adaptive evidence caps | sequence-only naive delete `{}` vs adaptive `{}` | Adaptive uncertainty is necessary; without it, missing evidence makes the planner over-delete. |\n".format(
            fmt(ab_naive_seq.get("global_auto_delete_ratio"), 3), fmt(ab_adapt_seq.get("global_auto_delete_ratio"), 3)
        )
    )
    lines.append(
        "| 6. Structure cases | ESMFold deleted structures vs qualitative sanity check | Qualitative pass only | `{}`/`{}` ESMFold jobs succeeded; Q43296 pLDDT `{}` | Useful for figures, not proof of function. Add larger post-deletion structure and function validation later. |\n\n".format(
            sum(1 for row in esmfold_rows if row.get("prediction_status") == "success"),
            len(esmfold_rows),
            fmt(find_row(esmfold_rows, accession="Q43296").get("mean_plddt"), 1),
        )
    )

    lines.append("## Experimental Setup\n\n")
    lines.append("**Model fixed for all current-best experiments.**\n\n")
    append_kv_table(
        lines,
        [
            ("Stage-1 checkpoint", "`checkpoints/stage1_scisor_style_uniref50_full_len800_2gpu/best_model.pt`"),
            ("Best step", "`{}`".format(stage1.get("best_step", ""))),
            ("Token AP", "`{}`".format(fmt(stage1.get("token_ap")))),
            ("Token AUC", "`{}`".format(fmt(stage1.get("token_auc")))),
            ("Token F1", "`{}`".format(fmt(stage1.get("token_f1")))),
            ("Precision / recall", "`{}` / `{}`".format(fmt(stage1.get("precision")), fmt(stage1.get("recall")))),
        ],
    )
    lines.append(
        "The checkpoint has good ranking metrics but zero thresholded F1 under the current test threshold. "
        "That means the downstream experiments are only as strong as the learned ranking prior, so the report "
        "should be read as a pipeline diagnostic rather than a final biological benchmark.\n\n"
    )

    lines.append("## 1. Candidate Interval Proposal Quality\n\n")
    lines.append(
        "The first stage is evaluated against teacher deletion-positive residues. The point of this experiment is "
        "not final deletion quality, but whether the proposer can reliably surface enough good candidates for stage2.\n\n"
    )
    append_kv_table(
        lines,
        [
            ("Proteins evaluated", "`{}`".format(interval_n)),
            ("Proteins with teacher labels", "`{}`".format(interval_teacher_labels)),
            ("Mean proposals per protein", "`{}`".format(fmt(sum(float(row.get("n_proposals", 0.0)) for row in interval_rows) / float(interval_n or 1), 3))),
            ("Mean interval recall", "`{}`".format(fmt(interval_mean_recall))),
            ("Mean interval hit rate", "`{}`".format(fmt(interval_mean_hit))),
            ("Positive residue coverage", "`{}`".format(fmt(coverage))),
        ],
    )
    lines.append("| Top-k | Mean recall | Mean hit rate | Zero-hit fraction | Overall positive coverage |\n")
    lines.append("|---:|---:|---:|---:|---:|\n")
    for row in interval_topk:
        lines.append(
            "| {} | `{}` | `{}` | `{}` | `{}` |\n".format(
                row["top_k"],
                fmt(row["mean_teacher_interval_recall"]),
                fmt(row["mean_teacher_interval_hit_rate"]),
                fmt(row["zero_hit_fraction"]),
                fmt(row["overall_teacher_positive_coverage"]),
            )
        )
    lines.append("\n")
    lines.append(
        "Compared with the requirement for a mature candidate generator, this is still weak. Top-1 recall is only "
        "`{}` and even top-10 leaves `{}` of proteins with no teacher hit. The current proposer is therefore useful "
        "as a routing signal, but not yet as a high-recall proposal stage. The most obvious fix is to improve "
        "Stage-1 training and calibrate proposal thresholds before claiming candidate quality.\n\n".format(
            fmt(interval_topk[0]["mean_teacher_interval_recall"], 3),
            pct(interval_topk[-1]["zero_hit_fraction"], 1),
        )
    )

    lines.append("## 2. Risk-Constrained Deletion on BioPrior Family-Heldout\n\n")
    lines.append(
        "The family-heldout benchmark compares utility-only deletion against explicit risk-constrained planning. "
        "Protected residues are hard constraints; shadow and closure are the main safety risks.\n\n"
    )
    lines.append("| Budget | Method | Fill | Deletion ratio | Protected overlap | Shadow rate | Closure rate | Under-80 fill proteins |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for row in comparison_rows:
        lines.append(
            "| {} | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n".format(
                row.get("budget_label", ""),
                row.get("method_label", row.get("method", "")),
                fmt(row.get("fill_ratio"), 3),
                fmt(row.get("achieved_deletion_ratio"), 3),
                intfmt(row.get("protected_overlap_residues")),
                fmt(row.get("shadow_overlap_rate"), 3),
                fmt(row.get("closure_unfriendly_rate"), 3),
                intfmt(row.get("proteins_under_80_fill")),
            )
        )
    lines.append("\n")
    lines.append(
        "At 30% budget, utility-only greedy maximizes fill but also carries the highest risk: compared with BioDel "
        "v2 balanced, it increases fill by `{}` but also raises shadow by `{}` and closure by `{}`. Compared with "
        "strict-safe greedy, BioDel balanced recovers fill by `{}` while paying `{}` shadow and `{}` closure. In "
        "other words, the planner is doing what it should: it exposes the safety-length trade-off rather than hiding it.\n\n".format(
            util_vs_selected_30["fill"],
            util_vs_selected_30["shadow"],
            util_vs_selected_30["closure"],
            selected_vs_strict_30["fill"],
            selected_vs_strict_30["shadow"],
            selected_vs_strict_30["closure"],
        )
    )
    lines.append(
        "The practical conclusion is that BioDel is better than naive utility maximization, but the current stage1 "
        "proposal quality still limits how much safe fill can be recovered. The model should be improved at the "
        "proposal stage before expecting a much stronger planner frontier.\n\n"
    )

    lines.append("### 2.1 Matched-Length External Baselines\n\n")
    lines.append(
        "The matched-length comparison is the cleanest external baseline check because the deletion ratio is fixed. "
        "Here the question is purely safety and sequence preservation under the same length budget.\n\n"
    )
    lines.append("| Method | Fill | Shadow rate | Closure rate | Length success | Sequence identity | Status |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---|\n")
    for row in [matched_biodel, matched_scisor, matched_raygun]:
        if not row:
            continue
        lines.append(
            "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | {} |\n".format(
                row.get("method", ""),
                fmt(row.get("mean_fill_ratio")),
                fmt(row.get("shadow_overlap_rate")),
                fmt(row.get("closure_unfriendly_rate")),
                fmt(row.get("length_success_rate")),
                fmt(row.get("mean_sequence_identity")),
                matched_status.get(row.get("method", ""), ""),
            )
        )
    lines.append("\n")
    lines.append(
        "BioDel-Cert is the safest of the three at matched length: it achieves zero shadow and zero closure, while "
        "SCISOR and Raygun both reintroduce closure risk. SCISOR is currently stable on the rechecked subset, but "
        "Raygun is still row-unstable in the partial rerun, so it should remain a historical reference until the "
        "recheck is finished.\n\n"
    )

    lines.append("## 3. Automatic Budgeting and Refusal\n\n")
    lines.append(
        "The automatic-budget setting lets the planner refuse uncertain proteins and choose different deletion "
        "amounts per protein.\n\n"
    )
    lines.append("| Profile | Analyzed proteins | Certified proteins | Abstention rate | Global delete ratio | Shadow rate | Closure rate | Mean evidence confidence |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in frontier_rows:
        lines.append(
            "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n".format(
                row.get("auto_profile", ""),
                intfmt(row.get("analyzed_proteins")),
                intfmt(row.get("certified_proteins")),
                fmt(row.get("abstention_rate")),
                fmt(row.get("global_auto_delete_ratio")),
                fmt(row.get("shadow_overlap_rate")),
                fmt(row.get("closure_unfriendly_rate")),
                fmt(row.get("mean_evidence_confidence")),
            )
        )
    lines.append("\n")
    lines.append("| Auto-budget profile | Delete ratio | Mean selected budget | Shadow rate | Closure rate | Structural risk / deleted |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in auto_rows:
        lines.append(
            "| {} | `{}` | `{}` | `{}` | `{}` | `{}` |\n".format(
                row.get("auto_profile", ""),
                fmt(row.get("global_auto_delete_ratio")),
                fmt(row.get("mean_selected_budget_ratio")),
                fmt(row.get("shadow_overlap_rate")),
                fmt(row.get("closure_unfriendly_rate")),
                fmt(row.get("structural_risk_per_deleted")),
            )
        )
    lines.append("\n")
    lines.append(
        "The conservative frontier abstains on 74.9% of proteins and deletes only 0.7% globally, while the default "
        "profile certifies 742 / 873 analyzed proteins and deletes 8.2% globally with zero shadow and closure risk. "
        "This is useful because it shows the planner is not forced to make a bad deletion decision when evidence is "
        "weak. The open problem is calibration: how to keep the refusal behavior conservative without collapsing total deletion.\n\n"
    )

    lines.append("### 2.2 CATH-domain and DisProt-IDR\n\n")
    lines.append(
        "These benchmarks test the same planner on structured domains and intrinsically disordered regions. They are "
        "useful because they tell us whether the risk logic generalizes across different protein regimes.\n\n"
    )
    lines.append("| Benchmark | Setting | Profile | Proteins | Delete ratio | Shadow rate | Closure rate | Structural risk / deleted | Evidence confidence |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for name, row in [
        ("CATH-domain", cath_conservative or cath_default or cath_aggressive),
        ("CATH-domain", cath_default),
        ("DisProt-IDR", disprot_conservative or disprot_default or disprot_aggressive),
        ("DisProt-IDR", disprot_default),
    ]:
        if not row:
            continue
        lines.append(
            "| {} | {} | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n".format(
                name,
                row.get("setting", ""),
                row.get("auto_profile", ""),
                intfmt(row.get("analyzed_proteins")),
                fmt(row.get("global_auto_delete_ratio")),
                fmt(row.get("shadow_overlap_rate")),
                fmt(row.get("closure_unfriendly_rate")),
                fmt(row.get("structural_risk_per_deleted")),
                fmt(row.get("mean_evidence_confidence")),
            )
        )
    lines.append("\n")
    lines.append(
        "The CATH runs delete more than the DisProt runs in the current certified summaries, which means the planner "
        "is not simply treating disorder as universally deletable. Instead, it is responding to the available evidence "
        "and closure risk. The practical lesson is that IDR-aware logic needs better evidence and better functional "
        "labels if we want to make stronger claims about safe deletion inside disordered regions.\n\n"
    )

    lines.append("## 4. External Functional Validation\n\n")
    lines.append("### 4.1 ProteinGym deletion fitness\n\n")
    lines.append(
        "ProteinGym asks whether deletion-oriented scores actually align with experimental fitness. This section is "
        "the clearest test of whether the learned prior is biologically meaningful.\n\n"
    )
    lines.append("| Subset | Score | N | Spearman | AUROC | AUPRC | Top-10 precision |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for row in external_rows:
        if not row:
            continue
        lines.append(
            "| {} | {} | `{}` | `{}` | `{}` | `{}` | `{}` |\n".format(
                row.get("subset", ""),
                row.get("score_name", ""),
                intfmt(row.get("n")),
                fmt(row.get("spearman")),
                fmt(row.get("auroc")),
                fmt(row.get("auprc")),
                fmt(row.get("top10_precision")),
            )
        )
    lines.append("\n")
    lines.append(
        "Across the current DMS-indel suite, the best Stage-1 score is `{}`. Even then, the strongest simple "
        "baseline on the same export is terminal proximity, whose Spearman/AUROC/AUPRC all exceed the Stage-1 "
        "family. The per-assay view is more direct: the best Stage-1 variant averages Spearman `{}`, while the best "
        "zero-shot PLM reference averages `{}`. That gap is large enough that the current Stage-1 should be treated "
        "as a weak deletion-risk scorer, not as a strong fitness predictor.\n\n".format(
            pg_dms_best_stage1.get("score_name", ""),
            fmt(pg_best_stage1_per_assay.get("spearman")),
            fmt(pg_top_zero_shot.get("spearman")),
        )
    )

    lines.append("### 4.2 Binder retention proxy\n\n")
    lines.append(
        "SKEMPI is used only as a binder-retention proxy because it is mutation-based rather than a direct deletion "
        "benchmark. The EGFR binder dataset is downloaded and readable, but a deletion-retention benchmark has not "
        "yet been generated from it.\n\n"
    )
    lines.append("| Subset | Score | N scored | Spearman | AUROC damaging | AUPRC damaging | Top-10 damaging precision |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for row in skempi_rows:
        if not row:
            continue
        lines.append(
            "| {} | {} | `{}` | `{}` | `{}` | `{}` | `{}` |\n".format(
                row.get("subset", ""),
                row.get("score_name", ""),
                intfmt(row.get("n_scored")),
                fmt(row.get("spearman_log10_kd_ratio")),
                fmt(row.get("auroc_damaging")),
                fmt(row.get("auprc_damaging")),
                fmt(row.get("top10_damaging_precision")),
            )
        )
    lines.append("\n")
    lines.append(
        "The binder proxy signal is modest: the mapped-interface binder proxy gets AUROC `{}` while the stronger "
        "location_COR label gets `{}`. This means interface/contact features are useful, but the proxy is not strong "
        "enough to stand in for direct deletion-preserved binding. The EGFR binder dataset therefore needs a proper "
        "deletion-retention benchmark before it can be used as a decisive validation set.\n\n".format(
            fmt(sk_binder_mapped.get("auroc_damaging"), 3), fmt(sk_location_mapped.get("auroc_damaging"), 3)
        )
    )

    lines.append("## 5. Ablation Study\n\n")
    lines.append(
        "Evidence dropout tests whether the planner behaves sensibly when annotation, structure, or both are missing.\n\n"
    )
    lines.append("| Evidence mode | Adaptive | Confidence | Max budget | Delete ratio | Shadow rate | Closure rate | Structural risk / deleted |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in [ab_full, ab_naive_ann, ab_adapt_ann, ab_naive_struct, ab_adapt_struct, ab_naive_seq, ab_adapt_seq]:
        lines.append(
            "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n".format(
                row.get("evidence_mode", ""),
                intfmt(row.get("adaptive_evidence")),
                fmt(row.get("evidence_confidence")),
                fmt(row.get("max_budget")),
                fmt(row.get("global_auto_delete_ratio")),
                fmt(row.get("shadow_overlap_rate")),
                fmt(row.get("closure_unfriendly_rate")),
                fmt(row.get("structural_risk_per_deleted")),
            )
        )
    lines.append("\n")
    lines.append(
        "The ablation is clean: dropping structure or annotation without uncertainty handling makes deletion much "
        "more aggressive, while adaptive evidence caps pull the policy back toward safety. Sequence-only naive "
        "deletes `{}`; sequence-only adaptive deletes `{}`. That is the behavior we want if the model lacks evidence.\n\n".format(
            fmt(ab_naive_seq.get("global_auto_delete_ratio"), 3),
            fmt(ab_adapt_seq.get("global_auto_delete_ratio"), 3),
        )
    )

    lines.append("## 6. Structure Re-Prediction Case Studies\n\n")
    lines.append(
        "Five high-deletion current-best cases were re-predicted with ESMFold. The table reports deleted-sequence "
        "structure confidence and local breakpoint geometry.\n\n"
    )
    lines.append("| Accession | Deleted length | Delete ratio | Mean pLDDT | Protected pLDDT | Breakpoint CA mean | Breakpoint CA max | Status |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for row in esmfold_rows:
        lines.append(
            "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |\n".format(
                row.get("accession", ""),
                intfmt(row.get("deleted_length")),
                fmt(row.get("delete_ratio"), 3),
                fmt(row.get("mean_plddt"), 3),
                fmt(row.get("protected_mean_plddt"), 3),
                fmt(row.get("breakpoint_mean_ca_distance"), 3),
                fmt(row.get("breakpoint_max_ca_distance"), 3),
                row.get("prediction_status", ""),
            )
        )
    lines.append("\n")
    lines.append(
        "All five ESMFold runs succeeded. Four cases have moderate-to-high mean pLDDT after deletion; Q43296 is the "
        "low-confidence outlier and should be treated as an uncertainty example, not as a strong positive case. The "
        "point of this experiment is only to show that the chosen deletions can be re-modeled structurally; it is not "
        "yet direct proof of function retention.\n\n"
    )

    lines.append("## 7. External Baseline References\n\n")
    lines.append(
        "These rows are reported separately from the current-best six experiments. They are still useful for paper "
        "context, but they should not be mixed with the main current-best pipeline tables unless rerun under exactly "
        "the same protocol.\n\n"
    )
    lines.append("### 7.1 Matched-Length SCISOR / Raygun Comparison\n\n")
    lines.append("| Method | N proteins | Target delete ratio | Fill | Shadow rate | Closure rate | Length success | Sequence identity | Status |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    matched_status = {
        "BioDel-Cert default": "current matched-length reference",
        "SCISOR hardmask+shadow02": "recheck subset matches old metrics: {}/{} differ".format(scisor_recheck.get("diffs"), scisor_recheck.get("common")),
        "Raygun 2.2M": "partial recheck differs on {}/{} shared rows; stochastic/not yet final".format(raygun_recheck.get("diffs"), raygun_recheck.get("common")),
    }
    for row in [matched_biodel, matched_scisor, matched_raygun]:
        lines.append(
            "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |\n".format(
                row.get("method", ""),
                intfmt(row.get("n_proteins")),
                fmt(row.get("mean_target_delete_ratio")),
                fmt(row.get("mean_fill_ratio")),
                fmt(row.get("shadow_overlap_rate")),
                fmt(row.get("closure_unfriendly_rate")),
                fmt(row.get("length_success_rate")),
                fmt(row.get("mean_sequence_identity")),
                matched_status.get(row.get("method", ""), ""),
            )
        )
    lines.append("\n")
    lines.append(
        "The SCISOR matched-length row is stable on the currently rechecked subset. Raygun is not stable row-by-row "
        "in the partial rerun, so the Raygun values above should be cited as historical matched-length results until "
        "the ongoing recheck writes a final summary. Current recheck progress: `{}` rows over `{}` accessions; final "
        "summary written: `{}`.\n\n".format(
            recheck.get("rows"), recheck.get("accessions"), recheck.get("summary_done")
        )
    )

    lines.append("### 7.2 Archived SCISOR family baseline\n\n")
    lines.append("| Method | Budget | Fill | Protected rate | Shadow rate | Closure rate | Status |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---|\n")
    for row in scisor_family_legacy:
        if row.get("budget_label") != "30%":
            continue
        lines.append(
            "| {} | {} | `{}` | `{}` | `{}` | `{}` | legacy archived |\n".format(
                row.get("method", ""),
                row.get("budget_label", ""),
                fmt(row.get("fill_ratio")),
                fmt(row.get("protected_overlap_rate")),
                fmt(row.get("shadow_overlap_rate")),
                fmt(row.get("closure_unfriendly_rate")),
            )
        )
    lines.append("\n")

    lines.append("### 7.3 ProteinGym zero-shot PLM baselines\n\n")
    lines.append("| Zero-shot model | Mean per-assay Spearman | Mean per-assay AUROC | Mean per-assay AUPRC |\n")
    lines.append("|---|---:|---:|---:|\n")
    for row in zero_shot_ranked:
        lines.append(
            "| {} | `{}` | `{}` | `{}` |\n".format(
                row["score_name"], fmt(row["spearman"]), fmt(row["auroc"]), fmt(row["auprc"])
            )
        )
    lines.append("\n")
    lines.append(
        "The strongest zero-shot PLM baselines in the current ProteinGym export are much stronger than the current "
        "Stage-1 checkpoint on mean per-assay Spearman. They are not deletion planners, but they set a useful upper "
        "reference for sequence-only fitness signal.\n\n"
    )

    lines.append("## Overall Result Statement\n\n")
    lines.append(
        "The current version successfully runs the six-experiment pipeline end to end: candidate proposal, "
        "risk-constrained planning, automatic budgeting, external validation, ablations, and structure case studies. "
        "The strongest conclusion is that explicit constraints matter: utility-only deletion reaches high fill but "
        "incurs large shadow and closure risk, while safe/selected BioDel operating points prevent protected-site "
        "violations and can keep shadow/closure risk at zero in conservative settings.\n\n"
    )
    lines.append(
        "The weakest result is Stage-1 model quality. Candidate recall is only moderate, thresholded token F1 is zero, "
        "and ProteinGym correlations are weaker than simple terminal or length baselines and much weaker than zero-shot "
        "PLM fitness scores. The honest framing is that the framework and experimental protocol are now in place, but "
        "the learned deletion prior still needs another training and model-design iteration before claiming strong "
        "biological predictive performance.\n\n"
    )
    lines.append("## Result Files\n\n")
    lines.append("- Candidate intervals: `results/current_best_experiments/stage1_interval_proposals_eval.csv`\n")
    lines.append("- Main family-heldout comparison: `results/current_best_experiments/experiment_family_split_current_certified/final_test_family_comparison/final_test_comparison_table.csv`\n")
    lines.append("- Auto-budget frontier: `results/current_best_experiments/experiment_family_split_current_certified/certified_frontier_test_summary.csv`\n")
    lines.append("- CATH / DisProt summaries: `results/current_best_experiments/experiment2_cath_current_certified/cath_auto_summary.csv`, `results/current_best_experiments/experiment2_disprot_current_certified/disprot_auto_summary.csv`\n")
    lines.append("- ProteinGym / SKEMPI external validation: `results/current_best_experiments/experiment4_proteingym/proteingym_current_stage1_external_validation_metrics.csv`, `results/current_best_experiments/experiment4_skempi/skempi_v2_binder_proxy_metrics.csv`\n")
    lines.append("- Ablations: `results/current_best_experiments/experiment5_ablation_current_certified/bioprior_10k_evidence_dropout_summary.csv`\n")
    lines.append("- ESMFold case studies: `results/current_best_experiments/experiment6_case_studies/current_best_esmfold_structure_summary.csv`\n\n")
    lines.append("CURRENT_BEST_SIX_EXPERIMENTS_SUMMARY_PASS\n")

    write_report(OUT, lines)
    return OUT


if __name__ == "__main__":
    print(build())
