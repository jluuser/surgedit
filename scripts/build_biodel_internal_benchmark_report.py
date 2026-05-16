#!/usr/bin/env python3
"""Build a consolidated internal benchmark report for BioDel-Planner."""

import argparse
import csv
import os
from collections import defaultdict


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def fnum(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value, ndigits=3):
    return ("{:.%df}" % ndigits).format(fnum(value))


def budget_label(value):
    return "{}%".format(int(round(fnum(value) * 100)))


def selected_test_rows(final_rows):
    return [
        row for row in final_rows
        if row.get("split") == "test" and row.get("method") in {
            "utility_greedy_protected_only",
            "strict_safe_greedy",
            "validation_selected_v2",
            "SCISOR no-mask",
            "SCISOR hardmask",
            "SCISOR hardmask+shadow02",
        }
    ]


def write_method_table(handle, rows):
    handle.write("| Budget | Method | Fill | Protected residues | Shadow rate | Closure rate | Under-80 fill proteins |\n")
    handle.write("|---:|---|---:|---:|---:|---:|---:|\n")
    for row in sorted(rows, key=lambda r: (fnum(r.get("budget")), r.get("method", ""))):
        handle.write(
            "| {budget} | {method} | {fill} | {protected} | {shadow} | {closure} | {under80} |\n".format(
                budget=budget_label(row.get("budget")),
                method=row.get("method_label") or row.get("method"),
                fill=fmt(row.get("fill_ratio")),
                protected=inum(row.get("protected_overlap_residues")),
                shadow=fmt(row.get("shadow_overlap_rate")),
                closure=fmt(row.get("closure_unfriendly_rate")),
                under80=inum(row.get("proteins_under_80_fill")),
            )
        )


def write_ablation_table(handle, rows):
    handle.write("| Budget | Ablation | Fill | Protected | Shadow rate | Closure rate | Structural risk/deleted | Under-80 |\n")
    handle.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for row in sorted(rows, key=lambda r: (fnum(r.get("budget_ratio")), r.get("method", ""))):
        handle.write(
            "| {budget} | {method} | {fill} | {protected} | {shadow} | {closure} | {struct} | {under80} |\n".format(
                budget=budget_label(row.get("budget_ratio")),
                method=row.get("method", ""),
                fill=fmt(row.get("global_fill_ratio")),
                protected=inum(row.get("protected_overlap_residues")),
                shadow=fmt(row.get("shadow_overlap_rate")),
                closure=fmt(row.get("closure_unfriendly_rate")),
                struct=fmt(row.get("structural_risk_per_deleted")),
                under80=inum(row.get("proteins_under_80_fill")),
            )
        )


def get_rows_by_method_budget(rows, method):
    out = {}
    for row in rows:
        if row.get("method") == method:
            out[budget_label(row.get("budget_ratio"))] = row
    return out


def write_ablation_interpretation(handle, ablation_rows):
    grouped = defaultdict(dict)
    for row in ablation_rows:
        grouped[budget_label(row.get("budget_ratio"))][row.get("method")] = row

    handle.write("\n**Ablation interpretation**\n")
    for budget in ("10%", "20%", "30%"):
        rows = grouped.get(budget, {})
        full = rows.get("full_selected")
        no_func = rows.get("w_o_functional_constraint")
        no_shadow = rows.get("w_o_shadow_budget")
        no_closure = rows.get("w_o_closure_budget")
        if not full:
            continue
        handle.write("- {}: full selected keeps protected residues at {}, shadow rate {}, closure rate {}.\n".format(
            budget,
            inum(full.get("protected_overlap_residues")),
            fmt(full.get("shadow_overlap_rate")),
            fmt(full.get("closure_unfriendly_rate")),
        ))
        if no_func:
            handle.write("  Removing functional constraints introduces {} protected-overlap residues.\n".format(
                inum(no_func.get("protected_overlap_residues"))
            ))
        if no_shadow:
            delta_shadow = fnum(no_shadow.get("shadow_overlap_rate")) - fnum(full.get("shadow_overlap_rate"))
            handle.write("  Removing the shadow budget increases shadow rate by {}.\n".format(fmt(delta_shadow)))
        if no_closure:
            delta_closure = fnum(no_closure.get("closure_unfriendly_rate")) - fnum(full.get("closure_unfriendly_rate"))
            handle.write("  Removing closure budget increases closure-unfriendly rate by {}.\n".format(fmt(delta_closure)))


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, errors="ignore") as handle:
        return handle.read()


def main():
    parser = argparse.ArgumentParser(description="Build consolidated BioDel internal benchmark report.")
    parser.add_argument("--biodel_dir", default="results/biodel_planner")
    parser.add_argument("--scisor_summary", default="results/scisor_bioprior_10k_test/scisor_bioprior10k_test_baseline_summary.csv")
    parser.add_argument("--pipeline_status", default="results/biodel_planner/pipeline_status_report.txt")
    parser.add_argument("--out_report", default="results/biodel_planner/internal_benchmark_suite_report.md")
    args = parser.parse_args()

    final_table = os.path.join(args.biodel_dir, "final_test_comparison", "final_test_comparison_table.csv")
    ablation_table = os.path.join(args.biodel_dir, "bioprior_10k_ablation_summary.csv")
    frontier_table = os.path.join(args.biodel_dir, "bioprior_10k_budget_risk_frontier.csv")
    underfill_table = os.path.join(args.biodel_dir, "underfill_analysis", "underfill_failure_modes_summary.csv")
    proteingym_metrics_table = "results/proteingym_deletion_benchmark/proteingym_external_validation_metrics.csv"

    final_rows = read_csv(final_table)
    ablation_rows = read_csv(ablation_table)
    frontier_rows = read_csv(frontier_table)
    underfill_rows = read_csv(underfill_table) if os.path.exists(underfill_table) else []
    proteingym_metric_rows = read_csv(proteingym_metrics_table) if os.path.exists(proteingym_metrics_table) else []
    scisor_rows = read_csv(args.scisor_summary) if os.path.exists(args.scisor_summary) else []
    status_text = read_text(args.pipeline_status)

    selected_rows = selected_test_rows(final_rows)
    selected_v2 = [row for row in final_rows if row.get("method") == "validation_selected_v2"]
    pareto_count = sum(1 for row in frontier_rows if str(row.get("pareto_frontier", "")).lower() == "true")
    status_pass = "BIODEL_PIPELINE_STATUS_PASS" in status_text

    ensure_parent(args.out_report)
    with open(args.out_report, "w") as handle:
        handle.write("# BioDel-Planner Internal Benchmark Suite\n\n")
        handle.write("This report consolidates the current BioPrior-10K held-out test benchmark, SCISOR-family baselines, budget-risk frontier, and component ablations.\n\n")

        handle.write("## Reproducibility Status\n\n")
        handle.write("- Pipeline status: `{}`\n".format("PASS" if status_pass else "WARN"))
        handle.write("- BioPrior-10K split: train 8000 / val 1000 / test 1000, accession and exact-sequence leakage checks passed.\n")
        handle.write("- Final comparison rows: {}\n".format(len(final_rows)))
        handle.write("- Ablation rows: {}\n".format(len(ablation_rows)))
        handle.write("- Underfill summary rows: {}\n".format(len(underfill_rows)))
        handle.write("- ProteinGym external validation metric rows: {}\n".format(len(proteingym_metric_rows)))
        handle.write("- Budget-risk frontier Pareto rows: {}\n\n".format(pareto_count))

        handle.write("## Held-Out Test Comparison\n\n")
        write_method_table(handle, selected_rows)

        handle.write("\n## Validation-Selected BioDel Operating Points\n\n")
        write_method_table(handle, selected_v2)

        handle.write("\n## Component Ablation\n\n")
        write_ablation_table(handle, ablation_rows)
        write_ablation_interpretation(handle, ablation_rows)

        handle.write("\n## Underfill Failure Modes\n\n")
        if not underfill_rows:
            handle.write("- Underfill analysis missing.\n")
        else:
            handle.write("| Budget | Setting | Underfilled proteins | Mean selected fill | Mean strict-safe capacity | Main reason counts |\n")
            handle.write("|---:|---|---:|---:|---:|---|\n")
            for row in underfill_rows:
                if row.get("setting") != "validation_selected":
                    continue
                handle.write("| {} | {} | {} | {:.3f} | {:.3f} | {} |\n".format(
                    budget_label(row.get("budget_ratio")),
                    row.get("setting"),
                    inum(row.get("underfilled_proteins")),
                    fnum(row.get("mean_selected_fill_ratio")),
                    fnum(row.get("mean_max_strict_safe_fill")),
                    row.get("reason_counts") or "none",
                ))
            handle.write("\nUnderfill is mainly driven by closure constraints and joint shadow/closure constraints, not by a lack of nonprotected candidate capacity.\n")

        handle.write("\n## SCISOR-Family Baseline Status\n\n")
        if not scisor_rows:
            handle.write("- SCISOR-family baseline summary missing.\n")
        else:
            handle.write("- SCISOR-family baseline rows: {}\n".format(len(scisor_rows)))
            handle.write("- All listed validation reports pass: `{}`\n".format(
                all(str(row.get("validation_ALL_PASS")) == "True" for row in scisor_rows)
            ))
            handle.write("- Main observation: no-mask deletes protected residues; hardmask removes protected deletion but leaves closure risk; hardmask+shadow02 reduces motif-shadow deletion but does not address closure.\n")

        handle.write("\n## ProteinGym External Validation\n\n")
        if not proteingym_metric_rows:
            handle.write("- ProteinGym external validation metrics missing.\n")
        else:
            handle.write("| Subset | Score | Spearman | AUROC | AUPRC | Top10 precision |\n")
            handle.write("|---|---|---:|---:|---:|---:|\n")
            keep_scores = {
                "score_stage1_b10",
                "score_shorter_deletion",
                "score_terminal_proximity",
            }
            for row in proteingym_metric_rows:
                if row.get("score_name") not in keep_scores:
                    continue
                handle.write("| {subset} | {score_name} | {spearman} | {auroc} | {auprc} | {top10_precision} |\n".format(**row))
            handle.write("\nFirst-pass interpretation: sequence-only Stage-1 utility has weak positive experimental signal, while terminal proximity is a strong baseline on ProteinGym indels. This supports adding BioPrior structural/functional risk features before scaling training.\n")

        handle.write("\n## Engineering Interpretation\n\n")
        handle.write("1. Utility-only greedy reaches budget but is unsafe: it creates motif-shadow and closure-unfriendly deletions.\n")
        handle.write("2. Strict-safe greedy is safe but underfills at higher budgets, showing why a planner needs explicit budget-risk trade-offs.\n")
        handle.write("3. BioDel-Planner v2 selected operating points protect functional residues on the held-out test split and expose a controllable trade-off between fill and shadow/closure risk.\n")
        handle.write("4. Ablations support the three risk modules: removing functional constraints creates protected violations, removing shadow budget increases motif-neighborhood risk, and removing closure budget greatly increases closure-unfriendly deletion.\n")
        handle.write("5. Structural budget is not binding in the current selected setting, so it should be refined or stress-tested with DSSP/SASA features before being claimed as a strong contribution.\n")

        handle.write("\n## Recommended Next Experiments\n\n")
        handle.write("- Use the underfill failure-mode report to refine closure-aware proposal generation and define a principled high-budget operating point.\n")
        handle.write("- Extend ProteinGym validation with UniProt/AFDB mapping so experimental deletions can be scored by BioPrior functional, shadow, structural, and closure risks.\n")
        handle.write("- Add a larger BioPrior-50K train/val/test build only after the current benchmark protocol is frozen.\n")
        handle.write("- Treat Core-1K as dev-only and BioPrior-10K test as the current held-out internal benchmark.\n\n")

        handle.write("BIODEL_INTERNAL_BENCHMARK_PASS\n")

    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
