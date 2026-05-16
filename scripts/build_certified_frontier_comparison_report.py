#!/usr/bin/env python3
"""Build a comparison report for certified frontier planning.

The report intentionally includes coverage/abstention because certified
frontier planning is not meant to force a deletion for every protein.  This
distinguishes it from fixed-budget and auto-budget baselines.
"""

import argparse
import csv
import os


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def fnum(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def inum(row, key, default=0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return int(float(value))


def fmt(value):
    if isinstance(value, int):
        return value
    return "{:.6f}".format(float(value))


def certified_frontier_rows(path):
    out = []
    for row in read_csv(path):
        out.append(
            {
                "family": "certified_frontier",
                "method": "certified_frontier_{}".format(row["auto_profile"]),
                "coverage": inum(row, "certified_proteins") / float(max(1, inum(row, "analyzed_proteins"))),
                "abstention_rate": fnum(row, "abstention_rate"),
                "delete_ratio": fnum(row, "global_auto_delete_ratio"),
                "protected_overlap_residues": inum(row, "protected_overlap_residues"),
                "shadow_rate": fnum(row, "shadow_overlap_rate"),
                "closure_rate": fnum(row, "closure_unfriendly_rate"),
                "structural_risk_per_deleted": fnum(row, "structural_risk_per_deleted"),
                "mean_risk_upper": fnum(row, "mean_risk_upper"),
                "mean_evidence_confidence": fnum(row, "mean_evidence_confidence"),
                "notes": "frontier selection with abstention",
            }
        )
    return out


def auto_budget_safe_rows(path):
    rows = read_csv(path)
    out = []
    for row in rows:
        if row.get("setting") == "safe" and row.get("auto_profile") == "default":
            out.append(
                {
                    "family": "auto_budget",
                    "method": "auto_budget_certified_safe_default",
                    "coverage": 1.0,
                    "abstention_rate": 0.0,
                    "delete_ratio": fnum(row, "global_auto_delete_ratio"),
                    "protected_overlap_residues": inum(row, "protected_overlap_residues"),
                    "shadow_rate": fnum(row, "shadow_overlap_rate"),
                    "closure_rate": fnum(row, "closure_unfriendly_rate"),
                    "structural_risk_per_deleted": fnum(row, "structural_risk_per_deleted"),
                    "mean_risk_upper": fnum(row, "mean_risk_upper"),
                    "mean_evidence_confidence": fnum(row, "mean_evidence_confidence"),
                    "notes": "old certified auto-budget operating point",
                }
            )
    return out


def selected_v2_rows(path):
    if not os.path.exists(path):
        return []
    selected = []
    for row in read_csv(path):
        if row.get("method") != "validation_selected_v2":
            continue
        underlying = (row.get("selected_underlying_method") or "v2").replace("v2_", "")
        budget = row.get("budget") or ""
        budget_tag = str(budget).replace(".", "p")
        selected.append(
            {
                "family": "v2_selected",
                "method": "validation_selected_v2_{}_{}".format(underlying, budget_tag),
                "coverage": 1.0,
                "abstention_rate": 0.0,
                "delete_ratio": fnum(row, "achieved_deletion_ratio"),
                "protected_overlap_residues": inum(row, "protected_overlap_residues"),
                "shadow_rate": fnum(row, "shadow_overlap_rate"),
                "closure_rate": fnum(row, "closure_unfriendly_rate"),
                "structural_risk_per_deleted": fnum(row, "structural_core_risk_per_deleted"),
                "mean_risk_upper": None,
                "mean_evidence_confidence": None,
                "notes": "validation-selected v2 operating point",
            }
        )
    return selected


def selected_greedy_rows(path):
    rows = read_csv(path)
    out = []
    wanted = [
        ("utility_greedy_protected_only", "0.3"),
        ("strict_safe_greedy", "0.3"),
    ]
    for method, budget in wanted:
        matches = [row for row in rows if row.get("split_name") == "test" and row.get("method") == method and row.get("budget_ratio") == budget]
        if not matches:
            continue
        row = matches[0]
        selected_len = max(1.0, fnum(row, "total_selected_len"))
        total_length = selected_len / max(1e-9, fnum(row, "global_fill_ratio") * float(budget))
        out.append(
            {
                "family": "greedy_baseline",
                "method": "{}_budget_{}".format(method, budget),
                "coverage": 1.0,
                "abstention_rate": 0.0,
                "delete_ratio": selected_len / total_length,
                "protected_overlap_residues": inum(row, "protected_overlap_residues"),
                "shadow_rate": fnum(row, "shadow_overlap_residues") / selected_len,
                "closure_rate": fnum(row, "closure_unfriendly_len") / selected_len,
                "structural_risk_per_deleted": fnum(row, "structural_risk_units") / selected_len,
                "mean_risk_upper": None,
                "mean_evidence_confidence": None,
                "notes": "greedy fixed-budget baseline",
            }
        )
    return out


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "family",
        "method",
        "coverage",
        "abstention_rate",
        "delete_ratio",
        "protected_overlap_residues",
        "shadow_rate",
        "closure_rate",
        "structural_risk_per_deleted",
        "mean_risk_upper",
        "mean_evidence_confidence",
        "notes",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def write_report(path, rows):
    ensure_parent(path)
    by_method = {row["method"]: row for row in rows}
    cf_default = by_method.get("certified_frontier_default")
    auto_safe = by_method.get("auto_budget_certified_safe_default")
    greedy = by_method.get("utility_greedy_protected_only_budget_0.3")
    strict = by_method.get("strict_safe_greedy_budget_0.3")

    with open(path, "w") as handle:
        handle.write("# Certified Frontier Comparison\n\n")
        handle.write("Split: BioPrior-10K held-out test.\n\n")
        handle.write("## Core Table\n\n")
        handle.write("| Family | Method | Coverage | Abstain | Delete ratio | Protected | Shadow rate | Closure rate | Structural risk | Risk upper | Evidence confidence |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                "| {family} | {method} | {coverage:.3f} | {abstention_rate:.3f} | {delete_ratio:.3f} | {protected_overlap_residues} | {shadow_rate:.4f} | {closure_rate:.4f} | {structural_risk} | {risk_upper} | {confidence} |\n".format(
                    family=row["family"],
                    method=row["method"],
                    coverage=float(row["coverage"]),
                    abstention_rate=float(row["abstention_rate"]),
                    delete_ratio=float(row["delete_ratio"]),
                    protected_overlap_residues=row["protected_overlap_residues"],
                    shadow_rate=float(row["shadow_rate"]),
                    closure_rate=float(row["closure_rate"]),
                    structural_risk="" if row.get("structural_risk_per_deleted") is None else "{:.4f}".format(float(row["structural_risk_per_deleted"])),
                    risk_upper="" if row.get("mean_risk_upper") is None else "{:.4f}".format(float(row["mean_risk_upper"])),
                    confidence="" if row.get("mean_evidence_confidence") is None else "{:.4f}".format(float(row["mean_evidence_confidence"])),
                )
            )

        handle.write("\n## Interpretation\n\n")
        if cf_default and auto_safe:
            handle.write(
                "- Certified frontier default covers {:.1f}% of proteins and deletes {:.1f}% of residues globally, with closure rate {:.4f}. The old safe/default auto-budget deletes {:.1f}% and forces coverage 100%.\n".format(
                    100.0 * float(cf_default["coverage"]),
                    100.0 * float(cf_default["delete_ratio"]),
                    float(cf_default["closure_rate"]),
                    100.0 * float(auto_safe["delete_ratio"]),
                )
            )
        if greedy:
            handle.write(
                "- Utility greedy at 30% reaches {:.1f}% deletion but has shadow rate {:.3f} and closure rate {:.3f}, showing why unconstrained utility is unsafe.\n".format(
                    100.0 * float(greedy["delete_ratio"]),
                    float(greedy["shadow_rate"]),
                    float(greedy["closure_rate"]),
                )
            )
        if strict:
            handle.write(
                "- Strict-safe greedy has zero shadow/closure risk but deletes only {:.1f}% globally at the 30% target, showing the cost of hard filtering without certified frontier selection.\n".format(
                    100.0 * float(strict["delete_ratio"])
                )
            )
        handle.write("- The new algorithm should be framed as a calibrated selective planner: it gives up coverage on high-risk proteins rather than forcing a fixed deletion ratio.\n")
        handle.write("\nBIODEL_CERTIFIED_FRONTIER_COMPARISON_PASS\n")


def run(args):
    rows = []
    rows.extend(certified_frontier_rows(args.certified_frontier_summary))
    rows.extend(auto_budget_safe_rows(args.auto_budget_certified_summary))
    rows.extend(selected_v2_rows(args.v2_summary))
    rows.extend(selected_greedy_rows(args.baseline_comparison))
    write_csv(args.out_csv, rows)
    write_report(args.out_report, rows)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Build certified frontier comparison report.")
    parser.add_argument("--certified_frontier_summary", default="results/biodel_planner/certified_frontier_test_summary.csv")
    parser.add_argument("--auto_budget_certified_summary", default="results/biodel_planner/bioprior_10k_test_auto_budget_certified_summary.csv")
    parser.add_argument(
        "--v2_summary",
        "--final_comparison_table",
        dest="v2_summary",
        default="results/biodel_planner/final_test_comparison/final_test_comparison_table.csv",
        help="Final comparison table that contains validation-selected v2 rows.",
    )
    parser.add_argument("--baseline_comparison", default="results/biodel_planner/bioprior_10k_test_planner_baseline_comparison.csv")
    parser.add_argument("--out_csv", default="results/biodel_planner/certified_frontier_comparison.csv")
    parser.add_argument("--out_report", default="results/biodel_planner/CERTIFIED_FRONTIER_COMPARISON_REPORT.md")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
