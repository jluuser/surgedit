#!/usr/bin/env python3
"""Build budget-risk frontier tables for BioDel-Planner experiments."""

import argparse
import csv
import json
import os
from collections import defaultdict


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def to_int(row, key, default=0):
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def enrich_row(row, split_name):
    selected_len = max(1.0, to_float(row, "total_selected_len"))
    target_len = max(1.0, to_float(row, "total_target_len"))
    shadow_rate = to_float(row, "shadow_overlap_residues") / selected_len
    closure_rate = to_float(row, "closure_unfriendly_len") / selected_len
    struct_rate = to_float(row, "structural_risk_units") / selected_len
    protected_violation = to_int(row, "protected_overlap_residues")
    row = dict(row)
    row["split_name"] = split_name
    row["target_fill_ratio"] = "{:.6f}".format(to_float(row, "total_selected_len") / target_len)
    row["shadow_rate_per_deleted"] = "{:.6f}".format(shadow_rate)
    row["closure_rate_per_deleted"] = "{:.6f}".format(closure_rate)
    row["structural_risk_per_deleted"] = "{:.6f}".format(struct_rate)
    row["protected_violation"] = protected_violation
    row["risk_score_shadow_closure"] = "{:.6f}".format(shadow_rate + closure_rate)
    return row


def dominates(a, b):
    """Return True if row a dominates row b for same split/budget.

    Higher fill is better; lower protected/shadow/closure risk is better.
    """
    if int(a["protected_violation"]) > int(b["protected_violation"]):
        return False
    if float(a["target_fill_ratio"]) < float(b["target_fill_ratio"]):
        return False
    if float(a["shadow_rate_per_deleted"]) > float(b["shadow_rate_per_deleted"]):
        return False
    if float(a["closure_rate_per_deleted"]) > float(b["closure_rate_per_deleted"]):
        return False
    return (
        int(a["protected_violation"]) < int(b["protected_violation"])
        or float(a["target_fill_ratio"]) > float(b["target_fill_ratio"])
        or float(a["shadow_rate_per_deleted"]) < float(b["shadow_rate_per_deleted"])
        or float(a["closure_rate_per_deleted"]) < float(b["closure_rate_per_deleted"])
    )


def mark_frontier(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split_name"], row["budget_ratio"])].append(row)
    out = []
    for _, group_rows in grouped.items():
        for row in group_rows:
            is_frontier = not any(dominates(other, row) for other in group_rows if other is not row)
            row = dict(row)
            row["pareto_frontier"] = str(is_frontier)
            out.append(row)
    return out


def load_selected_methods(path):
    with open(path) as handle:
        payload = json.load(handle)
    return {str(budget): item["selected_method"] for budget, item in payload.items()}


def write_frontier_csv(path, rows):
    ensure_parent(path)
    fields = [
        "split_name",
        "budget_ratio",
        "method",
        "target_fill_ratio",
        "mean_fill_ratio",
        "protected_violation",
        "shadow_overlap_residues",
        "shadow_rate_per_deleted",
        "closure_unfriendly_len",
        "closure_rate_per_deleted",
        "structural_risk_per_deleted",
        "risk_score_shadow_closure",
        "selected_segments",
        "pareto_frontier",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, args, rows, selected_methods):
    ensure_parent(path)
    test_rows = [row for row in rows if row["split_name"] == "test"]
    grouped = defaultdict(list)
    for row in test_rows:
        grouped[row["budget_ratio"]].append(row)

    with open(path, "w") as handle:
        handle.write("BioDel budget-risk frontier report\n\n")
        handle.write("val_comparison_csv: {}\n".format(args.val_comparison_csv))
        handle.write("test_comparison_csv: {}\n".format(args.test_comparison_csv))
        handle.write("selected_operating_points_json: {}\n\n".format(args.selected_operating_points_json))

        handle.write("Held-out test core table:\n")
        handle.write("budget,method,fill,shadow_rate,closure_rate,protected,pareto\n")
        for row in sorted(test_rows, key=lambda r: (float(r["budget_ratio"]), r["method"])):
            handle.write(
                "{},{},{:.4f},{:.4f},{:.4f},{},{}\n".format(
                    row["budget_ratio"],
                    row["method"],
                    float(row["target_fill_ratio"]),
                    float(row["shadow_rate_per_deleted"]),
                    float(row["closure_rate_per_deleted"]),
                    row["protected_violation"],
                    row["pareto_frontier"],
                )
            )

        handle.write("\nValidation-selected operating points evaluated on test:\n")
        handle.write("budget,selected_method,test_fill,test_shadow_rate,test_closure_rate\n")
        for budget, method in sorted(selected_methods.items(), key=lambda item: float(item[0])):
            matches = [row for row in test_rows if row["budget_ratio"] == budget and row["method"] == method]
            if not matches:
                continue
            row = matches[0]
            handle.write(
                "{},{},{:.4f},{:.4f},{:.4f}\n".format(
                    budget,
                    method,
                    float(row["target_fill_ratio"]),
                    float(row["shadow_rate_per_deleted"]),
                    float(row["closure_rate_per_deleted"]),
                )
            )

        handle.write("\nWhy planning is needed:\n")
        for budget in sorted(grouped, key=float):
            rows_by_method = {row["method"]: row for row in grouped[budget]}
            utility = rows_by_method.get("utility_greedy_protected_only")
            strict = rows_by_method.get("strict_safe_greedy")
            selected = rows_by_method.get(selected_methods.get(budget, ""))
            handle.write("- budget {}:\n".format(budget))
            if utility:
                handle.write(
                    "  utility-only reaches fill {:.3f} but deletes shadow rate {:.3f} and closure-unfriendly rate {:.3f}.\n".format(
                        float(utility["target_fill_ratio"]),
                        float(utility["shadow_rate_per_deleted"]),
                        float(utility["closure_rate_per_deleted"]),
                    )
                )
            if strict:
                handle.write(
                    "  strict-safe keeps shadow/closure at zero but fill is {:.3f}.\n".format(
                        float(strict["target_fill_ratio"])
                    )
                )
            if selected:
                handle.write(
                    "  validation-selected {} gives fill {:.3f}, shadow rate {:.3f}, closure rate {:.3f}.\n".format(
                        selected["method"],
                        float(selected["target_fill_ratio"]),
                        float(selected["shadow_rate_per_deleted"]),
                        float(selected["closure_rate_per_deleted"]),
                    )
                )

        handle.write("\nConclusion:\n")
        handle.write("BIODEL_BUDGET_RISK_FRONTIER_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Build BioDel budget-risk frontier report.")
    parser.add_argument("--val_comparison_csv", required=True)
    parser.add_argument("--test_comparison_csv", required=True)
    parser.add_argument("--selected_operating_points_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()

    rows = []
    rows.extend(enrich_row(row, "val") for row in read_csv(args.val_comparison_csv))
    rows.extend(enrich_row(row, "test") for row in read_csv(args.test_comparison_csv))
    rows = mark_frontier(rows)
    selected_methods = load_selected_methods(args.selected_operating_points_json)
    write_frontier_csv(args.out_csv, rows)
    write_report(args.out_report, args, rows, selected_methods)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
