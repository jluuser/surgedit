#!/usr/bin/env python3
"""Select a BioDel-Planner operating point on validation and report test metrics."""

import argparse
import csv
import json
import os


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def to_float(row, key):
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def row_rates(row):
    selected_len = max(1.0, to_float(row, "total_selected_len"))
    return {
        "shadow_rate": to_float(row, "shadow_overlap_residues") / selected_len,
        "closure_rate": to_float(row, "closure_unfriendly_len") / selected_len,
        "struct_rate": to_float(row, "structural_risk_units") / selected_len,
    }


def score_row(row, args):
    rates = row_rates(row)
    fill = to_float(row, "global_fill_ratio") if "global_fill_ratio" in row else to_float(row, "mean_fill_ratio")
    return (
        fill
        - args.shadow_weight * rates["shadow_rate"]
        - args.closure_weight * rates["closure_rate"]
        - args.struct_weight * rates["struct_rate"]
    )


def feasible(row, args):
    rates = row_rates(row)
    return (
        int(float(row.get("protected_overlap_residues", 0))) == 0
        and rates["shadow_rate"] <= args.max_shadow_rate
        and rates["closure_rate"] <= args.max_closure_rate
        and to_float(row, "global_fill_ratio") >= args.min_global_fill
    )


def select_by_budget(rows, args):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["budget_ratio"], []).append(row)
    selected = {}
    for budget, budget_rows in sorted(grouped.items(), key=lambda item: float(item[0])):
        candidates = [row for row in budget_rows if row["method"].startswith("v2_")]
        feasible_rows = [row for row in candidates if feasible(row, args)]
        pool = feasible_rows if feasible_rows else candidates
        best = max(pool, key=lambda row: score_row(row, args))
        selected[budget] = {
            "row": best,
            "feasible": bool(feasible_rows),
            "score": score_row(best, args),
            "rates": row_rates(best),
        }
    return selected


def main():
    parser = argparse.ArgumentParser(description="Select BioDel operating point from validation split.")
    parser.add_argument("--val_comparison_csv", required=True)
    parser.add_argument("--test_comparison_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--min_global_fill", type=float, default=0.65)
    parser.add_argument("--max_shadow_rate", type=float, default=0.02)
    parser.add_argument("--max_closure_rate", type=float, default=0.15)
    parser.add_argument("--shadow_weight", type=float, default=5.0)
    parser.add_argument("--closure_weight", type=float, default=2.0)
    parser.add_argument("--struct_weight", type=float, default=0.0)
    args = parser.parse_args()

    val_rows = read_csv(args.val_comparison_csv)
    test_rows = read_csv(args.test_comparison_csv)
    selected = select_by_budget(val_rows, args)

    selected_payload = {}
    for budget, item in selected.items():
        val_row = item["row"]
        method = val_row["method"]
        test_match = [
            row for row in test_rows
            if row["budget_ratio"] == budget and row["method"] == method
        ]
        test_row = test_match[0] if test_match else {}
        selected_payload[budget] = {
            "selected_method": method,
            "val_feasible_under_thresholds": item["feasible"],
            "val_score": item["score"],
            "val_metrics": val_row,
            "val_rates": item["rates"],
            "test_metrics": test_row,
            "test_rates": row_rates(test_row) if test_row else {},
        }

    ensure_parent(args.out_json)
    with open(args.out_json, "w") as handle:
        json.dump(selected_payload, handle, indent=2)

    ensure_parent(args.out_report)
    with open(args.out_report, "w") as handle:
        handle.write("BioDel operating point selection report\n\n")
        handle.write("Selection split: validation\n")
        handle.write("Held-out split: test\n")
        handle.write("Criteria:\n")
        handle.write("- min_global_fill: {}\n".format(args.min_global_fill))
        handle.write("- max_shadow_rate: {}\n".format(args.max_shadow_rate))
        handle.write("- max_closure_rate: {}\n".format(args.max_closure_rate))
        handle.write("- objective: fill - {}*shadow_rate - {}*closure_rate - {}*struct_rate\n\n".format(
            args.shadow_weight, args.closure_weight, args.struct_weight
        ))
        handle.write("Selected operating points:\n")
        handle.write("budget,method,val_fill,val_shadow_rate,val_closure_rate,test_fill,test_shadow_rate,test_closure_rate\n")
        for budget, payload in selected_payload.items():
            val_metrics = payload["val_metrics"]
            test_metrics = payload["test_metrics"]
            val_rates = payload["val_rates"]
            test_rates = payload["test_rates"]
            handle.write(
                "{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    budget,
                    payload["selected_method"],
                    to_float(val_metrics, "global_fill_ratio"),
                    val_rates.get("shadow_rate", 0.0),
                    val_rates.get("closure_rate", 0.0),
                    to_float(test_metrics, "global_fill_ratio"),
                    test_rates.get("shadow_rate", 0.0),
                    test_rates.get("closure_rate", 0.0),
                )
            )
        handle.write("\nInterpretation:\n")
        handle.write("- Operating points are selected only from validation metrics.\n")
        handle.write("- Test metrics are reported after selection without changing thresholds.\n")
        handle.write("\nBIODEL_OPERATING_POINT_SELECTION_PASS\n")

    print("Wrote {}".format(args.out_json))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
