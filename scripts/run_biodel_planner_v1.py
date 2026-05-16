#!/usr/bin/env python3
"""Run BioDel-Planner v1 baselines on segment candidates.

This first planner version compares unconstrained greedy scoring against a
risk-constrained non-overlapping segment selector under deletion budgets.
"""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict


METHODS = [
    "stage1_utility_greedy",
    "bioprior_score_greedy",
    "risk_constrained_stage1",
]


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def parse_budgets(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def read_segments(path):
    by_accession = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            row["_start"] = to_int(row["seg_start"])
            row["_end"] = to_int(row["seg_end"])
            row["_seg_len"] = to_int(row["seg_len"])
            row["_protein_length"] = to_int(row["protein_length"])
            by_accession[row["accession"]].append(row)
    return by_accession, fields


def overlaps_any(candidate, selected):
    start = candidate["_start"]
    end = candidate["_end"]
    for row in selected:
        if not (end < row["_start"] or start > row["_end"]):
            return True
    return False


def is_closure_safe(row):
    return row.get("closure_type") == "terminal" or to_bool(row.get("closure_friendly_8A"))


def is_risk_allowed(row, args):
    if to_bool(row.get("hard_reject")):
        return False
    if to_float(row.get("protected_overlap_fraction")) > 0:
        return False
    if not is_closure_safe(row):
        return False
    if to_float(row.get("shadow_overlap_fraction")) > args.max_shadow_overlap:
        return False
    if to_float(row.get("structural_core_risk_score")) > args.max_structural_core_risk:
        return False
    if to_float(row.get("motif_shadow_risk_score")) > args.max_motif_shadow_risk:
        return False
    return True


def score_row(row, method, args):
    utility = to_float(row.get("stage1_utility_score"))
    if method == "stage1_utility_greedy":
        return utility
    if method == "bioprior_score_greedy":
        return to_float(row.get("final_bioprior_score"))
    if method == "risk_constrained_stage1":
        risk_penalty = (
            args.shadow_penalty_weight * to_float(row.get("shadow_overlap_fraction"))
            + args.struct_penalty_weight * to_float(row.get("structural_core_risk_score"))
            + args.closure_penalty_weight * (0.0 if is_closure_safe(row) else 1.0)
        )
        return utility - risk_penalty
    raise ValueError(method)


def select_segments(rows, budget, method, args):
    if not rows:
        return [], 0
    protein_length = rows[0]["_protein_length"]
    target_len = int(math.floor(protein_length * budget))
    target_len = max(1, target_len)
    max_selected_len = int(math.floor(target_len + protein_length * args.allow_overshoot_fraction))
    max_selected_len = max(target_len, max_selected_len)

    candidates = rows
    if method == "risk_constrained_stage1":
        candidates = [row for row in rows if is_risk_allowed(row, args)]
    candidates = sorted(
        candidates,
        key=lambda row: (score_row(row, method, args), to_float(row.get("stage1_utility_score")), row["_seg_len"]),
        reverse=True,
    )

    selected = []
    selected_len = 0
    for row in candidates:
        seg_len = row["_seg_len"]
        if selected_len >= target_len:
            break
        if selected_len + seg_len > max_selected_len:
            continue
        if overlaps_any(row, selected):
            continue
        selected.append(row)
        selected_len += seg_len
    return selected, target_len


def summarize_selection(selected):
    return {
        "selected_segments": len(selected),
        "selected_len": sum(row["_seg_len"] for row in selected),
        "protected_overlap_segments": sum(1 for row in selected if to_float(row.get("protected_overlap_fraction")) > 0),
        "protected_overlap_residues": sum(to_int(row.get("n_protected_overlap")) for row in selected),
        "shadow_overlap_segments": sum(1 for row in selected if to_float(row.get("shadow_overlap_fraction")) > 0),
        "shadow_overlap_residues": sum(to_int(row.get("n_shadow_overlap")) for row in selected),
        "closure_unfriendly_segments": sum(1 for row in selected if not is_closure_safe(row)),
        "hard_reject_segments": sum(1 for row in selected if to_bool(row.get("hard_reject"))),
        "mean_shadow_overlap_fraction": mean([to_float(row.get("shadow_overlap_fraction")) for row in selected]),
        "mean_contact_density_8A": mean([to_float(row.get("mean_contact_density_8A")) for row in selected]),
        "mean_pLDDT": mean([to_float(row.get("mean_pLDDT")) for row in selected]),
        "mean_stage1_utility": mean([to_float(row.get("stage1_utility_score")) for row in selected]),
        "mean_final_bioprior_score": mean([to_float(row.get("final_bioprior_score")) for row in selected]),
    }


def write_selected(path, selected_rows, original_fields):
    ensure_parent(path)
    planner_fields = [
        "method",
        "budget_ratio",
        "target_delete_len",
        "selected_total_len_for_protein",
        "fill_ratio_for_protein",
        "planner_rank",
        "planner_score",
    ]
    fields = planner_fields + original_fields
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)


def write_summary_csv(path, rows):
    ensure_parent(path)
    fields = [
        "method",
        "budget_ratio",
        "total_proteins",
        "total_target_len",
        "total_selected_len",
        "mean_fill_ratio",
        "proteins_fill_ge_0.80",
        "proteins_fill_ge_0.95",
        "selected_segments",
        "protected_overlap_segments",
        "protected_overlap_residues",
        "shadow_overlap_segments",
        "shadow_overlap_residues",
        "closure_unfriendly_segments",
        "hard_reject_segments",
        "mean_shadow_overlap_fraction",
        "mean_contact_density_8A",
        "mean_pLDDT",
        "mean_stage1_utility",
        "mean_final_bioprior_score",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args):
    budgets = parse_budgets(args.budgets)
    by_accession, fields = read_segments(args.segments_csv)
    selected_output = []
    summary_rows = []
    per_method_budget = defaultdict(list)

    for method in METHODS:
        for budget in budgets:
            aggregate = Counter()
            fill_ratios = []
            metric_lists = defaultdict(list)
            for accession, rows in sorted(by_accession.items()):
                selected, target_len = select_segments(rows, budget, method, args)
                selected_len = sum(row["_seg_len"] for row in selected)
                fill_ratio = selected_len / float(target_len) if target_len else 0.0
                fill_ratios.append(fill_ratio)
                summary = summarize_selection(selected)
                aggregate["total_proteins"] += 1
                aggregate["total_target_len"] += target_len
                aggregate["total_selected_len"] += selected_len
                aggregate["proteins_fill_ge_0.80"] += 1 if fill_ratio >= 0.80 else 0
                aggregate["proteins_fill_ge_0.95"] += 1 if fill_ratio >= 0.95 else 0
                for key, value in summary.items():
                    if key.startswith("mean_"):
                        metric_lists[key].append(value)
                    else:
                        aggregate[key] += value
                for rank, row in enumerate(selected, start=1):
                    out = dict(row)
                    for private_key in list(out):
                        if private_key.startswith("_"):
                            del out[private_key]
                    out.update(
                        {
                            "method": method,
                            "budget_ratio": budget,
                            "target_delete_len": target_len,
                            "selected_total_len_for_protein": selected_len,
                            "fill_ratio_for_protein": "{:.6f}".format(fill_ratio),
                            "planner_rank": rank,
                            "planner_score": "{:.6f}".format(score_row(row, method, args)),
                        }
                    )
                    selected_output.append(out)
            summary_row = {
                "method": method,
                "budget_ratio": budget,
                "total_proteins": aggregate["total_proteins"],
                "total_target_len": aggregate["total_target_len"],
                "total_selected_len": aggregate["total_selected_len"],
                "mean_fill_ratio": mean(fill_ratios),
                "proteins_fill_ge_0.80": aggregate["proteins_fill_ge_0.80"],
                "proteins_fill_ge_0.95": aggregate["proteins_fill_ge_0.95"],
                "selected_segments": aggregate["selected_segments"],
                "protected_overlap_segments": aggregate["protected_overlap_segments"],
                "protected_overlap_residues": aggregate["protected_overlap_residues"],
                "shadow_overlap_segments": aggregate["shadow_overlap_segments"],
                "shadow_overlap_residues": aggregate["shadow_overlap_residues"],
                "closure_unfriendly_segments": aggregate["closure_unfriendly_segments"],
                "hard_reject_segments": aggregate["hard_reject_segments"],
                "mean_shadow_overlap_fraction": mean(metric_lists["mean_shadow_overlap_fraction"]),
                "mean_contact_density_8A": mean(metric_lists["mean_contact_density_8A"]),
                "mean_pLDDT": mean(metric_lists["mean_pLDDT"]),
                "mean_stage1_utility": mean(metric_lists["mean_stage1_utility"]),
                "mean_final_bioprior_score": mean(metric_lists["mean_final_bioprior_score"]),
            }
            summary_rows.append(summary_row)
            per_method_budget[(method, budget)] = summary_row

    write_selected(args.out_selected_csv, selected_output, fields)
    write_summary_csv(args.out_summary_csv, summary_rows)
    write_report(args.out_report, args, summary_rows, per_method_budget)
    print("Wrote {}".format(args.out_selected_csv))
    print("Wrote {}".format(args.out_summary_csv))
    print("Wrote {}".format(args.out_report))


def write_report(path, args, summary_rows, per_method_budget):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("BioDel-Planner v1 report\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("budgets: {}\n".format(args.budgets))
        handle.write("risk_constrained thresholds:\n")
        handle.write("- max_shadow_overlap: {}\n".format(args.max_shadow_overlap))
        handle.write("- max_structural_core_risk: {}\n".format(args.max_structural_core_risk))
        handle.write("- max_motif_shadow_risk: {}\n".format(args.max_motif_shadow_risk))
        handle.write("- allow_overshoot_fraction: {}\n\n".format(args.allow_overshoot_fraction))
        handle.write("Core table:\n")
        handle.write("method,budget,fill,protected_res,shadow_res,closure_unfriendly,hard_reject,selected_segments\n")
        for row in summary_rows:
            handle.write(
                "{method},{budget_ratio},{mean_fill_ratio:.4f},{protected_overlap_residues},{shadow_overlap_residues},{closure_unfriendly_segments},{hard_reject_segments},{selected_segments}\n".format(
                    **row
                )
            )
        handle.write("\nRisk-constrained vs stage1 greedy deltas:\n")
        for budget in sorted({row["budget_ratio"] for row in summary_rows}):
            greedy = per_method_budget.get(("stage1_utility_greedy", budget))
            constrained = per_method_budget.get(("risk_constrained_stage1", budget))
            if not greedy or not constrained:
                continue
            shadow_delta = greedy["shadow_overlap_residues"] - constrained["shadow_overlap_residues"]
            closure_delta = greedy["closure_unfriendly_segments"] - constrained["closure_unfriendly_segments"]
            fill_delta = constrained["mean_fill_ratio"] - greedy["mean_fill_ratio"]
            handle.write(
                "- budget {}: shadow_res_delta={}, closure_unfriendly_delta={}, fill_delta={:.4f}\n".format(
                    budget, shadow_delta, closure_delta, fill_delta
                )
            )
        handle.write("\nInterpretation:\n")
        handle.write("- stage1_utility_greedy tests whether learned utility alone is safe enough.\n")
        handle.write("- bioprior_score_greedy tests whether rule-derived BioPrior scoring alone is sufficient.\n")
        handle.write("- risk_constrained_stage1 tests the core planner claim: maximize learned utility under explicit biological risk constraints.\n")
        handle.write("\nBIODEL_PLANNER_V1_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run BioDel-Planner v1 on Stage-1-scored segment candidates.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--out_selected_csv", required=True)
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--max_shadow_overlap", type=float, default=0.25)
    parser.add_argument("--max_structural_core_risk", type=float, default=0.75)
    parser.add_argument("--max_motif_shadow_risk", type=float, default=0.25)
    parser.add_argument("--shadow_penalty_weight", type=float, default=0.5)
    parser.add_argument("--struct_penalty_weight", type=float, default=0.2)
    parser.add_argument("--closure_penalty_weight", type=float, default=0.5)
    parser.add_argument("--allow_overshoot_fraction", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
