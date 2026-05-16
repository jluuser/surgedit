#!/usr/bin/env python3
"""Compare simple segment-selection baselines against BioDel-Planner v2.

This script evaluates whether budgeted risk-constrained planning is necessary
by comparing v2 selected segments against greedy baselines that rank segments
by Stage-1 utility or BioPrior score under weaker local rules.
"""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict


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


def read_split_accessions(path):
    accessions = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if accession:
                accessions.append(accession)
    return accessions


def annotate_row(row):
    row = dict(row)
    row["_start"] = to_int(row["seg_start"])
    row["_end"] = to_int(row["seg_end"])
    row["_seg_len"] = to_int(row["seg_len"])
    row["_protein_length"] = to_int(row["protein_length"])
    row["_utility_mean"] = to_float(row.get("stage1_utility_score"))
    row["_utility_sum"] = row["_utility_mean"] * row["_seg_len"]
    row["_final_bioprior"] = to_float(row.get("final_bioprior_score"))
    row["_bioprior_sum"] = row["_final_bioprior"] * row["_seg_len"]
    row["_protected_overlap_res"] = to_int(row.get("n_protected_overlap"))
    row["_protected_overlap_frac"] = to_float(row.get("protected_overlap_fraction"))
    row["_shadow_res"] = to_int(row.get("n_shadow_overlap"))
    row["_shadow_frac"] = to_float(row.get("shadow_overlap_fraction"))
    row["_struct_risk"] = max(0.0, to_float(row.get("structural_core_risk_score")))
    row["_struct_units"] = row["_struct_risk"] * row["_seg_len"]
    row["_hard_reject"] = to_bool(row.get("hard_reject"))
    row["_closure_type"] = row.get("closure_type", "")
    row["_closure_friendly"] = to_bool(row.get("closure_friendly_8A"))
    row["_closure_unfriendly"] = row["_closure_type"] != "terminal" and not row["_closure_friendly"]
    row["_closure_units"] = row["_seg_len"] if row["_closure_unfriendly"] else 0
    return row


def read_segments(path, allowed_accessions):
    allowed = set(allowed_accessions)
    by_accession = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            accession = raw.get("accession")
            if accession not in allowed:
                continue
            row = annotate_row(raw)
            by_accession[accession].append(row)
    return dict(by_accession)


def overlaps(row, selected):
    for other in selected:
        if not (row["_end"] < other["_start"] or row["_start"] > other["_end"]):
            return True
    return False


def hard_allowed(row):
    if row["_hard_reject"]:
        return False
    if row["_protected_overlap_res"] > 0 or row["_protected_overlap_frac"] > 0:
        return False
    return True


def strict_safe_allowed(row, max_structural_core_risk):
    if not hard_allowed(row):
        return False
    if row["_shadow_res"] > 0:
        return False
    if row["_closure_unfriendly"]:
        return False
    if row["_struct_risk"] > max_structural_core_risk:
        return False
    return True


def greedy_select(rows, budget, args, method):
    if not rows:
        return [], 0
    protein_length = rows[0]["_protein_length"]
    target_len = max(1, int(math.floor(protein_length * budget)))
    max_selected_len = max(target_len, int(math.floor(target_len + protein_length * args.allow_overshoot_fraction)))
    if method == "utility_greedy_protected_only":
        candidates = [row for row in rows if hard_allowed(row)]
        candidates.sort(key=lambda r: (r["_utility_sum"], r["_utility_mean"], r["_seg_len"]), reverse=True)
    elif method == "bioprior_greedy_protected_only":
        candidates = [row for row in rows if hard_allowed(row)]
        candidates.sort(key=lambda r: (r["_bioprior_sum"], r["_final_bioprior"], r["_utility_sum"]), reverse=True)
    elif method == "strict_safe_greedy":
        candidates = [row for row in rows if strict_safe_allowed(row, args.strict_max_structural_core_risk)]
        candidates.sort(key=lambda r: (r["_utility_sum"] + 0.1 * r["_bioprior_sum"], r["_seg_len"]), reverse=True)
    else:
        raise ValueError("Unknown method {}".format(method))
    selected = []
    selected_len = 0
    for row in candidates[: args.max_candidates_per_protein]:
        if selected_len + row["_seg_len"] > max_selected_len:
            continue
        if overlaps(row, selected):
            continue
        selected.append(row)
        selected_len += row["_seg_len"]
        if selected_len >= target_len:
            break
    return selected, target_len


def read_v2_selected(path, allowed_accessions):
    allowed = set(allowed_accessions)
    grouped = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            accession = raw.get("accession")
            if accession not in allowed:
                continue
            row = annotate_row(raw)
            setting = raw["setting"]
            budget = float(raw["budget_ratio"])
            grouped[(setting, budget, accession)].append(row)
    return grouped


def summarize_method(method, budget, by_accession, selected_by_accession):
    agg = Counter()
    metric_lists = defaultdict(list)
    for accession, rows in sorted(by_accession.items()):
        if not rows:
            continue
        protein_length = rows[0]["_protein_length"]
        target_len = max(1, int(math.floor(protein_length * budget)))
        selected = selected_by_accession.get(accession, [])
        selected_len = sum(row["_seg_len"] for row in selected)
        fill_ratio = selected_len / float(target_len) if target_len else 0.0
        agg["analyzed_proteins"] += 1
        agg["total_target_len"] += target_len
        agg["total_selected_len"] += selected_len
        agg["proteins_fill_ge_0.80"] += 1 if fill_ratio >= 0.80 else 0
        agg["proteins_fill_ge_0.95"] += 1 if fill_ratio >= 0.95 else 0
        agg["selected_segments"] += len(selected)
        agg["protected_overlap_residues"] += sum(row["_protected_overlap_res"] for row in selected)
        agg["shadow_overlap_residues"] += sum(row["_shadow_res"] for row in selected)
        agg["shadow_overlap_segments"] += sum(1 for row in selected if row["_shadow_res"] > 0)
        agg["closure_unfriendly_segments"] += sum(1 for row in selected if row["_closure_unfriendly"])
        agg["closure_unfriendly_len"] += sum(row["_closure_units"] for row in selected)
        agg["structural_risk_units"] += sum(row["_struct_units"] for row in selected)
        metric_lists["fill_ratio"].append(fill_ratio)
        metric_lists["stage1_utility"].extend(row["_utility_mean"] for row in selected)
        metric_lists["final_bioprior"].extend(row["_final_bioprior"] for row in selected)
        metric_lists["shadow_fraction"].extend(row["_shadow_frac"] for row in selected)
        metric_lists["structural_core_risk"].extend(row["_struct_risk"] for row in selected)
    return {
        "method": method,
        "budget_ratio": budget,
        "analyzed_proteins": agg["analyzed_proteins"],
        "total_target_len": agg["total_target_len"],
        "total_selected_len": agg["total_selected_len"],
        "global_fill_ratio": agg["total_selected_len"] / float(agg["total_target_len"] or 1),
        "mean_fill_ratio": mean(metric_lists["fill_ratio"]),
        "proteins_fill_ge_0.80": agg["proteins_fill_ge_0.80"],
        "proteins_fill_ge_0.95": agg["proteins_fill_ge_0.95"],
        "selected_segments": agg["selected_segments"],
        "protected_overlap_residues": agg["protected_overlap_residues"],
        "shadow_overlap_residues": agg["shadow_overlap_residues"],
        "shadow_overlap_segments": agg["shadow_overlap_segments"],
        "closure_unfriendly_segments": agg["closure_unfriendly_segments"],
        "closure_unfriendly_len": agg["closure_unfriendly_len"],
        "structural_risk_units": agg["structural_risk_units"],
        "mean_stage1_utility": mean(metric_lists["stage1_utility"]),
        "mean_final_bioprior_score": mean(metric_lists["final_bioprior"]),
        "mean_shadow_overlap_fraction": mean(metric_lists["shadow_fraction"]),
        "mean_structural_core_risk": mean(metric_lists["structural_core_risk"]),
    }


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "split_name",
        "method",
        "budget_ratio",
        "analyzed_proteins",
        "total_target_len",
        "total_selected_len",
        "global_fill_ratio",
        "mean_fill_ratio",
        "proteins_fill_ge_0.80",
        "proteins_fill_ge_0.95",
        "selected_segments",
        "protected_overlap_residues",
        "shadow_overlap_residues",
        "shadow_overlap_segments",
        "closure_unfriendly_segments",
        "closure_unfriendly_len",
        "structural_risk_units",
        "mean_stage1_utility",
        "mean_final_bioprior_score",
        "mean_shadow_overlap_fraction",
        "mean_structural_core_risk",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, rows):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("BioDel planner baseline comparison\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("split_csv: {}\n".format(args.split_csv))
        handle.write("split_name: {}\n".format(args.split_name))
        handle.write("v2_selected_csv: {}\n".format(args.v2_selected_csv))
        handle.write("budgets: {}\n\n".format(args.budgets))
        handle.write("Core table:\n")
        handle.write("budget,method,fill,protected,shadow,closure_unfriendly_len,selected_segments\n")
        for row in rows:
            handle.write(
                "{budget_ratio},{method},{global_fill_ratio:.4f},{protected_overlap_residues},{shadow_overlap_residues},{closure_unfriendly_len},{selected_segments}\n".format(
                    **row
                )
            )
        handle.write("\nInterpretation:\n")
        handle.write("- utility_greedy_protected_only tests whether Stage-1 utility alone is sufficient after hard motif protection.\n")
        handle.write("- bioprior_greedy_protected_only tests simple segment scoring without global risk budgets.\n")
        handle.write("- strict_safe_greedy tests local hard risk filtering, which can underfill high budgets.\n")
        handle.write("- v2_* methods use global risk budgets and non-overlapping beam selection.\n\n")
        protected_ok = all(int(row["protected_overlap_residues"]) == 0 for row in rows if row["method"].startswith("v2_"))
        if protected_ok:
            handle.write("BIODEL_BASELINE_COMPARISON_PASS\n")
        else:
            handle.write("BIODEL_BASELINE_COMPARISON_WARN\n")


def run(args):
    budgets = parse_budgets(args.budgets)
    split_accessions = read_split_accessions(args.split_csv)
    by_accession = read_segments(args.segments_csv, split_accessions)
    v2_grouped = read_v2_selected(args.v2_selected_csv, split_accessions)
    rows = []
    greedy_methods = [
        "utility_greedy_protected_only",
        "bioprior_greedy_protected_only",
        "strict_safe_greedy",
    ]
    for budget in budgets:
        for method in greedy_methods:
            selected_by_accession = {}
            for accession, seg_rows in by_accession.items():
                selected, _ = greedy_select(seg_rows, budget, args, method)
                selected_by_accession[accession] = selected
            row = summarize_method(method, budget, by_accession, selected_by_accession)
            row["split_name"] = args.split_name
            rows.append(row)
        for setting in ["safe", "balanced", "aggressive"]:
            selected_by_accession = {
                accession: v2_grouped.get((setting, budget, accession), [])
                for accession in by_accession
            }
            row = summarize_method("v2_{}".format(setting), budget, by_accession, selected_by_accession)
            row["split_name"] = args.split_name
            rows.append(row)
    write_csv(args.out_csv, rows)
    write_report(args.out_report, args, rows)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare BioDel planner baselines.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--split_name", default="split")
    parser.add_argument("--v2_selected_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--max_candidates_per_protein", type=int, default=160)
    parser.add_argument("--allow_overshoot_fraction", type=float, default=0.02)
    parser.add_argument("--strict_max_structural_core_risk", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
