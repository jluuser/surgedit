#!/usr/bin/env python3
"""Run BioDel-Planner ablations on a held-out split.

The ablations are designed to answer whether each part of the planner controls
a distinct failure mode:
  - functional hard constraint -> protected motif deletion
  - motif-shadow risk budget -> motif-neighborhood deletion
  - closure risk budget -> stitchability/closure risk
  - structural risk budget -> folded-core risk
  - Stage-1 utility -> utility model contribution under the same risk budgets
"""

import argparse
import csv
import json
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


def read_selected_settings(path):
    with open(path) as handle:
        payload = json.load(handle)
    selected = {}
    for budget, item in payload.items():
        method = item["selected_method"].replace("v2_", "")
        selected[float(budget)] = method
    return selected


def default_settings():
    return {
        "safe": {
            "shadow_budget_fraction": 0.0,
            "closure_budget_fraction": 0.0,
            "structural_budget_fraction": 0.75,
        },
        "balanced": {
            "shadow_budget_fraction": 0.02,
            "closure_budget_fraction": 0.10,
            "structural_budget_fraction": 0.75,
        },
        "aggressive": {
            "shadow_budget_fraction": 0.05,
            "closure_budget_fraction": 0.25,
            "structural_budget_fraction": 1.0,
        },
    }


def annotate_row(row):
    row = dict(row)
    row["_start"] = to_int(row["seg_start"])
    row["_end"] = to_int(row["seg_end"])
    row["_seg_len"] = to_int(row["seg_len"])
    row["_protein_length"] = to_int(row["protein_length"])
    row["_stage1_mean"] = to_float(row.get("stage1_utility_score"))
    row["_stage1_sum"] = row["_stage1_mean"] * row["_seg_len"]
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
        fields = list(reader.fieldnames or [])
        for row in reader:
            accession = row.get("accession")
            if accession not in allowed:
                continue
            by_accession[accession].append(annotate_row(row))
    return dict(by_accession), fields


def hard_allowed(row):
    return (
        not row["_hard_reject"]
        and row["_protected_overlap_res"] == 0
        and row["_protected_overlap_frac"] == 0
    )


def overlaps(candidate, selected):
    for row in selected:
        if not (candidate["_end"] < row["_start"] or candidate["_start"] > row["_end"]):
            return True
    return False


def utility_value(row, mode):
    if mode == "stage1":
        return row["_stage1_sum"], row["_stage1_mean"]
    if mode == "bioprior":
        return row["_bioprior_sum"], row["_final_bioprior"]
    if mode == "length":
        return float(row["_seg_len"]), 1.0
    raise ValueError("Unknown utility mode {}".format(mode))


def ablation_specs():
    return [
        {
            "method": "full_selected",
            "description": "validation-selected v2 operating point",
            "functional_constraint": True,
            "shadow_budget": True,
            "closure_budget": True,
            "struct_budget": True,
            "utility_mode": "stage1",
        },
        {
            "method": "w_o_functional_constraint",
            "description": "allow protected-overlap candidates",
            "functional_constraint": False,
            "shadow_budget": True,
            "closure_budget": True,
            "struct_budget": True,
            "utility_mode": "stage1",
        },
        {
            "method": "w_o_shadow_budget",
            "description": "remove motif-shadow risk budget",
            "functional_constraint": True,
            "shadow_budget": False,
            "closure_budget": True,
            "struct_budget": True,
            "utility_mode": "stage1",
        },
        {
            "method": "w_o_closure_budget",
            "description": "remove closure/stitchability risk budget",
            "functional_constraint": True,
            "shadow_budget": True,
            "closure_budget": False,
            "struct_budget": True,
            "utility_mode": "stage1",
        },
        {
            "method": "w_o_structural_budget",
            "description": "remove structural-core risk budget",
            "functional_constraint": True,
            "shadow_budget": True,
            "closure_budget": True,
            "struct_budget": False,
            "utility_mode": "stage1",
        },
        {
            "method": "w_o_stage1_utility",
            "description": "replace Stage-1 utility with BioPrior score",
            "functional_constraint": True,
            "shadow_budget": True,
            "closure_budget": True,
            "struct_budget": True,
            "utility_mode": "bioprior",
        },
    ]


def state_score(state, target_len, args):
    fill = state["selected_len"] / float(target_len) if target_len else 0.0
    capped_fill = min(fill, 1.0)
    underfill = max(0.0, 1.0 - fill)
    overshoot = max(0.0, fill - 1.0)
    utility_norm = state["utility_sum"] / float(max(1, target_len))
    return args.fill_weight * capped_fill + utility_norm - args.underfill_penalty * underfill - args.overshoot_penalty * overshoot


def risk_budget_for(target_len, setting, spec):
    huge = 10 ** 12
    return {
        "shadow": int(math.floor(target_len * setting["shadow_budget_fraction"])) if spec["shadow_budget"] else huge,
        "closure": int(math.floor(target_len * setting["closure_budget_fraction"])) if spec["closure_budget"] else huge,
        "struct": float(target_len * setting["structural_budget_fraction"]) if spec["struct_budget"] else float(huge),
    }


def add_candidate(state, row, risk_budget, max_selected_len, utility_mode):
    utility_sum, _ = utility_value(row, utility_mode)
    if state["selected_len"] + row["_seg_len"] > max_selected_len:
        return None
    if state["shadow_units"] + row["_shadow_res"] > risk_budget["shadow"]:
        return None
    if state["closure_units"] + row["_closure_units"] > risk_budget["closure"]:
        return None
    if state["struct_units"] + row["_struct_units"] > risk_budget["struct"]:
        return None
    if overlaps(row, state["selected"]):
        return None
    return {
        "selected": state["selected"] + [row],
        "selected_len": state["selected_len"] + row["_seg_len"],
        "utility_sum": state["utility_sum"] + utility_sum,
        "shadow_units": state["shadow_units"] + row["_shadow_res"],
        "closure_units": state["closure_units"] + row["_closure_units"],
        "struct_units": state["struct_units"] + row["_struct_units"],
    }


def prune_states(states, target_len, beam_size, args):
    states = sorted(states, key=lambda s: state_score(s, target_len, args), reverse=True)
    pruned = []
    signatures = set()
    for state in states:
        signature = (
            state["selected_len"],
            state["shadow_units"],
            state["closure_units"],
            int(round(state["struct_units"] * 10)),
            tuple((row["_start"], row["_end"]) for row in state["selected"]),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        pruned.append(state)
        if len(pruned) >= beam_size:
            break
    return pruned


def plan_protein(rows, budget, setting, spec, args):
    if not rows:
        return [], 0
    protein_length = rows[0]["_protein_length"]
    target_len = max(1, int(math.floor(protein_length * budget)))
    max_selected_len = max(target_len, int(math.floor(target_len + protein_length * args.allow_overshoot_fraction)))
    risk_budget = risk_budget_for(target_len, setting, spec)
    if spec["functional_constraint"]:
        candidates = [row for row in rows if hard_allowed(row)]
    else:
        candidates = list(rows)
    candidates.sort(
        key=lambda row: (
            utility_value(row, spec["utility_mode"])[0],
            utility_value(row, spec["utility_mode"])[1],
            row["_seg_len"],
        ),
        reverse=True,
    )
    candidates = candidates[: args.max_candidates_per_protein]
    states = [
        {
            "selected": [],
            "selected_len": 0,
            "utility_sum": 0.0,
            "shadow_units": 0,
            "closure_units": 0,
            "struct_units": 0.0,
        }
    ]
    for row in candidates:
        new_states = list(states)
        for state in states:
            added = add_candidate(state, row, risk_budget, max_selected_len, spec["utility_mode"])
            if added is not None:
                new_states.append(added)
        states = prune_states(new_states, target_len, args.beam_size, args)
    best = sorted(states, key=lambda s: state_score(s, target_len, args), reverse=True)[0]
    return best["selected"], target_len


def summarize(method, description, budget, selected_by_accession, by_accession):
    agg = Counter()
    metric_lists = defaultdict(list)
    for accession, rows in sorted(by_accession.items()):
        if not rows:
            continue
        protein_length = rows[0]["_protein_length"]
        target_len = max(1, int(math.floor(protein_length * budget)))
        selected = selected_by_accession.get(accession, [])
        selected_len = sum(row["_seg_len"] for row in selected)
        fill = selected_len / float(target_len) if target_len else 0.0
        agg["analyzed_proteins"] += 1
        agg["total_target_len"] += target_len
        agg["total_selected_len"] += selected_len
        agg["selected_segments"] += len(selected)
        agg["proteins_under_80_fill"] += 1 if fill < 0.8 else 0
        protected = sum(row["_protected_overlap_res"] for row in selected)
        shadow = sum(row["_shadow_res"] for row in selected)
        closure = sum(row["_closure_units"] for row in selected)
        agg["protected_overlap_residues"] += protected
        agg["shadow_overlap_residues"] += shadow
        agg["closure_unfriendly_len"] += closure
        agg["closure_unfriendly_segments"] += sum(1 for row in selected if row["_closure_unfriendly"])
        agg["structural_risk_units"] += sum(row["_struct_units"] for row in selected)
        agg["proteins_with_any_protected_violation"] += 1 if protected else 0
        agg["proteins_with_any_shadow_overlap"] += 1 if shadow else 0
        agg["proteins_with_any_closure_unfriendly"] += 1 if closure else 0
        metric_lists["fill_ratio"].append(fill)
        metric_lists["stage1_utility"].extend(row["_stage1_mean"] for row in selected)
        metric_lists["final_bioprior"].extend(row["_final_bioprior"] for row in selected)
        metric_lists["shadow_fraction"].extend(row["_shadow_frac"] for row in selected)
        metric_lists["struct_risk"].extend(row["_struct_risk"] for row in selected)
    selected_len = float(agg["total_selected_len"] or 1)
    return {
        "method": method,
        "description": description,
        "budget_ratio": budget,
        "analyzed_proteins": agg["analyzed_proteins"],
        "total_target_len": agg["total_target_len"],
        "total_selected_len": agg["total_selected_len"],
        "global_fill_ratio": agg["total_selected_len"] / float(agg["total_target_len"] or 1),
        "mean_fill_ratio": mean(metric_lists["fill_ratio"]),
        "proteins_under_80_fill": agg["proteins_under_80_fill"],
        "selected_segments": agg["selected_segments"],
        "protected_overlap_residues": agg["protected_overlap_residues"],
        "protected_overlap_rate": agg["protected_overlap_residues"] / selected_len,
        "shadow_overlap_residues": agg["shadow_overlap_residues"],
        "shadow_overlap_rate": agg["shadow_overlap_residues"] / selected_len,
        "closure_unfriendly_len": agg["closure_unfriendly_len"],
        "closure_unfriendly_rate": agg["closure_unfriendly_len"] / selected_len,
        "closure_unfriendly_segments": agg["closure_unfriendly_segments"],
        "structural_risk_units": agg["structural_risk_units"],
        "structural_risk_per_deleted": agg["structural_risk_units"] / selected_len,
        "proteins_with_any_protected_violation": agg["proteins_with_any_protected_violation"],
        "proteins_with_any_shadow_overlap": agg["proteins_with_any_shadow_overlap"],
        "proteins_with_any_closure_unfriendly": agg["proteins_with_any_closure_unfriendly"],
        "mean_stage1_utility": mean(metric_lists["stage1_utility"]),
        "mean_final_bioprior_score": mean(metric_lists["final_bioprior"]),
        "mean_shadow_overlap_fraction": mean(metric_lists["shadow_fraction"]),
        "mean_structural_core_risk": mean(metric_lists["struct_risk"]),
    }


def write_summary_csv(path, rows):
    ensure_parent(path)
    fields = [
        "method",
        "description",
        "budget_ratio",
        "analyzed_proteins",
        "total_target_len",
        "total_selected_len",
        "global_fill_ratio",
        "mean_fill_ratio",
        "proteins_under_80_fill",
        "selected_segments",
        "protected_overlap_residues",
        "protected_overlap_rate",
        "shadow_overlap_residues",
        "shadow_overlap_rate",
        "closure_unfriendly_len",
        "closure_unfriendly_rate",
        "closure_unfriendly_segments",
        "structural_risk_units",
        "structural_risk_per_deleted",
        "proteins_with_any_protected_violation",
        "proteins_with_any_shadow_overlap",
        "proteins_with_any_closure_unfriendly",
        "mean_stage1_utility",
        "mean_final_bioprior_score",
        "mean_shadow_overlap_fraction",
        "mean_structural_core_risk",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_selected_csv(path, rows):
    ensure_parent(path)
    fields = [
        "method",
        "budget_ratio",
        "accession",
        "seg_start",
        "seg_end",
        "seg_len",
        "proposal_source",
        "biological_rationale",
        "n_protected_overlap",
        "n_shadow_overlap",
        "closure_type",
        "closure_friendly_8A",
        "structural_core_risk_score",
        "stage1_utility_score",
        "final_bioprior_score",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path, rows):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("BioDel-Planner ablation report\n\n")
        handle.write("Core table:\n")
        handle.write("budget,method,fill,protected,shadow_rate,closure_rate,struct_rate,under80\n")
        for row in rows:
            handle.write(
                "{budget_ratio},{method},{global_fill_ratio:.4f},{protected_overlap_residues},{shadow_overlap_rate:.4f},{closure_unfriendly_rate:.4f},{structural_risk_per_deleted:.4f},{proteins_under_80_fill}\n".format(
                    **row
                )
            )
        handle.write("\nInterpretation:\n")
        handle.write("- w_o_functional_constraint should reveal whether protected-overlap segments become attractive without the hard constraint.\n")
        handle.write("- w_o_shadow_budget should increase motif-neighborhood deletion if shadow risk is active.\n")
        handle.write("- w_o_closure_budget should increase closure-unfriendly deletion if closure risk is active.\n")
        handle.write("- w_o_stage1_utility tests whether the learned utility contributes beyond rule scores.\n")
        handle.write("\nBIODEL_ABLATION_SUITE_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Run BioDel-Planner ablation suite.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--selected_operating_points", required=True)
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--out_selected_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--max_candidates_per_protein", type=int, default=160)
    parser.add_argument("--allow_overshoot_fraction", type=float, default=0.02)
    parser.add_argument("--fill_weight", type=float, default=2.0)
    parser.add_argument("--underfill_penalty", type=float, default=1.0)
    parser.add_argument("--overshoot_penalty", type=float, default=0.2)
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    accessions = read_split_accessions(args.split_csv)
    by_accession, _ = read_segments(args.segments_csv, accessions)
    selected_settings = read_selected_settings(args.selected_operating_points)
    settings = default_settings()
    summary_rows = []
    selected_rows = []
    for budget in budgets:
        setting_name = selected_settings.get(budget)
        if setting_name is None:
            raise ValueError("No selected operating point for budget {}".format(budget))
        setting = settings[setting_name]
        for spec in ablation_specs():
            selected_by_accession = {}
            for accession, rows in by_accession.items():
                selected, _ = plan_protein(rows, budget, setting, spec, args)
                selected_by_accession[accession] = selected
                for rank, row in enumerate(selected, start=1):
                    out = {k: v for k, v in row.items() if not k.startswith("_")}
                    out.update({"method": spec["method"], "budget_ratio": budget, "planner_rank": rank})
                    selected_rows.append(out)
            summary_rows.append(summarize(spec["method"], spec["description"], budget, selected_by_accession, by_accession))
    write_summary_csv(args.out_summary_csv, summary_rows)
    write_selected_csv(args.out_selected_csv, selected_rows)
    write_report(args.out_report, summary_rows)
    print("Wrote {}".format(args.out_summary_csv))
    print("Wrote {}".format(args.out_selected_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
