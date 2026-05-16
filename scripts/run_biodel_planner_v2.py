#!/usr/bin/env python3
"""BioDel-Planner v2: global risk-budget beam planner.

v1 used hard per-segment thresholds. v2 keeps functional protection hard, but
allocates motif-shadow, closure, and structural-core risk as protein-level
budgets under each deletion ratio.
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


def parse_settings(text):
    settings = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, values = chunk.split(":", 1)
        parts = [float(x.strip()) for x in values.split(",")]
        if len(parts) != 3:
            raise ValueError("Setting must be name:shadow_frac,closure_frac,struct_frac")
        settings.append(
            {
                "setting": name.strip(),
                "shadow_budget_fraction": parts[0],
                "closure_budget_fraction": parts[1],
                "structural_budget_fraction": parts[2],
            }
        )
    return settings


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


def read_accession_filter(path, accession_column="accession"):
    if not path:
        return None
    allowed = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get(accession_column) or row.get("protein_id")
            if accession:
                allowed.add(accession)
    return allowed


def read_segments(path, allowed_accessions=None):
    by_accession = defaultdict(list)
    allowed = set(allowed_accessions or [])
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            if allowed and row.get("accession") not in allowed:
                continue
            row["_start"] = to_int(row["seg_start"])
            row["_end"] = to_int(row["seg_end"])
            row["_seg_len"] = to_int(row["seg_len"])
            row["_protein_length"] = to_int(row["protein_length"])
            row["_utility_mean"] = to_float(row.get("stage1_utility_score"))
            row["_utility_sum"] = row["_utility_mean"] * row["_seg_len"]
            row["_final_bioprior"] = to_float(row.get("final_bioprior_score"))
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
            by_accession[row["accession"]].append(row)
    for accession in by_accession:
        by_accession[accession].sort(
            key=lambda r: (r["_utility_sum"] + 0.05 * r["_seg_len"] + 0.02 * r["_final_bioprior"], r["_utility_mean"]),
            reverse=True,
        )
    return by_accession, fields


def overlaps(candidate, selected):
    for row in selected:
        if not (candidate["_end"] < row["_start"] or candidate["_start"] > row["_end"]):
            return True
    return False


def hard_allowed(row):
    if row["_hard_reject"]:
        return False
    if row["_protected_overlap_res"] > 0 or row["_protected_overlap_frac"] > 0:
        return False
    return True


def state_score(state, target_len, args):
    fill = state["selected_len"] / float(target_len) if target_len else 0.0
    capped_fill = min(fill, 1.0)
    underfill = max(0.0, 1.0 - fill)
    overshoot = max(0.0, fill - 1.0)
    utility_norm = state["utility_sum"] / float(max(1, target_len))
    return args.fill_weight * capped_fill + utility_norm - args.underfill_penalty * underfill - args.overshoot_penalty * overshoot


def add_candidate(state, row, risk_budget, max_selected_len):
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
    selected = state["selected"] + [row]
    return {
        "selected": selected,
        "selected_len": state["selected_len"] + row["_seg_len"],
        "utility_sum": state["utility_sum"] + row["_utility_sum"],
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


def choose_best(states, target_len, args):
    return sorted(states, key=lambda s: state_score(s, target_len, args), reverse=True)[0]


def plan_protein(rows, budget, setting, args):
    if not rows:
        return [], 0, {"shadow": 0, "closure": 0, "struct": 0.0}
    protein_length = rows[0]["_protein_length"]
    target_len = max(1, int(math.floor(protein_length * budget)))
    max_selected_len = max(target_len, int(math.floor(target_len + protein_length * args.allow_overshoot_fraction)))
    risk_budget = {
        "shadow": int(math.floor(target_len * setting["shadow_budget_fraction"])),
        "closure": int(math.floor(target_len * setting["closure_budget_fraction"])),
        "struct": float(target_len * setting["structural_budget_fraction"]),
    }
    candidates = [row for row in rows if hard_allowed(row)]
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
            added = add_candidate(state, row, risk_budget, max_selected_len)
            if added is not None:
                new_states.append(added)
        states = prune_states(new_states, target_len, args.beam_size, args)
    best = choose_best(states, target_len, args)
    return best["selected"], target_len, risk_budget


def summarize_selected(selected):
    return {
        "selected_segments": len(selected),
        "selected_len": sum(row["_seg_len"] for row in selected),
        "protected_overlap_residues": sum(row["_protected_overlap_res"] for row in selected),
        "shadow_overlap_residues": sum(row["_shadow_res"] for row in selected),
        "shadow_overlap_segments": sum(1 for row in selected if row["_shadow_res"] > 0),
        "closure_unfriendly_segments": sum(1 for row in selected if row["_closure_unfriendly"]),
        "closure_unfriendly_len": sum(row["_closure_units"] for row in selected),
        "structural_risk_units": sum(row["_struct_units"] for row in selected),
        "mean_stage1_utility": mean([row["_utility_mean"] for row in selected]),
        "mean_final_bioprior_score": mean([row["_final_bioprior"] for row in selected]),
        "mean_shadow_overlap_fraction": mean([row["_shadow_frac"] for row in selected]),
        "mean_structural_core_risk": mean([row["_struct_risk"] for row in selected]),
    }


def run(args):
    budgets = parse_budgets(args.budgets)
    settings = parse_settings(args.settings)
    allowed_accessions = read_accession_filter(args.accession_filter_csv, args.accession_column)
    by_accession, fields = read_segments(args.segments_csv, allowed_accessions)
    selected_rows = []
    summary_rows = []

    for setting in settings:
        for budget in budgets:
            agg = Counter()
            metric_lists = defaultdict(list)
            for accession, rows in sorted(by_accession.items()):
                selected, target_len, risk_budget = plan_protein(rows, budget, setting, args)
                summary = summarize_selected(selected)
                selected_len = summary["selected_len"]
                fill_ratio = selected_len / float(target_len) if target_len else 0.0
                agg["total_proteins"] += 1
                agg["total_target_len"] += target_len
                agg["total_selected_len"] += selected_len
                agg["proteins_fill_ge_0.80"] += 1 if fill_ratio >= 0.80 else 0
                agg["proteins_fill_ge_0.95"] += 1 if fill_ratio >= 0.95 else 0
                metric_lists["fill_ratio"].append(fill_ratio)
                for key, value in summary.items():
                    if key.startswith("mean_"):
                        metric_lists[key].append(value)
                    else:
                        agg[key] += value
                for rank, row in enumerate(selected, start=1):
                    out = {k: v for k, v in row.items() if not k.startswith("_")}
                    out.update(
                        {
                            "planner_version": "v2_global_risk_budget",
                            "setting": setting["setting"],
                            "budget_ratio": budget,
                            "target_delete_len": target_len,
                            "selected_total_len_for_protein": selected_len,
                            "fill_ratio_for_protein": "{:.6f}".format(fill_ratio),
                            "shadow_budget_for_protein": risk_budget["shadow"],
                            "closure_budget_for_protein": risk_budget["closure"],
                            "struct_budget_for_protein": "{:.6f}".format(risk_budget["struct"]),
                            "planner_rank": rank,
                        }
                    )
                    selected_rows.append(out)
            summary_rows.append(
                {
                    "setting": setting["setting"],
                    "budget_ratio": budget,
                    "shadow_budget_fraction": setting["shadow_budget_fraction"],
                    "closure_budget_fraction": setting["closure_budget_fraction"],
                    "structural_budget_fraction": setting["structural_budget_fraction"],
                    "total_proteins": agg["total_proteins"],
                    "total_target_len": agg["total_target_len"],
                    "total_selected_len": agg["total_selected_len"],
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
                    "mean_stage1_utility": mean(metric_lists["mean_stage1_utility"]),
                    "mean_final_bioprior_score": mean(metric_lists["mean_final_bioprior_score"]),
                    "mean_shadow_overlap_fraction": mean(metric_lists["mean_shadow_overlap_fraction"]),
                    "mean_structural_core_risk": mean(metric_lists["mean_structural_core_risk"]),
                }
            )
    write_selected(args.out_selected_csv, fields, selected_rows)
    write_summary(args.out_summary_csv, summary_rows)
    write_report(args.out_report, args, summary_rows)
    print("Wrote {}".format(args.out_selected_csv))
    print("Wrote {}".format(args.out_summary_csv))
    print("Wrote {}".format(args.out_report))


def write_selected(path, original_fields, rows):
    ensure_parent(path)
    planner_fields = [
        "planner_version",
        "setting",
        "budget_ratio",
        "target_delete_len",
        "selected_total_len_for_protein",
        "fill_ratio_for_protein",
        "shadow_budget_for_protein",
        "closure_budget_for_protein",
        "struct_budget_for_protein",
        "planner_rank",
    ]
    fields = planner_fields + original_fields
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path, rows):
    ensure_parent(path)
    fields = [
        "setting",
        "budget_ratio",
        "shadow_budget_fraction",
        "closure_budget_fraction",
        "structural_budget_fraction",
        "total_proteins",
        "total_target_len",
        "total_selected_len",
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
        for row in rows:
            writer.writerow(row)


def write_report(path, args, rows):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("BioDel-Planner v2 global risk-budget report\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("accession_filter_csv: {}\n".format(args.accession_filter_csv or ""))
        handle.write("settings: {}\n".format(args.settings))
        handle.write("budgets: {}\n".format(args.budgets))
        handle.write("beam_size: {}\n".format(args.beam_size))
        handle.write("max_candidates_per_protein: {}\n\n".format(args.max_candidates_per_protein))
        handle.write("Core table:\n")
        handle.write("setting,budget,fill,protected_res,shadow_res,closure_unfriendly_len,closure_unfriendly_segments,selected_segments\n")
        for row in rows:
            handle.write(
                "{setting},{budget_ratio},{mean_fill_ratio:.4f},{protected_overlap_residues},{shadow_overlap_residues},{closure_unfriendly_len},{closure_unfriendly_segments},{selected_segments}\n".format(
                    **row
                )
            )
        handle.write("\nInterpretation:\n")
        handle.write("- v2 keeps protected overlap as a hard constraint.\n")
        handle.write("- motif-shadow, closure, and structural risk are allocated as protein-level budgets, not per-segment hard filters.\n")
        handle.write("- Compare fill/risk trade-offs against v1 frontier to decide default risk budgets.\n")
        handle.write("\nBIODEL_PLANNER_V2_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run BioDel-Planner v2 global risk-budget beam search.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--accession_filter_csv", default=None)
    parser.add_argument("--accession_column", default="accession")
    parser.add_argument("--out_selected_csv", required=True)
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument(
        "--settings",
        default="safe:0.0,0.0,0.75;balanced:0.02,0.10,0.75;aggressive:0.05,0.25,1.0",
        help="Semicolon-separated name:shadow_budget_fraction,closure_budget_fraction,structural_budget_fraction.",
    )
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--max_candidates_per_protein", type=int, default=160)
    parser.add_argument("--allow_overshoot_fraction", type=float, default=0.02)
    parser.add_argument("--fill_weight", type=float, default=2.0)
    parser.add_argument("--underfill_penalty", type=float, default=1.0)
    parser.add_argument("--overshoot_penalty", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
