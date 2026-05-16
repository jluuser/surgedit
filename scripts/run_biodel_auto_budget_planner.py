#!/usr/bin/env python3
"""BioDel auto-budget planner.

This script reuses the v2 risk-budget beam search, but removes the need for a
user-provided deletion ratio. For each protein it builds a small deletion
frontier over candidate budgets, then selects the best risk-adjusted point for
conservative/default/aggressive use profiles.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_biodel_planner_v2 import (  # noqa: E402
    ensure_parent,
    mean,
    parse_budgets,
    parse_settings,
    plan_protein,
    read_accession_filter,
    read_segments,
    summarize_selected,
)


def safe_float(value, default=0.0):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_profiles(text):
    profiles = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, values = chunk.split(":", 1)
        parts = [float(x.strip()) for x in values.split(",")]
        if len(parts) != 3:
            raise ValueError("Profile must be name:delete_reward,risk_weight,length_penalty")
        profiles.append(
            {
                "auto_profile": name.strip(),
                "delete_reward": parts[0],
                "risk_weight": parts[1],
                "length_penalty": parts[2],
            }
        )
    return profiles


def selected_sums(selected):
    return {
        "utility_sum": sum(row["_utility_sum"] for row in selected),
        "bioprior_sum": sum(row["_final_bioprior"] * row["_seg_len"] for row in selected),
        "shadow_units": sum(row["_shadow_res"] for row in selected),
        "closure_units": sum(row["_closure_units"] for row in selected),
        "struct_units": sum(row["_struct_units"] for row in selected),
        "risk_upper_units": sum(safe_float(row.get("risk_upper"), row["_struct_risk"]) * row["_seg_len"] for row in selected),
        "risk_point_units": sum(safe_float(row.get("risk_point"), row["_struct_risk"]) * row["_seg_len"] for row in selected),
        "confidence_len": sum(safe_float(row.get("evidence_confidence"), 1.0) * row["_seg_len"] for row in selected),
    }


def frontier_row(accession, setting, budget, selected, target_len, risk_budget):
    protein_length = selected[0]["_protein_length"] if selected else None
    if protein_length is None:
        return None
    summary = summarize_selected(selected)
    sums = selected_sums(selected)
    selected_len = summary["selected_len"]
    actual_delete_ratio = selected_len / float(protein_length) if protein_length else 0.0
    fill_ratio = selected_len / float(target_len) if target_len else 0.0
    return {
        "accession": accession,
        "setting": setting["setting"],
        "candidate_budget_ratio": budget,
        "protein_length": protein_length,
        "target_delete_len": target_len,
        "selected_len": selected_len,
        "actual_delete_ratio": actual_delete_ratio,
        "fill_ratio": fill_ratio,
        "selected_segments": summary["selected_segments"],
        "protected_overlap_residues": summary["protected_overlap_residues"],
        "shadow_overlap_residues": summary["shadow_overlap_residues"],
        "closure_unfriendly_len": summary["closure_unfriendly_len"],
        "closure_unfriendly_segments": summary["closure_unfriendly_segments"],
        "structural_risk_units": summary["structural_risk_units"],
        "risk_upper_units": sums["risk_upper_units"],
        "risk_point_units": sums["risk_point_units"],
        "mean_risk_upper": sums["risk_upper_units"] / float(max(1, selected_len)),
        "mean_risk_point": sums["risk_point_units"] / float(max(1, selected_len)),
        "mean_evidence_confidence": sums["confidence_len"] / float(max(1, selected_len)),
        "utility_sum": sums["utility_sum"],
        "bioprior_sum": sums["bioprior_sum"],
        "mean_stage1_utility": summary["mean_stage1_utility"],
        "mean_final_bioprior_score": summary["mean_final_bioprior_score"],
        "mean_shadow_overlap_fraction": summary["mean_shadow_overlap_fraction"],
        "mean_structural_core_risk": summary["mean_structural_core_risk"],
        "shadow_budget_for_protein": risk_budget["shadow"],
        "closure_budget_for_protein": risk_budget["closure"],
        "struct_budget_for_protein": risk_budget["struct"],
        "_selected": selected,
    }


def empty_frontier_row(accession, rows, setting, budget, target_len, risk_budget):
    protein_length = rows[0]["_protein_length"] if rows else 0
    return {
        "accession": accession,
        "setting": setting["setting"],
        "candidate_budget_ratio": budget,
        "protein_length": protein_length,
        "target_delete_len": target_len,
        "selected_len": 0,
        "actual_delete_ratio": 0.0,
        "fill_ratio": 0.0,
        "selected_segments": 0,
        "protected_overlap_residues": 0,
        "shadow_overlap_residues": 0,
        "closure_unfriendly_len": 0,
        "closure_unfriendly_segments": 0,
        "structural_risk_units": 0.0,
        "risk_upper_units": 0.0,
        "risk_point_units": 0.0,
        "mean_risk_upper": 0.0,
        "mean_risk_point": 0.0,
        "mean_evidence_confidence": 0.0,
        "utility_sum": 0.0,
        "bioprior_sum": 0.0,
        "mean_stage1_utility": 0.0,
        "mean_final_bioprior_score": 0.0,
        "mean_shadow_overlap_fraction": 0.0,
        "mean_structural_core_risk": 0.0,
        "shadow_budget_for_protein": risk_budget["shadow"],
        "closure_budget_for_protein": risk_budget["closure"],
        "struct_budget_for_protein": risk_budget["struct"],
        "_selected": [],
    }


def risk_index(row, args):
    protein_length = float(max(1, int(row["protein_length"])))
    if getattr(args, "risk_index_mode", "observed") == "certificate_upper":
        return args.certificate_risk_weight * row.get("risk_upper_units", row["structural_risk_units"]) / protein_length
    if getattr(args, "risk_index_mode", "observed") == "certificate_point":
        return args.certificate_risk_weight * row.get("risk_point_units", row["structural_risk_units"]) / protein_length
    return (
        args.shadow_risk_weight * row["shadow_overlap_residues"] / protein_length
        + args.closure_risk_weight * row["closure_unfriendly_len"] / protein_length
        + args.structural_risk_weight * row["structural_risk_units"] / protein_length
    )


def score_frontier_row(row, profile, args):
    protein_length = float(max(1, int(row["protein_length"])))
    utility_norm = row["utility_sum"] / protein_length
    bioprior_norm = row["bioprior_sum"] / protein_length
    delete_ratio = row["actual_delete_ratio"]
    risk = risk_index(row, args)
    return (
        utility_norm
        + args.auto_bioprior_weight * bioprior_norm
        + profile["delete_reward"] * delete_ratio
        - profile["risk_weight"] * risk
        - profile["length_penalty"] * delete_ratio
    )


def choose_profile_row(frontier_rows, profile, args):
    scored = []
    for row in frontier_rows:
        item = dict(row)
        item["auto_score"] = score_frontier_row(row, profile, args)
        item["auto_risk_index"] = risk_index(row, args)
        scored.append(item)
    return sorted(
        scored,
        key=lambda row: (
            row["auto_score"],
            -row["auto_risk_index"],
            row["actual_delete_ratio"],
        ),
        reverse=True,
    )[0]


def strip_internal(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_frontier(path, rows):
    ensure_parent(path)
    fields = [
        "accession",
        "setting",
        "candidate_budget_ratio",
        "protein_length",
        "target_delete_len",
        "selected_len",
        "actual_delete_ratio",
        "fill_ratio",
        "selected_segments",
        "protected_overlap_residues",
        "shadow_overlap_residues",
        "closure_unfriendly_len",
        "closure_unfriendly_segments",
        "structural_risk_units",
        "risk_upper_units",
        "risk_point_units",
        "mean_risk_upper",
        "mean_risk_point",
        "mean_evidence_confidence",
        "utility_sum",
        "bioprior_sum",
        "mean_stage1_utility",
        "mean_final_bioprior_score",
        "mean_shadow_overlap_fraction",
        "mean_structural_core_risk",
        "shadow_budget_for_protein",
        "closure_budget_for_protein",
        "struct_budget_for_protein",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(strip_internal(row))


def write_selected(path, original_fields, rows):
    ensure_parent(path)
    planner_fields = [
        "planner_version",
        "setting",
        "auto_profile",
        "auto_selected_budget_ratio",
        "auto_actual_delete_ratio",
        "auto_score",
        "auto_risk_index",
        "target_delete_len",
        "selected_total_len_for_protein",
        "planner_rank",
    ]
    fields = planner_fields + original_fields
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_auto(selection_rows, by_accession, setting_name, profile_name):
    agg = Counter()
    lists = defaultdict(list)
    selected_by_accession = {row["accession"]: row for row in selection_rows}
    for accession, rows in sorted(by_accession.items()):
        if not rows:
            continue
        protein_length = rows[0]["_protein_length"]
        row = selected_by_accession.get(accession)
        if row is None:
            continue
        selected_len = int(row["selected_len"])
        selected_segments = int(row["selected_segments"])
        agg["analyzed_proteins"] += 1
        agg["total_protein_length"] += protein_length
        agg["total_selected_len"] += selected_len
        agg["selected_segments"] += selected_segments
        agg["protected_overlap_residues"] += int(row["protected_overlap_residues"])
        agg["shadow_overlap_residues"] += int(row["shadow_overlap_residues"])
        agg["closure_unfriendly_len"] += int(row["closure_unfriendly_len"])
        agg["closure_unfriendly_segments"] += int(row["closure_unfriendly_segments"])
        agg["structural_risk_units_x1000000"] += int(round(float(row["structural_risk_units"]) * 1000000))
        lists["selected_budget_ratio"].append(float(row["candidate_budget_ratio"]))
        lists["actual_delete_ratio"].append(float(row["actual_delete_ratio"]))
        lists["auto_score"].append(float(row["auto_score"]))
        lists["auto_risk_index"].append(float(row["auto_risk_index"]))
        lists["mean_stage1_utility"].append(float(row["mean_stage1_utility"]))
        lists["mean_final_bioprior_score"].append(float(row["mean_final_bioprior_score"]))
        lists["mean_shadow_overlap_fraction"].append(float(row["mean_shadow_overlap_fraction"]))
        lists["mean_structural_core_risk"].append(float(row["mean_structural_core_risk"]))
        lists["mean_risk_upper"].append(float(row.get("mean_risk_upper", 0.0)))
        lists["mean_risk_point"].append(float(row.get("mean_risk_point", 0.0)))
        lists["mean_evidence_confidence"].append(float(row.get("mean_evidence_confidence", 0.0)))
    selected_len = float(agg["total_selected_len"] or 1)
    return {
        "setting": setting_name,
        "auto_profile": profile_name,
        "analyzed_proteins": agg["analyzed_proteins"],
        "total_protein_length": agg["total_protein_length"],
        "total_selected_len": agg["total_selected_len"],
        "global_auto_delete_ratio": agg["total_selected_len"] / float(agg["total_protein_length"] or 1),
        "mean_auto_delete_ratio": mean(lists["actual_delete_ratio"]),
        "mean_selected_budget_ratio": mean(lists["selected_budget_ratio"]),
        "selected_segments": agg["selected_segments"],
        "protected_overlap_residues": agg["protected_overlap_residues"],
        "protected_overlap_rate": agg["protected_overlap_residues"] / selected_len,
        "shadow_overlap_residues": agg["shadow_overlap_residues"],
        "shadow_overlap_rate": agg["shadow_overlap_residues"] / selected_len,
        "closure_unfriendly_len": agg["closure_unfriendly_len"],
        "closure_unfriendly_rate": agg["closure_unfriendly_len"] / selected_len,
        "closure_unfriendly_segments": agg["closure_unfriendly_segments"],
        "structural_risk_units": agg["structural_risk_units_x1000000"] / 1000000.0,
        "structural_risk_per_deleted": (agg["structural_risk_units_x1000000"] / 1000000.0) / selected_len,
        "mean_auto_score": mean(lists["auto_score"]),
        "mean_auto_risk_index": mean(lists["auto_risk_index"]),
        "mean_stage1_utility": mean(lists["mean_stage1_utility"]),
        "mean_final_bioprior_score": mean(lists["mean_final_bioprior_score"]),
        "mean_shadow_overlap_fraction": mean(lists["mean_shadow_overlap_fraction"]),
        "mean_structural_core_risk": mean(lists["mean_structural_core_risk"]),
        "mean_risk_upper": mean(lists["mean_risk_upper"]),
        "mean_risk_point": mean(lists["mean_risk_point"]),
        "mean_evidence_confidence": mean(lists["mean_evidence_confidence"]),
    }


def write_summary(path, rows):
    ensure_parent(path)
    fields = [
        "setting",
        "auto_profile",
        "analyzed_proteins",
        "total_protein_length",
        "total_selected_len",
        "global_auto_delete_ratio",
        "mean_auto_delete_ratio",
        "mean_selected_budget_ratio",
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
        "mean_auto_score",
        "mean_auto_risk_index",
        "mean_stage1_utility",
        "mean_final_bioprior_score",
        "mean_shadow_overlap_fraction",
        "mean_structural_core_risk",
        "mean_risk_upper",
        "mean_risk_point",
        "mean_evidence_confidence",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, summary_rows):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("BioDel auto-budget planner report\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("accession_filter_csv: {}\n".format(args.accession_filter_csv or ""))
        handle.write("candidate_budgets: {}\n".format(args.candidate_budgets))
        handle.write("settings: {}\n".format(args.settings))
        handle.write("auto_profiles: {}\n\n".format(args.auto_profiles))
        handle.write("risk_index_mode: {}\n".format(args.risk_index_mode))
        handle.write("Core table:\n")
        handle.write("setting,profile,auto_delete_ratio,shadow_rate,closure_rate,struct_rate,risk_upper,confidence,segments\n")
        for row in summary_rows:
            handle.write(
                "{setting},{auto_profile},{global_auto_delete_ratio:.4f},{shadow_overlap_rate:.4f},{closure_unfriendly_rate:.4f},{structural_risk_per_deleted:.4f},{mean_risk_upper:.4f},{mean_evidence_confidence:.4f},{selected_segments}\n".format(
                    **row
                )
            )
        handle.write("\nInterpretation:\n")
        handle.write("- Auto-budget scans fixed-budget v2 plans and selects a risk-adjusted point per protein.\n")
        handle.write("- Conservative/default/aggressive profiles differ in deletion reward, risk weight, and length penalty.\n")
        handle.write("- Protected overlap remains forbidden because the underlying v2 planner keeps hard constraints.\n")
        handle.write("\nBIODEL_AUTO_BUDGET_PLANNER_PASS\n")


def run(args):
    settings = parse_settings(args.settings)
    profiles = parse_profiles(args.auto_profiles)
    budgets = parse_budgets(args.candidate_budgets)
    allowed_accessions = read_accession_filter(args.accession_filter_csv, args.accession_column)
    by_accession, fields = read_segments(args.segments_csv, allowed_accessions)
    if args.limit_proteins is not None:
        keep = set(sorted(by_accession)[: args.limit_proteins])
        by_accession = {accession: rows for accession, rows in by_accession.items() if accession in keep}

    frontier_rows = []
    selected_rows = []
    summary_rows = []
    profile_choices = defaultdict(list)

    for setting in settings:
        for accession, rows in sorted(by_accession.items()):
            accession_frontier = []
            for budget in budgets:
                selected, target_len, risk_budget = plan_protein(rows, budget, setting, args)
                if selected:
                    row = frontier_row(accession, setting, budget, selected, target_len, risk_budget)
                else:
                    row = empty_frontier_row(accession, rows, setting, budget, target_len, risk_budget)
                frontier_rows.append(row)
                accession_frontier.append(row)
            for profile in profiles:
                chosen = choose_profile_row(accession_frontier, profile, args)
                chosen["auto_profile"] = profile["auto_profile"]
                profile_choices[(setting["setting"], profile["auto_profile"])].append(chosen)
                for rank, segment in enumerate(chosen["_selected"], start=1):
                    out = {k: v for k, v in segment.items() if not k.startswith("_")}
                    out.update(
                        {
                            "planner_version": "auto_budget_v1",
                            "setting": setting["setting"],
                            "auto_profile": profile["auto_profile"],
                            "auto_selected_budget_ratio": chosen["candidate_budget_ratio"],
                            "auto_actual_delete_ratio": "{:.6f}".format(chosen["actual_delete_ratio"]),
                            "auto_score": "{:.6f}".format(chosen["auto_score"]),
                            "auto_risk_index": "{:.6f}".format(chosen["auto_risk_index"]),
                            "target_delete_len": chosen["target_delete_len"],
                            "selected_total_len_for_protein": chosen["selected_len"],
                            "planner_rank": rank,
                        }
                    )
                    selected_rows.append(out)

    for (setting_name, profile_name), rows in sorted(profile_choices.items()):
        summary_rows.append(summarize_auto(rows, by_accession, setting_name, profile_name))

    write_frontier(args.out_frontier_csv, frontier_rows)
    write_selected(args.out_selected_csv, fields, selected_rows)
    write_summary(args.out_summary_csv, summary_rows)
    write_report(args.out_report, args, summary_rows)
    print("Wrote {}".format(args.out_frontier_csv))
    print("Wrote {}".format(args.out_selected_csv))
    print("Wrote {}".format(args.out_summary_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Run BioDel auto-budget planner.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--accession_filter_csv", default=None)
    parser.add_argument("--accession_column", default="accession")
    parser.add_argument("--out_frontier_csv", required=True)
    parser.add_argument("--out_selected_csv", required=True)
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--candidate_budgets", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument(
        "--settings",
        default="safe:0.0,0.0,0.75;balanced:0.02,0.10,0.75;aggressive:0.05,0.25,1.0",
        help="Semicolon-separated name:shadow_budget_fraction,closure_budget_fraction,structural_budget_fraction.",
    )
    parser.add_argument(
        "--auto_profiles",
        default="conservative:0.25,3.0,0.25;default:0.55,1.5,0.10;aggressive:0.90,0.8,0.05",
        help="Semicolon-separated name:delete_reward,risk_weight,length_penalty.",
    )
    parser.add_argument("--auto_bioprior_weight", type=float, default=0.20)
    parser.add_argument("--shadow_risk_weight", type=float, default=1.0)
    parser.add_argument("--closure_risk_weight", type=float, default=1.0)
    parser.add_argument("--structural_risk_weight", type=float, default=0.25)
    parser.add_argument("--risk_index_mode", choices=["observed", "certificate_point", "certificate_upper"], default="observed")
    parser.add_argument("--certificate_risk_weight", type=float, default=1.0)
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--max_candidates_per_protein", type=int, default=160)
    parser.add_argument("--allow_overshoot_fraction", type=float, default=0.02)
    parser.add_argument("--fill_weight", type=float, default=2.0)
    parser.add_argument("--underfill_penalty", type=float, default=1.0)
    parser.add_argument("--overshoot_penalty", type=float, default=0.2)
    parser.add_argument("--limit_proteins", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
