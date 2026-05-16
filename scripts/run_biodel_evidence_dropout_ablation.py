#!/usr/bin/env python3
"""Run evidence-dropout ablations for the BioDel auto-budget planner.

The goal is to test whether BioDel can degrade gracefully when annotation or
structure evidence is missing. Each dropout has a naive and an adaptive version.
Naive versions remove evidence but keep the same candidate budget range.
Adaptive versions add uncertainty penalties and cap the maximum auto budget.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_biodel_auto_budget_planner import (  # noqa: E402
    choose_profile_row,
    empty_frontier_row,
    frontier_row,
    parse_profiles,
    risk_index,
    score_frontier_row,
)
from run_biodel_planner_v2 import (  # noqa: E402
    ensure_parent,
    mean,
    parse_budgets,
    parse_settings,
    plan_protein,
    read_accession_filter,
    read_segments,
)


def mode_specs():
    return [
        {
            "evidence_mode": "full_evidence",
            "description": "all evidence available",
            "drop_annotation": False,
            "drop_structure": False,
            "adaptive": False,
            "confidence": 1.00,
            "utility_scale": 1.00,
            "max_budget": None,
        },
        {
            "evidence_mode": "no_annotation_naive",
            "description": "annotation removed without adaptive uncertainty",
            "drop_annotation": True,
            "drop_structure": False,
            "adaptive": False,
            "confidence": 1.00,
            "utility_scale": 1.00,
            "max_budget": None,
        },
        {
            "evidence_mode": "no_annotation_adaptive",
            "description": "annotation removed with uncertainty-aware conservative cap",
            "drop_annotation": True,
            "drop_structure": False,
            "adaptive": True,
            "confidence": 0.75,
            "utility_scale": 0.85,
            "max_budget": 0.20,
        },
        {
            "evidence_mode": "no_structure_naive",
            "description": "structure removed without adaptive uncertainty",
            "drop_annotation": False,
            "drop_structure": True,
            "adaptive": False,
            "confidence": 1.00,
            "utility_scale": 1.00,
            "max_budget": None,
        },
        {
            "evidence_mode": "no_structure_adaptive",
            "description": "structure removed with uncertainty-aware conservative cap",
            "drop_annotation": False,
            "drop_structure": True,
            "adaptive": True,
            "confidence": 0.65,
            "utility_scale": 0.75,
            "max_budget": 0.15,
        },
        {
            "evidence_mode": "sequence_only_naive",
            "description": "annotation and structure removed without adaptive uncertainty",
            "drop_annotation": True,
            "drop_structure": True,
            "adaptive": False,
            "confidence": 1.00,
            "utility_scale": 1.00,
            "max_budget": None,
        },
        {
            "evidence_mode": "sequence_only_adaptive",
            "description": "sequence-only planning with conservative uncertainty cap",
            "drop_annotation": True,
            "drop_structure": True,
            "adaptive": True,
            "confidence": 0.45,
            "utility_scale": 0.60,
            "max_budget": 0.10,
        },
    ]


def parse_modes(text):
    requested = [item.strip() for item in text.split(",") if item.strip()]
    specs = {spec["evidence_mode"]: spec for spec in mode_specs()}
    unknown = [mode for mode in requested if mode not in specs]
    if unknown:
        raise ValueError("Unknown evidence modes: {}".format(",".join(unknown)))
    return [specs[mode] for mode in requested]


def transform_rows(rows, spec):
    out = []
    for row in rows:
        item = dict(row)
        item["_evidence_mode"] = spec["evidence_mode"]
        item["_evidence_confidence"] = float(spec["confidence"])
        item["_adaptive_evidence"] = bool(spec["adaptive"])
        item["_uncertainty_penalty"] = 1.0 - float(spec["confidence"])
        item["_utility_mean"] = item["_utility_mean"] * float(spec["utility_scale"])
        item["_utility_sum"] = item["_utility_mean"] * item["_seg_len"]
        item["_final_bioprior"] = item["_final_bioprior"] * float(spec["confidence"])
        if spec["drop_annotation"]:
            item["_protected_overlap_res"] = 0
            item["_protected_overlap_frac"] = 0.0
            item["_shadow_res"] = 0
            item["_shadow_frac"] = 0.0
            item["_hard_reject"] = False
        if spec["drop_structure"]:
            item["_struct_risk"] = 0.0
            item["_struct_units"] = 0.0
            item["_closure_unfriendly"] = False
            item["_closure_units"] = 0
            item["_closure_friendly"] = True
        out.append(item)
    return out


def budgets_for_spec(all_budgets, spec):
    if spec["max_budget"] is None:
        return list(all_budgets)
    return [budget for budget in all_budgets if budget <= float(spec["max_budget"])]


def selected_summary(row, profile, args):
    out = dict(row)
    out["auto_score"] = score_frontier_row(row, profile, args)
    out["auto_risk_index"] = risk_index(row, args)
    return out


def write_selected(path, original_fields, rows):
    ensure_parent(path)
    fields = [
        "planner_version",
        "evidence_mode",
        "adaptive_evidence",
        "evidence_confidence",
        "setting",
        "auto_profile",
        "auto_selected_budget_ratio",
        "auto_actual_delete_ratio",
        "auto_score",
        "auto_risk_index",
        "target_delete_len",
        "selected_total_len_for_protein",
        "planner_rank",
    ] + original_fields
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_frontier(path, rows):
    ensure_parent(path)
    fields = [
        "evidence_mode",
        "adaptive_evidence",
        "evidence_confidence",
        "setting",
        "auto_profile",
        "accession",
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
        "utility_sum",
        "bioprior_sum",
        "mean_stage1_utility",
        "mean_final_bioprior_score",
        "mean_shadow_overlap_fraction",
        "mean_structural_core_risk",
        "auto_score",
        "auto_risk_index",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if not key.startswith("_")})


def summarize_choice(rows, by_accession, spec, setting_name, profile_name):
    agg = Counter()
    lists = defaultdict(list)
    by_acc = {row["accession"]: row for row in rows}
    for accession, source_rows in sorted(by_accession.items()):
        if not source_rows or accession not in by_acc:
            continue
        row = by_acc[accession]
        protein_length = source_rows[0]["_protein_length"]
        selected_len = int(row["selected_len"])
        agg["analyzed_proteins"] += 1
        agg["total_protein_length"] += protein_length
        agg["total_selected_len"] += selected_len
        agg["selected_segments"] += int(row["selected_segments"])
        agg["protected_overlap_residues"] += int(row["protected_overlap_residues"])
        agg["shadow_overlap_residues"] += int(row["shadow_overlap_residues"])
        agg["closure_unfriendly_len"] += int(row["closure_unfriendly_len"])
        agg["closure_unfriendly_segments"] += int(row["closure_unfriendly_segments"])
        agg["structural_risk_units_x1000000"] += int(round(float(row["structural_risk_units"]) * 1000000))
        lists["candidate_budget_ratio"].append(float(row["candidate_budget_ratio"]))
        lists["actual_delete_ratio"].append(float(row["actual_delete_ratio"]))
        lists["auto_score"].append(float(row["auto_score"]))
        lists["auto_risk_index"].append(float(row["auto_risk_index"]))
        lists["stage1"].append(float(row["mean_stage1_utility"]))
        lists["bioprior"].append(float(row["mean_final_bioprior_score"]))
        lists["shadow_frac"].append(float(row["mean_shadow_overlap_fraction"]))
        lists["struct_risk"].append(float(row["mean_structural_core_risk"]))
    deleted = float(agg["total_selected_len"] or 1)
    return {
        "evidence_mode": spec["evidence_mode"],
        "description": spec["description"],
        "adaptive_evidence": int(bool(spec["adaptive"])),
        "evidence_confidence": spec["confidence"],
        "max_budget": "" if spec["max_budget"] is None else spec["max_budget"],
        "setting": setting_name,
        "auto_profile": profile_name,
        "analyzed_proteins": agg["analyzed_proteins"],
        "total_protein_length": agg["total_protein_length"],
        "total_selected_len": agg["total_selected_len"],
        "global_auto_delete_ratio": agg["total_selected_len"] / float(agg["total_protein_length"] or 1),
        "mean_auto_delete_ratio": mean(lists["actual_delete_ratio"]),
        "mean_selected_budget_ratio": mean(lists["candidate_budget_ratio"]),
        "selected_segments": agg["selected_segments"],
        "protected_overlap_residues": agg["protected_overlap_residues"],
        "protected_overlap_rate": agg["protected_overlap_residues"] / deleted,
        "shadow_overlap_residues": agg["shadow_overlap_residues"],
        "shadow_overlap_rate": agg["shadow_overlap_residues"] / deleted,
        "closure_unfriendly_len": agg["closure_unfriendly_len"],
        "closure_unfriendly_rate": agg["closure_unfriendly_len"] / deleted,
        "closure_unfriendly_segments": agg["closure_unfriendly_segments"],
        "structural_risk_units": agg["structural_risk_units_x1000000"] / 1000000.0,
        "structural_risk_per_deleted": (agg["structural_risk_units_x1000000"] / 1000000.0) / deleted,
        "mean_auto_score": mean(lists["auto_score"]),
        "mean_auto_risk_index": mean(lists["auto_risk_index"]),
        "mean_stage1_utility": mean(lists["stage1"]),
        "mean_final_bioprior_score": mean(lists["bioprior"]),
        "mean_shadow_overlap_fraction": mean(lists["shadow_frac"]),
        "mean_structural_core_risk": mean(lists["struct_risk"]),
    }


def write_summary(path, rows):
    ensure_parent(path)
    fields = [
        "evidence_mode",
        "description",
        "adaptive_evidence",
        "evidence_confidence",
        "max_budget",
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
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, rows):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("BioDel evidence dropout ablation report\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("accession_filter_csv: {}\n".format(args.accession_filter_csv or ""))
        handle.write("candidate_budgets: {}\n".format(args.candidate_budgets))
        handle.write("settings: {}\n".format(args.settings))
        handle.write("auto_profiles: {}\n\n".format(args.auto_profiles))
        handle.write("Core table:\n")
        handle.write("mode,adaptive,delete_ratio,shadow_rate,closure_rate,struct_rate,confidence,max_budget\n")
        for row in rows:
            handle.write(
                "{evidence_mode},{adaptive_evidence},{global_auto_delete_ratio:.4f},{shadow_overlap_rate:.4f},{closure_unfriendly_rate:.4f},{structural_risk_per_deleted:.4f},{evidence_confidence},{max_budget}\n".format(
                    **row
                )
            )
        handle.write("\nInterpretation:\n")
        handle.write("- Naive dropout removes evidence without reducing deletion ambition.\n")
        handle.write("- Adaptive dropout adds confidence penalties and budget caps so missing evidence leads to conservative plans.\n")
        handle.write("- This tests whether BioDel can remain usable for proteins with incomplete annotation or structure evidence.\n")
        handle.write("\nBIODEL_EVIDENCE_DROPOUT_ABLATION_PASS\n")


def run(args):
    all_budgets = parse_budgets(args.candidate_budgets)
    settings = parse_settings(args.settings)
    profiles = parse_profiles(args.auto_profiles)
    modes = parse_modes(args.evidence_modes)
    allowed = read_accession_filter(args.accession_filter_csv, args.accession_column)
    by_accession, fields = read_segments(args.segments_csv, allowed)
    if args.limit_proteins is not None:
        keep = set(sorted(by_accession)[: args.limit_proteins])
        by_accession = {accession: rows for accession, rows in by_accession.items() if accession in keep}

    summary_rows = []
    selected_rows = []
    frontier_rows = []

    for spec in modes:
        budgets = budgets_for_spec(all_budgets, spec)
        for setting in settings:
            for profile in profiles:
                choices = []
                for accession, rows in sorted(by_accession.items()):
                    transformed = transform_rows(rows, spec)
                    accession_frontier = []
                    for budget in budgets:
                        selected, target_len, risk_budget = plan_protein(transformed, budget, setting, args)
                        if selected:
                            row = frontier_row(accession, setting, budget, selected, target_len, risk_budget)
                        else:
                            row = empty_frontier_row(accession, transformed, setting, budget, target_len, risk_budget)
                        row["evidence_mode"] = spec["evidence_mode"]
                        row["adaptive_evidence"] = int(bool(spec["adaptive"]))
                        row["evidence_confidence"] = spec["confidence"]
                        row["auto_profile"] = profile["auto_profile"]
                        scored = selected_summary(row, profile, args)
                        frontier_rows.append(scored)
                        accession_frontier.append(row)
                    chosen = choose_profile_row(accession_frontier, profile, args)
                    chosen["evidence_mode"] = spec["evidence_mode"]
                    chosen["adaptive_evidence"] = int(bool(spec["adaptive"]))
                    chosen["evidence_confidence"] = spec["confidence"]
                    chosen["auto_profile"] = profile["auto_profile"]
                    choices.append(chosen)
                    for rank, segment in enumerate(chosen["_selected"], start=1):
                        out = {key: value for key, value in segment.items() if not key.startswith("_")}
                        out.update(
                            {
                                "planner_version": "evidence_dropout_auto_budget_v1",
                                "evidence_mode": spec["evidence_mode"],
                                "adaptive_evidence": int(bool(spec["adaptive"])),
                                "evidence_confidence": spec["confidence"],
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
                summary_rows.append(summarize_choice(choices, by_accession, spec, setting["setting"], profile["auto_profile"]))

    write_summary(args.out_summary_csv, summary_rows)
    write_selected(args.out_selected_csv, fields, selected_rows)
    write_frontier(args.out_frontier_csv, frontier_rows)
    write_report(args.out_report, args, summary_rows)
    print("Wrote {}".format(args.out_summary_csv))
    print("Wrote {}".format(args.out_selected_csv))
    print("Wrote {}".format(args.out_frontier_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Run BioDel evidence dropout ablation.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--accession_filter_csv", default=None)
    parser.add_argument("--accession_column", default="accession")
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--out_selected_csv", required=True)
    parser.add_argument("--out_frontier_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--candidate_budgets", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument(
        "--evidence_modes",
        default="full_evidence,no_annotation_naive,no_annotation_adaptive,no_structure_naive,no_structure_adaptive,sequence_only_naive,sequence_only_adaptive",
    )
    parser.add_argument("--settings", default="safe:0.0,0.0,0.75")
    parser.add_argument("--auto_profiles", default="default:0.55,1.5,0.10")
    parser.add_argument("--auto_bioprior_weight", type=float, default=0.20)
    parser.add_argument("--shadow_risk_weight", type=float, default=1.0)
    parser.add_argument("--closure_risk_weight", type=float, default=1.0)
    parser.add_argument("--structural_risk_weight", type=float, default=0.25)
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
