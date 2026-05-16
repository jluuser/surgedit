#!/usr/bin/env python3
"""Sweep BioDel-Planner risk thresholds to build budget-risk frontiers."""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


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
        for row in reader:
            row["_start"] = to_int(row["seg_start"])
            row["_end"] = to_int(row["seg_end"])
            row["_seg_len"] = to_int(row["seg_len"])
            row["_protein_length"] = to_int(row["protein_length"])
            row["_utility"] = to_float(row.get("stage1_utility_score"))
            row["_shadow"] = to_float(row.get("shadow_overlap_fraction"))
            row["_struct"] = to_float(row.get("structural_core_risk_score"))
            row["_motif_shadow"] = to_float(row.get("motif_shadow_risk_score"))
            row["_protected"] = to_float(row.get("protected_overlap_fraction"))
            row["_hard_reject"] = to_bool(row.get("hard_reject"))
            row["_closure_type"] = row.get("closure_type", "")
            row["_closure_friendly"] = to_bool(row.get("closure_friendly_8A"))
            row["_boundary_dist"] = to_float(row.get("boundary_ca_distance"), default=float("inf"))
            by_accession[row["accession"]].append(row)
    for accession in by_accession:
        by_accession[accession].sort(key=lambda row: (row["_utility"], row["_seg_len"]), reverse=True)
    return by_accession


def closure_allowed(row, closure_mode):
    if row["_closure_type"] == "terminal":
        return True
    if closure_mode == "strict8":
        return row["_closure_friendly"]
    if closure_mode == "any":
        return True
    if closure_mode.startswith("dist"):
        cutoff = float(closure_mode.replace("dist", ""))
        return row["_boundary_dist"] <= cutoff
    raise ValueError("Unsupported closure_mode: {}".format(closure_mode))


def is_closure_unfriendly(row):
    return row["_closure_type"] != "terminal" and not row["_closure_friendly"]


def allowed(row, max_shadow, max_struct, max_motif_shadow, closure_mode):
    if row["_hard_reject"]:
        return False
    if row["_protected"] > 0:
        return False
    if row["_shadow"] > max_shadow:
        return False
    if row["_struct"] > max_struct:
        return False
    if row["_motif_shadow"] > max_motif_shadow:
        return False
    if not closure_allowed(row, closure_mode):
        return False
    return True


def overlaps(row, selected):
    start = row["_start"]
    end = row["_end"]
    for other in selected:
        if not (end < other["_start"] or start > other["_end"]):
            return True
    return False


def select_for_protein(rows, budget, max_shadow, max_struct, max_motif_shadow, closure_mode, allow_overshoot_fraction):
    if not rows:
        return [], 0
    protein_length = rows[0]["_protein_length"]
    target_len = max(1, int(math.floor(protein_length * budget)))
    max_selected_len = max(target_len, int(math.floor(target_len + protein_length * allow_overshoot_fraction)))
    selected = []
    selected_len = 0
    for row in rows:
        if selected_len >= target_len:
            break
        if not allowed(row, max_shadow, max_struct, max_motif_shadow, closure_mode):
            continue
        if selected_len + row["_seg_len"] > max_selected_len:
            continue
        if overlaps(row, selected):
            continue
        selected.append(row)
        selected_len += row["_seg_len"]
    return selected, target_len


def summarize_selection(selected):
    return {
        "selected_segments": len(selected),
        "selected_len": sum(row["_seg_len"] for row in selected),
        "protected_overlap_residues": sum(to_int(row.get("n_protected_overlap")) for row in selected),
        "shadow_overlap_residues": sum(to_int(row.get("n_shadow_overlap")) for row in selected),
        "shadow_overlap_segments": sum(1 for row in selected if to_int(row.get("n_shadow_overlap")) > 0),
        "closure_unfriendly_segments": sum(1 for row in selected if is_closure_unfriendly(row)),
        "hard_reject_segments": sum(1 for row in selected if row["_hard_reject"]),
        "mean_stage1_utility": mean([row["_utility"] for row in selected]),
        "mean_shadow_overlap_fraction": mean([row["_shadow"] for row in selected]),
        "mean_structural_core_risk": mean([row["_struct"] for row in selected]),
        "mean_boundary_ca_distance": mean([row["_boundary_dist"] for row in selected if math.isfinite(row["_boundary_dist"])]),
    }


def run_setting(by_accession, budget, max_shadow, max_struct, max_motif_shadow, closure_mode, allow_overshoot_fraction):
    agg = Counter()
    lists = defaultdict(list)
    for accession, rows in by_accession.items():
        selected, target_len = select_for_protein(
            rows, budget, max_shadow, max_struct, max_motif_shadow, closure_mode, allow_overshoot_fraction
        )
        selected_len = sum(row["_seg_len"] for row in selected)
        fill = selected_len / float(target_len) if target_len else 0.0
        summary = summarize_selection(selected)
        agg["total_proteins"] += 1
        agg["total_target_len"] += target_len
        agg["total_selected_len"] += selected_len
        agg["proteins_fill_ge_0.80"] += 1 if fill >= 0.80 else 0
        agg["proteins_fill_ge_0.95"] += 1 if fill >= 0.95 else 0
        lists["fill_ratio"].append(fill)
        for key, value in summary.items():
            if key.startswith("mean_"):
                lists[key].append(value)
            else:
                agg[key] += value
    return {
        "budget_ratio": budget,
        "max_shadow_overlap": max_shadow,
        "max_structural_core_risk": max_struct,
        "max_motif_shadow_risk": max_motif_shadow,
        "closure_mode": closure_mode,
        "total_proteins": agg["total_proteins"],
        "total_target_len": agg["total_target_len"],
        "total_selected_len": agg["total_selected_len"],
        "mean_fill_ratio": mean(lists["fill_ratio"]),
        "proteins_fill_ge_0.80": agg["proteins_fill_ge_0.80"],
        "proteins_fill_ge_0.95": agg["proteins_fill_ge_0.95"],
        "selected_segments": agg["selected_segments"],
        "protected_overlap_residues": agg["protected_overlap_residues"],
        "shadow_overlap_residues": agg["shadow_overlap_residues"],
        "shadow_overlap_segments": agg["shadow_overlap_segments"],
        "closure_unfriendly_segments": agg["closure_unfriendly_segments"],
        "hard_reject_segments": agg["hard_reject_segments"],
        "mean_stage1_utility": mean(lists["mean_stage1_utility"]),
        "mean_shadow_overlap_fraction": mean(lists["mean_shadow_overlap_fraction"]),
        "mean_structural_core_risk": mean(lists["mean_structural_core_risk"]),
        "mean_boundary_ca_distance": mean(lists["mean_boundary_ca_distance"]),
    }


def dominates(a, b):
    return (
        a["mean_fill_ratio"] >= b["mean_fill_ratio"]
        and a["shadow_overlap_residues"] <= b["shadow_overlap_residues"]
        and a["closure_unfriendly_segments"] <= b["closure_unfriendly_segments"]
        and (
            a["mean_fill_ratio"] > b["mean_fill_ratio"]
            or a["shadow_overlap_residues"] < b["shadow_overlap_residues"]
            or a["closure_unfriendly_segments"] < b["closure_unfriendly_segments"]
        )
    )


def pareto_rows(rows):
    front = []
    for row in rows:
        if row["protected_overlap_residues"] != 0 or row["hard_reject_segments"] != 0:
            continue
        same_budget = [other for other in rows if other["budget_ratio"] == row["budget_ratio"]]
        if any(dominates(other, row) for other in same_budget):
            continue
        front.append(row)
    return sorted(front, key=lambda r: (r["budget_ratio"], -r["mean_fill_ratio"], r["shadow_overlap_residues"], r["closure_unfriendly_segments"]))


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "budget_ratio",
        "max_shadow_overlap",
        "max_structural_core_risk",
        "max_motif_shadow_risk",
        "closure_mode",
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
        "hard_reject_segments",
        "mean_stage1_utility",
        "mean_shadow_overlap_fraction",
        "mean_structural_core_risk",
        "mean_boundary_ca_distance",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def best_by_fill(rows, budget, min_fill):
    candidates = [
        row for row in rows
        if row["budget_ratio"] == budget
        and row["mean_fill_ratio"] >= min_fill
        and row["protected_overlap_residues"] == 0
        and row["hard_reject_segments"] == 0
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda r: (r["shadow_overlap_residues"] + 10 * r["closure_unfriendly_segments"], r["shadow_overlap_residues"], r["closure_unfriendly_segments"], -r["mean_fill_ratio"]),
    )[0]


def write_report(path, args, rows, front):
    ensure_parent(path)
    budgets = parse_float_list(args.budgets)
    with open(path, "w") as handle:
        handle.write("BioDel-Planner frontier sweep report\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("num_settings: {}\n".format(len(rows)))
        handle.write("num_pareto_rows: {}\n".format(len(front)))
        handle.write("shadow_thresholds: {}\n".format(args.shadow_thresholds))
        handle.write("structural_thresholds: {}\n".format(args.structural_thresholds))
        handle.write("motif_shadow_thresholds: {}\n".format(args.motif_shadow_thresholds))
        handle.write("closure_modes: {}\n\n".format(args.closure_modes))
        handle.write("Best protected-safe settings by minimum fill:\n")
        for budget in budgets:
            handle.write("budget {}:\n".format(budget))
            for min_fill in [0.95, 0.90, 0.80, 0.70, 0.50]:
                best = best_by_fill(rows, budget, min_fill)
                if best is None:
                    handle.write("- min_fill {}: no setting\n".format(min_fill))
                else:
                    handle.write(
                        "- min_fill {min_fill}: fill={mean_fill_ratio:.4f}, shadow_res={shadow_overlap_residues}, "
                        "closure_unfriendly={closure_unfriendly_segments}, shadow_thr={max_shadow_overlap}, "
                        "struct_thr={max_structural_core_risk}, motif_thr={max_motif_shadow_risk}, closure={closure_mode}\n".format(
                            min_fill=min_fill, **best
                        )
                    )
        handle.write("\nRepresentative Pareto rows:\n")
        for budget in budgets:
            handle.write("budget {}:\n".format(budget))
            subset = [row for row in front if row["budget_ratio"] == budget]
            for row in subset[:10]:
                handle.write(
                    "- fill={mean_fill_ratio:.4f}, shadow_res={shadow_overlap_residues}, closure_unfriendly={closure_unfriendly_segments}, "
                    "shadow_thr={max_shadow_overlap}, struct_thr={max_structural_core_risk}, motif_thr={max_motif_shadow_risk}, closure={closure_mode}\n".format(
                        **row
                    )
                )
        handle.write("\nInterpretation:\n")
        handle.write("- Strict closure and low shadow thresholds are very safe but under-fill 20/30% budgets.\n")
        handle.write("- Frontier rows quantify the trade-off needed for the next planner version: risk budgets rather than only hard per-segment thresholds.\n")
        handle.write("- Protected overlap remains hard forbidden in all settings.\n")
        handle.write("\nBIODEL_FRONTIER_SWEEP_PASS\n")


def run(args):
    by_accession = read_segments(args.segments_csv)
    budgets = parse_float_list(args.budgets)
    shadow_thresholds = parse_float_list(args.shadow_thresholds)
    structural_thresholds = parse_float_list(args.structural_thresholds)
    motif_shadow_thresholds = parse_float_list(args.motif_shadow_thresholds)
    closure_modes = parse_str_list(args.closure_modes)
    rows = []
    for budget in budgets:
        for max_shadow in shadow_thresholds:
            for max_struct in structural_thresholds:
                for max_motif_shadow in motif_shadow_thresholds:
                    for closure_mode in closure_modes:
                        rows.append(
                            run_setting(
                                by_accession,
                                budget,
                                max_shadow,
                                max_struct,
                                max_motif_shadow,
                                closure_mode,
                                args.allow_overshoot_fraction,
                            )
                        )
    front = pareto_rows(rows)
    write_csv(args.out_csv, rows)
    write_csv(args.out_pareto_csv, front)
    write_report(args.out_report, args, rows, front)
    print("Settings evaluated: {}".format(len(rows)))
    print("Pareto rows: {}".format(len(front)))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_pareto_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep BioDel-Planner risk thresholds.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_pareto_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--shadow_thresholds", default="0.0,0.1,0.25,0.5,1.0")
    parser.add_argument("--structural_thresholds", default="0.5,0.75,1.0,2.0")
    parser.add_argument("--motif_shadow_thresholds", default="0.25,0.5,1.0")
    parser.add_argument("--closure_modes", default="strict8,dist10,dist12,dist16,any")
    parser.add_argument("--allow_overshoot_fraction", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
