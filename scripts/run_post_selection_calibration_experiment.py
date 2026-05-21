#!/usr/bin/env python3
"""Evaluate post-selection calibration for deletion-frontier planning.

The central AAAI-facing claim is not just that BioDel can build a frontier.  It
is that a user may inspect that frontier and choose an operating point after the
fact, so the risk certificate should remain valid for the whole frontier, not
only for individual segments or one preselected plan.

This script compares four certificates on the same generated frontiers:

1. uncalibrated segment risk score,
2. segment-level conformal offset,
3. single-plan conformal offset,
4. frontier-level simultaneous conformal offset.

The evaluation reports both selected-plan violations and simultaneous frontier
violations on a held-out planning split.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from bioprior.certified_frontier import (  # noqa: E402
    CertifiedFrontierPlanner,
    SegmentCandidate,
    SelectionProfile,
    conformal_quantile,
    risk_proxy_from_row,
)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_accessions(path, column="accession"):
    accessions = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get(column) or row.get("protein_id")
            if accession:
                accessions.add(accession)
    return accessions


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def group_rows(rows, allowed_accessions):
    allowed = set(allowed_accessions)
    grouped = defaultdict(list)
    for row in rows:
        accession = row.get("accession") or row.get("protein_id") or ""
        if accession in allowed:
            grouped[accession].append(row)
    return dict(grouped)


def base_risk_from_row(row, field):
    value = row.get(field)
    if value not in (None, ""):
        return clamp(to_float(value))
    return clamp(risk_proxy_from_row(row))


def candidate_from_row(row, source_index, risk_field):
    start = to_int(row.get("seg_start"))
    end = to_int(row.get("seg_end"))
    length = to_int(row.get("seg_len"), end - start + 1)
    protein_length = to_int(row.get("protein_length"), 0)
    closure_type = row.get("closure_type", "")
    closure_bad = closure_type != "terminal" and not to_bool(row.get("closure_friendly_8A"))
    observed_risk = clamp(risk_proxy_from_row(row))
    base_risk = base_risk_from_row(row, risk_field)
    return SegmentCandidate(
        accession=row.get("accession") or row.get("protein_id") or "",
        start=start,
        end=end,
        length=length,
        protein_length=protein_length,
        utility=to_float(row.get("stage1_utility_score"), 0.0),
        bioprior=to_float(row.get("final_bioprior_score"), 0.0),
        risk_point=observed_risk,
        risk_upper=base_risk,
        evidence_confidence=clamp(to_float(row.get("evidence_confidence"), 1.0)),
        evidence_level=row.get("evidence_level") or "unknown",
        protected_overlap=to_int(row.get("n_protected_overlap"), 0),
        shadow_overlap=to_int(row.get("n_shadow_overlap"), 0),
        closure_unfriendly_len=length if closure_bad else 0,
        structural_risk_units=length * max(0.0, to_float(row.get("structural_core_risk_score"), 0.0)),
        source_index=source_index,
        row=dict(row),
    )


def build_frontiers(grouped_rows, planner, risk_field):
    frontiers = {}
    candidate_counts = {}
    for accession, rows in sorted(grouped_rows.items()):
        candidates = [
            candidate_from_row(row, index, risk_field)
            for index, row in enumerate(rows)
        ]
        candidate_counts[accession] = len(candidates)
        frontier = planner.build_frontier(candidates)
        if frontier:
            frontiers[accession] = frontier
    return frontiers, candidate_counts


def positive_residual(state):
    return max(0.0, state.risk_point_mean - state.risk_upper_mean)


def segment_residual(candidate):
    return max(0.0, candidate.risk_point - candidate.risk_upper)


def calibration_profile(args):
    return SelectionProfile(
        name="calibration_preselected",
        delete_reward=args.profile_delete_reward,
        risk_weight=args.profile_risk_weight,
        length_penalty=args.profile_length_penalty,
        confidence_penalty=args.profile_confidence_penalty,
        max_plan_risk=1.0,
        min_evidence_confidence=0.0,
        max_shadow_rate=1.0,
        max_closure_rate=1.0,
        max_structural_risk_per_deleted=1.0,
    )


def profile_score(state, q, args):
    upper = clamp(state.risk_upper_mean + q)
    uncertainty = 1.0 - state.evidence_confidence_mean
    return (
        state.value_norm
        + args.profile_delete_reward * state.delete_ratio
        - args.profile_risk_weight * upper * state.delete_ratio
        - args.profile_length_penalty * state.delete_ratio
        - args.profile_confidence_penalty * uncertainty * state.delete_ratio
    )


def choose_preselected_plan(frontier, args):
    profile = calibration_profile(args)
    feasible = [
        state for state in frontier
        if state.risk_upper_mean <= profile.max_plan_risk
        and state.evidence_confidence_mean >= profile.min_evidence_confidence
        and state.shadow_rate <= profile.max_shadow_rate
        and state.closure_rate <= profile.max_closure_rate
        and state.structural_risk_per_deleted <= profile.max_structural_risk_per_deleted
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda state: (
            profile_score(state, 0.0, args),
            -state.risk_upper_mean,
            state.delete_ratio,
        ),
    )


def calibrate_offsets(calibration_rows, calibration_frontiers, calibration_candidate_counts, risk_field, args):
    segment_scores = []
    for accession, rows in group_rows(calibration_rows, set(calibration_candidate_counts)).items():
        for index, row in enumerate(rows):
            candidate = candidate_from_row(row, index, risk_field)
            if candidate.protected_overlap > 0:
                continue
            segment_scores.append(segment_residual(candidate))

    plan_scores = []
    frontier_scores = []
    frontier_sizes = []
    for frontier in calibration_frontiers.values():
        frontier_sizes.append(len(frontier))
        chosen = choose_preselected_plan(frontier, args)
        if chosen is not None:
            plan_scores.append(positive_residual(chosen))
        frontier_scores.append(max(positive_residual(state) for state in frontier))

    return {
        "uncalibrated": 0.0,
        "segment_level": conformal_quantile(segment_scores, args.alpha),
        "single_plan": conformal_quantile(plan_scores, args.alpha),
        "frontier_level": conformal_quantile(frontier_scores, args.alpha),
    }, {
        "calibration_segments": len(segment_scores),
        "calibration_plans": len(plan_scores),
        "calibration_frontiers": len(frontier_scores),
        "mean_calibration_frontier_size": sum(frontier_sizes) / float(len(frontier_sizes) or 1),
        "max_calibration_frontier_size": max(frontier_sizes) if frontier_sizes else 0,
    }


def select_plan(frontier, q, rule, risk_cap, args):
    if not frontier:
        return None
    if rule == "max_deleted_certified":
        feasible = [
            state for state in frontier
            if clamp(state.risk_upper_mean + q) <= risk_cap
        ]
        if not feasible:
            return None
        return max(
            feasible,
            key=lambda state: (
                state.delete_ratio,
                state.value_norm,
                -clamp(state.risk_upper_mean + q),
            ),
        )
    if rule == "profile_certified":
        feasible = [
            state for state in frontier
            if clamp(state.risk_upper_mean + q) <= risk_cap
        ]
        if not feasible:
            return None
        return max(
            feasible,
            key=lambda state: (
                profile_score(state, q, args),
                -clamp(state.risk_upper_mean + q),
                state.delete_ratio,
            ),
        )
    if rule == "adversarial_frontier":
        return max(
            frontier,
            key=lambda state: (
                state.risk_point_mean - clamp(state.risk_upper_mean + q),
                state.delete_ratio,
                state.value_norm,
            ),
        )
    raise ValueError("Unknown selection rule {}".format(rule))


def evaluate_method(method, q, test_frontiers, rule, risk_cap, args):
    agg = Counter()
    selected_lengths = []
    protein_lengths = []
    selected_segments = []
    true_risks = []
    base_risks = []
    certified_uppers = []
    margins = []
    frontier_sizes = []
    for _, frontier in sorted(test_frontiers.items()):
        agg["analyzed_proteins"] += 1
        frontier_sizes.append(len(frontier))
        if any(state.risk_point_mean > clamp(state.risk_upper_mean + q) + args.epsilon for state in frontier):
            agg["frontier_simultaneous_violations"] += 1
        selected = select_plan(frontier, q, rule, risk_cap, args)
        if selected is None:
            agg["abstained_proteins"] += 1
            continue
        upper = clamp(selected.risk_upper_mean + q)
        agg["accepted_proteins"] += 1
        agg["selected_len"] += selected.selected_len
        agg["protein_length"] += selected.protein_length
        if selected.risk_point_mean > upper + args.epsilon:
            agg["selected_plan_violations"] += 1
        selected_lengths.append(selected.delete_ratio)
        protein_lengths.append(selected.protein_length)
        selected_segments.append(selected.selected_segments)
        true_risks.append(selected.risk_point_mean)
        base_risks.append(selected.risk_upper_mean)
        certified_uppers.append(upper)
        margins.append(selected.risk_point_mean - upper)

    analyzed = float(max(1, agg["analyzed_proteins"]))
    accepted = float(max(1, agg["accepted_proteins"]))
    return {
        "method": method,
        "selection_rule": rule,
        "selection_risk_cap": "{:.6f}".format(risk_cap),
        "calibration_offset": "{:.6f}".format(q),
        "analyzed_proteins": agg["analyzed_proteins"],
        "accepted_proteins": agg["accepted_proteins"],
        "abstained_proteins": agg["abstained_proteins"],
        "coverage": "{:.6f}".format(agg["accepted_proteins"] / analyzed),
        "selected_plan_violations": agg["selected_plan_violations"],
        "selected_plan_violation_rate": "{:.6f}".format(agg["selected_plan_violations"] / accepted),
        "frontier_simultaneous_violations": agg["frontier_simultaneous_violations"],
        "frontier_simultaneous_violation_rate": "{:.6f}".format(agg["frontier_simultaneous_violations"] / analyzed),
        "global_delete_ratio": "{:.6f}".format(agg["selected_len"] / float(max(1, agg["protein_length"]))),
        "mean_selected_delete_ratio": "{:.6f}".format(sum(selected_lengths) / float(len(selected_lengths) or 1)),
        "mean_selected_segments": "{:.6f}".format(sum(selected_segments) / float(len(selected_segments) or 1)),
        "mean_true_plan_risk": "{:.6f}".format(sum(true_risks) / float(len(true_risks) or 1)),
        "mean_base_plan_risk": "{:.6f}".format(sum(base_risks) / float(len(base_risks) or 1)),
        "mean_certified_upper": "{:.6f}".format(sum(certified_uppers) / float(len(certified_uppers) or 1)),
        "mean_true_minus_upper": "{:.6f}".format(sum(margins) / float(len(margins) or 1)),
        "mean_frontier_size": "{:.6f}".format(sum(frontier_sizes) / float(len(frontier_sizes) or 1)),
    }


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "method",
        "selection_rule",
        "selection_risk_cap",
        "calibration_offset",
        "analyzed_proteins",
        "accepted_proteins",
        "abstained_proteins",
        "coverage",
        "selected_plan_violations",
        "selected_plan_violation_rate",
        "frontier_simultaneous_violations",
        "frontier_simultaneous_violation_rate",
        "global_delete_ratio",
        "mean_selected_delete_ratio",
        "mean_selected_segments",
        "mean_true_plan_risk",
        "mean_base_plan_risk",
        "mean_certified_upper",
        "mean_true_minus_upper",
        "mean_frontier_size",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, offsets, calibration_stats, rows):
    ensure_parent(path)
    by_rule = defaultdict(list)
    for row in rows:
        by_rule[row["selection_rule"]].append(row)
    with open(path, "w") as handle:
        handle.write("Post-selection calibration experiment\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("calibration_filter_csv: {}\n".format(args.calibration_filter_csv))
        handle.write("planning_filter_csv: {}\n".format(args.planning_filter_csv))
        handle.write("risk_field: {}\n".format(args.risk_field))
        handle.write("alpha: {}\n".format(args.alpha))
        handle.write("selection_risk_caps: {}\n\n".format(args.selection_risk_caps or args.selection_risk_cap))
        handle.write("Calibration offsets:\n")
        for name in ["uncalibrated", "segment_level", "single_plan", "frontier_level"]:
            handle.write("- {}: {:.6f}\n".format(name, offsets[name]))
        handle.write("\nCalibration stats:\n")
        for key in sorted(calibration_stats):
            handle.write("- {}: {}\n".format(key, calibration_stats[key]))
        handle.write("\nCore table:\n")
        handle.write("cap,rule,method,coverage,selected_violation,frontier_violation,delete_ratio,true_risk,upper\n")
        for rule in sorted(by_rule):
            for row in by_rule[rule]:
                handle.write(
                    "{selection_risk_cap},{selection_rule},{method},{coverage},{selected_plan_violation_rate},"
                    "{frontier_simultaneous_violation_rate},{global_delete_ratio},"
                    "{mean_true_plan_risk},{mean_certified_upper}\n".format(**row)
                )
        handle.write("\nInterpretation:\n")
        handle.write("- segment_level calibrates individual intervals and ignores the fact that a user sees many plans.\n")
        handle.write("- single_plan calibrates one validation-selected plan per protein and is not a simultaneous guarantee for post-selection over a frontier.\n")
        handle.write("- frontier_level calibrates the maximum residual over each calibration frontier, so a later choice from that frontier is covered by the same certificate.\n")
        handle.write("\nBIODEL_POST_SELECTION_CALIBRATION_PASS\n")


def parse_rules(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def limit_grouped(grouped, limit):
    if limit is None:
        return grouped
    kept = {}
    for accession in sorted(grouped)[: int(limit)]:
        kept[accession] = grouped[accession]
    return kept


def run(args):
    all_rows = read_csv(args.segments_csv)
    calibration_accessions = read_accessions(args.calibration_filter_csv, args.accession_column)
    planning_accessions = read_accessions(args.planning_filter_csv, args.accession_column)
    calibration_rows = [row for row in all_rows if (row.get("accession") or row.get("protein_id")) in calibration_accessions]
    planning_rows = [row for row in all_rows if (row.get("accession") or row.get("protein_id")) in planning_accessions]

    planner = CertifiedFrontierPlanner(
        max_delete_ratio=args.max_delete_ratio,
        max_plan_risk=args.frontier_max_plan_risk,
        frontier_size=args.frontier_size,
        max_candidates=args.max_candidates,
        risk_bin_width=args.risk_bin_width,
        length_bin_size=args.length_bin_size,
        require_no_protected=not args.allow_protected_overlap,
    )
    calibration_grouped = limit_grouped(group_rows(calibration_rows, calibration_accessions), args.limit_calibration_proteins)
    planning_grouped = limit_grouped(group_rows(planning_rows, planning_accessions), args.limit_planning_proteins)
    calibration_frontiers, calibration_candidate_counts = build_frontiers(
        calibration_grouped,
        planner,
        args.risk_field,
    )
    test_frontiers, _ = build_frontiers(
        planning_grouped,
        planner,
        args.risk_field,
    )
    offsets, calibration_stats = calibrate_offsets(
        calibration_rows,
        calibration_frontiers,
        calibration_candidate_counts,
        args.risk_field,
        args,
    )
    rows = []
    risk_caps = parse_float_list(args.selection_risk_caps) if args.selection_risk_caps else [args.selection_risk_cap]
    for risk_cap in risk_caps:
        for rule in parse_rules(args.selection_rules):
            for method in ["uncalibrated", "segment_level", "single_plan", "frontier_level"]:
                rows.append(evaluate_method(method, offsets[method], test_frontiers, rule, risk_cap, args))
    write_csv(args.out_csv, rows)
    write_report(args.out_report, args, offsets, calibration_stats, rows)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Run post-selection calibration experiment.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--calibration_filter_csv", required=True)
    parser.add_argument("--planning_filter_csv", required=True)
    parser.add_argument("--accession_column", default="accession")
    parser.add_argument("--risk_field", default="risk_point")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--selection_risk_cap", type=float, default=0.35)
    parser.add_argument("--selection_risk_caps", default=None)
    parser.add_argument("--frontier_max_plan_risk", type=float, default=0.50)
    parser.add_argument("--selection_rules", default="max_deleted_certified,profile_certified,adversarial_frontier")
    parser.add_argument("--max_delete_ratio", type=float, default=0.30)
    parser.add_argument("--frontier_size", type=int, default=128)
    parser.add_argument("--max_candidates", type=int, default=220)
    parser.add_argument("--risk_bin_width", type=float, default=0.005)
    parser.add_argument("--length_bin_size", type=int, default=5)
    parser.add_argument("--allow_protected_overlap", action="store_true")
    parser.add_argument("--profile_delete_reward", type=float, default=0.55)
    parser.add_argument("--profile_risk_weight", type=float, default=1.5)
    parser.add_argument("--profile_length_penalty", type=float, default=0.10)
    parser.add_argument("--profile_confidence_penalty", type=float, default=0.50)
    parser.add_argument("--limit_calibration_proteins", type=int, default=None)
    parser.add_argument("--limit_planning_proteins", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
