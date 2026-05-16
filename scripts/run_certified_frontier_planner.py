#!/usr/bin/env python3
"""Run certified Pareto-frontier deletion planning.

This is the AAAI-facing planner entry point.  It separates three steps that were
previously mixed inside heuristic scripts:

1. calibrate evidence-conditional risk upper bounds on a calibration split,
2. build a Pareto frontier of non-overlapping deletion plans per protein,
3. select a certified operating point or abstain.
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
    RiskCalibrator,
    SelectionProfile,
    candidates_from_rows,
    group_rows_by_accession,
    plan_to_summary_row,
)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_accessions(path, column="accession"):
    if not path:
        return None
    accessions = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get(column) or row.get("protein_id")
            if accession:
                accessions.add(accession)
    return accessions


def filter_rows(rows, accessions):
    if not accessions:
        return list(rows)
    return [row for row in rows if row.get("accession") in accessions or row.get("protein_id") in accessions]


def parse_profiles(text):
    profiles = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, values = chunk.split(":", 1)
        parts = [float(item.strip()) for item in values.split(",")]
        if len(parts) not in (3, 5, 8):
            raise ValueError(
                "Profile must be "
                "name:delete_reward,risk_weight,length_penalty"
                "[,confidence_penalty,max_plan_risk"
                "[,max_shadow_rate,max_closure_rate,max_structural_risk_per_deleted]]"
            )
        confidence_penalty = parts[3] if len(parts) >= 4 else 0.5
        max_plan_risk = parts[4] if len(parts) >= 5 else 0.35
        max_shadow_rate = parts[5] if len(parts) >= 8 else 1.0
        max_closure_rate = parts[6] if len(parts) >= 8 else 1.0
        max_structural_risk = parts[7] if len(parts) >= 8 else 1.0
        profiles.append(
            SelectionProfile(
                name=name.strip(),
                delete_reward=parts[0],
                risk_weight=parts[1],
                length_penalty=parts[2],
                confidence_penalty=confidence_penalty,
                max_plan_risk=max_plan_risk,
                max_shadow_rate=max_shadow_rate,
                max_closure_rate=max_closure_rate,
                max_structural_risk_per_deleted=max_structural_risk,
            )
        )
    return profiles


def sort_frontier_rows(frontier):
    return sorted(
        frontier,
        key=lambda state: (
            state.accession,
            state.risk_upper_mean,
            -state.delete_ratio,
            -state.value_norm,
        ),
    )


def frontier_to_rows(frontier, frontier_limit=None):
    rows = []
    ordered = sort_frontier_rows(frontier)
    if frontier_limit is not None:
        ordered = ordered[: int(frontier_limit)]
    for rank, state in enumerate(ordered, start=1):
        selected_len = float(max(1, state.selected_len))
        rows.append(
            {
                "accession": state.accession,
                "frontier_rank": rank,
                "protein_length": state.protein_length,
                "selected_len": state.selected_len,
                "actual_delete_ratio": "{:.6f}".format(state.delete_ratio),
                "selected_segments": state.selected_segments,
                "value_norm": "{:.6f}".format(state.value_norm),
                "mean_risk_upper": "{:.6f}".format(state.risk_upper_mean),
                "mean_risk_point": "{:.6f}".format(state.risk_point_mean),
                "mean_evidence_confidence": "{:.6f}".format(state.evidence_confidence_mean),
                "protected_overlap_residues": state.protected_overlap,
                "shadow_overlap_residues": state.shadow_overlap,
                "shadow_overlap_rate": "{:.6f}".format(state.shadow_overlap / selected_len),
                "closure_unfriendly_len": state.closure_unfriendly_len,
                "closure_unfriendly_rate": "{:.6f}".format(state.closure_unfriendly_len / selected_len),
                "structural_risk_units": "{:.6f}".format(state.structural_risk_units),
                "structural_risk_per_deleted": "{:.6f}".format(state.structural_risk_units / selected_len),
            }
        )
    return rows


def selected_segment_rows(plan, profile_name, status):
    if plan is None:
        return []
    rows = []
    for rank, candidate in enumerate(plan.selected, start=1):
        out = dict(candidate.row)
        out.update(
            {
                "planner_version": "certified_frontier_v1",
                "auto_profile": profile_name,
                "selection_status": status,
                "planner_rank": rank,
                "selected_total_len_for_protein": plan.selected_len,
                "auto_actual_delete_ratio": "{:.6f}".format(plan.delete_ratio),
                "plan_mean_risk_upper": "{:.6f}".format(plan.risk_upper_mean),
                "plan_mean_evidence_confidence": "{:.6f}".format(plan.evidence_confidence_mean),
            }
        )
        rows.append(out)
    return rows


def write_csv(path, rows, preferred_fields=None):
    ensure_parent(path)
    fields = list(preferred_fields or [])
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(selected_rows, profiles):
    by_profile = defaultdict(list)
    for row in selected_rows:
        by_profile[row["auto_profile"]].append(row)
    out = []
    for profile in profiles:
        rows = by_profile.get(profile.name, [])
        analyzed = len(rows)
        accepted = [row for row in rows if row["selection_status"] == "certified"]
        total_length = sum(int(row.get("protein_length", 0) or 0) for row in rows)
        selected_len = sum(int(row.get("selected_len", 0) or 0) for row in accepted)
        selected_segments = sum(int(row.get("selected_segments", 0) or 0) for row in accepted)
        denom = float(max(1, selected_len))
        risks = [float(row.get("mean_risk_upper", 0.0) or 0.0) for row in accepted]
        confidences = [float(row.get("mean_evidence_confidence", 0.0) or 0.0) for row in accepted]
        structural_risk_units = sum(float(row.get("structural_risk_units", 0.0) or 0.0) for row in accepted)
        out.append(
            {
                "auto_profile": profile.name,
                "analyzed_proteins": analyzed,
                "certified_proteins": len(accepted),
                "abstained_proteins": analyzed - len(accepted),
                "abstention_rate": "{:.6f}".format((analyzed - len(accepted)) / float(max(1, analyzed))),
                "total_protein_length": total_length,
                "total_selected_len": selected_len,
                "global_auto_delete_ratio": "{:.6f}".format(selected_len / float(max(1, total_length))),
                "selected_segments": selected_segments,
                "protected_overlap_residues": sum(int(row.get("protected_overlap_residues", 0) or 0) for row in accepted),
                "shadow_overlap_residues": sum(int(row.get("shadow_overlap_residues", 0) or 0) for row in accepted),
                "shadow_overlap_rate": "{:.6f}".format(sum(int(row.get("shadow_overlap_residues", 0) or 0) for row in accepted) / denom),
                "closure_unfriendly_len": sum(int(row.get("closure_unfriendly_len", 0) or 0) for row in accepted),
                "closure_unfriendly_rate": "{:.6f}".format(sum(int(row.get("closure_unfriendly_len", 0) or 0) for row in accepted) / denom),
                "structural_risk_units": "{:.6f}".format(structural_risk_units),
                "structural_risk_per_deleted": "{:.6f}".format(structural_risk_units / denom),
                "mean_risk_upper": "{:.6f}".format(sum(risks) / float(max(1, len(risks)))),
                "mean_evidence_confidence": "{:.6f}".format(sum(confidences) / float(max(1, len(confidences)))),
            }
        )
    return out


def write_report(path, args, calibration_rows, selected_summary, calibrator):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("Certified frontier planner report\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("calibration_filter_csv: {}\n".format(args.calibration_filter_csv or ""))
        handle.write("planning_filter_csv: {}\n".format(args.planning_filter_csv or ""))
        handle.write("calibration_rows: {}\n".format(len(calibration_rows)))
        handle.write("alpha: {}\n".format(args.alpha))
        handle.write("global_calibration_offset: {:.6f}\n".format(calibrator.global_offset))
        handle.write("group_calibration_offsets: {}\n\n".format(calibrator.group_offsets))
        handle.write("Planner parameters:\n")
        handle.write("- max_delete_ratio: {}\n".format(args.max_delete_ratio))
        handle.write("- max_plan_risk: {}\n".format(args.max_plan_risk))
        handle.write("- frontier_size: {}\n".format(args.frontier_size))
        handle.write("- max_candidates: {}\n\n".format(args.max_candidates))
        handle.write("Core table:\n")
        handle.write("profile,certified,abstained,delete_ratio,shadow_rate,closure_rate,structural_risk,risk_upper,confidence\n")
        for row in selected_summary:
            handle.write(
                "{auto_profile},{certified_proteins},{abstained_proteins},{global_auto_delete_ratio},{shadow_overlap_rate},{closure_unfriendly_rate},{structural_risk_per_deleted},{mean_risk_upper},{mean_evidence_confidence}\n".format(
                    **row
                )
            )
        handle.write("\nInterpretation:\n")
        handle.write("- The planner first constructs a non-overlapping Pareto frontier of deletion plans.\n")
        handle.write("- Risk upper bounds are calibrated on a held-out calibration split before planning.\n")
        handle.write("- Selection is allowed to abstain when no frontier point satisfies the certified risk profile.\n")
        handle.write("\nBIODEL_CERTIFIED_FRONTIER_PLANNER_PASS\n")


def run(args):
    rows = read_csv(args.segments_csv)
    calibration_accessions = read_accessions(args.calibration_filter_csv, args.accession_column)
    planning_accessions = read_accessions(args.planning_filter_csv, args.accession_column)
    calibration_rows = filter_rows(rows, calibration_accessions) if calibration_accessions else rows
    planning_rows = filter_rows(rows, planning_accessions)
    if args.limit_proteins is not None:
        allowed = set(sorted({row.get("accession") for row in planning_rows})[: args.limit_proteins])
        planning_rows = [row for row in planning_rows if row.get("accession") in allowed]

    calibrator = RiskCalibrator.from_rows(calibration_rows, alpha=args.alpha)
    planner = CertifiedFrontierPlanner(
        max_delete_ratio=args.max_delete_ratio,
        max_plan_risk=args.max_plan_risk,
        frontier_size=args.frontier_size,
        max_candidates=args.max_candidates,
        risk_bin_width=args.risk_bin_width,
        length_bin_size=args.length_bin_size,
        require_no_protected=not args.allow_protected_overlap,
    )
    profiles = parse_profiles(args.profiles)
    grouped = group_rows_by_accession(planning_rows)

    all_frontier_rows = []
    protein_selection_rows = []
    segment_rows = []
    status_counts = Counter()

    for accession, accession_rows in sorted(grouped.items()):
        candidates = candidates_from_rows(accession_rows, calibrator=calibrator)
        frontier = planner.build_frontier(candidates)
        all_frontier_rows.extend(frontier_to_rows(frontier, args.frontier_rows_per_protein))
        for profile in profiles:
            chosen, status = planner.select(frontier, profile)
            status_counts[status] += 1
            summary = plan_to_summary_row(chosen, profile.name, status)
            if chosen is None:
                protein_length = int(accession_rows[0].get("protein_length") or accession_rows[0].get("length") or 0)
                summary.update(
                    {
                        "accession": accession,
                        "protein_length": protein_length,
                        "selected_len": 0,
                        "actual_delete_ratio": "0.000000",
                        "selected_segments": 0,
                    }
                )
            protein_selection_rows.append(summary)
            segment_rows.extend(selected_segment_rows(chosen, profile.name, status))

    summary_rows = summarize(protein_selection_rows, profiles)

    frontier_fields = [
        "accession", "frontier_rank", "protein_length", "selected_len", "actual_delete_ratio",
        "selected_segments", "value_norm", "mean_risk_upper", "mean_risk_point",
        "mean_evidence_confidence", "protected_overlap_residues", "shadow_overlap_residues",
        "shadow_overlap_rate", "closure_unfriendly_len", "closure_unfriendly_rate",
        "structural_risk_units", "structural_risk_per_deleted",
    ]
    selection_fields = [
        "accession", "auto_profile", "selection_status", "protein_length", "selected_len",
        "actual_delete_ratio", "selected_segments", "value_norm", "mean_risk_upper",
        "mean_risk_point", "mean_evidence_confidence", "protected_overlap_residues",
        "shadow_overlap_residues", "shadow_overlap_rate", "closure_unfriendly_len",
        "closure_unfriendly_rate", "structural_risk_units", "structural_risk_per_deleted",
    ]
    segment_fields = [
        "planner_version", "auto_profile", "selection_status", "planner_rank",
        "selected_total_len_for_protein", "auto_actual_delete_ratio", "plan_mean_risk_upper",
        "plan_mean_evidence_confidence",
    ]
    write_csv(args.out_frontier_csv, all_frontier_rows, frontier_fields)
    write_csv(args.out_protein_selection_csv, protein_selection_rows, selection_fields)
    write_csv(args.out_selected_segments_csv, segment_rows, segment_fields)
    write_csv(args.out_summary_csv, summary_rows)
    write_report(args.out_report, args, calibration_rows, summary_rows, calibrator)
    print("Wrote {}".format(args.out_frontier_csv))
    print("Wrote {}".format(args.out_protein_selection_csv))
    print("Wrote {}".format(args.out_selected_segments_csv))
    print("Wrote {}".format(args.out_summary_csv))
    print("Wrote {}".format(args.out_report))
    print("selection_status_counts={}".format(dict(status_counts)))


def parse_args():
    parser = argparse.ArgumentParser(description="Run certified Pareto-frontier deletion planning.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--calibration_filter_csv", default=None)
    parser.add_argument("--planning_filter_csv", default=None)
    parser.add_argument("--accession_column", default="accession")
    parser.add_argument("--out_frontier_csv", required=True)
    parser.add_argument("--out_protein_selection_csv", required=True)
    parser.add_argument("--out_selected_segments_csv", required=True)
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--max_delete_ratio", type=float, default=0.30)
    parser.add_argument("--max_plan_risk", type=float, default=0.35)
    parser.add_argument("--frontier_size", type=int, default=128)
    parser.add_argument("--max_candidates", type=int, default=220)
    parser.add_argument("--risk_bin_width", type=float, default=0.005)
    parser.add_argument("--length_bin_size", type=int, default=5)
    parser.add_argument("--frontier_rows_per_protein", type=int, default=32)
    parser.add_argument(
        "--profiles",
        default=(
            "conservative:0.25,3.0,0.25,0.75,0.20,0.000,0.000,0.55;"
            "default:0.55,1.5,0.10,0.50,0.35,0.010,0.000,0.60;"
            "aggressive:0.90,0.8,0.05,0.30,0.50,0.030,0.050,0.70"
        ),
        help=(
            "Semicolon-separated "
            "name:delete_reward,risk_weight,length_penalty,confidence_penalty,"
            "max_plan_risk,max_shadow_rate,max_closure_rate,max_structural_risk_per_deleted."
        ),
    )
    parser.add_argument("--allow_protected_overlap", action="store_true")
    parser.add_argument("--limit_proteins", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
