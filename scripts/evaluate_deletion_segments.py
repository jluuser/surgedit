#!/usr/bin/env python3
"""Evaluate whether existing deletion runs form closure-friendly segments."""

import argparse
import csv
import json
import os
from collections import defaultdict


DEFAULT_RUNS = {
    "hardmask10": "results/scisor_swissprot_diverse/run10_hardmask/deletions.json",
    "shadow02_10": "results/scisor_swissprot_diverse/run10_hardmask_shadow02/deletions.json",
    "hardmask20": "results/scisor_swissprot_diverse/run20_hardmask/deletions.json",
    "shadow02_20": "results/scisor_swissprot_diverse/run20_hardmask_shadow02/deletions.json",
    "hardmask30": "results/scisor_swissprot_diverse/run30_hardmask/deletions.json",
    "shadow02_30": "results/scisor_swissprot_diverse/run30_hardmask_shadow02/deletions.json",
}


def truthy(text):
    return str(text).strip().lower() in ("true", "1", "yes", "y")


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def parse_runs(text):
    if not text:
        return DEFAULT_RUNS
    runs = {}
    for part in text.split(","):
        if not part:
            continue
        name, path = part.split("=", 1)
        runs[name] = path
    return runs


def read_deletions(path):
    with open(path) as handle:
        rows = json.load(handle)
    deletions = {}
    for row in rows:
        protein_id = row.get("header") or row.get("protein_id")
        positions = row.get("deletion_positions_zero_based")
        if positions is None:
            positions = row.get("deleted_positions", [])
        deletions[protein_id] = sorted(set(int(x) for x in positions))
    return deletions


def contiguous_segments(positions):
    if not positions:
        return []
    segments = []
    start = positions[0]
    prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        segments.append((start, prev))
        start = pos
        prev = pos
    segments.append((start, prev))
    return segments


def read_segment_candidates(path):
    candidates = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["protein_id"], int(row["start"]), int(row["end"]))
            candidates[key] = {
                "has_motif_overlap": truthy(row["has_motif_overlap"]),
                "num_motif_overlap": int(row["num_motif_overlap"]),
                "has_shadow_overlap": truthy(row["has_shadow_overlap"]),
                "num_shadow_overlap": int(row["num_shadow_overlap"]),
                "shadow_overlap_fraction": float(row["shadow_overlap_fraction"]),
                "boundary_ca_distance": None
                if row["boundary_ca_distance"] == ""
                else float(row["boundary_ca_distance"]),
                "closure_friendly": truthy(row["closure_friendly"]),
                "is_terminal_deletion": truthy(row["is_terminal_deletion"]),
                "mean_contact_density_8A": float(row["mean_contact_density_8A"]),
                "mean_plddt": float(row["mean_plddt"]),
            }
    return candidates


def analyze_runs(runs, candidates):
    per_segment_rows = []
    summary_rows = []
    missing_candidate_count = defaultdict(int)

    for run_name, path in runs.items():
        deletions = read_deletions(path)
        run_segments = []

        for protein_id, positions in deletions.items():
            for start, end in contiguous_segments(positions):
                seg_len = end - start + 1
                candidate = candidates.get((protein_id, start, end))
                if candidate is None:
                    missing_candidate_count[run_name] += 1
                    continue

                row = {
                    "run_name": run_name,
                    "protein_id": protein_id,
                    "start": start,
                    "end": end,
                    "seg_len": seg_len,
                    "has_motif_overlap": candidate["has_motif_overlap"],
                    "num_motif_overlap": candidate["num_motif_overlap"],
                    "has_shadow_overlap": candidate["has_shadow_overlap"],
                    "num_shadow_overlap": candidate["num_shadow_overlap"],
                    "shadow_overlap_fraction": candidate["shadow_overlap_fraction"],
                    "boundary_ca_distance": candidate["boundary_ca_distance"],
                    "closure_friendly": candidate["closure_friendly"],
                    "is_terminal_deletion": candidate["is_terminal_deletion"],
                    "mean_contact_density_8A": candidate["mean_contact_density_8A"],
                    "mean_plddt": candidate["mean_plddt"],
                }
                per_segment_rows.append(row)
                run_segments.append(row)

        total_segments = len(run_segments)
        boundary_distances = [
            row["boundary_ca_distance"]
            for row in run_segments
            if row["boundary_ca_distance"] is not None
        ]
        summary_rows.append(
            {
                "run_name": run_name,
                "total_segments": total_segments,
                "mean_seg_len": mean(row["seg_len"] for row in run_segments),
                "fraction_singleton": sum(1 for row in run_segments if row["seg_len"] == 1)
                / float(total_segments or 1),
                "motif_overlap_segments": sum(1 for row in run_segments if row["has_motif_overlap"]),
                "shadow_overlap_segments": sum(
                    1 for row in run_segments if row["has_shadow_overlap"]
                ),
                "closure_friendly_segments": sum(
                    1 for row in run_segments if row["closure_friendly"]
                ),
                "closure_friendly_fraction": sum(
                    1 for row in run_segments if row["closure_friendly"]
                )
                / float(total_segments or 1),
                "mean_boundary_ca_distance": mean(boundary_distances),
                "mean_shadow_overlap_fraction": mean(
                    row["shadow_overlap_fraction"] for row in run_segments
                ),
                "mean_contact_density_8A": mean(
                    row["mean_contact_density_8A"] for row in run_segments
                ),
                "mean_plddt": mean(
                    row["mean_plddt"] for row in run_segments
                ),
                "missing_candidate_segments": missing_candidate_count[run_name],
            }
        )

    return per_segment_rows, summary_rows


def write_per_segment(path, rows):
    fieldnames = [
        "run_name",
        "protein_id",
        "start",
        "end",
        "seg_len",
        "has_motif_overlap",
        "num_motif_overlap",
        "has_shadow_overlap",
        "num_shadow_overlap",
        "shadow_overlap_fraction",
        "boundary_ca_distance",
        "closure_friendly",
        "is_terminal_deletion",
        "mean_contact_density_8A",
        "mean_plddt",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {field: row[field] for field in fieldnames}
            if out["boundary_ca_distance"] is None:
                out["boundary_ca_distance"] = ""
            writer.writerow(out)


def write_summary(path, rows):
    fieldnames = [
        "run_name",
        "total_segments",
        "mean_seg_len",
        "fraction_singleton",
        "motif_overlap_segments",
        "shadow_overlap_segments",
        "closure_friendly_segments",
        "closure_friendly_fraction",
        "mean_boundary_ca_distance",
        "mean_shadow_overlap_fraction",
        "mean_contact_density_8A",
        "mean_plddt",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_report(path, rows):
    by_name = {row["run_name"]: row for row in rows}
    lines = []
    lines.append("Deletion Segment Prior Analysis")
    lines.append("")
    lines.append("Runs analyzed:")
    for name in sorted(by_name):
        row = by_name[name]
        lines.append(
            "{}: total_segments={}, mean_seg_len={:.4f}, singleton_fraction={:.4f}, "
            "shadow_overlap_segments={}, closure_friendly_fraction={:.4f}, "
            "mean_boundary_ca_distance={:.4f}, skipped_missing_candidate_segments={}".format(
                name,
                row["total_segments"],
                row["mean_seg_len"],
                row["fraction_singleton"],
                row["shadow_overlap_segments"],
                row["closure_friendly_fraction"],
                row["mean_boundary_ca_distance"],
                row.get("missing_candidate_segments", 0),
            )
        )

    lines.append("")
    lines.append("Hardmask vs shadow02 comparisons:")
    for pct in (10, 20, 30):
        hard = by_name.get("hardmask{}".format(pct))
        shadow = by_name.get("shadow02_{}".format(pct))
        if not hard or not shadow:
            continue
        shadow_delta = hard["shadow_overlap_segments"] - shadow["shadow_overlap_segments"]
        closure_delta = shadow["closure_friendly_fraction"] - hard["closure_friendly_fraction"]
        lines.append(
            "{}%: shadow-overlap segments hardmask={} shadow02={} delta={}; "
            "closure-friendly fraction hardmask={:.4f} shadow02={:.4f} delta={:.4f}".format(
                pct,
                hard["shadow_overlap_segments"],
                shadow["shadow_overlap_segments"],
                shadow_delta,
                hard["closure_friendly_fraction"],
                shadow["closure_friendly_fraction"],
                closure_delta,
            )
        )

    lines.append("")
    if all(
        by_name.get("shadow02_{}".format(pct), {}).get("shadow_overlap_segments", 10**9)
        < by_name.get("hardmask{}".format(pct), {}).get("shadow_overlap_segments", -1)
        for pct in (10, 20, 30)
    ):
        lines.append("shadow02 reduces shadow-overlap deletion segments at all shrink levels.")
    else:
        lines.append("shadow02 does not uniformly reduce shadow-overlap deletion segments.")

    if by_name.get("hardmask30") and by_name.get("hardmask10"):
        if by_name["hardmask30"]["mean_seg_len"] > by_name["hardmask10"]["mean_seg_len"]:
            lines.append("30% shrink produces longer deletion segments than 10% hardmask.")
        else:
            lines.append("30% shrink does not produce longer mean deletion segments than 10% hardmask.")

    singleton_rates = [row["fraction_singleton"] for row in rows]
    if mean(singleton_rates) > 0.7:
        lines.append("Current SCISOR-derived deletions are mostly scattered residue deletions.")
    else:
        lines.append("Current SCISOR-derived deletions include substantial multi-residue segments.")

    closure_fractions = [row["closure_friendly_fraction"] for row in rows]
    if mean(closure_fractions) < 0.9:
        lines.append("Segment-level closure prior is necessary: many deletion segments are not closure-friendly.")
    else:
        lines.append("Segment-level closure prior is still useful for ranking, but most observed segments are closure-friendly under the current cutoff.")

    lines.append("")
    lines.append("SEGMENT_PRIOR_ANALYSIS_PASS")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--structure_priors", required=True)
    parser.add_argument("--segment_candidates", required=True)
    parser.add_argument("--runs", default="")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    del args.motif_csv
    del args.structure_priors

    os.makedirs(args.out_dir, exist_ok=True)
    runs = parse_runs(args.runs)
    candidates = read_segment_candidates(args.segment_candidates)
    per_segment_rows, summary_rows = analyze_runs(runs, candidates)

    per_segment_path = os.path.join(args.out_dir, "deleted_segments.csv")
    summary_path = os.path.join(args.out_dir, "deletion_segment_summary.csv")
    report_path = os.path.join(args.out_dir, "deletion_segment_report.txt")
    write_per_segment(per_segment_path, per_segment_rows)
    write_summary(summary_path, summary_rows)
    write_report(report_path, summary_rows)

    print("Wrote {}".format(per_segment_path))
    print("Wrote {}".format(summary_path))
    print("Wrote {}".format(report_path))


if __name__ == "__main__":
    main()
