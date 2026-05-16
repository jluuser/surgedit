#!/usr/bin/env python3
"""Diagnose whether Stage-1 utility scores carry useful segment-level signal."""

import argparse
import csv
import math
import os
from collections import defaultdict


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def median(values):
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def summarize_group(rows):
    utility = [to_float(row["stage1_utility_score"]) for row in rows]
    return {
        "n_segments": len(rows),
        "mean_stage1_utility": mean(utility),
        "median_stage1_utility": median(utility),
        "mean_final_bioprior_score": mean([to_float(row["final_bioprior_score"]) for row in rows]),
        "mean_shadow_overlap_fraction": mean([to_float(row["shadow_overlap_fraction"]) for row in rows]),
        "mean_protected_overlap_fraction": mean([to_float(row["protected_overlap_fraction"]) for row in rows]),
        "mean_contact_density_8A": mean([to_float(row["mean_contact_density_8A"]) for row in rows]),
        "mean_pLDDT": mean([to_float(row["mean_pLDDT"]) for row in rows]),
        "closure_friendly_fraction": mean([1.0 if to_bool(row["closure_friendly_8A"]) else 0.0 for row in rows]),
        "hard_reject_fraction": mean([1.0 if to_bool(row["hard_reject"]) else 0.0 for row in rows]),
    }


def write_group_csv(path, grouped):
    ensure_parent(path)
    fields = [
        "group_type",
        "group_value",
        "n_segments",
        "mean_stage1_utility",
        "median_stage1_utility",
        "mean_final_bioprior_score",
        "mean_shadow_overlap_fraction",
        "mean_protected_overlap_fraction",
        "mean_contact_density_8A",
        "mean_pLDDT",
        "closure_friendly_fraction",
        "hard_reject_fraction",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in grouped:
            writer.writerow(row)


def format_summary(row):
    return (
        "{group_type}={group_value}: n={n_segments}, "
        "U_mean={mean_stage1_utility:.4f}, BioPrior={mean_final_bioprior_score:.4f}, "
        "shadow={mean_shadow_overlap_fraction:.4f}, protected={mean_protected_overlap_fraction:.4f}, "
        "closure={closure_friendly_fraction:.3f}, hard_reject={hard_reject_fraction:.3f}"
    ).format(**row)


def analyze(args):
    with open(args.segments_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))

    group_specs = []
    by_source = defaultdict(list)
    by_hard_reject = defaultdict(list)
    by_closure = defaultdict(list)
    by_shadow_bin = defaultdict(list)
    by_protected = defaultdict(list)
    by_status = defaultdict(list)

    for row in rows:
        by_source[row.get("proposal_source", "")].append(row)
        by_hard_reject[str(to_bool(row.get("hard_reject", "")))].append(row)
        closure_value = "closure_friendly" if to_bool(row.get("closure_friendly_8A", "")) else "closure_unfriendly"
        if row.get("closure_type") == "terminal":
            closure_value = "terminal"
        by_closure[closure_value].append(row)
        shadow = to_float(row.get("shadow_overlap_fraction"))
        if shadow == 0:
            shadow_bin = "shadow_0"
        elif shadow < 0.25:
            shadow_bin = "shadow_0_0.25"
        elif shadow < 0.5:
            shadow_bin = "shadow_0.25_0.5"
        else:
            shadow_bin = "shadow_ge_0.5"
        by_shadow_bin[shadow_bin].append(row)
        by_protected["protected_overlap" if to_float(row.get("protected_overlap_fraction")) > 0 else "no_protected_overlap"].append(row)
        by_status[row.get("stage1_scoring_status", "")].append(row)

    for group_type, grouped in [
        ("proposal_source", by_source),
        ("hard_reject", by_hard_reject),
        ("closure_class", by_closure),
        ("shadow_bin", by_shadow_bin),
        ("protected_overlap", by_protected),
        ("stage1_scoring_status", by_status),
    ]:
        for group_value, group_rows in sorted(grouped.items()):
            summary = summarize_group(group_rows)
            summary["group_type"] = group_type
            summary["group_value"] = group_value
            group_specs.append(summary)

    utilities = [to_float(row["stage1_utility_score"]) for row in rows]
    correlations = {
        "corr_stage1_vs_final_bioprior_score": pearson(utilities, [to_float(row["final_bioprior_score"]) for row in rows]),
        "corr_stage1_vs_shadow_overlap_fraction": pearson(utilities, [to_float(row["shadow_overlap_fraction"]) for row in rows]),
        "corr_stage1_vs_protected_overlap_fraction": pearson(utilities, [to_float(row["protected_overlap_fraction"]) for row in rows]),
        "corr_stage1_vs_mean_contact_density_8A": pearson(utilities, [to_float(row["mean_contact_density_8A"]) for row in rows]),
        "corr_stage1_vs_mean_pLDDT": pearson(utilities, [to_float(row["mean_pLDDT"]) for row in rows]),
    }

    write_group_csv(args.out_csv, group_specs)
    ensure_parent(args.out_report)
    with open(args.out_report, "w") as handle:
        handle.write("Stage-1 utility signal diagnostic\n\n")
        handle.write("segments_csv: {}\n".format(args.segments_csv))
        handle.write("total_segments: {}\n".format(len(rows)))
        handle.write("mean_stage1_utility: {:.6f}\n".format(mean(utilities)))
        handle.write("median_stage1_utility: {:.6f}\n".format(median(utilities)))
        handle.write("\nCorrelations:\n")
        for key, value in correlations.items():
            handle.write("- {}: {:.6f}\n".format(key, value))
        handle.write("\nBy proposal_source:\n")
        for row in sorted([r for r in group_specs if r["group_type"] == "proposal_source"], key=lambda r: r["mean_stage1_utility"], reverse=True):
            handle.write("- {}\n".format(format_summary(row)))
        handle.write("\nBy risk bins:\n")
        for group_type in ["protected_overlap", "shadow_bin", "closure_class", "hard_reject"]:
            handle.write("{}:\n".format(group_type))
            for row in sorted([r for r in group_specs if r["group_type"] == group_type], key=lambda r: r["group_value"]):
                handle.write("- {}\n".format(format_summary(row)))
        handle.write("\nInterpretation:\n")
        handle.write("- Stage-1 utility is a sequence redundancy signal learned from synthetic insertions; it is not a safety model.\n")
        handle.write("- If high-risk bins also have high utility, downstream planning must enforce explicit risk constraints.\n")
        handle.write("- Use this report to decide whether U_stage1 is complementary to BioPrior risk features.\n")
        handle.write("\nSTAGE1_UTILITY_DIAGNOSTIC_PASS\n")
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Stage-1 utility signal on segment candidates.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
