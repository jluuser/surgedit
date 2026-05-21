#!/usr/bin/env python3
"""Bootstrap ProteinGym v1.3 indel score comparisons across assays."""

import argparse
import csv
import os
import random
from collections import defaultdict


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def fnum(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / float(len(values)) if values else None


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return values[low] * (1.0 - frac) + values[high] * frac


def fmt(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def build_metric_map(rows, metric):
    grouped = defaultdict(dict)
    for row in rows:
        value = fnum(row.get(metric))
        if value is None:
            continue
        grouped[row["DMS_id"]][row["score_name"]] = value
    return grouped


def paired_assays(grouped, method, baseline):
    assays = []
    for assay, scores in grouped.items():
        if method in scores and baseline in scores:
            assays.append(assay)
    return sorted(assays)


def bootstrap_comparison(grouped, metric, method, baseline, n_boot, seed):
    assays = paired_assays(grouped, method, baseline)
    rng = random.Random(seed)
    observed_method = mean(grouped[assay][method] for assay in assays)
    observed_baseline = mean(grouped[assay][baseline] for assay in assays)
    observed_delta = None
    if observed_method is not None and observed_baseline is not None:
        observed_delta = observed_method - observed_baseline
    deltas = []
    for _ in range(n_boot):
        sample = [rng.choice(assays) for _ in assays]
        method_mean = mean(grouped[assay][method] for assay in sample)
        baseline_mean = mean(grouped[assay][baseline] for assay in sample)
        if method_mean is not None and baseline_mean is not None:
            deltas.append(method_mean - baseline_mean)
    p_le_zero = sum(1 for delta in deltas if delta <= 0.0) / float(len(deltas) or 1)
    p_ge_zero = sum(1 for delta in deltas if delta >= 0.0) / float(len(deltas) or 1)
    p_two_sided = min(1.0, 2.0 * min(p_le_zero, p_ge_zero))
    return {
        "metric": metric,
        "method": method,
        "baseline": baseline,
        "n_assays": len(assays),
        "observed_method_mean": fmt(observed_method),
        "observed_baseline_mean": fmt(observed_baseline),
        "observed_delta": fmt(observed_delta),
        "ci95_low": fmt(quantile(deltas, 0.025)),
        "ci95_high": fmt(quantile(deltas, 0.975)),
        "p_bootstrap_delta_le_zero": fmt(p_le_zero),
        "p_bootstrap_two_sided": fmt(p_two_sided),
        "n_boot": n_boot,
    }


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "metric",
        "method",
        "baseline",
        "n_assays",
        "observed_method_mean",
        "observed_baseline_mean",
        "observed_delta",
        "ci95_low",
        "ci95_high",
        "p_bootstrap_delta_le_zero",
        "p_bootstrap_two_sided",
        "n_boot",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, args):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("# ProteinGym v1.3 Indel Bootstrap Score Comparison\n\n")
        handle.write("- Input: `{}`\n".format(args.per_assay_csv))
        handle.write("- Bootstrap samples: {}\n".format(args.n_boot))
        handle.write("- Unit: DMS assay, sampled with replacement\n\n")
        handle.write("| Metric | Method | Baseline | Assays | Delta | 95% CI | P(delta <= 0) |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                "| {metric} | {method} | {baseline} | {n_assays} | {observed_delta} | [{ci95_low}, {ci95_high}] | {p_bootstrap_delta_le_zero} |\n".format(**row)
            )
        handle.write("\nInterpretation:\n")
        handle.write("- Positive deltas mean the BioDel score has higher mean per-assay metric than the baseline.\n")
        handle.write("- This test is assay-level, so large assays do not dominate the interval.\n")
        handle.write("\nPROTEINGYM_V13_BOOTSTRAP_SCORE_COMPARISON_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap ProteinGym v1.3 score comparisons.")
    parser.add_argument("--per_assay_csv", default="results/proteingym_v13_indel/proteingym_v13_step3_per_assay_metrics_finetuned_stage1_certified.csv")
    parser.add_argument("--methods", default="risk_certified_biodel_score,full_biodel_score")
    parser.add_argument("--baselines", default="terminal_baseline,bioprior_risk_only,stage1_utility_only")
    parser.add_argument("--metrics", default="spearman_favorable_DMS,auroc_DMS_score_bin_1,auprc_DMS_score_bin_1")
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out_csv", default="results/proteingym_v13_indel/proteingym_v13_bootstrap_score_comparison.csv")
    parser.add_argument("--out_report", default="results/proteingym_v13_indel/PROTEINGYM_V13_BOOTSTRAP_SCORE_COMPARISON_REPORT.md")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_csv(args.per_assay_csv)
    out = []
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    for metric_index, metric in enumerate(metrics):
        grouped = build_metric_map(rows, metric)
        for method in methods:
            for baseline in baselines:
                out.append(
                    bootstrap_comparison(
                        grouped,
                        metric,
                        method,
                        baseline,
                        args.n_boot,
                        args.seed + 1000 * metric_index + 17 * len(out),
                    )
                )
    write_csv(args.out_csv, out)
    write_report(args.out_report, out, args)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
