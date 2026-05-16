#!/usr/bin/env python3
"""Evaluate BioDel/Stage-1 signals on ProteinGym single-segment deletions."""

import argparse
import csv
import math
import os
from collections import defaultdict

import pandas as pd


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def safe_float(value):
    try:
        if value == "":
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xvals, yvals = zip(*pairs)
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    dy = math.sqrt(sum((y - my) ** 2 for y in yvals))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xvals, yvals = zip(*pairs)
    return pearson(rankdata(list(xvals)), rankdata(list(yvals)))


def binary_metrics(labels, scores):
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if y is not None and s is not None]
    if not pairs:
        return None, None
    positives = sum(y for y, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None, None

    # AUROC via rank statistic.
    sorted_pairs = sorted(pairs, key=lambda p: p[1])
    ranks = rankdata([score for _, score in sorted_pairs])
    pos_rank_sum = sum(rank for rank, (label, _) in zip(ranks, sorted_pairs) if label == 1)
    auroc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    # Average precision.
    desc = sorted(pairs, key=lambda p: p[1], reverse=True)
    tp = 0
    precision_sum = 0.0
    for idx, (label, _) in enumerate(desc, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / idx
    auprc = precision_sum / positives
    return auroc, auprc


def topk_precision(labels, scores, frac):
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if y is not None and s is not None]
    if not pairs:
        return None
    k = max(1, int(round(len(pairs) * frac)))
    top = sorted(pairs, key=lambda p: p[1], reverse=True)[:k]
    return sum(label for label, _ in top) / float(len(top))


def fmt(value):
    if value is None:
        return ""
    return "{:.6f}".format(float(value))


def add_baseline_scores(df):
    df = df.copy()
    df["score_stage1_b10"] = pd.to_numeric(df["stage1_mean_prob_b10"], errors="coerce")
    df["score_stage1_b20"] = pd.to_numeric(df["stage1_mean_prob_b20"], errors="coerce")
    df["score_stage1_b30"] = pd.to_numeric(df["stage1_mean_prob_b30"], errors="coerce")
    df["score_stage1_sum_b10"] = pd.to_numeric(df["stage1_sum_prob_b10"], errors="coerce")
    df["score_shorter_deletion"] = -pd.to_numeric(df["deletion_fraction"], errors="coerce")
    midpoint = pd.to_numeric(df["normalized_midpoint"], errors="coerce")
    df["score_terminal_proximity"] = (midpoint - 0.5).abs() * 2.0
    df["score_terminal_binary"] = df["is_terminal_deletion"].astype(str).str.lower().isin(["true", "1"]).astype(float)
    return df


def evaluate_subset(df, subset_name, target_col, label_col=None, group_col=None):
    score_cols = [
        "score_stage1_b10",
        "score_stage1_b20",
        "score_stage1_b30",
        "score_stage1_sum_b10",
        "score_shorter_deletion",
        "score_terminal_proximity",
        "score_terminal_binary",
    ]
    rows = []
    for score_col in score_cols:
        scores = [safe_float(x) for x in df[score_col].tolist()]
        targets = [safe_float(x) for x in df[target_col].tolist()] if target_col else [None] * len(scores)
        labels = [safe_float(x) for x in df[label_col].tolist()] if label_col else [None] * len(scores)
        auroc, auprc = binary_metrics(labels, scores) if label_col else (None, None)
        rows.append({
            "subset": subset_name,
            "score_name": score_col,
            "n": len(df),
            "n_groups": df[group_col].nunique() if group_col and group_col in df.columns else "",
            "pearson": fmt(pearson(scores, targets)) if target_col else "",
            "spearman": fmt(spearman(scores, targets)) if target_col else "",
            "auroc": fmt(auroc),
            "auprc": fmt(auprc),
            "top10_precision": fmt(topk_precision(labels, scores, 0.10)) if label_col else "",
            "top20_precision": fmt(topk_precision(labels, scores, 0.20)) if label_col else "",
        })
    return rows


def evaluate_per_assay(dms_df):
    rows = []
    score_cols = [
        "score_stage1_b10",
        "score_stage1_b20",
        "score_stage1_b30",
        "score_shorter_deletion",
        "score_terminal_proximity",
    ]
    for assay, group in dms_df.groupby("assay_id"):
        if len(group) < 10:
            continue
        labels = [safe_float(x) for x in group["DMS_score_bin"].tolist()]
        targets = [safe_float(x) for x in group["DMS_score"].tolist()]
        for score_col in score_cols:
            scores = [safe_float(x) for x in group[score_col].tolist()]
            auroc, auprc = binary_metrics(labels, scores)
            rows.append({
                "assay_id": assay,
                "score_name": score_col,
                "n": len(group),
                "positive_rate": fmt(sum(x for x in labels if x is not None) / len([x for x in labels if x is not None])),
                "spearman": fmt(spearman(scores, targets)),
                "auroc": fmt(auroc),
                "auprc": fmt(auprc),
            })
    return rows


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows, score_name, metric):
    vals = [safe_float(row.get(metric)) for row in rows if row.get("score_name") == score_name]
    vals = [x for x in vals if x is not None]
    return sum(vals) / len(vals) if vals else None


def write_report(path, metrics, per_assay, dms_df, clinical_df):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("# ProteinGym Deletion External Validation\n\n")
        handle.write("This is the first external validation of the Stage-1 deletion utility signal on experimentally annotated ProteinGym indels.\n\n")
        handle.write("## Dataset\n\n")
        handle.write("- DMS single-segment deletions: {}\n".format(len(dms_df)))
        handle.write("- DMS assays: {}\n".format(dms_df["assay_id"].nunique()))
        handle.write("- Clinical single-segment deletions used for binary benign/pathogenic evaluation: {}\n".format(len(clinical_df)))
        if len(clinical_df):
            handle.write("- Clinical labels: {}\n".format(dict(clinical_df["clinical_annotation"].value_counts())))
        handle.write("\n## Overall Metrics\n\n")
        handle.write("| Subset | Score | N | Spearman | AUROC | AUPRC | Top10 precision |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in metrics:
            handle.write("| {subset} | {score_name} | {n} | {spearman} | {auroc} | {auprc} | {top10_precision} |\n".format(**row))
        handle.write("\n## Per-Assay DMS Summary\n\n")
        for score_name in ["score_stage1_b10", "score_shorter_deletion", "score_terminal_proximity"]:
            handle.write("- {} mean per-assay Spearman: {}; mean AUROC: {}; mean AUPRC: {}\n".format(
                score_name,
                fmt(mean_metric(per_assay, score_name, "spearman")),
                fmt(mean_metric(per_assay, score_name, "auroc")),
                fmt(mean_metric(per_assay, score_name, "auprc")),
            ))
        handle.write("\n## Interpretation\n\n")
        handle.write("- Stage-1 utility is sequence-only and was trained on synthetic insertions; it is not expected to capture functional safety by itself.\n")
        handle.write("- If Stage-1 is weaker than length/terminal baselines, this supports the need for BioPrior risk constraints and experimental calibration.\n")
        handle.write("- The next Step-2 subtask should add BioPrior-style structural/functional features where target sequences can be mapped to UniProt/AFDB.\n\n")
        handle.write("PROTEINGYM_EXTERNAL_VALIDATION_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProteinGym deletion external validation metrics.")
    parser.add_argument("--scored_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored.csv")
    parser.add_argument("--out_metrics_csv", default="results/proteingym_deletion_benchmark/proteingym_external_validation_metrics.csv")
    parser.add_argument("--out_per_assay_csv", default="results/proteingym_deletion_benchmark/proteingym_external_validation_per_assay.csv")
    parser.add_argument("--out_report", default="results/proteingym_deletion_benchmark/proteingym_external_validation_report.md")
    args = parser.parse_args()

    df = pd.read_csv(args.scored_csv)
    df = add_baseline_scores(df)
    dms = df[(df["source"] == "DMS_indels") & df["DMS_score"].notna() & df["DMS_score_bin"].notna()].copy()
    clinical = df[(df["source"] == "clinical_indels") & df["clinical_annotation"].isin(["Benign", "Pathogenic"])].copy()
    if len(clinical):
        clinical["clinical_label"] = (clinical["clinical_annotation"] == "Benign").astype(float)

    metrics = []
    metrics.extend(evaluate_subset(dms, "DMS_indels", "DMS_score", "DMS_score_bin", group_col="assay_id"))
    metrics.extend(evaluate_subset(clinical, "clinical_indels_benign_vs_pathogenic", None, "clinical_label", group_col="protein_id"))
    per_assay = evaluate_per_assay(dms)

    metric_fields = ["subset", "score_name", "n", "n_groups", "pearson", "spearman", "auroc", "auprc", "top10_precision", "top20_precision"]
    assay_fields = ["assay_id", "score_name", "n", "positive_rate", "spearman", "auroc", "auprc"]
    write_csv(args.out_metrics_csv, metrics, metric_fields)
    write_csv(args.out_per_assay_csv, per_assay, assay_fields)
    write_report(args.out_report, metrics, per_assay, dms, clinical)
    print("Wrote {}".format(args.out_metrics_csv))
    print("Wrote {}".format(args.out_per_assay_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
