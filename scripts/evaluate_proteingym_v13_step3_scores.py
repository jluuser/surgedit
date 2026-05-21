#!/usr/bin/env python3
"""Evaluate Step-3 BioDel and ProteinGym zero-shot baselines on v1.3 indels."""

import argparse
import csv
import math
import os
from collections import defaultdict


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def safe_float(value):
    try:
        if value in ("", None):
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
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
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
    sorted_pairs = sorted(pairs, key=lambda p: p[1])
    ranks = rankdata([score for _, score in sorted_pairs])
    pos_rank_sum = sum(rank for rank, (label, _) in zip(ranks, sorted_pairs) if label == 1)
    auroc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    desc = sorted(pairs, key=lambda p: p[1], reverse=True)
    tp = 0
    precision_sum = 0.0
    for idx, (label, _) in enumerate(desc, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / idx
    return auroc, precision_sum / positives


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


def read_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def score_columns(fields):
    fixed = [
        ("stage1_utility_only", "stage1_utility_score"),
        ("terminal_baseline", "terminal_proximity_score"),
        ("bioprior_risk_only", "bioprior_risk_only_favorable_score"),
        ("full_biodel_score", "full_biodel_score"),
        ("risk_upper_only", "risk_upper_favorable_score"),
        ("risk_certified_biodel_score", "risk_certified_biodel_score"),
        ("compatibility_model_score", "compatibility_score"),
        ("compatibility_model_biodel_score", "compatibility_biodel_score"),
    ]
    cols = [(name, col) for name, col in fixed if col in fields]
    for field in fields:
        if field.startswith("zero_shot_") and field != "zero_shot_scores_found":
            cols.append(("proteingym_zero_shot:{}".format(field.replace("zero_shot_", "")), field))
    return cols


def evaluate_rows(rows, fields, group_col):
    score_defs = score_columns(fields)
    out = []
    labels = [safe_float(row.get("DMS_score_bin")) for row in rows]
    targets = [safe_float(row.get("DMS_score")) for row in rows]
    damage_targets = [-x if x is not None else None for x in targets]
    for score_name, col in score_defs:
        scores = [safe_float(row.get(col)) for row in rows]
        auroc, auprc = binary_metrics(labels, scores)
        out.append({
            "subset": "all_dms_single_segment_deletions",
            "score_name": score_name,
            "score_column": col,
            "n": len(rows),
            "n_scored": sum(1 for value in scores if value is not None),
            "n_groups": len(set(row.get(group_col, "") for row in rows)) if group_col else "",
            "pearson_favorable_DMS": fmt(pearson(scores, targets)),
            "spearman_favorable_DMS": fmt(spearman(scores, targets)),
            "spearman_damage_neg_DMS": fmt(spearman(scores, damage_targets)),
            "auroc_DMS_score_bin_1": fmt(auroc),
            "auprc_DMS_score_bin_1": fmt(auprc),
            "top10_precision_bin_1": fmt(topk_precision(labels, scores, 0.10)),
            "top20_precision_bin_1": fmt(topk_precision(labels, scores, 0.20)),
        })
    return out


def evaluate_per_assay(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["DMS_id"]].append(row)
    out = []
    selected = [
        ("stage1_utility_only", "stage1_utility_score"),
        ("terminal_baseline", "terminal_proximity_score"),
        ("bioprior_risk_only", "bioprior_risk_only_favorable_score"),
        ("full_biodel_score", "full_biodel_score"),
        ("risk_upper_only", "risk_upper_favorable_score"),
        ("risk_certified_biodel_score", "risk_certified_biodel_score"),
        ("compatibility_model_score", "compatibility_score"),
        ("compatibility_model_biodel_score", "compatibility_biodel_score"),
    ]
    selected.extend((name, col) for name, col in score_columns(fields) if name.startswith("proteingym_zero_shot:"))
    for assay, group in grouped.items():
        if len(group) < 10:
            continue
        labels = [safe_float(row.get("DMS_score_bin")) for row in group]
        targets = [safe_float(row.get("DMS_score")) for row in group]
        for score_name, col in selected:
            scores = [safe_float(row.get(col)) for row in group]
            if sum(1 for value in scores if value is not None) < 10:
                continue
            auroc, auprc = binary_metrics(labels, scores)
            out.append({
                "DMS_id": assay,
                "score_name": score_name,
                "score_column": col,
                "n": len(group),
                "n_scored": sum(1 for value in scores if value is not None),
                "spearman_favorable_DMS": fmt(spearman(scores, targets)),
                "auroc_DMS_score_bin_1": fmt(auroc),
                "auprc_DMS_score_bin_1": fmt(auprc),
            })
    return out


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_metric(rows, score_name, metric):
    vals = [safe_float(row.get(metric)) for row in rows if row.get("score_name") == score_name]
    vals = [value for value in vals if value is not None]
    return sum(vals) / len(vals) if vals else None


def write_report(path, rows, metrics, per_assay):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("# ProteinGym v1.3 Indel Step-3 External Validation\n\n")
        handle.write("- DMS single-segment deletion rows: {}\n".format(len(rows)))
        handle.write("- DMS assays: {}\n".format(len(set(row.get("DMS_id", "") for row in rows))))
        handle.write("- Structural BioPrior rows: {}\n".format(sum(1 for row in rows if row.get("structure_feature_status") == "success")))
        handle.write("- Swiss-Prot feature rows: {}\n\n".format(sum(1 for row in rows if "success" in row.get("swissprot_feature_status", ""))))
        handle.write("## Overall\n\n")
        handle.write("| Score | N scored | Spearman vs DMS | AUROC bin=1 | AUPRC bin=1 | Top10 bin=1 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in metrics:
            handle.write("| {score_name} | {n_scored} | {spearman_favorable_DMS} | {auroc_DMS_score_bin_1} | {auprc_DMS_score_bin_1} | {top10_precision_bin_1} |\n".format(**row))
        handle.write("\n## Mean Per-Assay Summary\n\n")
        for score_name in ["stage1_utility_only", "terminal_baseline", "bioprior_risk_only", "full_biodel_score", "risk_upper_only", "risk_certified_biodel_score"]:
            handle.write("- {}: mean Spearman {}; mean AUROC {}; mean AUPRC {}\n".format(
                score_name,
                fmt(mean_metric(per_assay, score_name, "spearman_favorable_DMS")),
                fmt(mean_metric(per_assay, score_name, "auroc_DMS_score_bin_1")),
                fmt(mean_metric(per_assay, score_name, "auprc_DMS_score_bin_1")),
            ))
        zero_names = sorted(set(row["score_name"] for row in per_assay if row["score_name"].startswith("proteingym_zero_shot:")))
        if zero_names:
            best = sorted(
                (
                    (
                        mean_metric(per_assay, name, "spearman_favorable_DMS"),
                        mean_metric(per_assay, name, "auroc_DMS_score_bin_1"),
                        name,
                    )
                    for name in zero_names
                ),
                key=lambda item: (item[0] is not None, item[0] if item[0] is not None else -999),
                reverse=True,
            )[:10]
            handle.write("\nTop zero-shot baselines by mean per-assay Spearman:\n")
            for spearman_value, auroc_value, name in best:
                handle.write("- {}: mean Spearman {}; mean AUROC {}\n".format(name, fmt(spearman_value), fmt(auroc_value)))
        handle.write("\nPROTEINGYM_V13_STEP3_EVALUATION_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProteinGym v1.3 Step-3 scores.")
    parser.add_argument("--input_csv", default="results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_biodel_features.csv")
    parser.add_argument("--out_metrics_csv", default="results/proteingym_v13_indel/proteingym_v13_step3_metrics.csv")
    parser.add_argument("--out_per_assay_csv", default="results/proteingym_v13_indel/proteingym_v13_step3_per_assay_metrics.csv")
    parser.add_argument("--out_report", default="results/proteingym_v13_indel/proteingym_v13_step3_external_validation_report.md")
    args = parser.parse_args()

    rows, fields = read_rows(args.input_csv)
    metrics = evaluate_rows(rows, fields, "DMS_id")
    per_assay = evaluate_per_assay(rows, fields)
    metric_fields = [
        "subset",
        "score_name",
        "score_column",
        "n",
        "n_scored",
        "n_groups",
        "pearson_favorable_DMS",
        "spearman_favorable_DMS",
        "spearman_damage_neg_DMS",
        "auroc_DMS_score_bin_1",
        "auprc_DMS_score_bin_1",
        "top10_precision_bin_1",
        "top20_precision_bin_1",
    ]
    assay_fields = [
        "DMS_id",
        "score_name",
        "score_column",
        "n",
        "n_scored",
        "spearman_favorable_DMS",
        "auroc_DMS_score_bin_1",
        "auprc_DMS_score_bin_1",
    ]
    write_csv(args.out_metrics_csv, metrics, metric_fields)
    write_csv(args.out_per_assay_csv, per_assay, assay_fields)
    write_report(args.out_report, rows, metrics, per_assay)
    print("Wrote {}".format(args.out_metrics_csv))
    print("Wrote {}".format(args.out_per_assay_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
