#!/usr/bin/env python3
"""Evaluate Swiss-Prot functional-risk features on ProteinGym deletions."""

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
        if value == "":
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def truth(value):
    return str(value).lower() in ("true", "1", "yes")


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
    positives = sum(label for label, _ in pairs)
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


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def feature_scores(row):
    protected_count = safe_float(row.get("protected_overlap_count", ""))
    protected_fraction = safe_float(row.get("protected_overlap_fraction", ""))
    shadow_count = safe_float(row.get("sequence_shadow_count", ""))
    shadow_fraction = safe_float(row.get("sequence_shadow_overlap_fraction", ""))
    protected_bool = 1.0 if truth(row.get("has_protected_overlap", "")) else 0.0
    shadow_bool = 1.0 if truth(row.get("has_sequence_shadow", "")) else 0.0
    functional_any = 1.0 if protected_bool or shadow_bool else 0.0
    functional_weighted = protected_bool * 2.0 + shadow_bool
    return {
        "risk_protected_bool": protected_bool,
        "risk_shadow_bool": shadow_bool,
        "risk_functional_any": functional_any,
        "risk_functional_weighted": functional_weighted,
        "risk_protected_count": protected_count,
        "risk_protected_fraction": protected_fraction,
        "risk_shadow_count": shadow_count,
        "risk_shadow_fraction": shadow_fraction,
    }


def annotated(row):
    return truth(row.get("has_swissprot_features", ""))


def evaluate_subset(rows, subset, label_mode):
    score_names = [
        "risk_protected_bool",
        "risk_shadow_bool",
        "risk_functional_any",
        "risk_functional_weighted",
        "risk_protected_count",
        "risk_protected_fraction",
        "risk_shadow_count",
        "risk_shadow_fraction",
    ]
    out = []
    scored = [(row, feature_scores(row)) for row in rows]
    for score_name in score_names:
        scores = [scores[score_name] for _, scores in scored]
        dms_score = [safe_float(row.get("DMS_score", "")) for row, _ in scored]
        dms_damage = [-x if x is not None else None for x in dms_score]
        dms_bin = [safe_float(row.get("DMS_score_bin", "")) for row, _ in scored]
        dms_bin_inverted = [1.0 - x if x is not None else None for x in dms_bin]
        clinical_pathogenic = [
            1.0 if row.get("clinical_annotation") == "Pathogenic" else 0.0 if row.get("clinical_annotation") == "Benign" else None
            for row, _ in scored
        ]

        if label_mode == "dms":
            auroc_bin, auprc_bin = binary_metrics(dms_bin, scores)
            auroc_inv, auprc_inv = binary_metrics(dms_bin_inverted, scores)
            out.append({
                "subset": subset,
                "score_name": score_name,
                "n": len(rows),
                "n_annotated": sum(1 for row in rows if annotated(row)),
                "spearman_dms_score": fmt(spearman(scores, dms_score)),
                "spearman_dms_damage": fmt(spearman(scores, dms_damage)),
                "auroc_dms_score_bin": fmt(auroc_bin),
                "auprc_dms_score_bin": fmt(auprc_bin),
                "auroc_inverted_dms_score_bin": fmt(auroc_inv),
                "auprc_inverted_dms_score_bin": fmt(auprc_inv),
                "auroc_clinical_pathogenic": "",
                "auprc_clinical_pathogenic": "",
                "top10_clinical_pathogenic": "",
            })
        else:
            auroc_path, auprc_path = binary_metrics(clinical_pathogenic, scores)
            out.append({
                "subset": subset,
                "score_name": score_name,
                "n": len(rows),
                "n_annotated": sum(1 for row in rows if annotated(row)),
                "spearman_dms_score": "",
                "spearman_dms_damage": "",
                "auroc_dms_score_bin": "",
                "auprc_dms_score_bin": "",
                "auroc_inverted_dms_score_bin": "",
                "auprc_inverted_dms_score_bin": "",
                "auroc_clinical_pathogenic": fmt(auroc_path),
                "auprc_clinical_pathogenic": fmt(auprc_path),
                "top10_clinical_pathogenic": fmt(topk_precision(clinical_pathogenic, scores, 0.10)),
            })
    return out


def group_summary(rows):
    groups = defaultdict(lambda: {"rows": 0, "annotated": 0, "protected": 0, "shadow": 0})
    for row in rows:
        key = row.get("source", "")
        groups[key]["rows"] += 1
        if annotated(row):
            groups[key]["annotated"] += 1
        if truth(row.get("has_protected_overlap", "")):
            groups[key]["protected"] += 1
        if truth(row.get("has_sequence_shadow", "")):
            groups[key]["shadow"] += 1
    return groups


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_lookup(metrics, subset, score_name):
    for row in metrics:
        if row["subset"] == subset and row["score_name"] == score_name:
            return row
    return {}


def write_report(path, args, rows, metrics):
    ensure_parent(path)
    dms = [row for row in rows if row.get("source") == "DMS_indels"]
    dms_annotated = [row for row in dms if annotated(row)]
    clinical = [
        row for row in rows
        if row.get("source") == "clinical_indels" and row.get("clinical_annotation") in ("Benign", "Pathogenic")
    ]
    clinical_annotated = [row for row in clinical if annotated(row)]
    groups = group_summary(rows)

    with open(path, "w") as handle:
        handle.write("# ProteinGym Swiss-Prot Functional-Risk Validation\n\n")
        handle.write("Input: `{}`\n\n".format(args.input_csv))
        handle.write("## Coverage\n\n")
        handle.write("| Source | Rows | Swiss-Prot annotated | Protected overlap | Sequence shadow |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for source in sorted(groups):
            item = groups[source]
            handle.write("| {} | {} | {} | {} | {} |\n".format(
                source, item["rows"], item["annotated"], item["protected"], item["shadow"]
            ))

        handle.write("\n## DMS Metrics\n\n")
        handle.write("Higher risk scores are expected to correlate positively with `-DMS_score` if lower DMS scores mean stronger damage. Both `DMS_score_bin` orientations are reported because the bin convention is dataset-defined.\n\n")
        handle.write("| Subset | Score | N | Spearman vs DMS | Spearman vs damage | AUROC bin | AUROC inverted bin |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for subset in ("DMS_all", "DMS_annotated"):
            for score_name in ("risk_protected_bool", "risk_shadow_bool", "risk_functional_any", "risk_functional_weighted"):
                row = metric_lookup(metrics, subset, score_name)
                handle.write("| {subset} | {score_name} | {n} | {spearman_dms_score} | {spearman_dms_damage} | {auroc_dms_score_bin} | {auroc_inverted_dms_score_bin} |\n".format(**row))

        handle.write("\n## Clinical Metrics\n\n")
        handle.write("Clinical AUROC uses Pathogenic as the positive label.\n\n")
        handle.write("| Subset | Score | N | AUROC pathogenic | AUPRC pathogenic | Top10 pathogenic precision |\n")
        handle.write("|---|---|---:|---:|---:|---:|\n")
        for subset in ("Clinical_all", "Clinical_annotated"):
            for score_name in ("risk_protected_bool", "risk_shadow_bool", "risk_functional_any", "risk_functional_weighted"):
                row = metric_lookup(metrics, subset, score_name)
                handle.write("| {subset} | {score_name} | {n} | {auroc_clinical_pathogenic} | {auprc_clinical_pathogenic} | {top10_clinical_pathogenic} |\n".format(**row))

        handle.write("\n## Interpretation\n\n")
        handle.write("- Exact Swiss-Prot annotation covers {} / {} DMS deletion rows and {} / {} clinical benign/pathogenic rows.\n".format(
            len(dms_annotated), len(dms), len(clinical_annotated), len(clinical)
        ))
        handle.write("- Protected-overlap and sequence-window shadow are functional-risk annotations, not learned sequence utility scores.\n")
        handle.write("- This is a stronger external validation setup than Stage-1-only scoring because it directly tests biologically annotated functional sites on ProteinGym deletions.\n")
        handle.write("- Rows without exact Swiss-Prot matches should be treated as missing annotations, not low-risk examples, in conservative analyses.\n\n")
        handle.write("PROTEINGYM_SWISSPROT_FEATURE_VALIDATION_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ProteinGym Swiss-Prot functional-risk annotations.")
    parser.add_argument("--input_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_mapped_features.csv")
    parser.add_argument("--out_metrics_csv", default="results/proteingym_deletion_benchmark/proteingym_swissprot_feature_validation_metrics.csv")
    parser.add_argument("--out_report", default="results/proteingym_deletion_benchmark/proteingym_swissprot_feature_validation_report.md")
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    dms = [row for row in rows if row.get("source") == "DMS_indels"]
    dms_annotated = [row for row in dms if annotated(row)]
    clinical = [
        row for row in rows
        if row.get("source") == "clinical_indels" and row.get("clinical_annotation") in ("Benign", "Pathogenic")
    ]
    clinical_annotated = [row for row in clinical if annotated(row)]

    metrics = []
    metrics.extend(evaluate_subset(dms, "DMS_all", "dms"))
    metrics.extend(evaluate_subset(dms_annotated, "DMS_annotated", "dms"))
    metrics.extend(evaluate_subset(clinical, "Clinical_all", "clinical"))
    metrics.extend(evaluate_subset(clinical_annotated, "Clinical_annotated", "clinical"))

    fields = [
        "subset",
        "score_name",
        "n",
        "n_annotated",
        "spearman_dms_score",
        "spearman_dms_damage",
        "auroc_dms_score_bin",
        "auprc_dms_score_bin",
        "auroc_inverted_dms_score_bin",
        "auprc_inverted_dms_score_bin",
        "auroc_clinical_pathogenic",
        "auprc_clinical_pathogenic",
        "top10_clinical_pathogenic",
    ]
    write_csv(args.out_metrics_csv, metrics, fields)
    write_report(args.out_report, args, rows, metrics)
    print("Wrote {}".format(args.out_metrics_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
