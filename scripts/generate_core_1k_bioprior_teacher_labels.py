#!/usr/bin/env python3
"""Generate rule-based BioPrior teacher labels and pairwise ranking pairs."""

import argparse
import csv
import math
import os
import random
from collections import Counter, defaultdict


BUDGETS = [0.10, 0.20, 0.30]
POSITIVE_SOURCES = {
    "terminal_tail_prior": 0.9,
    "surface_flexible_loop_prior": 1.0,
    "disorder_like_prior": 0.7,
    "linker_compressibility_prior": 0.8,
}
NEGATIVE_SOURCES = {
    "functional_core_protection_prior",
    "structural_core_protection_prior",
    "motif_neighborhood_support_prior",
    "geometric_closure_prior",
}
PRIOR_SCORE_COLUMNS = [
    "terminal_tail_score",
    "surface_flexible_loop_score",
    "disorder_like_score",
    "linker_compressibility_score",
    "functional_core_risk_score",
    "structural_core_risk_score",
    "motif_shadow_risk_score",
    "geometric_closure_score",
]


def f(row, key, default=0.0):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def i(row, key, default=0):
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def read_core_lengths(path):
    lengths = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            accession = row.get("accession") or row.get("protein_id")
            lengths[accession] = int(row["length"])
    return lengths


def read_segments(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            row["seg_start_int"] = i(row, "seg_start")
            row["seg_end_int"] = i(row, "seg_end")
            row["seg_len_int"] = i(row, "seg_len")
            rows.append(row)
    return rows


def classify_negative_type(row):
    if f(row, "protected_overlap_fraction") > 0 or f(row, "functional_core_risk_score") > 0:
        return "protected_overlap_negative"
    if f(row, "motif_shadow_risk_score") >= 0.5 or f(row, "shadow_overlap_fraction") >= 0.5:
        return "motif_shadow_negative"
    if f(row, "structural_core_risk_score") >= 0.5:
        return "structural_core_negative"
    if row.get("closure_type") == "internal_unfriendly" or row.get("closure_friendly_8A") == "False":
        return "closure_unfriendly_negative"
    return "random_unselected_negative"


def compute_teacher_score(row):
    reject_reasons = []
    hard_reject = truthy(row.get("hard_reject", "False"))
    if hard_reject:
        reject_reasons.append("hard_reject")
    if f(row, "protected_overlap_fraction") > 0:
        reject_reasons.append("protected_overlap")
    if f(row, "functional_core_risk_score") > 0:
        reject_reasons.append("functional_core_risk")
    if row.get("closure_type") == "internal_unfriendly":
        reject_reasons.append("closure_unfriendly")

    score = f(row, "final_bioprior_score")
    source = row["proposal_source"]
    score += POSITIVE_SOURCES.get(source, 0.0)

    score += 0.8 * f(row, "terminal_tail_score")
    score += 0.9 * f(row, "surface_flexible_loop_score")
    score += 0.7 * f(row, "disorder_like_score")
    score += 0.8 * f(row, "linker_compressibility_score")

    if row.get("closure_type") == "terminal":
        score += 0.8
    elif row.get("closure_friendly_8A") == "True":
        score += 0.6
    else:
        score -= 4.0

    score -= 6.0 * f(row, "functional_core_risk_score")
    score -= 3.0 * f(row, "motif_shadow_risk_score")
    score -= 2.5 * f(row, "structural_core_risk_score")
    score -= 1.5 * f(row, "shadow_overlap_fraction")
    score -= 0.5 * f(row, "high_pLDDT_fraction")

    if source in NEGATIVE_SOURCES:
        score -= 1.5
    if f(row, "mean_distance_to_protected", 999.0) >= 10.0:
        score += 0.4

    if "hard_reject" in reject_reasons or "protected_overlap" in reject_reasons or "functional_core_risk" in reject_reasons:
        eligible = False
    elif "closure_unfriendly" in reject_reasons:
        eligible = False
    elif f(row, "motif_shadow_risk_score") >= 0.7:
        eligible = False
        reject_reasons.append("motif_shadow_heavy")
    elif f(row, "structural_core_risk_score") >= 0.85:
        eligible = False
        reject_reasons.append("structural_core_heavy")
    else:
        eligible = source not in NEGATIVE_SOURCES

    if not eligible and f(row, "final_bioprior_score") > 1.0:
        reject_reasons.append("high_score_but_rejected")

    if eligible:
        if source == "terminal_tail_prior":
            reason = "safe terminal truncation prior"
        elif source == "surface_flexible_loop_prior":
            reason = "motif-far low-contact closure-friendly loop prior"
        elif source == "disorder_like_prior":
            reason = "low-pLDDT disorder-like region without functional overlap"
        elif source == "linker_compressibility_prior":
            reason = "closure-friendly motif-far linker compression prior"
        else:
            reason = "eligible BioPrior segment"
    else:
        reason = "rejected: " + ";".join(reject_reasons or [classify_negative_type(row)])

    return score, eligible, reason, ";".join(sorted(set(reject_reasons)))


def overlaps_any(row, selected):
    start = row["seg_start_int"]
    end = row["seg_end_int"]
    for other in selected:
        if not (end < other["seg_start_int"] or start > other["seg_end_int"]):
            return True
    return False


def select_for_budget(rows, protein_length, budget_ratio):
    target = int(math.floor(protein_length * budget_ratio))
    eligible = [row for row in rows if row["teacher_eligible"]]
    eligible.sort(
        key=lambda row: (
            -row["teacher_score_float"],
            row["seg_len_int"],
            row["seg_start_int"],
        )
    )
    selected = []
    selected_len = 0
    for row in eligible:
        if overlaps_any(row, selected):
            continue
        if selected_len + row["seg_len_int"] > target:
            continue
        selected.append(row)
        selected_len += row["seg_len_int"]
    return selected, selected_len, target


def sample_pairwise(rows_by_accession_budget, rng, max_negatives):
    pairs = []
    pair_id = 0
    for (accession, budget_ratio), rows in rows_by_accession_budget.items():
        positives = [row for row in rows if row["teacher_label"] == "1"]
        negatives_by_type = defaultdict(list)
        for row in rows:
            if row["teacher_label"] == "1":
                continue
            negative_type = row["negative_type"]
            negatives_by_type[negative_type].append(row)
        for positive in positives:
            selected_negatives = []
            for negative_type in [
                "protected_overlap_negative",
                "motif_shadow_negative",
                "structural_core_negative",
                "closure_unfriendly_negative",
                "high_score_but_rejected_negative",
                "random_unselected_negative",
            ]:
                bucket = negatives_by_type.get(negative_type, [])
                if bucket:
                    selected_negatives.append(rng.choice(bucket))
                if len(selected_negatives) >= max_negatives:
                    break
            while len(selected_negatives) < max_negatives and negatives_by_type:
                bucket = rng.choice(list(negatives_by_type.values()))
                if bucket:
                    selected_negatives.append(rng.choice(bucket))
                else:
                    break
            dedup = []
            seen = set()
            for negative in selected_negatives:
                key = (negative["seg_start"], negative["seg_end"], negative["proposal_source"])
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(negative)
            for negative in dedup:
                pair_id += 1
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "accession": accession,
                        "budget_ratio": budget_ratio,
                        "positive_seg_start": positive["seg_start"],
                        "positive_seg_end": positive["seg_end"],
                        "positive_proposal_source": positive["proposal_source"],
                        "positive_teacher_score": positive["teacher_score"],
                        "negative_seg_start": negative["seg_start"],
                        "negative_seg_end": negative["seg_end"],
                        "negative_proposal_source": negative["proposal_source"],
                        "negative_teacher_score": negative["teacher_score"],
                        "negative_type": negative["negative_type"],
                    }
                )
    return pairs


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(len(values) - 1, lo + 1)
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def write_summary(path, label_rows, pair_rows):
    by_budget = defaultdict(list)
    for row in label_rows:
        by_budget[row["budget_ratio"]].append(row)
    selected = [row for row in label_rows if row["teacher_label"] == "1"]
    hard_reject = [row for row in label_rows if row["hard_reject"] == "True"]
    pair_type_counts = Counter(row["negative_type"] for row in pair_rows)
    selected_source_counts = Counter(row["proposal_source"] for row in selected)
    selected_shadow = [float(row["shadow_overlap_fraction"]) for row in selected]
    selected_struct = [float(row["structural_core_risk_score"]) for row in selected]
    selected_closure_friendly = sum(1 for row in selected if row["closure_friendly_8A"] == "True")
    selected_protected_overlap = sum(1 for row in selected if float(row["protected_overlap_fraction"]) > 0)

    fill_by_accession_budget = {}
    for row in label_rows:
        key = (row["accession"], row["budget_ratio"])
        fill_by_accession_budget[key] = float(row["fill_ratio"])
    low_fill = [(a, b, f) for (a, b), f in fill_by_accession_budget.items() if f < 0.5]

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as handle:
        handle.write("Core-1K BioPrior teacher label summary\n\n")
        handle.write("Teacher biological deletion principles:\n")
        handle.write("1. Prefer terminal tails when protected overlap and structural risk are low.\n")
        handle.write("2. Prefer motif-far low-contact closure-friendly surface/flexible loop-like segments.\n")
        handle.write("3. Prefer low-pLDDT disorder-like regions only when functional overlap is absent.\n")
        handle.write("4. Prefer closure-friendly motif-far linker-compressible segments.\n")
        handle.write("5. Reject protected motif/functional overlap and non-terminal closure-unfriendly segments.\n")
        handle.write("6. Strongly penalize motif-shadow-heavy and high-contact/high-pLDDT structural-core segments.\n")
        handle.write("7. SCISOR score is not used in this v1 teacher; future SCISOR segment evidence should be additive only.\n\n")

        handle.write("Label rows: {}\n".format(len(label_rows)))
        handle.write("Selected positive rows: {}\n".format(len(selected)))
        handle.write("Pairwise pairs: {}\n\n".format(len(pair_rows)))
        handle.write("Average fill ratio by budget:\n")
        for budget in sorted(by_budget, key=lambda x: float(x)):
            values = {row["accession"]: float(row["fill_ratio"]) for row in by_budget[budget]}.values()
            handle.write("- {}: {:.6f}\n".format(budget, mean(values)))
        handle.write("\nTeacher selected segment source distribution:\n")
        for source, count in selected_source_counts.most_common():
            handle.write("- {}: {}\n".format(source, count))
        handle.write("\nProtected overlap in selected segments: {}\n".format(selected_protected_overlap))
        handle.write("Selected shadow overlap fraction distribution:\n")
        handle.write("- p50: {:.6f}\n".format(percentile(selected_shadow, 0.50)))
        handle.write("- p75: {:.6f}\n".format(percentile(selected_shadow, 0.75)))
        handle.write("- max: {:.6f}\n".format(max(selected_shadow) if selected_shadow else 0.0))
        handle.write("Selected structural risk distribution:\n")
        handle.write("- p50: {:.6f}\n".format(percentile(selected_struct, 0.50)))
        handle.write("- p75: {:.6f}\n".format(percentile(selected_struct, 0.75)))
        handle.write("- max: {:.6f}\n".format(max(selected_struct) if selected_struct else 0.0))
        handle.write(
            "Selected closure-friendly fraction: {:.6f}\n".format(
                selected_closure_friendly / float(len(selected) or 1)
            )
        )
        handle.write("\nHard reject rows: {}\n".format(len(hard_reject)))
        reject_counts = Counter(row["reject_reason"] for row in hard_reject)
        for reason, count in reject_counts.most_common(20):
            handle.write("- {}: {}\n".format(reason or "hard_reject", count))
        handle.write("\nPairwise negative type distribution:\n")
        for negative_type, count in pair_type_counts.most_common():
            handle.write("- {}: {}\n".format(negative_type, count))
        handle.write("\nLow fill ratio (<0.5) examples:\n")
        for accession, budget, fill in low_fill[:50]:
            handle.write("- {}\t{}\t{:.6f}\n".format(accession, budget, fill))
        if len(low_fill) > 50:
            handle.write("- ... {} more\n".format(len(low_fill) - 50))
        handle.write("\nCORE_1K_BIOPRIOR_TEACHER_PASS\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BioPrior teacher labels and pairwise pairs.")
    parser.add_argument("--segments_csv", required=True)
    parser.add_argument("--core_csv", required=True)
    parser.add_argument("--config", default="configs/bioprior_v1.yaml")
    parser.add_argument("--out_labels", required=True)
    parser.add_argument("--out_pairs", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--max_negatives_per_positive", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    protein_lengths = read_core_lengths(args.core_csv)
    segment_rows = read_segments(args.segments_csv)
    by_accession = defaultdict(list)
    for row in segment_rows:
        score, eligible, reason, reject_reason = compute_teacher_score(row)
        row["teacher_score_float"] = score
        row["teacher_score"] = "{:.6f}".format(score)
        row["teacher_eligible"] = eligible
        row["teacher_base_reason"] = reason
        row["teacher_reject_reason"] = reject_reason
        by_accession[row["accession"]].append(row)

    label_rows = []
    rows_by_accession_budget = {}
    for accession, rows in by_accession.items():
        protein_length = protein_lengths.get(accession, i(rows[0], "protein_length"))
        for budget in BUDGETS:
            selected, selected_len, target_len = select_for_budget(rows, protein_length, budget)
            selected_keys = {
                (row["seg_start"], row["seg_end"], row["proposal_source"]) for row in selected
            }
            selected_rank = {
                (row["seg_start"], row["seg_end"], row["proposal_source"]): rank + 1
                for rank, row in enumerate(sorted(selected, key=lambda r: -r["teacher_score_float"]))
            }
            budget_key = "{:.2f}".format(budget)
            budget_rows = []
            for row in rows:
                key = (row["seg_start"], row["seg_end"], row["proposal_source"])
                is_selected = key in selected_keys
                negative_type = ""
                if not is_selected:
                    if "high_score_but_rejected" in row["teacher_reject_reason"]:
                        negative_type = "high_score_but_rejected_negative"
                    elif not row["teacher_eligible"]:
                        negative_type = classify_negative_type(row)
                    else:
                        negative_type = "random_unselected_negative"
                out = dict(row)
                out["budget_ratio"] = budget_key
                out["teacher_label"] = "1" if is_selected else "0"
                out["selected_by_teacher"] = str(is_selected)
                out["teacher_rank"] = selected_rank.get(key, "")
                out["selection_reason"] = row["teacher_base_reason"] if is_selected else "not selected"
                out["reject_reason"] = row["teacher_reject_reason"]
                out["selected_length_for_protein_budget"] = selected_len
                out["target_budget_length"] = target_len
                out["fill_ratio"] = "{:.6f}".format(selected_len / float(target_len or 1))
                out["negative_type"] = negative_type
                budget_rows.append(out)
                label_rows.append(out)
            rows_by_accession_budget[(accession, budget_key)] = budget_rows

    rng = random.Random(args.seed)
    pair_rows = sample_pairwise(rows_by_accession_budget, rng, args.max_negatives_per_positive)

    label_fields = [
        "accession",
        "budget_ratio",
        "seg_start",
        "seg_end",
        "seg_len",
        "proposal_source",
        "biological_rationale",
        "final_bioprior_score",
        "teacher_score",
        "teacher_label",
        "selected_by_teacher",
        "teacher_rank",
        "selection_reason",
        "reject_reason",
        "selected_length_for_protein_budget",
        "target_budget_length",
        "fill_ratio",
    ] + PRIOR_SCORE_COLUMNS + [
        "protected_overlap_fraction",
        "shadow_overlap_fraction",
        "mean_contact_density_8A",
        "mean_pLDDT",
        "closure_type",
        "closure_friendly_8A",
        "hard_reject",
        "negative_type",
    ]
    pair_fields = [
        "pair_id",
        "accession",
        "budget_ratio",
        "positive_seg_start",
        "positive_seg_end",
        "positive_proposal_source",
        "positive_teacher_score",
        "negative_seg_start",
        "negative_seg_end",
        "negative_proposal_source",
        "negative_teacher_score",
        "negative_type",
    ]

    for path in [args.out_labels, args.out_pairs, args.summary_txt]:
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
    with open(args.out_labels, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fields, extrasaction="ignore")
        writer.writeheader()
        for row in label_rows:
            writer.writerow(row)
    with open(args.out_pairs, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(row)
    write_summary(args.summary_txt, label_rows, pair_rows)

    print("Segments: {}".format(len(segment_rows)))
    print("Label rows: {}".format(len(label_rows)))
    print("Positive labels: {}".format(sum(1 for row in label_rows if row["teacher_label"] == "1")))
    print("Pairwise pairs: {}".format(len(pair_rows)))
    print("Wrote {}".format(args.out_labels))
    print("Wrote {}".format(args.out_pairs))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
