#!/usr/bin/env python3
"""Evaluate simple baselines on CATH S40 and DisProt BioDel benchmarks."""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def fnum(value, default=0.0):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bval(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_accession(rows):
    out = defaultdict(list)
    for row in rows:
        out[row["accession"]].append(row)
    return out


def overlaps(candidate, selected):
    start = inum(candidate["seg_start"])
    end = inum(candidate["seg_end"])
    for row in selected:
        if not (end < inum(row["seg_start"]) or start > inum(row["seg_end"])):
            return True
    return False


def hard_allowed(row):
    return not bval(row.get("hard_reject")) and inum(row.get("n_protected_overlap")) == 0 and fnum(row.get("protected_overlap_fraction")) == 0.0


def metric_value(row, dataset, metric):
    if dataset == "cath":
        if metric == "domain_core":
            return fnum(row.get("domain_core_overlap_fraction"))
        if metric == "terminal":
            return fnum(row.get("domain_terminal_overlap_fraction"))
        if metric == "boundary":
            return 1.0 if inum(row.get("domain_boundary_crossing")) > 0 else 0.0
    if dataset == "disprot":
        if metric == "idr":
            return fnum(row.get("idr_overlap_fraction"))
        if metric == "functional_idr":
            return fnum(row.get("functional_idr_overlap_fraction"))
        if metric == "ordered":
            return fnum(row.get("ordered_overlap_fraction"))
    return 0.0


def method_sort_key(row, method, rng):
    if method == "terminal":
        return (fnum(row.get("terminal_overlap_fraction")), fnum(row.get("stage1_utility_score")), fnum(row.get("seg_len")))
    if method == "stage1_only":
        return (fnum(row.get("stage1_utility_score")), fnum(row.get("seg_len")))
    if method == "bioprior_only":
        return (fnum(row.get("final_bioprior_score")), fnum(row.get("stage1_utility_score")))
    if method == "risk_certified_score":
        return (fnum(row.get("risk_certified_biodel_score")), fnum(row.get("stage1_utility_score")))
    if method == "random":
        return (rng.random(),)
    raise ValueError(method)


def select_greedy(rows, target_len, method, rng, safe_filter=False):
    candidates = list(rows)
    if safe_filter:
        candidates = [row for row in candidates if hard_allowed(row)]
    candidates.sort(key=lambda row: method_sort_key(row, method, rng), reverse=True)
    selected = []
    selected_len = 0
    for row in candidates:
        seg_len = inum(row.get("seg_len"))
        if selected_len + seg_len > target_len:
            continue
        if overlaps(row, selected):
            continue
        selected.append(row)
        selected_len += seg_len
        if selected_len >= target_len:
            break
    return selected


def selected_from_biodel(rows):
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row.get("setting"), row.get("auto_profile"))].append(row)
    return by_key


def summarize_selection(dataset, method, selected_by_accession, all_by_accession):
    agg = Counter()
    metric_sums = Counter()
    lists = defaultdict(list)
    source_counts = Counter()
    for accession, all_rows in all_by_accession.items():
        selected = selected_by_accession.get(accession, [])
        protein_length = inum(all_rows[0].get("protein_length"))
        selected_len = sum(inum(row.get("seg_len")) for row in selected)
        agg["proteins"] += 1
        agg["total_protein_length"] += protein_length
        agg["selected_len"] += selected_len
        agg["segments"] += len(selected)
        agg["protected_res"] += sum(inum(row.get("n_protected_overlap")) for row in selected)
        agg["hard_reject_segments"] += sum(1 for row in selected if bval(row.get("hard_reject")))
        agg["closure_unfriendly_segments"] += sum(
            1 for row in selected if str(row.get("closure_friendly_8A")).lower() not in ("true", "1", "yes")
        )
        lists["stage1"].extend(fnum(row.get("stage1_utility_score")) for row in selected)
        lists["bioprior"].extend(fnum(row.get("final_bioprior_score")) for row in selected)
        lists["risk_upper"].extend(fnum(row.get("risk_upper")) for row in selected)
        for row in selected:
            seg_len = inum(row.get("seg_len"))
            source_counts[row.get("proposal_source", "")] += 1
            if dataset == "cath":
                metric_sums["domain_core_len"] += metric_value(row, dataset, "domain_core") * seg_len
                metric_sums["terminal_len"] += metric_value(row, dataset, "terminal") * seg_len
                agg["boundary_segments"] += int(metric_value(row, dataset, "boundary") > 0)
            else:
                metric_sums["idr_len"] += metric_value(row, dataset, "idr") * seg_len
                metric_sums["functional_idr_len"] += metric_value(row, dataset, "functional_idr") * seg_len
                metric_sums["ordered_len"] += metric_value(row, dataset, "ordered") * seg_len
    denom = float(agg["selected_len"] or 1)
    out = {
        "dataset": dataset,
        "method": method,
        "proteins": agg["proteins"],
        "selected_len": agg["selected_len"],
        "global_delete_ratio": agg["selected_len"] / float(agg["total_protein_length"] or 1),
        "segments": agg["segments"],
        "protected_overlap_rate": agg["protected_res"] / denom,
        "hard_reject_segments": agg["hard_reject_segments"],
        "closure_unfriendly_segments": agg["closure_unfriendly_segments"],
        "mean_stage1_utility": sum(lists["stage1"]) / float(len(lists["stage1"]) or 1),
        "mean_final_bioprior_score": sum(lists["bioprior"]) / float(len(lists["bioprior"]) or 1),
        "mean_risk_upper": sum(lists["risk_upper"]) / float(len(lists["risk_upper"]) or 1),
        "top_sources": "; ".join("{}={}".format(k, v) for k, v in source_counts.most_common(3)),
    }
    if dataset == "cath":
        out.update(
            {
                "domain_core_overlap_rate": metric_sums["domain_core_len"] / denom,
                "terminal_overlap_rate": metric_sums["terminal_len"] / denom,
                "boundary_crossing_segments": agg["boundary_segments"],
                "idr_overlap_rate": "",
                "functional_idr_overlap_rate": "",
                "ordered_overlap_rate": "",
            }
        )
    else:
        out.update(
            {
                "domain_core_overlap_rate": "",
                "terminal_overlap_rate": "",
                "boundary_crossing_segments": "",
                "idr_overlap_rate": metric_sums["idr_len"] / denom,
                "functional_idr_overlap_rate": metric_sums["functional_idr_len"] / denom,
                "ordered_overlap_rate": metric_sums["ordered_len"] / denom,
            }
        )
    return out


def benchmark_dataset(dataset, candidates_csv, biodel_selected_csv, target_ratio, seed):
    rows = read_csv(candidates_csv)
    by_acc = group_by_accession(rows)
    rng = random.Random(seed)
    out = []
    methods = [
        ("random", False),
        ("terminal", False),
        ("stage1_only", False),
        ("bioprior_only", False),
        ("risk_certified_score", False),
        ("stage1_only_safe", True),
        ("bioprior_only_safe", True),
    ]
    for method, safe_filter in methods:
        selected_by_acc = {}
        base_method = method.replace("_safe", "")
        for accession, acc_rows in by_acc.items():
            protein_length = inum(acc_rows[0].get("protein_length"))
            target_len = max(1, int(round(protein_length * target_ratio)))
            selected_by_acc[accession] = select_greedy(acc_rows, target_len, base_method, rng, safe_filter=safe_filter)
        out.append(summarize_selection(dataset, method, selected_by_acc, by_acc))

    biodel_rows = read_csv(biodel_selected_csv)
    for key, group in selected_from_biodel(biodel_rows).items():
        setting, profile = key
        if setting != "safe" or profile not in ("conservative", "default", "aggressive"):
            continue
        selected_by_acc = group_by_accession(group)
        out.append(summarize_selection(dataset, "biodel_cert_{}_{}".format(setting, profile), selected_by_acc, by_acc))
    return out


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "dataset", "method", "proteins", "selected_len", "global_delete_ratio", "segments",
        "protected_overlap_rate", "hard_reject_segments", "closure_unfriendly_segments",
        "mean_stage1_utility", "mean_final_bioprior_score", "mean_risk_upper",
        "domain_core_overlap_rate", "terminal_overlap_rate", "boundary_crossing_segments",
        "idr_overlap_rate", "functional_idr_overlap_rate", "ordered_overlap_rate",
        "top_sources",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("# CATH/DisProt Baseline Comparison\n\n")
        for dataset in ("cath", "disprot"):
            handle.write("## {}\n\n".format(dataset.upper()))
            subset = [row for row in rows if row["dataset"] == dataset]
            if dataset == "cath":
                handle.write("method,delete_ratio,domain_core_overlap,terminal_overlap,boundary_segments,hard_reject\n")
                for row in subset:
                    handle.write(
                        "{},{:.4f},{:.4f},{:.4f},{},{}\n".format(
                            row["method"],
                            fnum(row["global_delete_ratio"]),
                            fnum(row["domain_core_overlap_rate"]),
                            fnum(row["terminal_overlap_rate"]),
                            row["boundary_crossing_segments"],
                            row["hard_reject_segments"],
                        )
                    )
            else:
                handle.write("method,delete_ratio,idr_overlap,functional_idr_overlap,ordered_overlap,hard_reject\n")
                for row in subset:
                    handle.write(
                        "{},{:.4f},{:.4f},{:.4f},{:.4f},{}\n".format(
                            row["method"],
                            fnum(row["global_delete_ratio"]),
                            fnum(row["idr_overlap_rate"]),
                            fnum(row["functional_idr_overlap_rate"]),
                            fnum(row["ordered_overlap_rate"]),
                            row["hard_reject_segments"],
                        )
                    )
            handle.write("\n")
        handle.write("BIODEL_CATH_DISPROT_BASELINE_COMPARISON_PASS\n")


def run(args):
    rows = []
    rows.extend(
        benchmark_dataset(
            "cath",
            args.cath_candidates_csv,
            args.cath_biodel_selected_csv,
            args.target_ratio,
            args.seed,
        )
    )
    rows.extend(
        benchmark_dataset(
            "disprot",
            args.disprot_candidates_csv,
            args.disprot_biodel_selected_csv,
            args.target_ratio,
            args.seed + 17,
        )
    )
    write_csv(args.out_csv, rows)
    write_report(args.out_report, rows)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CATH/DisProt baselines against BioDel-Cert.")
    parser.add_argument("--cath_candidates_csv", default="data/features/cath_s40_domain_biodel_segments_stage1_finetuned_certified.csv")
    parser.add_argument("--cath_biodel_selected_csv", default="results/biodel_planner/cath_s40_domain_auto_budget_certified_selected_segments.csv")
    parser.add_argument("--disprot_candidates_csv", default="data/features/disprot_idr_biodel_segments_stage1_finetuned_certified.csv")
    parser.add_argument("--disprot_biodel_selected_csv", default="results/biodel_planner/disprot_idr_auto_budget_certified_selected_segments.csv")
    parser.add_argument("--out_csv", default="results/biodel_planner/cath_disprot_baseline_comparison.csv")
    parser.add_argument("--out_report", default="results/biodel_planner/CATH_DISPROT_BASELINE_COMPARISON_REPORT.md")
    parser.add_argument("--target_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
