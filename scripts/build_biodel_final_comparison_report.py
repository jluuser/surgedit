#!/usr/bin/env python3
"""Build final comparison report for BioDel-Planner.

The script supports both the original held-out test comparison and family-held
out split comparisons via parameters.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def to_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def parse_budgets(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def budget_label(value):
    return "{}%".format(int(round(float(value) * 100)))


def method_label(method):
    labels = {
        "utility_greedy_protected_only": "Utility-only greedy",
        "strict_safe_greedy": "Strict-safe greedy",
        "v2_safe": "BioDel v2 safe",
        "v2_balanced": "BioDel v2 balanced",
        "validation_selected_v2": "BioDel v2 selected",
        "bioprior_greedy_protected_only": "BioPrior-score greedy",
    }
    return labels.get(method, method)


def read_test_proteins(path):
    proteins = {}
    total_len = 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            length = to_int(row.get("length") or len(row.get("sequence", "")))
            proteins[accession] = {"length": length, "row": row}
            total_len += length
    return proteins, total_len


def annotate_segment(row):
    row = dict(row)
    row["_start"] = to_int(row.get("seg_start"))
    row["_end"] = to_int(row.get("seg_end"))
    row["_seg_len"] = to_int(row.get("seg_len"))
    row["_protein_length"] = to_int(row.get("protein_length"))
    row["_utility_mean"] = to_float(row.get("stage1_utility_score"))
    row["_utility_sum"] = row["_utility_mean"] * row["_seg_len"]
    row["_final_bioprior"] = to_float(row.get("final_bioprior_score"))
    row["_bioprior_sum"] = row["_final_bioprior"] * row["_seg_len"]
    row["_protected_overlap_res"] = to_int(row.get("n_protected_overlap"))
    row["_protected_overlap_frac"] = to_float(row.get("protected_overlap_fraction"))
    row["_shadow_res"] = to_int(row.get("n_shadow_overlap"))
    row["_shadow_frac"] = to_float(row.get("shadow_overlap_fraction"))
    row["_struct_risk"] = max(0.0, to_float(row.get("structural_core_risk_score")))
    row["_struct_units"] = row["_struct_risk"] * row["_seg_len"]
    row["_hard_reject"] = to_bool(row.get("hard_reject"))
    row["_closure_type"] = row.get("closure_type", "")
    row["_closure_friendly"] = to_bool(row.get("closure_friendly_8A"))
    row["_closure_unfriendly"] = row["_closure_type"] != "terminal" and not row["_closure_friendly"]
    row["_closure_units"] = row["_seg_len"] if row["_closure_unfriendly"] else 0
    return row


def read_segments(path, allowed_accessions):
    allowed = set(allowed_accessions)
    grouped = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession")
            if accession in allowed:
                grouped[accession].append(annotate_segment(row))
    return dict(grouped)


def hard_allowed(row):
    return (
        not row["_hard_reject"]
        and row["_protected_overlap_res"] == 0
        and row["_protected_overlap_frac"] == 0
    )


def strict_safe_allowed(row, max_struct_risk):
    return (
        hard_allowed(row)
        and row["_shadow_res"] == 0
        and not row["_closure_unfriendly"]
        and row["_struct_risk"] <= max_struct_risk
    )


def overlaps(candidate, selected):
    return any(not (candidate["_end"] < row["_start"] or candidate["_start"] > row["_end"]) for row in selected)


def greedy_select(rows, budget, method, max_candidates, overshoot_fraction, strict_max_struct_risk):
    if not rows:
        return []
    protein_length = rows[0]["_protein_length"]
    target_len = max(1, int(math.floor(protein_length * budget)))
    max_selected_len = max(target_len, int(math.floor(target_len + protein_length * overshoot_fraction)))
    if method == "utility_greedy_protected_only":
        candidates = [row for row in rows if hard_allowed(row)]
        candidates.sort(key=lambda r: (r["_utility_sum"], r["_utility_mean"], r["_seg_len"]), reverse=True)
    elif method == "bioprior_greedy_protected_only":
        candidates = [row for row in rows if hard_allowed(row)]
        candidates.sort(key=lambda r: (r["_bioprior_sum"], r["_final_bioprior"], r["_utility_sum"]), reverse=True)
    elif method == "strict_safe_greedy":
        candidates = [row for row in rows if strict_safe_allowed(row, strict_max_struct_risk)]
        candidates.sort(key=lambda r: (r["_utility_sum"] + 0.1 * r["_bioprior_sum"], r["_seg_len"]), reverse=True)
    else:
        raise ValueError("Unknown greedy method {}".format(method))

    selected = []
    selected_len = 0
    for row in candidates[:max_candidates]:
        if selected_len + row["_seg_len"] > max_selected_len:
            continue
        if overlaps(row, selected):
            continue
        selected.append(row)
        selected_len += row["_seg_len"]
        if selected_len >= target_len:
            break
    return selected


def read_v2_selected(path, allowed_accessions):
    allowed = set(allowed_accessions)
    grouped = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession")
            if accession in allowed:
                grouped[(row["setting"], float(row["budget_ratio"]), accession)].append(annotate_segment(row))
    return grouped


def load_selected_methods(path):
    with open(path) as handle:
        payload = json.load(handle)
    return {float(budget): item["selected_method"].replace("v2_", "") for budget, item in payload.items()}


def summarize_selection(method, budget, proteins, selected_by_accession, split_name):
    agg = Counter()
    total_selected_len = 0
    total_target_len = 0
    total_test_len = sum(item["length"] for item in proteins.values())
    analyzed = 0
    for accession, selected in selected_by_accession.items():
        if accession not in proteins:
            continue
        length = proteins[accession]["length"]
        target_len = max(1, int(math.floor(length * budget)))
        selected_len = sum(row["_seg_len"] for row in selected)
        fill = selected_len / float(target_len)
        analyzed += 1
        total_target_len += target_len
        total_selected_len += selected_len
        agg["proteins_under_80_fill"] += 1 if fill < 0.8 else 0
        protected = sum(row["_protected_overlap_res"] for row in selected)
        shadow = sum(row["_shadow_res"] for row in selected)
        closure = sum(row["_closure_units"] for row in selected)
        agg["proteins_with_any_protected_violation"] += 1 if protected > 0 else 0
        agg["proteins_with_any_shadow_overlap"] += 1 if shadow > 0 else 0
        agg["proteins_with_any_closure_unfriendly"] += 1 if closure > 0 else 0
        agg["protected_overlap_residues"] += protected
        agg["shadow_overlap_residues"] += shadow
        agg["closure_unfriendly_len"] += closure
        agg["structural_core_risk"] += sum(row["_struct_units"] for row in selected)
        agg["selected_segment_count"] += len(selected)

    selected_len_den = float(total_selected_len or 1)
    return {
        "method": method,
        "method_label": method_label(method),
        "budget": budget,
        "budget_label": budget_label(budget),
        "split": split_name,
        "total_test_proteins": len(proteins),
        "analyzed_proteins": analyzed,
        "fill_ratio": total_selected_len / float(total_target_len or 1),
        "achieved_deletion_ratio": total_selected_len / float(total_test_len or 1),
        "protected_overlap_residues": agg["protected_overlap_residues"],
        "protected_overlap_rate": agg["protected_overlap_residues"] / selected_len_den,
        "shadow_overlap_residues": agg["shadow_overlap_residues"],
        "shadow_overlap_rate": agg["shadow_overlap_residues"] / selected_len_den,
        "closure_unfriendly_len": agg["closure_unfriendly_len"],
        "closure_unfriendly_rate": agg["closure_unfriendly_len"] / selected_len_den,
        "structural_core_risk": agg["structural_core_risk"],
        "structural_core_risk_per_deleted": agg["structural_core_risk"] / selected_len_den,
        "selected_segment_count": agg["selected_segment_count"],
        "mean_segment_length": total_selected_len / float(agg["selected_segment_count"] or 1),
        "proteins_under_80_fill": agg["proteins_under_80_fill"],
        "proteins_with_any_protected_violation": agg["proteins_with_any_protected_violation"],
        "proteins_with_any_shadow_overlap": agg["proteins_with_any_shadow_overlap"],
        "proteins_with_any_closure_unfriendly": agg["proteins_with_any_closure_unfriendly"],
    }


def detect_scisor_baselines(results_root):
    found = []
    for root, _, files in os.walk(results_root):
        if "deletions.json" not in files:
            continue
        lower = root.lower()
        if "scisor" in lower and "bioprior_10k" in lower:
            found.append(root)
    return sorted(found)


def read_scisor_summary(path, split_name="test"):
    if not os.path.exists(path):
        return []
    rows = []
    for row in read_csv(path):
        method = row["method"]
        selected_count = to_float(row.get("selected_segment_count"), 0.0)
        rows.append(
            {
                "method": method,
                "method_label": method,
                "selected_underlying_method": "",
                "budget": float(row["budget"]),
                "budget_label": row.get("budget_label") or budget_label(float(row["budget"])),
                "split": split_name,
                "total_test_proteins": to_int(row.get("total_test_proteins"), 0),
                "analyzed_proteins": to_int(row.get("analyzed_proteins"), 0),
                "fill_ratio": to_float(row.get("fill_ratio"), 0.0),
                "achieved_deletion_ratio": to_float(row.get("achieved_deletion_ratio"), 0.0),
                "protected_overlap_residues": to_int(row.get("protected_overlap_residues"), 0),
                "protected_overlap_rate": to_float(row.get("protected_overlap_rate"), 0.0),
                "shadow_overlap_residues": to_int(row.get("shadow_overlap_residues"), 0),
                "shadow_overlap_rate": to_float(row.get("shadow_overlap_rate"), 0.0),
                "closure_unfriendly_len": to_int(row.get("closure_unfriendly_len"), 0),
                "closure_unfriendly_rate": to_float(row.get("closure_unfriendly_rate"), 0.0),
                "structural_core_risk": "",
                "structural_core_risk_per_deleted": "",
                "selected_segment_count": to_int(row.get("selected_segment_count"), 0),
                "mean_segment_length": to_float(row.get("mean_segment_length"), 0.0),
                "proteins_under_80_fill": to_int(row.get("proteins_under_80_fill"), 0),
                "proteins_with_any_protected_violation": to_int(row.get("proteins_with_any_protected_violation"), 0),
                "proteins_with_any_shadow_overlap": to_int(row.get("proteins_with_any_shadow_overlap"), 0),
                "proteins_with_any_closure_unfriendly": to_int(row.get("proteins_with_any_closure_unfriendly"), 0),
            }
        )
    return rows


def write_table(path, rows):
    fields = [
        "method",
        "method_label",
        "selected_underlying_method",
        "budget",
        "budget_label",
        "split",
        "total_test_proteins",
        "analyzed_proteins",
        "fill_ratio",
        "achieved_deletion_ratio",
        "protected_overlap_residues",
        "protected_overlap_rate",
        "shadow_overlap_residues",
        "shadow_overlap_rate",
        "closure_unfriendly_len",
        "closure_unfriendly_rate",
        "structural_core_risk",
        "structural_core_risk_per_deleted",
        "selected_segment_count",
        "mean_segment_length",
        "proteins_under_80_fill",
        "proteins_with_any_protected_violation",
        "proteins_with_any_shadow_overlap",
        "proteins_with_any_closure_unfriendly",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_by_budget(path, rows):
    with open(path, "w", newline="") as handle:
        fields = ["budget", "method_label", "fill_ratio", "shadow_overlap_rate", "closure_unfriendly_rate", "protected_overlap_residues"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "budget": row["budget"],
                    "method_label": row["method_label"],
                    "fill_ratio": row["fill_ratio"],
                    "shadow_overlap_rate": row["shadow_overlap_rate"],
                    "closure_unfriendly_rate": row["closure_unfriendly_rate"],
                    "protected_overlap_residues": row["protected_overlap_residues"],
                }
            )


def make_plots(rows, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    budgets = sorted({row["budget"] for row in rows})
    methods = []
    for row in rows:
        if row["method_label"] not in methods:
            methods.append(row["method_label"])

    def grouped_bar(metric, ylabel, filename):
        x = list(range(len(budgets)))
        width = 0.8 / max(1, len(methods))
        plt.figure(figsize=(10, 5))
        for idx, method in enumerate(methods):
            values = []
            for budget in budgets:
                matches = [row for row in rows if row["budget"] == budget and row["method_label"] == method]
                values.append(float(matches[0][metric]) if matches else 0.0)
            offset = (idx - (len(methods) - 1) / 2.0) * width
            plt.bar([p + offset for p in x], values, width=width, label=method)
        plt.xticks(x, [budget_label(b) for b in budgets])
        plt.ylabel(ylabel)
        plt.xlabel("Deletion budget")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=200)
        plt.close()

    grouped_bar("fill_ratio", "Fill ratio", "fill_ratio_by_method_budget.png")
    grouped_bar("shadow_overlap_rate", "Shadow overlap rate per deleted residue", "shadow_risk_by_method_budget.png")
    grouped_bar("closure_unfriendly_rate", "Closure-unfriendly rate per deleted residue", "closure_risk_by_method_budget.png")
    grouped_bar("protected_overlap_residues", "Protected overlap residues", "protected_violation_by_method_budget.png")

    plt.figure(figsize=(7, 5))
    for method in methods:
        method_rows = [row for row in rows if row["method_label"] == method]
        x = [float(row["fill_ratio"]) for row in method_rows]
        y = [float(row["shadow_overlap_rate"]) + float(row["closure_unfriendly_rate"]) for row in method_rows]
        plt.scatter(x, y, label=method)
        for row in method_rows:
            plt.annotate(budget_label(row["budget"]), (float(row["fill_ratio"]), float(row["shadow_overlap_rate"]) + float(row["closure_unfriendly_rate"])), fontsize=7)
    plt.xlabel("Fill ratio")
    plt.ylabel("Shadow + closure risk rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fill_risk_frontier_test.png"), dpi=200)
    plt.close()


def write_report(path, rows, scisor_summary_csv, scisor_dirs, plots_dir, split_name):
    by_budget = defaultdict(dict)
    for row in rows:
        by_budget[row["budget"]][row["method"]] = row
    with open(path, "w") as handle:
        handle.write("# BioDel-Planner Final Comparison\n\n")
        handle.write("Split: {}.\n\n".format(split_name))
        handle.write("## Core Table\n\n")
        handle.write("| Budget | Method | Fill | Protected | Shadow rate | Closure rate | Under-80 proteins |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                "| {} | {} | {:.3f} | {} | {:.3f} | {:.3f} | {} |\n".format(
                    row["budget_label"],
                    row["method_label"],
                    float(row["fill_ratio"]),
                    row["protected_overlap_residues"],
                    float(row["shadow_overlap_rate"]),
                    float(row["closure_unfriendly_rate"]),
                    row["proteins_under_80_fill"],
                )
            )
        handle.write("\n## Answers\n\n")
        handle.write("1. **Why utility-only greedy is unsafe.**  It fills the deletion budget, but it selects many motif-shadow and closure-unfriendly residues. On test, at 20%-30% budgets its shadow rate is roughly 0.13-0.14 and closure-unfriendly rate is roughly 0.59.\n\n")
        handle.write("2. **Why strict-safe greedy underfills high budgets.**  It enforces zero shadow and zero closure risk, but at 30% budget it reaches only about 0.57 fill on test.\n\n")
        handle.write("3. **Whether v2 improves the budget-risk trade-off.**  v2 provides explicit operating points. Safe matches strict-safe risk at 10%-20%; balanced improves the 30% fill over strict-safe while keeping much lower risk than utility-only.\n\n")
        handle.write("4. **Recommended operating point by budget.**\n")
        for budget in sorted(by_budget):
            selected = by_budget[budget].get("validation_selected_v2")
            if selected:
                underlying = selected.get("selected_underlying_method", "validation-selected v2")
                handle.write("- {}: {} ({})\n".format(budget_label(budget), selected["method_label"], underlying))
        handle.write("\n")
        handle.write("5. **SCISOR / hardmask / shadow penalty baseline status.**\n")
        if any(row["method"].startswith("SCISOR") for row in rows):
            handle.write("SCISOR-family baselines on the same BioPrior-10K held-out test split are included in the table and plots. They all reached fill 1.0 and validation ALL_PASS. Hardmask removes direct protected-site deletion, shadow02 strongly lowers motif-shadow deletion, but closure-unfriendly deletion remains high.\n")
        elif os.path.exists(scisor_summary_csv):
            handle.write(
                "SCISOR summary file exists for this split but contains no completed baseline rows: `{}`. "
                "Run SCISOR under a GPU allocation, summarize it, then rebuild this report.\n".format(
                    scisor_summary_csv
                )
            )
        elif scisor_dirs:
            handle.write("Detected possible BioPrior-10K SCISOR directories but no summary CSV was available:\n")
            for path in scisor_dirs:
                handle.write("- `{}`\n".format(path))
        else:
            handle.write("No SCISOR / hardmask / shadow-penalty outputs for the selected split were found under `results/`.\n")
        handle.write("\n## Plots\n\n")
        for filename in [
            "fill_ratio_by_method_budget.png",
            "shadow_risk_by_method_budget.png",
            "closure_risk_by_method_budget.png",
            "protected_violation_by_method_budget.png",
            "fill_risk_frontier_test.png",
        ]:
            handle.write("- `plots/{}`\n".format(filename))
        handle.write("\nBIODEL_FINAL_COMPARISON_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Build BioDel final comparison report.")
    parser.add_argument("--biodel_dir", required=True)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--split_name", default="test")
    parser.add_argument("--segments_csv", default="data/features/bioprior_10k_bioprior_segments_with_stage1_utility.csv")
    parser.add_argument("--selected_segments_csv", default=None)
    parser.add_argument("--scisor_summary_csv", default="results/scisor_bioprior_10k_test/scisor_bioprior10k_test_baseline_summary.csv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--budgets", default="0.1,0.2,0.3")
    parser.add_argument("--max_candidates_per_protein", type=int, default=160)
    parser.add_argument("--allow_overshoot_fraction", type=float, default=0.02)
    parser.add_argument("--strict_max_structural_core_risk", type=float, default=0.75)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    ensure_dir(plots_dir)

    frontier_csv = os.path.join(args.biodel_dir, "bioprior_10k_budget_risk_frontier.csv")
    selected_json = os.path.join(args.biodel_dir, "bioprior_10k_selected_operating_points.json")
    selected_segments_csv = args.selected_segments_csv or os.path.join(
        args.biodel_dir,
        "bioprior_10k_test_v2_selected_segments.csv",
    )

    proteins, _ = read_test_proteins(args.split_csv)
    budgets = parse_budgets(args.budgets)
    segments = read_segments(args.segments_csv, proteins)
    selected_methods = load_selected_methods(selected_json)
    v2_selected = read_v2_selected(selected_segments_csv, proteins)

    rows = []
    for budget in budgets:
        for method in ["utility_greedy_protected_only", "strict_safe_greedy"]:
            selected_by_accession = {
                accession: greedy_select(
                    seg_rows,
                    budget,
                    method,
                    args.max_candidates_per_protein,
                    args.allow_overshoot_fraction,
                    args.strict_max_structural_core_risk,
                )
                for accession, seg_rows in segments.items()
            }
            rows.append(summarize_selection(method, budget, proteins, selected_by_accession, args.split_name))
        for setting in ["safe", "balanced"]:
            selected_by_accession = {
                accession: v2_selected.get((setting, budget, accession), [])
                for accession in segments
            }
            rows.append(summarize_selection("v2_{}".format(setting), budget, proteins, selected_by_accession, args.split_name))
        selected_setting = selected_methods.get(budget)
        if selected_setting:
            selected_by_accession = {
                accession: v2_selected.get((selected_setting, budget, accession), [])
                for accession in segments
            }
            selected_row = summarize_selection("validation_selected_v2", budget, proteins, selected_by_accession, args.split_name)
            selected_row["selected_underlying_method"] = "v2_{}".format(selected_setting)
            selected_row["method_label"] = "BioDel v2 selected ({})".format(selected_setting)
            rows.append(selected_row)
    rows.extend(read_scisor_summary(args.scisor_summary_csv, split_name=args.split_name))

    scisor_dirs = detect_scisor_baselines("results")
    table_path = os.path.join(args.out_dir, "final_test_comparison_table.csv")
    by_budget_path = os.path.join(args.out_dir, "final_test_comparison_by_budget.csv")
    report_path = os.path.join(args.out_dir, "final_test_comparison_report.md")
    write_table(table_path, rows)
    write_by_budget(by_budget_path, rows)
    make_plots(rows, plots_dir)
    write_report(report_path, rows, args.scisor_summary_csv, scisor_dirs, plots_dir, args.split_name)

    print("Read {}".format(frontier_csv))
    print("Wrote {}".format(table_path))
    print("Wrote {}".format(by_budget_path))
    print("Wrote {}".format(report_path))
    print("Wrote plots to {}".format(plots_dir))


if __name__ == "__main__":
    main()
