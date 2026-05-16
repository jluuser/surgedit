#!/usr/bin/env python3
"""Analyze BioDel-Planner underfill failure modes on a held-out split."""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict


def fnum(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def boolish(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_test_proteins(path):
    proteins = {}
    for row in read_csv(path):
        accession = row.get("accession") or row.get("protein_id")
        if not accession:
            continue
        proteins[accession] = {
            "accession": accession,
            "length": inum(row.get("length") or row.get("protein_length")),
            "primary_protected_type": row.get("primary_protected_type", ""),
            "n_protected": inum(row.get("n_protected")),
        }
    return proteins


def load_selected_ops(path):
    with open(path) as handle:
        raw = json.load(handle)
    out = {}
    for budget, payload in raw.items():
        method = payload.get("selected_method", "")
        setting = method.replace("v2_", "")
        out[str(fnum(budget))] = setting
    return out


def segment_from_row(row):
    return {
        "start": inum(row["seg_start"]),
        "end": inum(row["seg_end"]),
        "len": inum(row["seg_len"]),
        "protected": inum(row.get("n_protected_overlap")),
        "shadow": inum(row.get("n_shadow_overlap")),
        "shadow_fraction": fnum(row.get("shadow_overlap_fraction")),
        "closure_friendly": boolish(row.get("closure_friendly_8A")),
        "closure_type": row.get("closure_type", ""),
        "hard_reject": boolish(row.get("hard_reject")),
        "struct_risk": fnum(row.get("structural_core_risk_score")),
        "score": fnum(row.get("stage1_utility_score") or row.get("final_bioprior_score")),
        "source": row.get("proposal_source", ""),
    }


def load_candidate_segments(path, accessions):
    candidates = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession")
            if accession not in accessions:
                continue
            candidates[accession].append(segment_from_row(row))
    return candidates


def load_selected_lengths(path):
    selected = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    selected_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession")
            setting = row.get("setting")
            budget = str(fnum(row.get("budget_ratio")))
            length = inum(row.get("seg_len"))
            selected[setting][budget][accession] += length
            selected_counts[setting][budget][accession] += 1
    return selected, selected_counts


def max_nonoverlap_len(segments, protein_length):
    """Maximum total covered length from non-overlapping closed intervals."""
    if not segments or protein_length <= 0:
        return 0
    by_end = defaultdict(list)
    for seg in segments:
        start = max(0, seg["start"])
        end = min(protein_length - 1, seg["end"])
        if start <= end:
            by_end[end + 1].append((start, end + 1, end - start + 1))
    dp = [0] * (protein_length + 1)
    for pos in range(1, protein_length + 1):
        best = dp[pos - 1]
        for start, end_excl, length in by_end.get(pos, []):
            candidate = dp[start] + length
            if candidate > best:
                best = candidate
        dp[pos] = best
    return dp[protein_length]


def is_nonprotected(seg):
    return not seg["hard_reject"] and seg["protected"] == 0


def is_no_shadow(seg):
    return is_nonprotected(seg) and seg["shadow"] == 0


def is_closure_friendly(seg):
    return is_nonprotected(seg) and seg["closure_friendly"]


def is_strict_safe(seg):
    return is_nonprotected(seg) and seg["shadow"] == 0 and seg["closure_friendly"]


def classify_underfill(row):
    if fnum(row["selected_fill_ratio"]) >= 0.8:
        return "not_underfilled"
    target_80 = fnum(row["target_delete_len"]) * 0.8
    if fnum(row["max_nonprotected_len"]) < target_80:
        return "proposal_or_functional_capacity_limited"
    if fnum(row["max_no_shadow_len"]) < target_80 and fnum(row["max_closure_friendly_len"]) >= target_80:
        return "shadow_constraint_limited"
    if fnum(row["max_closure_friendly_len"]) < target_80 and fnum(row["max_no_shadow_len"]) >= target_80:
        return "closure_constraint_limited"
    if fnum(row["max_strict_safe_len"]) < target_80:
        return "joint_shadow_closure_limited"
    return "planner_scoring_or_global_budget_limited"


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["budget_ratio"], row["setting"])].append(row)
    out = []
    for (budget, setting), items in sorted(grouped.items(), key=lambda x: (fnum(x[0][0]), x[0][1])):
        total = len(items)
        under = [r for r in items if fnum(r["selected_fill_ratio"]) < 0.8]
        reasons = Counter(r["underfill_reason"] for r in under)
        out.append({
            "budget_ratio": budget,
            "setting": setting,
            "proteins": total,
            "underfilled_proteins": len(under),
            "underfilled_fraction": "{:.6f}".format(len(under) / total if total else 0.0),
            "mean_selected_fill_ratio": "{:.6f}".format(sum(fnum(r["selected_fill_ratio"]) for r in items) / total if total else 0.0),
            "mean_max_nonprotected_fill": "{:.6f}".format(sum(fnum(r["max_nonprotected_fill"]) for r in items) / total if total else 0.0),
            "mean_max_no_shadow_fill": "{:.6f}".format(sum(fnum(r["max_no_shadow_fill"]) for r in items) / total if total else 0.0),
            "mean_max_closure_friendly_fill": "{:.6f}".format(sum(fnum(r["max_closure_friendly_fill"]) for r in items) / total if total else 0.0),
            "mean_max_strict_safe_fill": "{:.6f}".format(sum(fnum(r["max_strict_safe_fill"]) for r in items) / total if total else 0.0),
            "reason_counts": ";".join("{}:{}".format(k, v) for k, v in sorted(reasons.items())),
        })
    return out


def write_report(path, summary_rows, protein_rows):
    ensure_dir(os.path.dirname(path))
    selected_rows = [r for r in protein_rows if r["setting"] == "validation_selected"]
    with open(path, "w") as handle:
        handle.write("# BioDel Underfill Failure-Mode Analysis\n\n")
        handle.write("This analysis asks why selected BioDel-Planner settings underfill the target deletion budget on the BioPrior-10K held-out test split.\n\n")
        handle.write("## Summary Table\n\n")
        handle.write("| Budget | Setting | Proteins | Underfilled | Mean selected fill | Max nonprotected fill | Max strict-safe fill | Top reasons |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---|\n")
        for row in summary_rows:
            handle.write("| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {} |\n".format(
                "{}%".format(int(round(fnum(row["budget_ratio"]) * 100))),
                row["setting"],
                row["proteins"],
                row["underfilled_proteins"],
                fnum(row["mean_selected_fill_ratio"]),
                fnum(row["mean_max_nonprotected_fill"]),
                fnum(row["mean_max_strict_safe_fill"]),
                row["reason_counts"] or "none",
            ))
        handle.write("\n## Validation-Selected Operating Point\n\n")
        by_budget = defaultdict(list)
        for row in selected_rows:
            by_budget[row["budget_ratio"]].append(row)
        for budget in sorted(by_budget, key=fnum):
            items = by_budget[budget]
            under = [r for r in items if fnum(r["selected_fill_ratio"]) < 0.8]
            reasons = Counter(r["underfill_reason"] for r in under)
            handle.write("- {}%: underfilled {}/{} proteins; reasons: {}.\n".format(
                int(round(fnum(budget) * 100)),
                len(under),
                len(items),
                ", ".join("{}={}".format(k, v) for k, v in sorted(reasons.items())) or "none",
            ))
        handle.write("\n## Interpretation\n\n")
        handle.write("- If max nonprotected fill is low, proposal capacity or functional constraints are the bottleneck.\n")
        handle.write("- If no-shadow or closure-friendly capacity is low, the biological risk constraints are the bottleneck.\n")
        handle.write("- If strict-safe capacity is high but selected fill is low, the planner scoring/global risk budget is leaving safe candidates unused.\n")
        handle.write("- This report should guide whether the next change should expand proposal generation, relax operating points, or improve utility scoring.\n\n")
        handle.write("BIODEL_UNDERFILL_ANALYSIS_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze BioDel underfill failure modes.")
    parser.add_argument("--segments_csv", default="data/features/bioprior_10k_bioprior_segments_with_stage1_utility.csv")
    parser.add_argument("--test_csv", default="data/processed/bioprior_10k_splits/test.csv")
    parser.add_argument("--selected_segments_csv", default="results/biodel_planner/bioprior_10k_test_v2_selected_segments.csv")
    parser.add_argument("--selected_ops_json", default="results/biodel_planner/bioprior_10k_selected_operating_points.json")
    parser.add_argument("--out_dir", default="results/biodel_planner/underfill_analysis")
    args = parser.parse_args()

    proteins = load_test_proteins(args.test_csv)
    selected_ops = load_selected_ops(args.selected_ops_json)
    candidates = load_candidate_segments(args.segments_csv, set(proteins))
    selected_lengths, selected_counts = load_selected_lengths(args.selected_segments_csv)

    settings = ["safe", "balanced", "aggressive"]
    budgets = ["0.1", "0.2", "0.3"]
    rows = []
    for accession, meta in proteins.items():
        length = meta["length"]
        if length <= 0 or accession not in candidates:
            continue
        segs = candidates[accession]
        nonprotected = [s for s in segs if is_nonprotected(s)]
        no_shadow = [s for s in segs if is_no_shadow(s)]
        closure_friendly = [s for s in segs if is_closure_friendly(s)]
        strict_safe = [s for s in segs if is_strict_safe(s)]

        capacity = {
            "max_nonprotected_len": max_nonoverlap_len(nonprotected, length),
            "max_no_shadow_len": max_nonoverlap_len(no_shadow, length),
            "max_closure_friendly_len": max_nonoverlap_len(closure_friendly, length),
            "max_strict_safe_len": max_nonoverlap_len(strict_safe, length),
        }
        for budget in budgets:
            target = int(length * fnum(budget))
            if target <= 0:
                continue
            for setting in settings:
                selected_len = selected_lengths[setting][budget].get(accession, 0)
                row = {
                    "accession": accession,
                    "budget_ratio": budget,
                    "setting": setting,
                    "protein_length": length,
                    "primary_protected_type": meta["primary_protected_type"],
                    "n_protected": meta["n_protected"],
                    "candidate_segments": len(segs),
                    "nonprotected_segments": len(nonprotected),
                    "strict_safe_segments": len(strict_safe),
                    "target_delete_len": target,
                    "selected_len": selected_len,
                    "selected_segments": selected_counts[setting][budget].get(accession, 0),
                    "selected_fill_ratio": "{:.6f}".format(selected_len / target),
                    **capacity,
                }
                for key in ("max_nonprotected_len", "max_no_shadow_len", "max_closure_friendly_len", "max_strict_safe_len"):
                    row[key.replace("_len", "_fill")] = "{:.6f}".format(capacity[key] / target)
                row["underfill_reason"] = classify_underfill(row)
                rows.append(row)

            selected_setting = selected_ops.get(budget)
            if selected_setting:
                selected_len = selected_lengths[selected_setting][budget].get(accession, 0)
                row = {
                    "accession": accession,
                    "budget_ratio": budget,
                    "setting": "validation_selected",
                    "protein_length": length,
                    "primary_protected_type": meta["primary_protected_type"],
                    "n_protected": meta["n_protected"],
                    "candidate_segments": len(segs),
                    "nonprotected_segments": len(nonprotected),
                    "strict_safe_segments": len(strict_safe),
                    "target_delete_len": target,
                    "selected_len": selected_len,
                    "selected_segments": selected_counts[selected_setting][budget].get(accession, 0),
                    "selected_fill_ratio": "{:.6f}".format(selected_len / target),
                    **capacity,
                }
                for key in ("max_nonprotected_len", "max_no_shadow_len", "max_closure_friendly_len", "max_strict_safe_len"):
                    row[key.replace("_len", "_fill")] = "{:.6f}".format(capacity[key] / target)
                row["underfill_reason"] = classify_underfill(row)
                rows.append(row)

    summary_rows = summarize(rows)
    fields = [
        "accession", "budget_ratio", "setting", "protein_length", "primary_protected_type", "n_protected",
        "candidate_segments", "nonprotected_segments", "strict_safe_segments", "target_delete_len",
        "selected_len", "selected_segments", "selected_fill_ratio", "max_nonprotected_len",
        "max_no_shadow_len", "max_closure_friendly_len", "max_strict_safe_len",
        "max_nonprotected_fill", "max_no_shadow_fill", "max_closure_friendly_fill",
        "max_strict_safe_fill", "underfill_reason",
    ]
    summary_fields = [
        "budget_ratio", "setting", "proteins", "underfilled_proteins", "underfilled_fraction",
        "mean_selected_fill_ratio", "mean_max_nonprotected_fill", "mean_max_no_shadow_fill",
        "mean_max_closure_friendly_fill", "mean_max_strict_safe_fill", "reason_counts",
    ]
    out_csv = os.path.join(args.out_dir, "underfill_failure_modes_by_protein.csv")
    out_summary = os.path.join(args.out_dir, "underfill_failure_modes_summary.csv")
    out_report = os.path.join(args.out_dir, "underfill_failure_modes_report.md")
    write_csv(out_csv, rows, fields)
    write_csv(out_summary, summary_rows, summary_fields)
    write_report(out_report, summary_rows, rows)
    print("Wrote {}".format(out_csv))
    print("Wrote {}".format(out_summary))
    print("Wrote {}".format(out_report))


if __name__ == "__main__":
    main()
