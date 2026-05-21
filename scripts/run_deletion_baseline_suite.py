#!/usr/bin/env python3
"""Run BioDel deletion baseline suite on a family-held-out split.

This is the paper-facing fixed-budget benchmark harness.  It evaluates simple
diagnostic baselines, learned-utility greedy baselines, BioDel planners, and
SCISOR summaries in a single schema so the AAAI experiments are not scattered
across separate result files.
"""

import argparse
import csv
import math
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
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def maybe_float(value):
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def inum(value, default=0):
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bval(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / float(len(values)) if values else 0.0


def fmt(value):
    if value in ("", None):
        return ""
    return "{:.6f}".format(float(value))


def stable_hash(text):
    value = 2166136261
    for byte in str(text).encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def budget_label(value):
    if value in ("", None, "auto"):
        return "auto"
    return "{}%".format(int(round(float(value) * 100)))


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_split_proteins(path):
    proteins = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if not accession:
                continue
            length = inum(row.get("length"), 0)
            if length <= 0:
                length = len(row.get("sequence", ""))
            proteins[accession] = {
                "length": length,
                "row": row,
            }
    return proteins


def annotate_segment(row):
    row = dict(row)
    start = inum(row.get("seg_start"))
    end = inum(row.get("seg_end"))
    seg_len = inum(row.get("seg_len"), end - start + 1)
    protein_length = inum(row.get("protein_length"), 0)
    row["_start"] = start
    row["_end"] = end
    row["_seg_len"] = seg_len
    row["_protein_length"] = protein_length
    row["_stage1"] = fnum(row.get("stage1_utility_score"))
    row["_stage1_sum"] = row["_stage1"] * seg_len
    row["_bioprior"] = fnum(row.get("final_bioprior_score"))
    row["_bioprior_sum"] = row["_bioprior"] * seg_len
    row["_compatibility"] = fnum(row.get("compatibility_score"))
    row["_compatibility_sum"] = row["_compatibility"] * seg_len
    row["_risk_certified"] = fnum(row.get("risk_certified_biodel_score"))
    row["_risk_certified_sum"] = row["_risk_certified"] * seg_len
    row["_terminal"] = max(fnum(row.get("terminal_overlap_fraction")), fnum(row.get("terminal_tail_score")))
    row["_disorder"] = fnum(row.get("disorder_like_score"))
    row["_low_plddt"] = fnum(row.get("low_pLDDT_fraction"))
    row["_mean_plddt"] = fnum(row.get("mean_pLDDT"), 100.0)
    row["_surface"] = fnum(row.get("surface_flexible_loop_score"))
    row["_contact"] = fnum(row.get("mean_contact_density_8A"), 999.0)
    row["_protected"] = inum(row.get("n_protected_overlap"))
    row["_protected_frac"] = fnum(row.get("protected_overlap_fraction"))
    row["_shadow"] = inum(row.get("n_shadow_overlap"))
    row["_shadow_frac"] = fnum(row.get("shadow_overlap_fraction"))
    row["_hard_reject"] = bval(row.get("hard_reject"))
    closure_type = row.get("closure_type", "")
    closure_bad = closure_type != "terminal" and not bval(row.get("closure_friendly_8A"))
    row["_closure_bad"] = closure_bad
    row["_closure_units"] = seg_len if closure_bad else 0
    row["_structural_risk"] = max(0.0, fnum(row.get("structural_core_risk_score")))
    row["_structural_units"] = row["_structural_risk"] * seg_len
    row["_risk_upper"] = maybe_float(row.get("risk_upper"))
    row["_risk_upper_units"] = (row["_risk_upper"] or 0.0) * seg_len
    row["_evidence_confidence"] = maybe_float(row.get("evidence_confidence"))
    row["_evidence_confidence_units"] = (row["_evidence_confidence"] or 0.0) * seg_len
    return row


def read_segments(path, proteins):
    allowed = set(proteins)
    grouped = defaultdict(list)
    total_rows = 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for raw in reader:
            accession = raw.get("accession") or raw.get("protein_id")
            if accession not in allowed:
                continue
            grouped[accession].append(annotate_segment(raw))
            total_rows += 1
    return dict(grouped), fields, total_rows


def hard_allowed(row):
    return (
        not row["_hard_reject"]
        and row["_protected"] == 0
        and row["_protected_frac"] == 0.0
    )


def strict_safe_allowed(row, max_structural_risk):
    return (
        hard_allowed(row)
        and row["_shadow"] == 0
        and not row["_closure_bad"]
        and row["_structural_risk"] <= max_structural_risk
    )


def overlaps(candidate, selected):
    return any(not (candidate["_end"] < row["_start"] or candidate["_start"] > row["_end"]) for row in selected)


def method_label(method):
    labels = {
        "random_candidate": "Random candidate intervals",
        "terminal_trim": "Terminal trimming",
        "low_plddt_first": "Low-pLDDT first",
        "disorder_first": "Disorder first",
        "surface_loop_first": "Surface-loop first",
        "low_contact_first": "Low-contact first",
        "utility_greedy_protected": "Stage-1 utility greedy",
        "bioprior_greedy_protected": "BioPrior-score greedy",
        "risk_certified_greedy_protected": "Risk-certified-score greedy",
        "compatibility_greedy_protected": "Learned compatibility greedy",
        "strict_safe_greedy": "Strict-safe greedy",
        "compatibility_strict_safe": "Learned compatibility strict-safe",
        "biodel_v2_safe": "BioDel v2 safe",
        "biodel_v2_balanced": "BioDel v2 balanced",
        "biodel_v2_aggressive": "BioDel v2 aggressive",
        "biodel_cert_frontier_conservative": "BioDel-Cert frontier conservative",
        "biodel_cert_frontier_default": "BioDel-Cert frontier default",
        "biodel_cert_frontier_aggressive": "BioDel-Cert frontier aggressive",
        "biodel_cert_learned_frontier_conservative": "BioDel-Cert learned frontier conservative",
        "biodel_cert_learned_frontier_default": "BioDel-Cert learned frontier default",
        "biodel_cert_learned_frontier_aggressive": "BioDel-Cert learned frontier aggressive",
    }
    return labels.get(method, method)


def method_category(method):
    if method.startswith("SCISOR"):
        return "external_generative_baseline"
    if method.startswith("biodel_cert"):
        return "auto_certified_planner"
    if method.startswith("biodel_v2"):
        return "fixed_budget_planner"
    if "compatibility" in method:
        return "learned_policy_baseline"
    if method in ("random_candidate", "terminal_trim", "low_plddt_first", "disorder_first", "surface_loop_first", "low_contact_first"):
        return "diagnostic_heuristic"
    if method == "strict_safe_greedy":
        return "diagnostic_safety_baseline"
    return "diagnostic_greedy"


def sorted_candidates(rows, method, rng, args):
    if method == "random_candidate":
        candidates = list(rows)
        rng.shuffle(candidates)
        return candidates
    if method == "terminal_trim":
        return sorted(rows, key=lambda row: (row["_terminal"], row["_seg_len"], row["_stage1"]), reverse=True)
    if method == "low_plddt_first":
        return sorted(rows, key=lambda row: (row["_low_plddt"], -row["_mean_plddt"], row["_disorder"], row["_seg_len"]), reverse=True)
    if method == "disorder_first":
        return sorted(rows, key=lambda row: (row["_disorder"], row["_low_plddt"], row["_seg_len"]), reverse=True)
    if method == "surface_loop_first":
        return sorted(rows, key=lambda row: (row["_surface"], -row["_contact"], row["_stage1"], row["_seg_len"]), reverse=True)
    if method == "low_contact_first":
        return sorted(rows, key=lambda row: (-row["_contact"], row["_surface"], row["_seg_len"]), reverse=True)
    if method == "utility_greedy_protected":
        candidates = [row for row in rows if hard_allowed(row)]
        return sorted(candidates, key=lambda row: (row["_stage1_sum"], row["_stage1"], row["_seg_len"]), reverse=True)
    if method == "bioprior_greedy_protected":
        candidates = [row for row in rows if hard_allowed(row)]
        return sorted(candidates, key=lambda row: (row["_bioprior_sum"], row["_bioprior"], row["_stage1_sum"]), reverse=True)
    if method == "risk_certified_greedy_protected":
        candidates = [row for row in rows if hard_allowed(row)]
        return sorted(candidates, key=lambda row: (row["_risk_certified_sum"], row["_risk_certified"], row["_stage1_sum"]), reverse=True)
    if method == "compatibility_greedy_protected":
        candidates = [row for row in rows if hard_allowed(row)]
        return sorted(candidates, key=lambda row: (row["_compatibility_sum"], row["_compatibility"], row["_stage1_sum"]), reverse=True)
    if method == "strict_safe_greedy":
        candidates = [row for row in rows if strict_safe_allowed(row, args.strict_max_structural_core_risk)]
        return sorted(candidates, key=lambda row: (row["_stage1_sum"] + 0.1 * row["_bioprior_sum"], row["_seg_len"]), reverse=True)
    if method == "compatibility_strict_safe":
        candidates = [row for row in rows if strict_safe_allowed(row, args.strict_max_structural_core_risk)]
        return sorted(candidates, key=lambda row: (row["_compatibility_sum"], row["_compatibility"], row["_stage1_sum"]), reverse=True)
    raise ValueError("Unknown method {}".format(method))


def select_fixed_budget(rows, budget, method, args, accession):
    if not rows:
        return []
    protein_length = rows[0]["_protein_length"]
    target_len = max(1, int(math.floor(protein_length * float(budget))))
    max_selected_len = max(target_len, int(math.floor(target_len + protein_length * args.allow_overshoot_fraction)))
    rng = random.Random(args.seed + stable_hash("{}:{}:{}".format(accession, method, budget)))
    candidates = sorted_candidates(rows, method, rng, args)
    if method not in ("random_candidate",):
        candidates = candidates[: args.max_candidates_per_protein]
    selected = []
    selected_len = 0
    for row in candidates:
        seg_len = row["_seg_len"]
        if selected_len + seg_len > max_selected_len:
            continue
        if overlaps(row, selected):
            continue
        selected.append(row)
        selected_len += seg_len
        if selected_len >= target_len:
            break
    return selected


def summarize_selection(method, budget, proteins, selected_by_accession, split_name, budget_type="fixed", status_by_accession=None, notes="", source_path=""):
    status_by_accession = status_by_accession or {}
    accessions = sorted(selected_by_accession)
    agg = Counter()
    lists = defaultdict(list)
    source_counts = Counter()
    total_protein_len = 0
    total_target_len = 0
    total_selected_len = 0
    certified = 0
    abstained = 0
    for accession in accessions:
        meta = proteins.get(accession, {})
        protein_length = inum(meta.get("length"), 0)
        selected = selected_by_accession.get(accession, [])
        selected_len = sum(row["_seg_len"] for row in selected)
        total_protein_len += protein_length
        total_selected_len += selected_len
        if budget_type == "fixed":
            target_len = max(1, int(math.floor(protein_length * float(budget))))
            total_target_len += target_len
            fill = selected_len / float(target_len or 1)
            agg["proteins_under_80_fill"] += 1 if fill < 0.80 else 0
        status = status_by_accession.get(accession, "")
        if status == "certified":
            certified += 1
        elif status.startswith("abstain"):
            abstained += 1
        elif budget_type == "auto":
            if selected_len > 0:
                certified += 1
            else:
                abstained += 1

        protected = sum(row["_protected"] for row in selected)
        shadow = sum(row["_shadow"] for row in selected)
        closure = sum(row["_closure_units"] for row in selected)
        structural = sum(row["_structural_units"] for row in selected)
        agg["selected_segments"] += len(selected)
        agg["protected_overlap_residues"] += protected
        agg["shadow_overlap_residues"] += shadow
        agg["closure_unfriendly_len"] += closure
        agg["structural_core_risk"] += structural
        agg["proteins_with_any_protected_violation"] += 1 if protected > 0 else 0
        agg["proteins_with_any_shadow_overlap"] += 1 if shadow > 0 else 0
        agg["proteins_with_any_closure_unfriendly"] += 1 if closure > 0 else 0
        terminal_len = sum(row["_seg_len"] * row["_terminal"] for row in selected)
        low_plddt_len = sum(row["_seg_len"] * row["_low_plddt"] for row in selected)
        disorder_len = sum(row["_seg_len"] * row["_disorder"] for row in selected)
        lists["terminal_len"].append(terminal_len)
        lists["low_plddt_len"].append(low_plddt_len)
        lists["disorder_len"].append(disorder_len)
        lists["stage1"].extend(row["_stage1"] for row in selected)
        lists["bioprior"].extend(row["_bioprior"] for row in selected)
        lists["compatibility"].extend(row["_compatibility"] for row in selected)
        lists["risk_upper_units"].append(sum(row["_risk_upper_units"] for row in selected))
        lists["evidence_confidence_units"].append(sum(row["_evidence_confidence_units"] for row in selected))
        for row in selected:
            source_counts[row.get("proposal_source", "")] += 1

    selected_den = float(total_selected_len or 1)
    analyzed = len(accessions)
    fill_ratio = total_selected_len / float(total_target_len or 1) if budget_type == "fixed" else ""
    return {
        "benchmark_group": "bioprior_family_split",
        "method": method,
        "method_label": method_label(method),
        "method_category": method_category(method),
        "budget_ratio": "" if budget_type == "auto" else fmt(budget),
        "budget_label": "auto" if budget_type == "auto" else budget_label(budget),
        "budget_type": budget_type,
        "split": split_name,
        "total_split_proteins": len(proteins),
        "analyzed_proteins": analyzed,
        "certified_proteins": certified if budget_type == "auto" else "",
        "abstained_proteins": abstained if budget_type == "auto" else "",
        "abstention_rate": fmt(abstained / float(analyzed or 1)) if budget_type == "auto" else "",
        "total_target_len": total_target_len if budget_type == "fixed" else "",
        "total_analyzed_length": total_protein_len,
        "total_selected_len": total_selected_len,
        "fill_ratio": fmt(fill_ratio) if fill_ratio != "" else "",
        "achieved_deletion_ratio": fmt(total_selected_len / float(total_protein_len or 1)),
        "selected_segment_count": agg["selected_segments"],
        "mean_segment_length": fmt(total_selected_len / float(agg["selected_segments"] or 1)),
        "protected_overlap_residues": agg["protected_overlap_residues"],
        "protected_overlap_rate": fmt(agg["protected_overlap_residues"] / selected_den),
        "shadow_overlap_residues": agg["shadow_overlap_residues"],
        "shadow_overlap_rate": fmt(agg["shadow_overlap_residues"] / selected_den),
        "closure_unfriendly_len": agg["closure_unfriendly_len"],
        "closure_unfriendly_rate": fmt(agg["closure_unfriendly_len"] / selected_den),
        "structural_core_risk": fmt(agg["structural_core_risk"]),
        "structural_core_risk_per_deleted": fmt(agg["structural_core_risk"] / selected_den),
        "mean_risk_upper_per_deleted": fmt(sum(lists["risk_upper_units"]) / selected_den),
        "mean_evidence_confidence_per_deleted": fmt(sum(lists["evidence_confidence_units"]) / selected_den),
        "mean_stage1_utility": fmt(mean(lists["stage1"])),
        "mean_final_bioprior_score": fmt(mean(lists["bioprior"])),
        "mean_compatibility_score": fmt(mean(lists["compatibility"])),
        "proteins_under_80_fill": agg["proteins_under_80_fill"] if budget_type == "fixed" else "",
        "proteins_with_any_protected_violation": agg["proteins_with_any_protected_violation"],
        "proteins_with_any_shadow_overlap": agg["proteins_with_any_shadow_overlap"],
        "proteins_with_any_closure_unfriendly": agg["proteins_with_any_closure_unfriendly"],
        "terminal_selected_rate": fmt(sum(lists["terminal_len"]) / selected_den),
        "low_plddt_selected_rate": fmt(sum(lists["low_plddt_len"]) / selected_den),
        "disorder_selected_rate": fmt(sum(lists["disorder_len"]) / selected_den),
        "top_sources": "; ".join("{}={}".format(k or "missing", v) for k, v in source_counts.most_common(5)),
        "source_path": source_path,
        "notes": notes,
    }


def summarize_scisor_rows(path, split_name):
    if not path or not os.path.exists(path):
        return []
    out = []
    for row in read_csv(path):
        selected_len = fnum(row.get("achieved_deletion_ratio")) * fnum(row.get("total_test_proteins"), 0.0)
        method = row.get("method", "")
        out.append(
            {
                "benchmark_group": "bioprior_family_split",
                "method": method,
                "method_label": method,
                "method_category": "external_generative_baseline",
                "budget_ratio": fmt(row.get("budget")),
                "budget_label": row.get("budget_label") or budget_label(row.get("budget")),
                "budget_type": "fixed",
                "split": split_name,
                "total_split_proteins": row.get("total_test_proteins", ""),
                "analyzed_proteins": row.get("analyzed_proteins", ""),
                "certified_proteins": "",
                "abstained_proteins": "",
                "abstention_rate": "",
                "total_target_len": "",
                "total_analyzed_length": "",
                "total_selected_len": "",
                "fill_ratio": fmt(row.get("fill_ratio")),
                "achieved_deletion_ratio": fmt(row.get("achieved_deletion_ratio")),
                "selected_segment_count": row.get("selected_segment_count", ""),
                "mean_segment_length": fmt(row.get("mean_segment_length")),
                "protected_overlap_residues": row.get("protected_overlap_residues", ""),
                "protected_overlap_rate": fmt(row.get("protected_overlap_rate")),
                "shadow_overlap_residues": row.get("shadow_overlap_residues", ""),
                "shadow_overlap_rate": fmt(row.get("shadow_overlap_rate")),
                "closure_unfriendly_len": row.get("closure_unfriendly_len", ""),
                "closure_unfriendly_rate": fmt(row.get("closure_unfriendly_rate")),
                "structural_core_risk": "",
                "structural_core_risk_per_deleted": "",
                "mean_risk_upper_per_deleted": "",
                "mean_evidence_confidence_per_deleted": "",
                "mean_stage1_utility": "",
                "mean_final_bioprior_score": "",
                "mean_compatibility_score": "",
                "proteins_under_80_fill": row.get("proteins_under_80_fill", ""),
                "proteins_with_any_protected_violation": row.get("proteins_with_any_protected_violation", ""),
                "proteins_with_any_shadow_overlap": row.get("proteins_with_any_shadow_overlap", ""),
                "proteins_with_any_closure_unfriendly": row.get("proteins_with_any_closure_unfriendly", ""),
                "terminal_selected_rate": "",
                "low_plddt_selected_rate": "",
                "disorder_selected_rate": "",
                "top_sources": "",
                "source_path": row.get("run_dir", path),
                "notes": "SCISOR summary; selected_len is not recomputed in this suite. ratio_marker={:.6f}".format(selected_len),
            }
        )
    return out


def read_v2_selected(path, proteins):
    if not path or not os.path.exists(path):
        return {}
    allowed = set(proteins)
    grouped = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            accession = raw.get("accession") or raw.get("protein_id")
            if accession not in allowed:
                continue
            row = annotate_segment(raw)
            grouped[(raw.get("setting"), fnum(raw.get("budget_ratio")), accession)].append(row)
    return grouped


def has_v2_budget_setting(grouped, setting, budget, segments):
    for accession in segments:
        if grouped.get((setting, budget, accession)):
            return True
    return False


def read_certified_selected(path, proteins):
    if not path or not os.path.exists(path):
        return defaultdict(list)
    allowed = set(proteins)
    grouped = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            accession = raw.get("accession") or raw.get("protein_id")
            if accession not in allowed:
                continue
            row = annotate_segment(raw)
            grouped[(raw.get("auto_profile"), accession)].append(row)
    return grouped


def read_certified_selection(path, proteins):
    if not path or not os.path.exists(path):
        return {}, {}
    status = {}
    lengths = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if accession not in proteins:
                continue
            profile = row.get("auto_profile")
            status[(profile, accession)] = row.get("selection_status", "")
            lengths[accession] = {"length": inum(row.get("protein_length"), proteins[accession]["length"])}
    return status, lengths


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "benchmark_group",
        "method",
        "method_label",
        "method_category",
        "budget_ratio",
        "budget_label",
        "budget_type",
        "split",
        "total_split_proteins",
        "analyzed_proteins",
        "certified_proteins",
        "abstained_proteins",
        "abstention_rate",
        "total_target_len",
        "total_analyzed_length",
        "total_selected_len",
        "fill_ratio",
        "achieved_deletion_ratio",
        "selected_segment_count",
        "mean_segment_length",
        "protected_overlap_residues",
        "protected_overlap_rate",
        "shadow_overlap_residues",
        "shadow_overlap_rate",
        "closure_unfriendly_len",
        "closure_unfriendly_rate",
        "structural_core_risk",
        "structural_core_risk_per_deleted",
        "mean_risk_upper_per_deleted",
        "mean_evidence_confidence_per_deleted",
        "mean_stage1_utility",
        "mean_final_bioprior_score",
        "mean_compatibility_score",
        "proteins_under_80_fill",
        "proteins_with_any_protected_violation",
        "proteins_with_any_shadow_overlap",
        "proteins_with_any_closure_unfriendly",
        "terminal_selected_rate",
        "low_plddt_selected_rate",
        "disorder_selected_rate",
        "top_sources",
        "source_path",
        "notes",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, args, candidate_rows, candidate_proteins, fields):
    ensure_parent(path)
    fixed = [row for row in rows if row["budget_type"] == "fixed"]
    auto = [row for row in rows if row["budget_type"] == "auto"]
    focus_methods = [
        "random_candidate",
        "terminal_trim",
        "low_plddt_first",
        "disorder_first",
        "surface_loop_first",
        "compatibility_greedy_protected",
        "utility_greedy_protected",
        "strict_safe_greedy",
        "biodel_v2_safe",
        "biodel_v2_balanced",
        "SCISOR hardmask+shadow02",
    ]
    with open(path, "w") as handle:
        handle.write("# Deletion Baseline Suite\n\n")
        handle.write("- Split CSV: `{}`\n".format(args.split_csv))
        handle.write("- Segment CSV: `{}`\n".format(args.segments_csv))
        handle.write("- Candidate proteins with segments: {}\n".format(candidate_proteins))
        handle.write("- Candidate rows: {}\n".format(candidate_rows))
        handle.write("- Candidate fields include compatibility_score: {}\n\n".format("compatibility_score" in fields))
        handle.write("## Fixed-Budget Main Table\n\n")
        handle.write("| Budget | Method | Fill | Delete ratio | Protected | Shadow rate | Closure rate | Struct risk | Under-80 |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for budget in sorted({row["budget_ratio"] for row in fixed if row["budget_ratio"]}):
            budget_rows = [row for row in fixed if row["budget_ratio"] == budget]
            by_method = {row["method"]: row for row in budget_rows}
            for method in focus_methods:
                row = by_method.get(method)
                if not row:
                    continue
                handle.write(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                        row["budget_label"],
                        row["method_label"],
                        row["fill_ratio"],
                        row["achieved_deletion_ratio"],
                        row["protected_overlap_residues"],
                        row["shadow_overlap_rate"],
                        row["closure_unfriendly_rate"],
                        row["structural_core_risk_per_deleted"],
                        row["proteins_under_80_fill"],
                    )
                )
        handle.write("\n## Auto-Budget Certified Frontiers\n\n")
        handle.write("| Method | Certified | Abstained | Delete ratio | Risk upper | Shadow rate | Closure rate | Struct risk |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in auto:
            handle.write(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                    row["method_label"],
                    row["certified_proteins"],
                    row["abstained_proteins"],
                    row["achieved_deletion_ratio"],
                    row["mean_risk_upper_per_deleted"],
                    row["shadow_overlap_rate"],
                    row["closure_unfriendly_rate"],
                    row["structural_core_risk_per_deleted"],
                )
            )
        handle.write("\n## Interpretation\n\n")
        handle.write("- Random, terminal, low-pLDDT, disorder, surface-loop, and low-contact baselines test whether simple biological heuristics are enough.\n")
        handle.write("- Stage-1 and learned-compatibility greedy baselines test whether a trainable score alone can replace the planner.\n")
        handle.write("- Strict-safe greedy tests the opposite failure mode: safe but often unable to fill high deletion budgets.\n")
        handle.write("- BioDel v2 and BioDel-Cert rows test interval planning, risk constraints, automatic budget selection, and abstention.\n")
        handle.write("- SCISOR rows are imported from the completed family-held-out GPU baseline summary when present.\n\n")
        handle.write("BIODEL_DELETION_BASELINE_SUITE_PASS\n")


def run(args):
    proteins = read_split_proteins(args.split_csv)
    segments, fields, candidate_rows = read_segments(args.segments_csv, proteins)
    candidate_proteins = len(segments)
    rows = []
    methods = [
        "random_candidate",
        "terminal_trim",
        "low_plddt_first",
        "disorder_first",
        "surface_loop_first",
        "low_contact_first",
        "utility_greedy_protected",
        "bioprior_greedy_protected",
        "risk_certified_greedy_protected",
        "compatibility_greedy_protected",
        "strict_safe_greedy",
        "compatibility_strict_safe",
    ]
    budgets = [float(item.strip()) for item in args.budgets.split(",") if item.strip()]
    for budget in budgets:
        for method in methods:
            selected_by_accession = {}
            for accession, seg_rows in segments.items():
                selected_by_accession[accession] = select_fixed_budget(seg_rows, budget, method, args, accession)
            rows.append(
                summarize_selection(
                    method,
                    budget,
                    proteins,
                    selected_by_accession,
                    args.split_name,
                    budget_type="fixed",
                    source_path=args.segments_csv,
                )
            )

    v2_selected = read_v2_selected(args.v2_selected_csv, proteins)
    if v2_selected:
        for budget in budgets:
            for setting in ("safe", "balanced", "aggressive"):
                if not has_v2_budget_setting(v2_selected, setting, budget, segments):
                    continue
                method = "biodel_v2_{}".format(setting)
                selected_by_accession = {
                    accession: v2_selected.get((setting, budget, accession), [])
                    for accession in segments
                }
                rows.append(
                    summarize_selection(
                        method,
                        budget,
                        proteins,
                        selected_by_accession,
                        args.split_name,
                        budget_type="fixed",
                        source_path=args.v2_selected_csv,
                    )
                )

    for prefix, selection_csv, selected_csv in [
        ("biodel_cert_frontier", args.certified_selection_csv, args.certified_selected_segments_csv),
        ("biodel_cert_learned_frontier", args.learned_certified_selection_csv, args.learned_certified_selected_segments_csv),
    ]:
        selected = read_certified_selected(selected_csv, proteins)
        status, _ = read_certified_selection(selection_csv, proteins)
        if not selected and not status:
            continue
        profiles = sorted({profile for profile, _ in set(selected) | set(status)})
        for profile in profiles:
            selected_by_accession = {
                accession: selected.get((profile, accession), [])
                for accession in segments
            }
            status_by_accession = {
                accession: status.get((profile, accession), "")
                for accession in segments
            }
            rows.append(
                summarize_selection(
                    "{}_{}".format(prefix, profile),
                    "auto",
                    proteins,
                    selected_by_accession,
                    args.split_name,
                    budget_type="auto",
                    status_by_accession=status_by_accession,
                    source_path=selected_csv,
                )
            )

    rows.extend(summarize_scisor_rows(args.scisor_summary_csv, args.split_name))
    write_csv(args.out_csv, rows)
    write_report(args.out_report, rows, args, candidate_rows, candidate_proteins, fields)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Run deletion baseline suite.")
    parser.add_argument("--segments_csv", default="results/deletion_compatibility_model_v1/bioprior_10k_segments_compatibility_scored.csv")
    parser.add_argument("--split_csv", default="data/processed/bioprior_10k_family_splits/test.csv")
    parser.add_argument("--split_name", default="family_test")
    parser.add_argument("--v2_selected_csv", default="results/biodel_planner/family_split/bioprior_10k_test_v2_selected_segments.csv")
    parser.add_argument("--certified_selection_csv", default="results/biodel_planner/family_split/certified_frontier_test_protein_selection.csv")
    parser.add_argument("--certified_selected_segments_csv", default="results/biodel_planner/family_split/certified_frontier_test_selected_segments.csv")
    parser.add_argument("--learned_certified_selection_csv", default="results/deletion_compatibility_model_v1/family_certified_selection_utilityonly.csv")
    parser.add_argument("--learned_certified_selected_segments_csv", default="results/deletion_compatibility_model_v1/family_certified_selected_segments_utilityonly.csv")
    parser.add_argument("--scisor_summary_csv", default="results/scisor_bioprior_10k_family_test/scisor_bioprior10k_family_test_baseline_summary.csv")
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--max_candidates_per_protein", type=int, default=220)
    parser.add_argument("--allow_overshoot_fraction", type=float, default=0.02)
    parser.add_argument("--strict_max_structural_core_risk", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out_csv", default="results/biodel_planner/deletion_baseline_suite_summary.csv")
    parser.add_argument("--out_report", default="results/biodel_planner/DELETION_BASELINE_SUITE_REPORT.md")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
