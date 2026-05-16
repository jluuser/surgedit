#!/usr/bin/env python3
"""Annotate BioDel candidate segments with risk certificates.

The certificate converts point BioPrior risks into a conservative interval:

    risk_lower <= risk_point <= risk_upper

Missing evidence widens the interval. This gives the planner a principled
alternative to treating unknown annotation/structure as safe.
"""

import argparse
import csv
import os
from collections import Counter


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def safe_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def has_field(row, field):
    return field in row and row.get(field) not in ("", None)


def has_annotation(row):
    fields = [
        "n_protected_overlap",
        "protected_overlap_fraction",
        "n_shadow_overlap",
        "shadow_overlap_fraction",
        "functional_core_risk_score",
        "motif_shadow_risk_score",
    ]
    return any(has_field(row, field) for field in fields)


def has_structure(row):
    fields = [
        "mean_pLDDT",
        "min_pLDDT",
        "mean_contact_density_8A",
        "boundary_ca_distance",
        "closure_type",
        "structural_core_risk_score",
        "geometric_closure_score",
    ]
    return any(has_field(row, field) for field in fields)


def component_values(row):
    functional = safe_float(row.get("functional_core_risk_score"))
    if functional is None:
        protected_frac = safe_float(row.get("protected_overlap_fraction"))
        protected_count = safe_int(row.get("n_protected_overlap"))
        functional = 1.0 if protected_count > 0 else (protected_frac if protected_frac is not None else None)

    structural = safe_float(row.get("structural_core_risk_score"))

    motif = safe_float(row.get("motif_shadow_risk_score"))
    if motif is None:
        motif = safe_float(row.get("shadow_overlap_fraction"))

    closure = safe_float(row.get("geometric_closure_score"))
    if closure is None:
        closure_type = row.get("closure_type", "")
        closure_friendly = safe_bool(row.get("closure_friendly_8A"))
        if closure_type:
            closure = 0.0 if closure_type == "terminal" or closure_friendly else 1.0

    return {
        "functional": None if functional is None else clamp(functional),
        "structural": None if structural is None else clamp(structural),
        "motif_shadow": None if motif is None else clamp(motif),
        "closure": None if closure is None else clamp(closure),
    }


def evidence_level(has_ann, has_struct, has_stage1):
    if has_stage1 and has_ann and has_struct:
        return "annotation_structure"
    if has_stage1 and has_struct:
        return "structure_aware"
    if has_stage1 and has_ann:
        return "annotation_aware"
    if has_stage1:
        return "sequence_only"
    return "insufficient"


def annotate_row(row, args):
    has_ann = has_annotation(row)
    has_struct = has_structure(row)
    has_stage1 = row.get("stage1_scoring_status") == "success" or row.get("stage1_utility_score") not in ("", None)
    components = component_values(row)
    observed = [value for value in components.values() if value is not None]
    risk_point = sum(observed) / float(len(observed)) if observed else args.sequence_only_base_risk

    missing_annotation_penalty = 0.0 if has_ann else args.missing_annotation_penalty
    missing_structure_penalty = 0.0 if has_struct else args.missing_structure_penalty
    low_plddt_fraction = safe_float(row.get("low_pLDDT_fraction"), 0.0) if has_struct else 0.0
    low_plddt_penalty = clamp(low_plddt_fraction) * args.low_plddt_penalty
    hard_reject_penalty = args.hard_reject_penalty if safe_bool(row.get("hard_reject")) else 0.0

    uncertainty = (
        missing_annotation_penalty
        + missing_structure_penalty
        + low_plddt_penalty
        + hard_reject_penalty
    )
    confidence = clamp(1.0 - args.confidence_scale * uncertainty)
    lower_margin = args.lower_margin_scale * (1.0 - confidence)
    risk_lower = clamp(risk_point - lower_margin)
    risk_upper = clamp(risk_point + uncertainty)
    if safe_int(row.get("n_protected_overlap")) > 0 or safe_bool(row.get("hard_reject")):
        risk_upper = 1.0
        confidence = min(confidence, 0.40)

    level = evidence_level(has_ann, has_struct, has_stage1)
    if risk_upper >= args.abstain_risk_threshold and confidence <= args.low_confidence_threshold:
        status = "abstain_recommended"
    elif confidence <= args.low_confidence_threshold:
        status = "low_confidence"
    else:
        status = "certified"

    stage1 = safe_float(row.get("stage1_utility_score"), 0.0)
    bioprior = safe_float(row.get("final_bioprior_score"), 0.0)
    certified_score = stage1 + bioprior - args.certified_risk_weight * risk_upper

    out = dict(row)
    out.update(
        {
            "risk_point": "{:.6f}".format(risk_point),
            "risk_lower": "{:.6f}".format(risk_lower),
            "risk_upper": "{:.6f}".format(risk_upper),
            "risk_interval_width": "{:.6f}".format(risk_upper - risk_lower),
            "evidence_level": level,
            "evidence_confidence": "{:.6f}".format(confidence),
            "has_annotation_evidence": int(has_ann),
            "has_structure_evidence": int(has_struct),
            "missing_annotation_penalty": "{:.6f}".format(missing_annotation_penalty),
            "missing_structure_penalty": "{:.6f}".format(missing_structure_penalty),
            "low_plddt_uncertainty_penalty": "{:.6f}".format(low_plddt_penalty),
            "risk_certificate_status": status,
            "risk_upper_favorable_score": "{:.6f}".format(-risk_upper),
            "risk_certified_biodel_score": "{:.6f}".format(certified_score),
            "risk_certificate_components": ";".join(
                "{}={}".format(key, "" if value is None else "{:.6f}".format(value))
                for key, value in components.items()
            ),
        }
    )
    return out


def write_summary(path, args, rows):
    counts = Counter(row["risk_certificate_status"] for row in rows)
    levels = Counter(row["evidence_level"] for row in rows)
    uppers = [safe_float(row["risk_upper"], 0.0) for row in rows]
    widths = [safe_float(row["risk_interval_width"], 0.0) for row in rows]
    confidences = [safe_float(row["evidence_confidence"], 0.0) for row in rows]
    with open(path, "w") as handle:
        handle.write("BioDel risk certificate annotation summary\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("rows: {}\n".format(len(rows)))
        handle.write("status_counts: {}\n".format(dict(counts)))
        handle.write("evidence_level_counts: {}\n".format(dict(levels)))
        handle.write("mean_risk_upper: {:.6f}\n".format(sum(uppers) / float(len(uppers) or 1)))
        handle.write("mean_interval_width: {:.6f}\n".format(sum(widths) / float(len(widths) or 1)))
        handle.write("mean_evidence_confidence: {:.6f}\n".format(sum(confidences) / float(len(confidences) or 1)))
        handle.write("\nBIODEL_RISK_CERTIFICATE_PASS\n")


def run(args):
    with open(args.input_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [annotate_row(row, args) for row in reader]
    added = [
        "risk_point",
        "risk_lower",
        "risk_upper",
        "risk_interval_width",
        "evidence_level",
        "evidence_confidence",
        "has_annotation_evidence",
        "has_structure_evidence",
        "missing_annotation_penalty",
        "missing_structure_penalty",
        "low_plddt_uncertainty_penalty",
        "risk_certificate_status",
        "risk_upper_favorable_score",
        "risk_certified_biodel_score",
        "risk_certificate_components",
    ]
    ensure_parent(args.out_csv)
    out_fields = fields + [field for field in added if field not in fields]
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    ensure_parent(args.summary_txt)
    write_summary(args.summary_txt, args, rows)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate BioDel candidates with risk certificate intervals.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--missing_annotation_penalty", type=float, default=0.20)
    parser.add_argument("--missing_structure_penalty", type=float, default=0.35)
    parser.add_argument("--low_plddt_penalty", type=float, default=0.20)
    parser.add_argument("--hard_reject_penalty", type=float, default=0.30)
    parser.add_argument("--sequence_only_base_risk", type=float, default=0.50)
    parser.add_argument("--confidence_scale", type=float, default=0.80)
    parser.add_argument("--lower_margin_scale", type=float, default=0.25)
    parser.add_argument("--certified_risk_weight", type=float, default=1.0)
    parser.add_argument("--low_confidence_threshold", type=float, default=0.50)
    parser.add_argument("--abstain_risk_threshold", type=float, default=0.80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
