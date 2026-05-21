#!/usr/bin/env python3
"""Annotate learned Stage-1 interval proposals with BioPrior evidence.

The SCISOR-style Stage-1 proposer emits contiguous intervals from a learned
deletion posterior.  Those intervals are not yet safe deletion candidates: they
must be re-scored against functional annotations, motif neighborhoods,
structure-core features, and boundary closure geometry before certified
planning.  This script performs that evidence join and writes the same segment
schema used by the existing BioDel planner.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from bioprior.utils import load_bioprior_config  # noqa: E402
from build_core_1k_bioprior_segments import (  # noqa: E402
    aggregate_segment,
    canonical_residue_features,
    parse_pdb_ca_coords,
    read_residue_csv,
)


BASE_FIELDS = [
    "accession",
    "protein_length",
    "seg_start",
    "seg_end",
    "seg_len",
    "proposal_source",
    "biological_rationale",
    "n_protected_overlap",
    "protected_overlap_fraction",
    "n_shadow_overlap",
    "shadow_overlap_fraction",
    "min_distance_to_protected",
    "mean_distance_to_protected",
    "mean_contact_density_8A",
    "max_contact_density_8A",
    "mean_motif_contact_count_8A",
    "max_motif_contact_count_8A",
    "mean_pLDDT",
    "min_pLDDT",
    "low_pLDDT_fraction",
    "high_pLDDT_fraction",
    "terminal_overlap_fraction",
    "boundary_ca_distance",
    "closure_type",
    "closure_friendly_8A",
    "terminal_tail_score",
    "surface_flexible_loop_score",
    "disorder_like_score",
    "linker_compressibility_score",
    "functional_core_risk_score",
    "structural_core_risk_score",
    "motif_shadow_risk_score",
    "geometric_closure_score",
    "final_bioprior_score",
    "hard_reject",
    "reject_reason",
]


STAGE1_FIELDS = [
    "stage1_interval_rank",
    "stage1_posterior_mean",
    "stage1_posterior_sum",
    "stage1_posterior_max",
    "stage1_utility_score",
    "stage1_utility_budget",
    "stage1_scoring_status",
]


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        accession = row.get("accession") or row.get("protein_id")
        if accession:
            grouped[accession].append(row)
    return dict(grouped)


def stage1_extras(row):
    return {field: row.get(field, "") for field in STAGE1_FIELDS}


def fallback_row(row, reason):
    out = {field: row.get(field, "") for field in BASE_FIELDS}
    out.update(stage1_extras(row))
    out["bioprior_annotation_status"] = reason
    return out


def segment_from_row(row):
    start = safe_int(row.get("seg_start"))
    end = safe_int(row.get("seg_end"))
    return {
        "seg_start": start,
        "seg_end": end,
        "seg_len": end - start + 1,
        "proposal_source": row.get("proposal_source") or "stage1_scisor_style_posterior",
        "biological_rationale": row.get("biological_rationale")
        or "learned SCISOR-style deletion posterior interval proposal",
    }


def annotate_accession(accession, rows, residue_rows, structure_dir, config, args, coord_cache):
    out_rows = []
    if not residue_rows:
        return [fallback_row(row, "no_residue_features") for row in rows]

    residue_rows = sorted(residue_rows, key=lambda row: safe_int(row["residue_index_0based"]))
    protein_length = safe_int(rows[0].get("protein_length"), len(residue_rows))
    if len(residue_rows) != protein_length:
        return [fallback_row(row, "residue_length_mismatch") for row in rows]

    pdb_path = os.path.join(structure_dir, accession + ".pdb")
    if not os.path.exists(pdb_path):
        return [fallback_row(row, "missing_pdb") for row in rows]
    if accession not in coord_cache:
        coord_cache[accession] = parse_pdb_ca_coords(pdb_path)
    coords = coord_cache[accession]
    if len(coords) != protein_length:
        return [fallback_row(row, "coord_length_mismatch") for row in rows]

    features = canonical_residue_features(residue_rows)
    for row in rows:
        start = safe_int(row.get("seg_start"))
        end = safe_int(row.get("seg_end"))
        if start < 0 or end >= protein_length or start > end:
            out_rows.append(fallback_row(row, "segment_oob"))
            continue
        segment = segment_from_row(row)
        annotated = aggregate_segment(accession, protein_length, segment, features, coords, config)
        annotated.update(stage1_extras(row))
        annotated["bioprior_annotation_status"] = "success"
        out_rows.append(annotated)
    return out_rows


def write_summary(path, args, input_rows, output_rows):
    status_counts = Counter(row.get("bioprior_annotation_status", "") for row in output_rows)
    source_counts = Counter(row.get("proposal_source", "") for row in output_rows)
    hard_reject = sum(1 for row in output_rows if row.get("hard_reject") == "True")
    annotated = status_counts.get("success", 0)
    with open(path, "w") as handle:
        handle.write("Stage-1 interval BioPrior annotation summary\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("residue_csv: {}\n".format(args.residue_csv))
        handle.write("structure_dir: {}\n".format(args.structure_dir))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("input_rows: {}\n".format(len(input_rows)))
        handle.write("output_rows: {}\n".format(len(output_rows)))
        handle.write("annotated_rows: {}\n".format(annotated))
        handle.write("annotation_status_counts: {}\n".format(dict(status_counts)))
        handle.write("proposal_source_counts: {}\n".format(dict(source_counts)))
        handle.write("hard_reject_rows: {}\n".format(hard_reject))
        handle.write("\n{}\n".format("STAGE1_INTERVAL_BIOPRIOR_ANNOTATION_PASS" if output_rows else "STAGE1_INTERVAL_BIOPRIOR_ANNOTATION_WARN"))


def run(args):
    input_rows, _ = read_rows(args.input_csv)
    grouped = group_rows(input_rows)
    selected_accessions = sorted(grouped)
    if args.limit_proteins is not None:
        selected_accessions = selected_accessions[: args.limit_proteins]
        selected = set(selected_accessions)
        input_rows = [row for row in input_rows if (row.get("accession") or row.get("protein_id")) in selected]
        grouped = group_rows(input_rows)

    config = load_bioprior_config(args.config)
    residue_groups = read_residue_csv(args.residue_csv, selected_accessions)
    coord_cache = {}
    output_rows = []
    for accession in selected_accessions:
        rows = grouped.get(accession, [])
        if not rows:
            continue
        output_rows.extend(
            annotate_accession(
                accession,
                rows,
                residue_groups.get(accession),
                args.structure_dir,
                config,
                args,
                coord_cache,
            )
        )

    if args.drop_unannotated:
        output_rows = [row for row in output_rows if row.get("bioprior_annotation_status") == "success"]

    fields = BASE_FIELDS + [field for field in STAGE1_FIELDS if field not in BASE_FIELDS] + ["bioprior_annotation_status"]
    ensure_parent(args.out_csv)
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    ensure_parent(args.summary_txt)
    write_summary(args.summary_txt, args, input_rows, output_rows)
    print("Input rows: {}".format(len(input_rows)))
    print("Output rows: {}".format(len(output_rows)))
    print("Annotated rows: {}".format(sum(1 for row in output_rows if row.get("bioprior_annotation_status") == "success")))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate Stage-1 learned interval proposals with BioPrior evidence.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--residue_csv", required=True)
    parser.add_argument("--structure_dir", required=True)
    parser.add_argument("--config", default="configs/bioprior_v1.yaml")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--limit_proteins", type=int, default=None)
    parser.add_argument("--drop_unannotated", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
