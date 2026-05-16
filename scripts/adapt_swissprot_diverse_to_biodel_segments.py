#!/usr/bin/env python3
"""Convert Swiss-Prot diverse segment candidates to BioDel planner schema."""

import argparse
import csv
import os
from collections import Counter


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


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def read_metadata(path):
    out = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            if accession:
                out[accession] = row
    return out


def terminal_tail_score(row):
    if bval(row.get("is_terminal_deletion")):
        return 1.0
    return 0.0


def structural_risk(row):
    contact = clamp(fnum(row.get("mean_contact_density_8A")) / 20.0)
    plddt = clamp(fnum(row.get("mean_plddt")) / 100.0)
    return clamp(0.5 * contact + 0.5 * plddt)


def closure_type(row):
    if bval(row.get("is_terminal_deletion")):
        return "terminal"
    return "internal_friendly" if bval(row.get("closure_friendly")) else "internal_unfriendly"


def final_bioprior(row):
    motif_overlap = fnum(row.get("motif_overlap_fraction"))
    shadow = fnum(row.get("shadow_overlap_fraction"))
    struct = structural_risk(row)
    closure_penalty = 0.0 if closure_type(row) in ("terminal", "internal_friendly") else 1.0
    flexible = 1.0 - clamp(fnum(row.get("mean_plddt")) / 100.0)
    terminal = terminal_tail_score(row)
    return (
        0.8 * terminal
        + 0.5 * flexible
        + 0.5 * (1.0 - struct)
        + 0.4 * (1.0 - shadow)
        - 1.2 * motif_overlap
        - 0.5 * closure_penalty
    )


def convert_row(row, meta):
    accession = row["protein_id"]
    start = inum(row["start"])
    end = inum(row["end"])
    seg_len = inum(row["seg_len"])
    protected_overlap = inum(row.get("num_motif_overlap"))
    shadow_overlap = inum(row.get("num_shadow_overlap"))
    ctype = closure_type(row)
    closure_friendly = ctype == "terminal" or ctype == "internal_friendly"
    struct = structural_risk(row)
    motif_shadow = fnum(row.get("shadow_overlap_fraction"))
    functional = 1.0 if protected_overlap > 0 else 0.0
    terminal = terminal_tail_score(row)
    flexible = 1.0 - clamp(fnum(row.get("mean_plddt")) / 100.0)
    out = {
        "accession": accession,
        "protein_length": meta.get("length") or "",
        "seg_start": start,
        "seg_end": end,
        "seg_len": seg_len,
        "proposal_source": "swissprot_diverse_segment_candidate",
        "biological_rationale": "Swiss-Prot diverse held-out stress candidate",
        "n_protected_overlap": protected_overlap,
        "protected_overlap_fraction": row.get("motif_overlap_fraction", ""),
        "n_shadow_overlap": shadow_overlap,
        "shadow_overlap_fraction": row.get("shadow_overlap_fraction", ""),
        "min_distance_to_protected": row.get("min_distance_to_motif", ""),
        "mean_distance_to_protected": row.get("mean_distance_to_motif", ""),
        "mean_contact_density_8A": row.get("mean_contact_density_8A", ""),
        "max_contact_density_8A": row.get("max_contact_density_8A", ""),
        "mean_motif_contact_count_8A": row.get("num_motif_contact_overlap", ""),
        "max_motif_contact_count_8A": row.get("num_motif_contact_overlap", ""),
        "mean_pLDDT": row.get("mean_plddt", ""),
        "min_pLDDT": row.get("min_plddt", ""),
        "low_pLDDT_fraction": 1.0 if fnum(row.get("mean_plddt")) < 50.0 else 0.0,
        "high_pLDDT_fraction": 1.0 if fnum(row.get("mean_plddt")) >= 70.0 else 0.0,
        "terminal_overlap_fraction": 1.0 if terminal else 0.0,
        "boundary_ca_distance": row.get("boundary_ca_distance", ""),
        "closure_type": ctype,
        "closure_friendly_8A": str(closure_friendly),
        "terminal_tail_score": "{:.6f}".format(terminal),
        "surface_flexible_loop_score": "{:.6f}".format(flexible),
        "disorder_like_score": "{:.6f}".format(flexible),
        "linker_compressibility_score": "{:.6f}".format(max(terminal, flexible)),
        "functional_core_risk_score": "{:.6f}".format(functional),
        "structural_core_risk_score": "{:.6f}".format(struct),
        "motif_shadow_risk_score": "{:.6f}".format(motif_shadow),
        "geometric_closure_score": "{:.6f}".format(0.0 if closure_friendly else 1.0),
        "final_bioprior_score": "{:.6f}".format(final_bioprior(row)),
        "hard_reject": str(protected_overlap > 0),
        "reject_reason": "protected_motif_overlap" if protected_overlap > 0 else "",
    }
    return out


def write_rows(path, rows):
    ensure_parent(path)
    fields = [
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
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, args, rows):
    stats = Counter()
    for row in rows:
        stats["rows"] += 1
        stats["proteins"] += 0
        if row["hard_reject"] == "True":
            stats["hard_reject_rows"] += 1
        if row["closure_type"] == "internal_unfriendly":
            stats["closure_unfriendly_rows"] += 1
    proteins = len(set(row["accession"] for row in rows))
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("Swiss-Prot diverse BioDel segment adaptation summary\n\n")
        handle.write("input_segments: {}\n".format(args.input_segments_csv))
        handle.write("metadata_csv: {}\n".format(args.metadata_csv))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("proteins: {}\n".format(proteins))
        handle.write("rows: {}\n".format(len(rows)))
        handle.write("hard_reject_rows: {}\n".format(stats["hard_reject_rows"]))
        handle.write("closure_unfriendly_rows: {}\n".format(stats["closure_unfriendly_rows"]))
        handle.write("\nSWISSPROT_DIVERSE_BIODEL_ADAPT_PASS\n")


def run(args):
    metadata = read_metadata(args.metadata_csv)
    rows = []
    with open(args.input_segments_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            meta = metadata.get(row["protein_id"])
            if not meta:
                continue
            rows.append(convert_row(row, meta))
    write_rows(args.out_csv, rows)
    write_summary(args.summary_txt, args, rows)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


def parse_args():
    parser = argparse.ArgumentParser(description="Adapt Swiss-Prot diverse candidates to BioDel segment schema.")
    parser.add_argument("--input_segments_csv", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
