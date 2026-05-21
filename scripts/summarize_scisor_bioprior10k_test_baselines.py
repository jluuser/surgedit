#!/usr/bin/env python3
"""Summarize SCISOR-family BioPrior-10K held-out test baselines."""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def to_int(value, default=0):
    if value is None or value == "":
        return default
    return int(float(value))


def parse_positions(text):
    if not text:
        return set()
    return {int(token) for token in text.split(";") if token.strip()}


def read_test_metadata(path):
    proteins = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            proteins[accession] = {
                "length": to_int(row.get("length") or len(row.get("sequence", ""))),
                "protected": parse_positions(row.get("protected_positions", "")),
            }
    return proteins


def read_residue_priors(path):
    priors = defaultdict(lambda: {"shadow": set(), "motif_contact": set()})
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            pos = to_int(row.get("residue_index_0based") or row.get("position"))
            if row.get("is_motif_shadow_8A") == "True":
                priors[accession]["shadow"].add(pos)
            if to_int(row.get("motif_contact_count_8A"), 0) > 0:
                priors[accession]["motif_contact"].add(pos)
    return dict(priors)


def parse_pdb_ca_coords(path):
    coords = []
    seen = set()
    with open(path, "rb") as handle:
        for raw_line in handle:
            if not raw_line.startswith(b"ATOM"):
                continue
            line = raw_line.decode("ascii", "ignore")
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in (" ", "A"):
                continue
            key = (line[21:22], line[22:26].strip(), line[26:27])
            if key in seen:
                continue
            seen.add(key)
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def deleted_segments(positions):
    positions = sorted(set(positions))
    if not positions:
        return []
    segments = []
    start = prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            segments.append((start, prev))
            start = prev = pos
    segments.append((start, prev))
    return segments


def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def closure_unfriendly_len(accession, length, positions, coord_cache, structure_dir, cutoff):
    if accession not in coord_cache:
        pdb_path = os.path.join(structure_dir, accession + ".pdb")
        if os.path.exists(pdb_path):
            coords = parse_pdb_ca_coords(pdb_path)
            coord_cache[accession] = coords if len(coords) == length else None
        else:
            coord_cache[accession] = None
    coords = coord_cache[accession]
    if coords is None:
        return 0, 0, 0
    unfriendly_len = 0
    unfriendly_segments = 0
    total_segments = 0
    for start, end in deleted_segments(positions):
        total_segments += 1
        if start == 0 or end == length - 1:
            continue
        left = start - 1
        right = end + 1
        if left < 0 or right >= len(coords):
            continue
        if dist(coords[left], coords[right]) > cutoff:
            unfriendly_segments += 1
            unfriendly_len += end - start + 1
    return unfriendly_len, unfriendly_segments, total_segments


def get_positions(entry):
    for key in ("deleted_positions", "deletion_positions_zero_based", "deleted_positions_zero_based"):
        if key in entry:
            return entry[key]
    return []


def validation_all_pass(path):
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        return any(line.strip() == "ALL_PASS" for line in handle)


def summarize_run(run_dir, method, budget, proteins, priors, structure_dir, closure_cutoff):
    with open(os.path.join(run_dir, "deletions.json")) as handle:
        entries = json.load(handle)
    by_header = {entry["header"].split("|", 1)[0]: entry for entry in entries}
    coord_cache = {}
    agg = Counter()
    total_selected = 0
    total_target = 0
    total_len = sum(item["length"] for item in proteins.values())
    analyzed = 0
    for accession, meta in proteins.items():
        entry = by_header.get(accession)
        if entry is None:
            agg["missing_entries"] += 1
            continue
        length = meta["length"]
        positions = set(get_positions(entry))
        deleted_count = len(positions)
        target = int(math.ceil(length * budget / 100.0))
        fill = deleted_count / float(target or 1)
        protected_deleted = positions & meta["protected"]
        prior = priors.get(accession, {"shadow": set(), "motif_contact": set()})
        shadow_deleted = positions & prior["shadow"]
        motif_contact_deleted = positions & prior["motif_contact"]
        closure_len, closure_segments, total_segments = closure_unfriendly_len(
            accession, length, positions, coord_cache, structure_dir, closure_cutoff
        )
        analyzed += 1
        total_selected += deleted_count
        total_target += target
        agg["selected_segment_count"] += total_segments
        agg["protected_overlap_residues"] += len(protected_deleted)
        agg["shadow_overlap_residues"] += len(shadow_deleted)
        agg["motif_contact_overlap_residues"] += len(motif_contact_deleted)
        agg["closure_unfriendly_len"] += closure_len
        agg["closure_unfriendly_segments"] += closure_segments
        agg["proteins_under_80_fill"] += 1 if fill < 0.8 else 0
        agg["proteins_with_any_protected_violation"] += 1 if protected_deleted else 0
        agg["proteins_with_any_shadow_overlap"] += 1 if shadow_deleted else 0
        agg["proteins_with_any_closure_unfriendly"] += 1 if closure_len > 0 else 0
    denom = float(total_selected or 1)
    return {
        "method": method,
        "budget": budget / 100.0,
        "budget_label": "{}%".format(budget),
        "split": "test",
        "total_test_proteins": len(proteins),
        "analyzed_proteins": analyzed,
        "fill_ratio": total_selected / float(total_target or 1),
        "achieved_deletion_ratio": total_selected / float(total_len or 1),
        "protected_overlap_residues": agg["protected_overlap_residues"],
        "protected_overlap_rate": agg["protected_overlap_residues"] / denom,
        "shadow_overlap_residues": agg["shadow_overlap_residues"],
        "shadow_overlap_rate": agg["shadow_overlap_residues"] / denom,
        "motif_contact_overlap_residues": agg["motif_contact_overlap_residues"],
        "motif_contact_overlap_rate": agg["motif_contact_overlap_residues"] / denom,
        "closure_unfriendly_len": agg["closure_unfriendly_len"],
        "closure_unfriendly_rate": agg["closure_unfriendly_len"] / denom,
        "selected_segment_count": agg["selected_segment_count"],
        "mean_segment_length": total_selected / float(agg["selected_segment_count"] or 1),
        "proteins_under_80_fill": agg["proteins_under_80_fill"],
        "proteins_with_any_protected_violation": agg["proteins_with_any_protected_violation"],
        "proteins_with_any_shadow_overlap": agg["proteins_with_any_shadow_overlap"],
        "proteins_with_any_closure_unfriendly": agg["proteins_with_any_closure_unfriendly"],
        "validation_ALL_PASS": validation_all_pass(os.path.join(run_dir, "validation_report.txt")),
        "missing_entries": agg["missing_entries"],
        "run_dir": run_dir,
    }


def write_csv(path, rows):
    ensure_parent(path)
    fields = [
        "method",
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
        "motif_contact_overlap_residues",
        "motif_contact_overlap_rate",
        "closure_unfriendly_len",
        "closure_unfriendly_rate",
        "selected_segment_count",
        "mean_segment_length",
        "proteins_under_80_fill",
        "proteins_with_any_protected_violation",
        "proteins_with_any_shadow_overlap",
        "proteins_with_any_closure_unfriendly",
        "validation_ALL_PASS",
        "missing_entries",
        "run_dir",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, expected_count):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("SCISOR-family BioPrior-10K test baseline summary\n\n")
        handle.write("expected_runs: {}\n".format(expected_count))
        handle.write("observed_runs: {}\n\n".format(len(rows)))
        handle.write("method,budget,fill,protected,shadow_rate,closure_rate,validation\n")
        for row in rows:
            handle.write(
                "{},{},{:.4f},{},{:.4f},{:.4f},{}\n".format(
                    row["method"],
                    row["budget_label"],
                    row["fill_ratio"],
                    row["protected_overlap_residues"],
                    row["shadow_overlap_rate"],
                    row["closure_unfriendly_rate"],
                    row["validation_ALL_PASS"],
                )
            )
        all_pass = (
            len(rows) == expected_count
            and expected_count > 0
            and all(row["validation_ALL_PASS"] for row in rows)
        )
        if len(rows) != expected_count:
            handle.write(
                "\nWARNING: observed {} SCISOR runs, expected {}.\n".format(
                    len(rows), expected_count
                )
            )
        handle.write("\n{}\n".format("SCISOR_BIOPRIOR10K_BASELINES_PASS" if all_pass else "SCISOR_BIOPRIOR10K_BASELINES_WARN"))


def main():
    parser = argparse.ArgumentParser(description="Summarize SCISOR BioPrior-10K test baselines.")
    parser.add_argument("--test_csv", default="data/processed/bioprior_10k_splits/test.csv")
    parser.add_argument("--residue_priors", default="data/features/bioprior_10k_test_residue_biopriors.csv")
    parser.add_argument("--structure_dir", default="data/structures/afdb_bioprior_10k")
    parser.add_argument("--scisor_root", default="results/scisor_bioprior_10k_test")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--closure_cutoff", type=float, default=8.0)
    parser.add_argument("--budgets", default="10,20,30")
    args = parser.parse_args()

    proteins = read_test_metadata(args.test_csv)
    priors = read_residue_priors(args.residue_priors)
    rows = []
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    method_specs = [
        ("SCISOR no-mask", "nomask"),
        ("SCISOR hardmask", "hardmask"),
        ("SCISOR hardmask+shadow02", "hardmask_shadow02"),
    ]
    for budget in budgets:
        for method, suffix in [
            ("SCISOR no-mask", "nomask"),
            ("SCISOR hardmask", "hardmask"),
            ("SCISOR hardmask+shadow02", "hardmask_shadow02"),
        ]:
            run_dir = os.path.join(args.scisor_root, "run{:02d}_{}".format(budget, suffix))
            if not os.path.exists(os.path.join(run_dir, "deletions.json")):
                continue
            rows.append(summarize_run(run_dir, method, budget, proteins, priors, args.structure_dir, args.closure_cutoff))
    write_csv(args.out_csv, rows)
    write_report(args.out_report, rows, expected_count=len(budgets) * len(method_specs))
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
