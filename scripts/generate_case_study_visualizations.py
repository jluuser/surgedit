#!/usr/bin/env python3
"""Generate case-study residue selections and PyMOL scripts."""

import argparse
import csv
import json
import os
from collections import defaultdict


RUNS = {
    10: {
        "hardmask": "results/scisor_swissprot_diverse/run10_hardmask/deletions.json",
        "shadow02": "results/scisor_swissprot_diverse/run10_hardmask_shadow02/deletions.json",
    },
    20: {
        "hardmask": "results/scisor_swissprot_diverse/run20_hardmask/deletions.json",
        "shadow02": "results/scisor_swissprot_diverse/run20_hardmask_shadow02/deletions.json",
    },
    30: {
        "hardmask": "results/scisor_swissprot_diverse/run30_hardmask/deletions.json",
        "shadow02": "results/scisor_swissprot_diverse/run30_hardmask_shadow02/deletions.json",
    },
}

PREFERRED_CANDIDATES = {"P22791", "F9VMT6", "C5C9D1", "B2K9Z2", "A9UTP3"}


def parse_positions(text):
    if not text:
        return set()
    return set(int(x) for x in text.split(";") if x != "")


def truthy(text):
    return str(text).strip().lower() in ("true", "1", "yes", "y")


def read_motif_csv(path):
    proteins = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_id = row["protein_id"]
            sequence = row["sequence"]
            proteins[protein_id] = {
                "protein_id": protein_id,
                "primary_protected_type": row.get("primary_protected_type", ""),
                "protected_type": row.get("protected_type", ""),
                "sequence": sequence,
                "length": int(row.get("length") or len(sequence)),
                "motif_positions": parse_positions(row.get("protected_positions", "")),
            }
    return proteins


def read_structure_priors(path):
    priors = defaultdict(dict)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("structure_status") != "success":
                continue
            protein_id = row["protein_id"]
            position = int(row["position"])
            priors[protein_id][position] = {
                "position": position,
                "aa": row["aa"],
                "is_motif": truthy(row.get("is_motif")),
                "is_shadow": truthy(row.get("is_motif_shadow_8A")),
                "contact_density": int(float(row.get("contact_density_8A") or 0)),
                "motif_contact_count": int(float(row.get("motif_contact_count_8A") or 0)),
                "plddt": float(row.get("plddt") or 0.0),
            }
    return dict(priors)


def read_deletions(path):
    with open(path) as handle:
        rows = json.load(handle)
    deletions = {}
    for row in rows:
        protein_id = row.get("header") or row.get("protein_id")
        positions = row.get("deletion_positions_zero_based")
        if positions is None:
            positions = row.get("deleted_positions", [])
        deletions[protein_id] = set(int(x) for x in positions)
    return deletions


def pymol_resi_expr(positions):
    residues = sorted(pos + 1 for pos in positions)
    if not residues:
        return "none"
    return "resi " + "+".join(str(pos) for pos in residues)


def pml_select(name, positions):
    return "select {}, {}".format(name, pymol_resi_expr(positions))


def count_shadow_deleted(deleted, shadow_positions):
    return len(deleted & shadow_positions)


def choose_cases(proteins, priors, deletions_by_run, pdb_dir):
    candidates = []
    for protein_id, info in proteins.items():
        pdb_path = os.path.join(pdb_dir, "{}.pdb".format(protein_id))
        if not os.path.exists(pdb_path) or os.path.getsize(pdb_path) == 0:
            continue
        if protein_id not in priors:
            continue

        shadow_positions = {pos for pos, row in priors[protein_id].items() if row["is_shadow"]}
        num_motif = len(info["motif_positions"])
        num_shadow = len(shadow_positions)
        if num_motif < 1 or num_motif > 30:
            continue
        if num_shadow < 3 or num_shadow > 180:
            continue

        metrics = {}
        hard_total = 0
        shadow02_total = 0
        delta_total = 0
        positive_delta_runs = 0
        for pct in (10, 20, 30):
            hard = count_shadow_deleted(deletions_by_run[pct]["hardmask"].get(protein_id, set()), shadow_positions)
            shadow02 = count_shadow_deleted(deletions_by_run[pct]["shadow02"].get(protein_id, set()), shadow_positions)
            delta = hard - shadow02
            metrics[pct] = {"hardmask": hard, "shadow02": shadow02, "delta": delta}
            hard_total += hard
            shadow02_total += shadow02
            delta_total += delta
            if delta > 0:
                positive_delta_runs += 1

        if hard_total == 0 or delta_total <= 0:
            continue

        score = (
            8.0 * delta_total
            + 2.0 * hard_total
            - 1.5 * shadow02_total
            + 5.0 * positive_delta_runs
        )
        if protein_id in PREFERRED_CANDIDATES:
            score += 8.0
        if 8 <= num_shadow <= 80:
            score += 5.0

        candidates.append(
            {
                "protein_id": protein_id,
                "primary_protected_type": info["primary_protected_type"],
                "length": info["length"],
                "num_motif": num_motif,
                "num_shadow": num_shadow,
                "pdb_path": pdb_path,
                "metrics": metrics,
                "score": score,
            }
        )

    candidates.sort(key=lambda row: (-row["score"], row["protein_id"]))
    selected = []
    used_types = set()

    for candidate in candidates:
        if len(selected) == 3:
            break
        primary_type = candidate["primary_protected_type"]
        if primary_type in used_types:
            continue
        selected.append(candidate)
        used_types.add(primary_type)

    for candidate in candidates:
        if len(selected) == 3:
            break
        if candidate["protein_id"] in {row["protein_id"] for row in selected}:
            continue
        selected.append(candidate)

    return selected


def write_selected_cases(path, selected):
    fieldnames = [
        "protein_id",
        "primary_protected_type",
        "length",
        "num_motif",
        "num_shadow",
        "hardmask_shadow_deleted_10",
        "shadow02_shadow_deleted_10",
        "delta_10",
        "hardmask_shadow_deleted_20",
        "shadow02_shadow_deleted_20",
        "delta_20",
        "hardmask_shadow_deleted_30",
        "shadow02_shadow_deleted_30",
        "delta_30",
        "pdb_path",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            out = {
                "protein_id": row["protein_id"],
                "primary_protected_type": row["primary_protected_type"],
                "length": row["length"],
                "num_motif": row["num_motif"],
                "num_shadow": row["num_shadow"],
                "pdb_path": row["pdb_path"],
            }
            for pct in (10, 20, 30):
                out["hardmask_shadow_deleted_{}".format(pct)] = row["metrics"][pct]["hardmask"]
                out["shadow02_shadow_deleted_{}".format(pct)] = row["metrics"][pct]["shadow02"]
                out["delta_{}".format(pct)] = row["metrics"][pct]["delta"]
            writer.writerow(out)


def write_residue_sets(case_dir, protein_id, proteins, priors, deletions_by_run):
    fieldnames = [
        "position",
        "aa",
        "is_motif",
        "is_shadow",
        "deleted_hardmask_10",
        "deleted_shadow02_10",
        "deleted_hardmask_20",
        "deleted_shadow02_20",
        "deleted_hardmask_30",
        "deleted_shadow02_30",
        "contact_density_8A",
        "motif_contact_count_8A",
        "plddt",
    ]
    path = os.path.join(case_dir, "residue_sets.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pos in range(proteins[protein_id]["length"]):
            row = priors[protein_id][pos]
            writer.writerow(
                {
                    "position": pos,
                    "aa": row["aa"],
                    "is_motif": row["is_motif"],
                    "is_shadow": row["is_shadow"],
                    "deleted_hardmask_10": pos in deletions_by_run[10]["hardmask"].get(protein_id, set()),
                    "deleted_shadow02_10": pos in deletions_by_run[10]["shadow02"].get(protein_id, set()),
                    "deleted_hardmask_20": pos in deletions_by_run[20]["hardmask"].get(protein_id, set()),
                    "deleted_shadow02_20": pos in deletions_by_run[20]["shadow02"].get(protein_id, set()),
                    "deleted_hardmask_30": pos in deletions_by_run[30]["hardmask"].get(protein_id, set()),
                    "deleted_shadow02_30": pos in deletions_by_run[30]["shadow02"].get(protein_id, set()),
                    "contact_density_8A": row["contact_density"],
                    "motif_contact_count_8A": row["motif_contact_count"],
                    "plddt": "{:.2f}".format(row["plddt"]),
                }
            )
    return path


def write_selections(case_dir, protein_id, proteins, priors, deletions_by_run):
    motif = set(proteins[protein_id]["motif_positions"])
    shadow = {pos for pos, row in priors[protein_id].items() if row["is_shadow"]}
    lines = []
    lines.append("# {} residue selections".format(protein_id))
    lines.append("# Internal positions are 0-based. PyMOL residues below are 1-based.")
    lines.append(pml_select("motif_residues", motif))
    lines.append(pml_select("motif_shadow_residues", shadow))
    for pct in (10, 20, 30):
        hard = deletions_by_run[pct]["hardmask"].get(protein_id, set())
        shadow02 = deletions_by_run[pct]["shadow02"].get(protein_id, set())
        avoided = (hard - shadow02) & shadow
        lines.append(pml_select("hardmask_deleted_{}".format(pct), hard))
        lines.append(pml_select("shadow02_deleted_{}".format(pct), shadow02))
        lines.append(pml_select("avoided_shadow_{}".format(pct), avoided))
    path = os.path.join(case_dir, "selections.txt")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def write_pymol_script(case_dir, protein_id, pdb_path, proteins, priors, deletions_by_run):
    motif = set(proteins[protein_id]["motif_positions"])
    shadow = {pos for pos, row in priors[protein_id].items() if row["is_shadow"]}
    hard_all = set()
    shadow02_all = set()
    avoided_all = set()
    for pct in (10, 20, 30):
        hard = deletions_by_run[pct]["hardmask"].get(protein_id, set())
        shadow02 = deletions_by_run[pct]["shadow02"].get(protein_id, set())
        hard_all |= hard
        shadow02_all |= shadow02
        avoided_all |= (hard - shadow02) & shadow

    path = os.path.join(case_dir, "visualize_{}.pml".format(protein_id))
    png_path = os.path.join(case_dir, "{}_case_study.png".format(protein_id))
    lines = [
        "reinitialize",
        "load {}, {}".format(pdb_path, protein_id),
        "hide everything",
        "show cartoon, {}".format(protein_id),
        "bg_color white",
        "color gray80, {}".format(protein_id),
        pml_select("motif_residues", motif),
        pml_select("motif_shadow_residues", shadow),
        pml_select("hardmask_deleted_all", hard_all),
        pml_select("shadow02_deleted_all", shadow02_all),
        pml_select("avoided_shadow_all", avoided_all),
    ]
    for pct in (10, 20, 30):
        hard = deletions_by_run[pct]["hardmask"].get(protein_id, set())
        shadow02 = deletions_by_run[pct]["shadow02"].get(protein_id, set())
        avoided = (hard - shadow02) & shadow
        lines.extend(
            [
                pml_select("hardmask_deleted_{}".format(pct), hard),
                pml_select("shadow02_deleted_{}".format(pct), shadow02),
                pml_select("avoided_shadow_{}".format(pct), avoided),
            ]
        )
    lines.extend(
        [
            "show sticks, motif_shadow_residues",
            "color yellow, motif_shadow_residues",
            "show spheres, hardmask_deleted_all",
            "color blue, hardmask_deleted_all",
            "show spheres, shadow02_deleted_all",
            "color green, shadow02_deleted_all",
            "show spheres, avoided_shadow_all",
            "color magenta, avoided_shadow_all",
            "show spheres, motif_residues",
            "color red, motif_residues",
            "set sphere_scale, 0.35, hardmask_deleted_all",
            "set sphere_scale, 0.35, shadow02_deleted_all",
            "set sphere_scale, 0.50, avoided_shadow_all",
            "set sphere_scale, 0.55, motif_residues",
            "set stick_radius, 0.12, motif_shadow_residues",
            "set cartoon_transparency, 0.15",
            "set ray_opaque_background, off",
            "orient {}".format(protein_id),
            "# Optional rendering command:",
            "# png {}, width=2200, height=1600, dpi=300, ray=1".format(png_path),
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def write_report(path, selected):
    lines = []
    lines.append("Case Study Visualization Report")
    lines.append("")
    lines.append(
        "Selected 3 proteins by ranking high hardmask motif-shadow deletion, large shadow02 improvement, available AlphaFold PDB, moderate motif/shadow set size, and diverse primary_protected_type where possible."
    )
    lines.append("")
    for row in selected:
        lines.append("{} ({}, length {})".format(row["protein_id"], row["primary_protected_type"], row["length"]))
        lines.append("  num_motif: {}".format(row["num_motif"]))
        lines.append("  num_shadow: {}".format(row["num_shadow"]))
        for pct in (10, 20, 30):
            metrics = row["metrics"][pct]
            lines.append(
                "  {}%: hardmask_shadow_deleted={}, shadow02_shadow_deleted={}, reduction={}".format(
                    pct, metrics["hardmask"], metrics["shadow02"], metrics["delta"]
                )
            )
        lines.append(
            "  interpretation: shadow prior redirects deletions away from residues spatially adjacent to the protected motif while direct motif residues remain protected."
        )
        lines.append(
            "  PyMOL script: results/case_studies/{}/visualize_{}.pml".format(
                row["protein_id"], row["protein_id"]
            )
        )
        lines.append("")
    lines.append("CASE_STUDY_VISUALIZATION_READY")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--structure_priors", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    proteins = read_motif_csv(args.motif_csv)
    priors = read_structure_priors(args.structure_priors)
    deletions_by_run = {
        pct: {method: read_deletions(path) for method, path in methods.items()}
        for pct, methods in RUNS.items()
    }

    os.makedirs(args.out_dir, exist_ok=True)
    selected = choose_cases(proteins, priors, deletions_by_run, args.pdb_dir)
    if len(selected) < 3:
        raise RuntimeError("Could not select 3 case proteins; selected {}".format(len(selected)))

    selected_path = os.path.join(args.out_dir, "selected_cases.csv")
    write_selected_cases(selected_path, selected)

    for row in selected:
        protein_id = row["protein_id"]
        case_dir = os.path.join(args.out_dir, protein_id)
        os.makedirs(case_dir, exist_ok=True)
        write_residue_sets(case_dir, protein_id, proteins, priors, deletions_by_run)
        write_selections(case_dir, protein_id, proteins, priors, deletions_by_run)
        write_pymol_script(case_dir, protein_id, row["pdb_path"], proteins, priors, deletions_by_run)

    report_path = os.path.join(args.out_dir, "case_study_report.txt")
    write_report(report_path, selected)
    print("Selected cases: {}".format(", ".join(row["protein_id"] for row in selected)))
    print("Wrote {}".format(selected_path))
    print("Wrote {}".format(report_path))


if __name__ == "__main__":
    main()
