#!/usr/bin/env python3
"""Build segment-level deletion candidates with closure/stitchability features."""

import argparse
import csv
import math
import os
from collections import Counter, defaultdict


PDB_DIR = "data/raw/structures/alphafold_swissprot_diverse"


def parse_positions(text):
    if not text:
        return set()
    return set(int(x) for x in text.split(";") if x != "")


def truthy(text):
    return str(text).strip().lower() in ("true", "1", "yes", "y")


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def read_motif_csv(path):
    proteins = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_id = row["protein_id"]
            sequence = row["sequence"]
            proteins[protein_id] = {
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
            pos = int(row["position"])
            priors[protein_id][pos] = {
                "aa": row["aa"],
                "is_motif": truthy(row["is_motif"]),
                "is_shadow": truthy(row["is_motif_shadow_8A"]),
                "distance_to_motif": float(row["distance_to_motif"]),
                "contact_density": float(row["contact_density_8A"]),
                "motif_contact_count": float(row["motif_contact_count_8A"]),
                "plddt": float(row["plddt"]),
            }
    return dict(priors)


def read_ca_coords(pdb_path):
    coords = []
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
    return coords


def output_paths(out_csv):
    root, _ = os.path.splitext(out_csv)
    return root + "_summary.txt"


def bool_text(value):
    return "True" if value else "False"


def blank_or_float(value):
    if value is None:
        return ""
    return "{:.4f}".format(value)


def build_candidates(args):
    proteins = read_motif_csv(args.motif_csv)
    priors = read_structure_priors(args.structure_priors)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    fieldnames = [
        "protein_id",
        "start",
        "end",
        "seg_len",
        "sequence_segment",
        "num_motif_overlap",
        "motif_overlap_fraction",
        "has_motif_overlap",
        "num_shadow_overlap",
        "shadow_overlap_fraction",
        "has_shadow_overlap",
        "num_motif_contact_overlap",
        "motif_contact_overlap_fraction",
        "mean_contact_density_8A",
        "max_contact_density_8A",
        "mean_plddt",
        "min_plddt",
        "mean_distance_to_motif",
        "min_distance_to_motif",
        "is_terminal_deletion",
        "boundary_left",
        "boundary_right",
        "boundary_ca_distance",
        "closure_friendly",
        "boundary_contact_density_mean",
        "boundary_plddt_mean",
    ]

    processed = 0
    skipped = []
    total_segments = 0
    len_counts = Counter()
    motif_overlap_count = 0
    shadow_overlap_count = 0
    closure_friendly_count = 0
    terminal_count = 0
    boundary_distances = []
    closure_shadow_fractions = []
    nonclosure_shadow_fractions = []
    examples = []

    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for protein_id in sorted(proteins):
            info = proteins[protein_id]
            length = info["length"]
            if protein_id not in priors:
                skipped.append((protein_id, "no_structure_prior"))
                continue
            if len(priors[protein_id]) != length:
                skipped.append((protein_id, "prior_length_mismatch"))
                continue

            pdb_path = os.path.join(PDB_DIR, protein_id + ".pdb")
            if not os.path.exists(pdb_path):
                skipped.append((protein_id, "missing_pdb"))
                continue
            coords = read_ca_coords(pdb_path)
            if len(coords) != length:
                skipped.append((protein_id, "pdb_ca_length_mismatch"))
                continue

            processed += 1
            sequence = info["sequence"]
            protein_prior = priors[protein_id]

            for start in range(0, length, args.stride):
                max_end = min(length - 1, start + args.max_seg_len - 1)
                for end in range(start + args.min_seg_len - 1, max_end + 1):
                    positions = list(range(start, end + 1))
                    seg_len = len(positions)
                    rows = [protein_prior[pos] for pos in positions]

                    num_motif = sum(1 for row in rows if row["is_motif"])
                    num_shadow = sum(1 for row in rows if row["is_shadow"])
                    num_motif_contact = sum(1 for row in rows if row["motif_contact_count"] > 0)
                    contact_values = [row["contact_density"] for row in rows]
                    plddt_values = [row["plddt"] for row in rows]
                    distance_values = [row["distance_to_motif"] for row in rows]

                    is_terminal = start == 0 or end == length - 1
                    boundary_left = None
                    boundary_right = None
                    boundary_distance = None
                    boundary_contact_density_mean = None
                    boundary_plddt_mean = None
                    if is_terminal:
                        closure_friendly = True
                    else:
                        boundary_left = start - 1
                        boundary_right = end + 1
                        boundary_distance = distance(coords[boundary_left], coords[boundary_right])
                        boundary_contact_density_mean = mean(
                            [
                                protein_prior[boundary_left]["contact_density"],
                                protein_prior[boundary_right]["contact_density"],
                            ]
                        )
                        boundary_plddt_mean = mean(
                            [
                                protein_prior[boundary_left]["plddt"],
                                protein_prior[boundary_right]["plddt"],
                            ]
                        )
                        closure_friendly = boundary_distance <= args.closure_cutoff
                        boundary_distances.append(boundary_distance)

                    shadow_fraction = num_shadow / float(seg_len)
                    row = {
                        "protein_id": protein_id,
                        "start": start,
                        "end": end,
                        "seg_len": seg_len,
                        "sequence_segment": sequence[start : end + 1],
                        "num_motif_overlap": num_motif,
                        "motif_overlap_fraction": "{:.6f}".format(num_motif / float(seg_len)),
                        "has_motif_overlap": bool_text(num_motif > 0),
                        "num_shadow_overlap": num_shadow,
                        "shadow_overlap_fraction": "{:.6f}".format(shadow_fraction),
                        "has_shadow_overlap": bool_text(num_shadow > 0),
                        "num_motif_contact_overlap": num_motif_contact,
                        "motif_contact_overlap_fraction": "{:.6f}".format(
                            num_motif_contact / float(seg_len)
                        ),
                        "mean_contact_density_8A": "{:.4f}".format(mean(contact_values)),
                        "max_contact_density_8A": "{:.4f}".format(max(contact_values)),
                        "mean_plddt": "{:.4f}".format(mean(plddt_values)),
                        "min_plddt": "{:.4f}".format(min(plddt_values)),
                        "mean_distance_to_motif": "{:.4f}".format(mean(distance_values)),
                        "min_distance_to_motif": "{:.4f}".format(min(distance_values)),
                        "is_terminal_deletion": bool_text(is_terminal),
                        "boundary_left": "" if boundary_left is None else boundary_left,
                        "boundary_right": "" if boundary_right is None else boundary_right,
                        "boundary_ca_distance": blank_or_float(boundary_distance),
                        "closure_friendly": bool_text(closure_friendly),
                        "boundary_contact_density_mean": blank_or_float(
                            boundary_contact_density_mean
                        ),
                        "boundary_plddt_mean": blank_or_float(boundary_plddt_mean),
                    }
                    writer.writerow(row)

                    total_segments += 1
                    len_counts[seg_len] += 1
                    motif_overlap_count += int(num_motif > 0)
                    shadow_overlap_count += int(num_shadow > 0)
                    closure_friendly_count += int(closure_friendly)
                    terminal_count += int(is_terminal)
                    if closure_friendly:
                        closure_shadow_fractions.append(shadow_fraction)
                    else:
                        nonclosure_shadow_fractions.append(shadow_fraction)

                    risk_score = (
                        100.0 * int(num_motif > 0)
                        + 25.0 * shadow_fraction
                        + mean(contact_values)
                        - 0.03 * mean(plddt_values)
                        + (0.0 if closure_friendly else 10.0)
                    )
                    examples.append(
                        {
                            "protein_id": protein_id,
                            "start": start,
                            "end": end,
                            "seg_len": seg_len,
                            "has_motif_overlap": num_motif > 0,
                            "num_motif_overlap": num_motif,
                            "shadow_overlap_fraction": shadow_fraction,
                            "boundary_ca_distance": boundary_distance,
                            "closure_friendly": closure_friendly,
                            "risk_score": risk_score,
                        }
                    )

    summary_path = output_paths(args.out_csv)
    write_summary(
        summary_path,
        processed,
        skipped,
        total_segments,
        len_counts,
        motif_overlap_count,
        shadow_overlap_count,
        closure_friendly_count,
        terminal_count,
        boundary_distances,
        closure_shadow_fractions,
        nonclosure_shadow_fractions,
        examples,
    )
    return summary_path


def format_example(example):
    dist = example["boundary_ca_distance"]
    dist_text = "" if dist is None else "{:.4f}".format(dist)
    return "{protein_id},{start},{end},{seg_len},{has_motif_overlap},{shadow_overlap_fraction:.4f},{dist},{closure_friendly}".format(
        dist=dist_text, **example
    )


def write_summary(
    path,
    processed,
    skipped,
    total_segments,
    len_counts,
    motif_overlap_count,
    shadow_overlap_count,
    closure_friendly_count,
    terminal_count,
    boundary_distances,
    closure_shadow_fractions,
    nonclosure_shadow_fractions,
    examples,
):
    low_risk = sorted(
        examples,
        key=lambda x: (
            x["has_motif_overlap"],
            x["shadow_overlap_fraction"],
            not x["closure_friendly"],
            x["risk_score"],
            x["protein_id"],
            x["start"],
        ),
    )[:10]
    high_risk = sorted(
        examples,
        key=lambda x: (
            -x["num_motif_overlap"],
            -x["shadow_overlap_fraction"],
            x["closure_friendly"],
            -x["risk_score"],
            x["protein_id"],
            x["start"],
        ),
    )[:10]

    with open(path, "w") as handle:
        handle.write("Segment Candidate Summary\n\n")
        handle.write("successful_proteins: {}\n".format(processed))
        handle.write("skipped_proteins: {}\n".format(len(skipped)))
        if skipped:
            handle.write(
                "skipped_examples: {}\n".format(
                    "; ".join("{}:{}".format(pid, reason) for pid, reason in skipped[:20])
                )
            )
        handle.write("total_candidate_segments: {}\n".format(total_segments))
        handle.write(
            "segment_length_distribution: {}\n".format(
                ", ".join("{}:{}".format(k, len_counts[k]) for k in sorted(len_counts))
            )
        )
        handle.write(
            "has_motif_overlap_segments: {} ({:.4f})\n".format(
                motif_overlap_count, motif_overlap_count / float(total_segments or 1)
            )
        )
        handle.write(
            "has_shadow_overlap_segments: {} ({:.4f})\n".format(
                shadow_overlap_count, shadow_overlap_count / float(total_segments or 1)
            )
        )
        handle.write(
            "closure_friendly_segments: {} ({:.4f})\n".format(
                closure_friendly_count, closure_friendly_count / float(total_segments or 1)
            )
        )
        handle.write(
            "terminal_deletion_segments: {} ({:.4f})\n".format(
                terminal_count, terminal_count / float(total_segments or 1)
            )
        )
        handle.write("mean_boundary_ca_distance: {:.4f}\n".format(mean(boundary_distances)))
        handle.write(
            "closure_friendly_mean_shadow_overlap_fraction: {:.4f}\n".format(
                mean(closure_shadow_fractions)
            )
        )
        handle.write(
            "non_closure_friendly_mean_shadow_overlap_fraction: {:.4f}\n".format(
                mean(nonclosure_shadow_fractions)
            )
        )
        handle.write("\nLowest risk segment examples:\n")
        handle.write(
            "protein_id,start,end,seg_len,has_motif_overlap,shadow_overlap_fraction,boundary_ca_distance,closure_friendly\n"
        )
        for example in low_risk:
            handle.write(format_example(example) + "\n")
        handle.write("\nHighest risk segment examples:\n")
        handle.write(
            "protein_id,start,end,seg_len,num_motif_overlap,shadow_overlap_fraction,boundary_ca_distance,closure_friendly\n"
        )
        for example in high_risk:
            dist = example["boundary_ca_distance"]
            dist_text = "" if dist is None else "{:.4f}".format(dist)
            handle.write(
                "{},{},{},{},{},{:.4f},{},{}\n".format(
                    example["protein_id"],
                    example["start"],
                    example["end"],
                    example["seg_len"],
                    example["num_motif_overlap"],
                    example["shadow_overlap_fraction"],
                    dist_text,
                    example["closure_friendly"],
                )
            )
        handle.write("\nSEGMENT_CANDIDATE_BUILD_PASS\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--structure_priors", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--min_seg_len", type=int, default=1)
    parser.add_argument("--max_seg_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--closure_cutoff", type=float, default=8.0)
    args = parser.parse_args()

    if args.min_seg_len < 1:
        raise ValueError("--min_seg_len must be >= 1")
    if args.max_seg_len < args.min_seg_len:
        raise ValueError("--max_seg_len must be >= --min_seg_len")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    summary_path = build_candidates(args)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(summary_path))


if __name__ == "__main__":
    main()
