#!/usr/bin/env python3
"""Create a family-held-out BioPrior split using CD-HIT clusters.

The existing BioPrior-10K split is exact-sequence leakage free, but it does not
control family-level similarity. This script adds a stricter cluster-held-out
protocol by clustering all proteins with CD-HIT and assigning whole clusters to
train/val/test. It keeps the same biological strata used by the current split
report, but the unit of assignment is now a protein cluster rather than a single
accession.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict


LENGTH_BINS = [
    ("100-200", 100, 200),
    ("201-400", 201, 400),
    ("401-600", 401, 600),
    ("601-800", 601, 800),
]

PRIMARY_ORDER = [
    "active site",
    "binding site",
    "site",
    "short sequence motif",
]


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def length_bin(length):
    for label, lo, hi in LENGTH_BINS:
        if lo <= length <= hi:
            return label
    return "outside"


def seq_hash(sequence):
    return hashlib.sha1(sequence.encode("ascii")).hexdigest()


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        for row in rows:
            accession = row["accession"]
            handle.write(">{}\n".format(accession))
            sequence = row["sequence"]
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i : i + 80] + "\n")


def read_stage1_hashes(path):
    if not path:
        return set()
    hashes = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequence = row.get("sequence", "")
            if sequence:
                hashes.add(seq_hash(sequence))
    return hashes


def read_cdhit_clusters(path):
    clusters = []
    current = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                if current:
                    clusters.append(current)
                    current = []
                continue
            if ">" not in line or "..." not in line:
                continue
            member = line.split(">", 1)[1].split("...", 1)[0].strip()
            if member:
                current.append(member)
    if current:
        clusters.append(current)
    return clusters


def word_length_for_identity(identity):
    if identity <= 0.4:
        return 2
    if identity <= 0.5:
        return 3
    if identity <= 0.6:
        return 4
    return 5


def default_cdhit_executable():
    found = shutil.which("cd-hit")
    if found:
        return found
    fallback = "/public/home/zhangyangroup/chengshiz/mambaforge/bin/cd-hit"
    return fallback


def run_cdhit(input_fasta, out_prefix, identity, coverage, threads, memory_mb, executable):
    cmd = [
        executable,
        "-i",
        input_fasta,
        "-o",
        out_prefix,
        "-c",
        "{:.3f}".format(identity),
        "-n",
        str(word_length_for_identity(identity)),
        "-G",
        "1",
        "-A",
        "{:.3f}".format(coverage),
        "-d",
        "0",
        "-T",
        str(threads),
        "-M",
        str(memory_mb),
    ]
    print("Running CD-HIT:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_prefix + ".clstr"


def summarize_rows(rows):
    primary = Counter(row.get("primary_protected_type", "") for row in rows)
    length_bins = Counter(length_bin(int(row["length"])) for row in rows)
    protected_counts = [int(row["n_protected"]) for row in rows]
    lengths = [int(row["length"]) for row in rows]
    return {
        "n": len(rows),
        "primary": dict(primary),
        "length_bins": dict(length_bins),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mean": sum(lengths) / float(len(lengths)) if lengths else 0.0,
        "protected_min": min(protected_counts) if protected_counts else 0,
        "protected_max": max(protected_counts) if protected_counts else 0,
        "protected_mean": sum(protected_counts) / float(len(protected_counts)) if protected_counts else 0.0,
    }


def make_cluster_groups(rows, cluster_members):
    by_accession = {row["accession"]: row for row in rows}
    groups = []
    missing = []
    for cluster_id, members in enumerate(cluster_members):
        cluster_rows = []
        for accession in members:
            row = by_accession.get(accession)
            if row is None:
                missing.append(accession)
                continue
            cluster_rows.append(row)
        if not cluster_rows:
            continue
        primary_counts = Counter(row.get("primary_protected_type", "") for row in cluster_rows)
        length_counts = Counter(length_bin(int(row["length"])) for row in cluster_rows)
        accessions = sorted(row["accession"] for row in cluster_rows)
        groups.append(
            {
                "cluster_id": cluster_id,
                "accessions": accessions,
                "rows": cluster_rows,
                "size": len(cluster_rows),
                "primary_counts": primary_counts,
                "length_counts": length_counts,
                "signature": seq_hash("|".join(accessions)),
                "mixed_primary_count": sum(1 for count in primary_counts.values() if count > 0),
            }
        )
    return groups, missing


def build_targets(rows, train_frac, val_frac):
    test_frac = 1.0 - train_frac - val_frac
    if test_frac < -1e-9:
        raise ValueError("train_frac + val_frac must be <= 1.0")
    fractions = {
        "train": float(train_frac),
        "val": float(val_frac),
        "test": float(test_frac),
    }
    total = len(rows)
    primary_counts = Counter(row.get("primary_protected_type", "") for row in rows)
    length_counts = Counter(length_bin(int(row["length"])) for row in rows)

    primary_order = list(PRIMARY_ORDER)
    extra_primary = sorted(label for label in primary_counts if label not in PRIMARY_ORDER)
    primary_order.extend(extra_primary)

    target = {}
    for split, frac in fractions.items():
        target[split] = {
            "total": total * frac,
            "primary": {label: primary_counts.get(label, 0) * frac for label in primary_order},
            "length": {label: length_counts.get(label, 0) * frac for label in [item[0] for item in LENGTH_BINS] + ["outside"]},
        }
    return target, primary_order


def projected_score(assigned, cluster, target, primary_order):
    total_target = max(1.0, target["total"])
    total_fill = (assigned["total"] + cluster["size"]) / total_target

    primary_fills = []
    for label in primary_order:
        denom = max(1.0, target["primary"].get(label, 0.0))
        numer = assigned["primary"].get(label, 0) + cluster["primary_counts"].get(label, 0)
        primary_fills.append(numer / denom)
    primary_fill = sum(primary_fills) / float(len(primary_fills) or 1)

    length_labels = list(target["length"].keys())
    length_fills = []
    for label in length_labels:
        denom = max(1.0, target["length"].get(label, 0.0))
        numer = assigned["length"].get(label, 0) + cluster["length_counts"].get(label, 0)
        length_fills.append(numer / denom)
    length_fill = sum(length_fills) / float(len(length_fills) or 1)

    worst_fill = max(total_fill, primary_fill, length_fill)
    tie_breaker = total_fill + 0.50 * primary_fill + 0.25 * length_fill
    return worst_fill, tie_breaker


def assign_cluster_splits(cluster_groups, target, primary_order, seed):
    # Largest clusters are assigned first so the small remaining clusters can
    # absorb the residual imbalance.  Seed controls tie-breaking among clusters
    # of the same size.
    rng = random.Random(seed)
    cluster_groups = [
        dict(cluster, seed_noise=rng.random())
        for cluster in cluster_groups
    ]
    ordered = sorted(cluster_groups, key=lambda item: (-item["size"], item["seed_noise"], item["cluster_id"]))

    split_order = ["train", "val", "test"]
    assigned = {
        split: {
            "total": 0,
            "primary": Counter(),
            "length": Counter(),
            "clusters": [],
        }
        for split in split_order
    }

    for cluster in ordered:
        best_split = None
        best_key = None
        for split in split_order:
            score = projected_score(assigned[split], cluster, target[split], primary_order)
            key = (score[0], score[1], -target[split]["total"], split_order.index(split))
            if best_key is None or key < best_key:
                best_key = key
                best_split = split
        assigned[best_split]["clusters"].append(cluster)
        assigned[best_split]["total"] += cluster["size"]
        assigned[best_split]["primary"].update(cluster["primary_counts"])
        assigned[best_split]["length"].update(cluster["length_counts"])

    # Build row-level split map.
    split_rows = {split: [] for split in split_order}
    cluster_assignments = []
    row_split = {}
    for split in split_order:
        for cluster in assigned[split]["clusters"]:
            for row in cluster["rows"]:
                row = dict(row)
                row["split"] = split
                row["cluster_id"] = cluster["cluster_id"]
                row["cluster_signature"] = cluster["signature"]
                row["cluster_size"] = cluster["size"]
                split_rows[split].append(row)
                row_split[row["accession"]] = split
            cluster_assignments.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "cluster_signature": cluster["signature"],
                    "cluster_size": cluster["size"],
                    "split": split,
                    "accessions": ";".join(cluster["accessions"]),
                    "primary_counts": json.dumps(dict(cluster["primary_counts"]), sort_keys=True),
                    "length_counts": json.dumps(dict(cluster["length_counts"]), sort_keys=True),
                    "mixed_primary_count": cluster["mixed_primary_count"],
                }
            )
    for split in split_rows:
        split_rows[split].sort(key=lambda row: row["accession"])
    cluster_assignments.sort(key=lambda row: int(row["cluster_id"]))
    return split_rows, cluster_assignments, row_split


def split_overlaps(split_rows):
    accession_sets = {split: set(row["accession"] for row in rows) for split, rows in split_rows.items()}
    sequence_sets = {split: set(seq_hash(row["sequence"]) for row in rows) for split, rows in split_rows.items()}
    cluster_sets = {split: set(row["cluster_signature"] for row in rows) for split, rows in split_rows.items()}
    checks = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        checks["accession_overlap_{}_{}".format(a, b)] = sorted(accession_sets[a] & accession_sets[b])
        checks["exact_sequence_overlap_{}_{}".format(a, b)] = sorted(sequence_sets[a] & sequence_sets[b])
        checks["cluster_overlap_{}_{}".format(a, b)] = sorted(cluster_sets[a] & cluster_sets[b])
    return checks


def write_split_json(path, split_rows):
    payload = {
        split: [row["accession"] for row in rows]
        for split, rows in split_rows.items()
    }
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_report(path, args, cluster_groups, split_rows, summaries, leakage, stage1_overlap):
    ensure_dir(os.path.dirname(path))
    all_leak_free = all(len(value) == 0 for value in leakage.values())
    total_clusters = len(cluster_groups)
    singleton_clusters = sum(1 for cluster in cluster_groups if cluster["size"] == 1)
    mixed_clusters = sum(1 for cluster in cluster_groups if cluster["mixed_primary_count"] > 1)
    largest_cluster = max((cluster["size"] for cluster in cluster_groups), default=0)
    mean_cluster_size = sum(cluster["size"] for cluster in cluster_groups) / float(max(1, total_clusters))

    with open(path, "w") as handle:
        handle.write("BioPrior family-held-out split and leakage report\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("out_dir: {}\n".format(args.out_dir))
        handle.write("seed: {}\n".format(args.seed))
        handle.write("split fractions: train={} val={} test={}\n".format(args.train_frac, args.val_frac, 1.0 - args.train_frac - args.val_frac))
        handle.write("split unit: CD-HIT cluster / protein accession\n")
        handle.write("identity clustering: cd-hit c={} coverage={} word_length={}\n".format(args.identity, args.coverage, word_length_for_identity(args.identity)))
        handle.write("cluster assignment strategy: greedy fill balancing total, primary-type, and length-bin loads\n\n")

        handle.write("Cluster statistics:\n")
        handle.write("- total clusters: {}\n".format(total_clusters))
        handle.write("- singleton clusters: {}\n".format(singleton_clusters))
        handle.write("- mixed-primary clusters: {}\n".format(mixed_clusters))
        handle.write("- largest cluster size: {}\n".format(largest_cluster))
        handle.write("- mean cluster size: {:.2f}\n\n".format(mean_cluster_size))

        for split in ["train", "val", "test"]:
            summary = summaries[split]
            handle.write("{}:\n".format(split))
            handle.write("- proteins: {}\n".format(summary["n"]))
            handle.write("- primary distribution: {}\n".format(summary["primary"]))
            handle.write("- length bins: {}\n".format(summary["length_bins"]))
            handle.write("- length min/max/mean: {}/{}/{:.2f}\n".format(summary["length_min"], summary["length_max"], summary["length_mean"]))
            handle.write("- protected count min/max/mean: {}/{}/{:.2f}\n".format(summary["protected_min"], summary["protected_max"], summary["protected_mean"]))
        handle.write("\nLeakage checks:\n")
        for key, values in sorted(leakage.items()):
            handle.write("- {}: {}\n".format(key, len(values)))
        if stage1_overlap is not None:
            handle.write("\nStage-1 exact sequence overlap with this family-held-out split:\n")
            for split in ["train", "val", "test"]:
                handle.write("- {}: {}\n".format(split, stage1_overlap[split]))
        handle.write("\nCluster assignment note:\n")
        handle.write("- Clusters are kept intact across splits; no cluster id appears in more than one split.\n")
        handle.write("\n{}\n".format("BIOPRIOR_FAMILY_SPLIT_LEAKAGE_PASS" if all_leak_free else "BIOPRIOR_FAMILY_SPLIT_LEAKAGE_WARN"))


def run(args):
    rows, fieldnames = read_csv(args.input_csv)
    ensure_dir(args.out_dir)

    fasta_path = os.path.join(args.out_dir, "input.fasta")
    write_fasta(fasta_path, rows)

    cdhit_executable = args.cdhit_executable or default_cdhit_executable()
    if not os.path.exists(cdhit_executable):
        raise RuntimeError("Could not find cd-hit executable: {}".format(cdhit_executable))

    cluster_prefix = os.path.join(
        args.out_dir,
        "cdhit_identity{:.2f}_coverage{:.2f}".format(args.identity, args.coverage),
    )
    if args.reuse_clstr:
        clstr_path = args.reuse_clstr
    else:
        clstr_path = run_cdhit(
            fasta_path,
            cluster_prefix,
            args.identity,
            args.coverage,
            args.threads,
            args.memory_mb,
            cdhit_executable,
        )

    cluster_members = read_cdhit_clusters(clstr_path)
    cluster_groups, missing = make_cluster_groups(rows, cluster_members)
    if missing:
        print("WARNING: {} cluster members were missing from the input CSV".format(len(missing)))

    target, primary_order = build_targets(rows, args.train_frac, args.val_frac)
    split_rows, cluster_assignments, row_split = assign_cluster_splits(cluster_groups, target, primary_order, args.seed)

    fieldnames_with_split = list(fieldnames)
    for extra in ["split", "cluster_id", "cluster_signature", "cluster_size"]:
        if extra not in fieldnames_with_split:
            fieldnames_with_split.append(extra)

    output_rows = []
    for split in ["train", "val", "test"]:
        for row in split_rows[split]:
            output_rows.append(dict(row))

    assignments_csv = os.path.join(args.out_dir, "split_assignments.csv")
    write_csv(assignments_csv, output_rows, fieldnames_with_split)
    write_split_json(os.path.join(args.out_dir, "splits.json"), split_rows)
    write_csv(
        os.path.join(args.out_dir, "cluster_assignments.csv"),
        cluster_assignments,
        ["cluster_id", "cluster_signature", "cluster_size", "split", "accessions", "primary_counts", "length_counts", "mixed_primary_count"],
    )

    for split in ["train", "val", "test"]:
        split_path = os.path.join(args.out_dir, "{}.csv".format(split))
        split_fasta = os.path.join(args.out_dir, "{}.fasta".format(split))
        write_csv(split_path, split_rows[split], fieldnames_with_split)
        write_fasta(split_fasta, split_rows[split])

    summaries = {split: summarize_rows(split_rows[split]) for split in ["train", "val", "test"]}
    leakage = split_overlaps(split_rows)
    stage1_overlap = None
    if args.stage1_csv:
        stage1_hashes = read_stage1_hashes(args.stage1_csv)
        stage1_overlap = {
            split: sum(1 for row in split_rows[split] if seq_hash(row["sequence"]) in stage1_hashes)
            for split in ["train", "val", "test"]
        }
    write_report(os.path.join(args.out_dir, "split_leakage_report.txt"), args, cluster_groups, split_rows, summaries, leakage, stage1_overlap)

    print("Wrote {}".format(assignments_csv))
    print("Wrote {}".format(os.path.join(args.out_dir, "split_leakage_report.txt")))
    print("Cluster members: {}".format(len(cluster_members)))
    for split in ["train", "val", "test"]:
        print("{} proteins: {}".format(split, len(split_rows[split])))


def parse_args():
    parser = argparse.ArgumentParser(description="Create a family-held-out BioPrior split using CD-HIT clusters.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--memory_mb", type=int, default=0)
    parser.add_argument("--cdhit_executable", default=None)
    parser.add_argument("--reuse_clstr", default=None, help="Reuse an existing CD-HIT .clstr file instead of rerunning clustering.")
    parser.add_argument("--stage1_csv", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
