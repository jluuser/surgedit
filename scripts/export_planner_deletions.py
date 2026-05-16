#!/usr/bin/env python3
"""Export BioDel planner selected segments as shrunk FASTA + deletions JSON."""

import argparse
import csv
import json
import os
from collections import defaultdict


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def base_header(header):
    return header.split("|", 1)[0]


def read_fasta(path):
    records = []
    header = None
    chunks = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((base_header(header[1:]), "".join(chunks)))
                header = line
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            records.append((base_header(header[1:]), "".join(chunks)))
    return records


def read_selected_segments(path):
    grouped = defaultdict(list)
    settings = set()
    budgets = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            setting = row["setting"]
            budget = float(row["budget_ratio"])
            accession = row["accession"]
            start = int(row["seg_start"])
            end = int(row["seg_end"])
            grouped[(setting, budget, accession)].append((start, end, row))
            settings.add(setting)
            budgets.add(budget)
    return grouped, sorted(settings), sorted(budgets)


def merge_positions(segments, sequence_length):
    positions = set()
    for start, end, _ in segments:
        for pos in range(start, end + 1):
            if 0 <= pos < sequence_length:
                positions.add(pos)
    return sorted(positions)


def deletion_header_field(sequence, positions):
    return ",".join("{}{}".format(sequence[pos], pos) for pos in positions)


def write_fasta(path, records):
    with open(path, "w") as handle:
        for header, sequence in records:
            handle.write(">{}\n".format(header))
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i : i + 80] + "\n")


def write_report(path, rows, setting, budget):
    total = len(rows)
    deleted_counts = [row["deletion_count"] for row in rows]
    proteins_with_deletion = sum(1 for count in deleted_counts if count > 0)
    orig_total = sum(row["original_length"] for row in rows)
    deleted_total = sum(deleted_counts)
    fill = deleted_total / float(orig_total * budget) if orig_total and budget else 0.0
    with open(path, "w") as handle:
        handle.write("Planner deletion export report\n\n")
        handle.write("setting: {}\n".format(setting))
        handle.write("budget_ratio: {}\n".format(budget))
        handle.write("total_proteins: {}\n".format(total))
        handle.write("proteins_with_deletion: {}\n".format(proteins_with_deletion))
        handle.write("total_original_length: {}\n".format(orig_total))
        handle.write("total_deleted_residues: {}\n".format(deleted_total))
        handle.write("global_fill_ratio_vs_nominal_budget: {:.6f}\n".format(fill))
        handle.write("min_deleted_per_protein: {}\n".format(min(deleted_counts) if deleted_counts else 0))
        handle.write("max_deleted_per_protein: {}\n".format(max(deleted_counts) if deleted_counts else 0))
        handle.write("\nBIODEL_PLANNER_DELETION_EXPORT_PASS\n")


def export(args):
    records = read_fasta(args.core_fasta)
    sequences = dict(records)
    selected, settings, budgets = read_selected_segments(args.selected_segments_csv)
    if args.settings:
        settings = [x.strip() for x in args.settings.split(",") if x.strip()]
    if args.budgets:
        budgets = [float(x.strip()) for x in args.budgets.split(",") if x.strip()]
    ensure_dir(args.out_root)
    manifest_rows = []
    for setting in settings:
        for budget in budgets:
            pct = int(round(budget * 100))
            out_dir = os.path.join(args.out_root, "{}_run{:02d}".format(setting, pct))
            ensure_dir(out_dir)
            fasta_path = os.path.join(out_dir, "shrunk_sequences.fasta")
            json_path = os.path.join(out_dir, "deletions.json")
            report_path = os.path.join(out_dir, "planner_deletion_report.txt")
            shrunk_records = []
            deletion_entries = []
            report_rows = []
            for accession, sequence in records:
                segments = selected.get((setting, budget, accession), [])
                positions = merge_positions(segments, len(sequence))
                position_set = set(positions)
                shrunk = "".join(aa for idx, aa in enumerate(sequence) if idx not in position_set)
                header_field = deletion_header_field(sequence, positions)
                fasta_header = "{}|planner={}|budget={:.2f}|deleted={}".format(
                    accession, setting, budget, len(positions)
                )
                shrunk_records.append((fasta_header, shrunk))
                deletion_entries.append(
                    {
                        "header": accession,
                        "original_length": len(sequence),
                        "new_length": len(shrunk),
                        "deletion_positions_zero_based": positions,
                        "deleted_positions": positions,
                        "deletions_header_field": header_field,
                        "planner_setting": setting,
                        "budget_ratio": budget,
                        "num_selected_segments": len(segments),
                    }
                )
                report_rows.append(
                    {
                        "accession": accession,
                        "original_length": len(sequence),
                        "new_length": len(shrunk),
                        "deletion_count": len(positions),
                        "num_selected_segments": len(segments),
                    }
                )
            write_fasta(fasta_path, shrunk_records)
            with open(json_path, "w") as handle:
                json.dump(deletion_entries, handle, indent=2)
            write_report(report_path, report_rows, setting, budget)
            manifest_rows.append(
                {
                    "setting": setting,
                    "budget_ratio": budget,
                    "out_dir": out_dir,
                    "shrunk_fasta": fasta_path,
                    "deletions_json": json_path,
                    "report": report_path,
                }
            )
    manifest_path = os.path.join(args.out_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as handle:
        fields = ["setting", "budget_ratio", "out_dir", "shrunk_fasta", "deletions_json", "report"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print("Exported {} planner deletion runs".format(len(manifest_rows)))
    print("Wrote {}".format(manifest_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Export planner selected segments to deletion FASTA/JSON.")
    parser.add_argument("--core_fasta", required=True)
    parser.add_argument("--selected_segments_csv", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--budgets", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
