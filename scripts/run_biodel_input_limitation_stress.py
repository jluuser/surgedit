#!/usr/bin/env python3
"""Stress-test BioDel input gating and limitation behavior."""

import argparse
import csv
import os
from collections import Counter


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def max_char_fraction(seq):
    if not seq:
        return 0.0
    counts = Counter(seq)
    return max(counts.values()) / float(len(seq))


def classify(seq, min_len, max_len):
    seq = "".join(seq.split()).upper()
    nonstandard = sorted(set(seq) - STANDARD_AA)
    reasons = []
    status = "accepted"
    route = "full_pipeline_candidate"
    if not seq:
        return "reject", "empty", "none"
    if nonstandard:
        return "reject", "nonstandard_symbols:{}".format("".join(nonstandard)), "none"
    if len(seq) < min_len:
        status = "degrade"
        route = "low_confidence_sequence_only"
        reasons.append("shorter_than_training_min")
    if len(seq) > max_len:
        status = "degrade"
        route = "chunk_or_reject_for_stage1"
        reasons.append("longer_than_stage1_max_len")
    if set(seq).issubset(set("ACGT")) and len(set(seq)) >= 3 and len(seq) >= 50:
        status = "review"
        route = "manual_type_check_required"
        reasons.append("dna_like_alphabet_ambiguous_with_amino_acids")
    if max_char_fraction(seq) >= 0.60:
        status = "degrade" if status == "accepted" else status
        route = "low_complexity_conservative"
        reasons.append("low_complexity")
    if not reasons:
        reasons.append("standard_protein_like")
    return status, ";".join(reasons), route


def built_in_cases():
    return [
        ("normal_enzyme_like", "MSELILASTSSARRALMDGLRLPYRAEAPGVDEVVAPHLSVTEAVRELASRKARAVHQRHPEAWVLGADQLVEVAGEVLSKPVDRNAAREQLRKLVGHTHAIHTGVCLVGPGGKVLDAVETTRLTFYRVKEEELERYLDLNEWE"),
        ("too_short_peptide", "MKTLLILAV"),
        ("nonstandard_aa", "MKTLLUBZXO"),
        ("stop_symbol", "MKTLL*ILAVVV"),
        ("dna_like_acgt", "ATGCGTACGTAGCTAGCTAGCGTACGTAGCTAGCTAGCGTACGTAGCTAGCTAGCGTACGTAGCTAGCTA"),
        ("low_complexity_poly_gly", "G" * 120),
        ("long_stage1_limit", "M" + "ACDEFGHIKLMNPQRSTVWY" * 70),
    ]


def run(args):
    rows = []
    for name, seq in built_in_cases():
        status, reason, route = classify(seq, args.min_len, args.max_len)
        rows.append(
            {
                "case_id": name,
                "length": len(seq),
                "status": status,
                "route": route,
                "reason": reason,
                "sequence_preview": seq[:80],
            }
        )
    ensure_parent(args.out_csv)
    fields = ["case_id", "length", "status", "route", "reason", "sequence_preview"]
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["status"] for row in rows)
    ensure_parent(args.out_report)
    with open(args.out_report, "w") as handle:
        handle.write("BioDel input limitation stress test\n\n")
        handle.write("min_len: {}\n".format(args.min_len))
        handle.write("max_len: {}\n".format(args.max_len))
        handle.write("status_counts: {}\n\n".format(dict(counts)))
        handle.write("Key limitations:\n")
        handle.write("- Non-standard symbols are rejected.\n")
        handle.write("- Very short or very long sequences require low-confidence/degraded handling.\n")
        handle.write("- DNA-like A/C/G/T strings are ambiguous because these are also valid amino-acid one-letter codes.\n")
        handle.write("- Low-complexity inputs should trigger conservative mode.\n")
        handle.write("\nBIODEL_INPUT_LIMITATION_STRESS_PASS\n")
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


def parse_args():
    parser = argparse.ArgumentParser(description="Run BioDel input limitation stress test.")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    parser.add_argument("--min_len", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
