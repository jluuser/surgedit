#!/usr/bin/env python3
"""Build a CATH S40 domain-aware BioDel benchmark.

The benchmark treats each CATH S40 domain sequence as a compact structural
unit. Since these are domain-level sequences rather than full proteins, the
main question is conservative: does BioDel avoid deleting the central domain
core and prefer terminal/boundary-compatible segments when it must delete?
"""

import argparse
import csv
import os
from collections import Counter, OrderedDict


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def parse_lengths(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def read_fasta(path):
    records = OrderedDict()
    header = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records[header] = "".join(chunks)
    return records


def cath_domain_id(header):
    token = header.split("|")[-1]
    return token.split("/")[0]


def unknown_fraction(sequence):
    if not sequence:
        return 1.0
    return sum(1 for aa in sequence if aa not in AA20) / float(len(sequence))


def overlap_len(a0, a1, b0, b1):
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo + 1)


def add_segment(segments, seen, start, end, source):
    if start < 0 or end < start:
        return
    key = (start, end, source)
    if key in seen:
        return
    seen.add(key)
    segments.append((start, end, source))


def build_segments(length, lengths, core_margin):
    core_start = int(round(length * core_margin))
    core_end = length - core_start - 1
    terminal_end = max(0, int(round(length * 0.10)) - 1)
    c_terminal_start = min(length - 1, length - terminal_end - 1)
    segments = []
    seen = set()

    for seg_len in lengths:
        if seg_len > length:
            continue
        add_segment(segments, seen, 0, seg_len - 1, "n_terminal_domain_edge")
        add_segment(segments, seen, length - seg_len, length - 1, "c_terminal_domain_edge")

        for center_frac, source in [
            (0.25, "domain_boundary_proxy"),
            (0.50, "domain_core_challenge"),
            (0.75, "domain_boundary_proxy"),
        ]:
            center = int(round((length - 1) * center_frac))
            start = max(0, min(length - seg_len, center - seg_len // 2))
            add_segment(segments, seen, start, start + seg_len - 1, source)

        for boundary, source in [
            (core_start, "core_boundary_crossing_challenge"),
            (core_end, "core_boundary_crossing_challenge"),
        ]:
            start = max(0, min(length - seg_len, boundary - seg_len // 2))
            add_segment(segments, seen, start, start + seg_len - 1, source)

    return segments, core_start, core_end, terminal_end, c_terminal_start


def fmt(value):
    return "{:.6f}".format(float(value))


def convert_segment(domain_id, sequence, start, end, source, core_start, core_end, terminal_end, c_terminal_start, args):
    length = len(sequence)
    seg_len = end - start + 1
    core_overlap = overlap_len(start, end, core_start, core_end)
    nterm_overlap = overlap_len(start, end, 0, terminal_end)
    cterm_overlap = overlap_len(start, end, c_terminal_start, length - 1)
    terminal_overlap = nterm_overlap + cterm_overlap
    core_fraction = core_overlap / float(seg_len)
    terminal_fraction = terminal_overlap / float(seg_len)
    boundary_crossing = int(start < core_start <= end or start <= core_end < end)
    closure_friendly = source in ("n_terminal_domain_edge", "c_terminal_domain_edge")
    hard_reject = core_fraction >= args.hard_reject_core_fraction
    structural_risk = min(1.0, 0.85 * core_fraction + 0.15 * boundary_crossing)
    bioprior = (
        0.8 * terminal_fraction
        + 0.3 * (1.0 - core_fraction)
        - 1.2 * core_fraction
        - 0.4 * boundary_crossing
        - (0.3 if not closure_friendly else 0.0)
    )
    return {
        "accession": domain_id,
        "protein_length": length,
        "seg_start": start,
        "seg_end": end,
        "seg_len": seg_len,
        "proposal_source": source,
        "biological_rationale": "CATH S40 domain-core protection benchmark",
        "n_protected_overlap": core_overlap,
        "protected_overlap_fraction": fmt(core_fraction),
        "n_shadow_overlap": boundary_crossing * seg_len,
        "shadow_overlap_fraction": fmt(boundary_crossing),
        "min_distance_to_protected": 0 if core_overlap else min(abs(end - core_start), abs(start - core_end)),
        "mean_distance_to_protected": "",
        "mean_contact_density_8A": "",
        "max_contact_density_8A": "",
        "mean_motif_contact_count_8A": "",
        "max_motif_contact_count_8A": "",
        "mean_pLDDT": "",
        "min_pLDDT": "",
        "low_pLDDT_fraction": "",
        "high_pLDDT_fraction": "",
        "terminal_overlap_fraction": fmt(terminal_fraction),
        "boundary_ca_distance": "",
        "closure_type": "terminal" if closure_friendly else "internal_compact_domain",
        "closure_friendly_8A": str(closure_friendly),
        "terminal_tail_score": fmt(terminal_fraction),
        "surface_flexible_loop_score": fmt(max(0.0, terminal_fraction - core_fraction)),
        "disorder_like_score": fmt(terminal_fraction * 0.25),
        "linker_compressibility_score": fmt(terminal_fraction),
        "functional_core_risk_score": fmt(core_fraction),
        "structural_core_risk_score": fmt(structural_risk),
        "motif_shadow_risk_score": fmt(boundary_crossing),
        "geometric_closure_score": fmt(0.0 if closure_friendly else 1.0),
        "final_bioprior_score": fmt(bioprior),
        "hard_reject": str(hard_reject),
        "reject_reason": "domain_core_overlap" if hard_reject else "",
        "domain_core_start": core_start,
        "domain_core_end": core_end,
        "domain_core_overlap_fraction": fmt(core_fraction),
        "domain_boundary_crossing": boundary_crossing,
        "domain_terminal_overlap_fraction": fmt(terminal_fraction),
    }


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    records = read_fasta(args.s40_fasta)
    allowed = None
    if args.s40_list:
        with open(args.s40_list) as handle:
            allowed = set(line.strip() for line in handle if line.strip())

    metadata_rows = []
    segment_rows = []
    stats = Counter()
    lengths = parse_lengths(args.segment_lengths)

    for header, sequence in records.items():
        domain_id = cath_domain_id(header)
        if allowed is not None and domain_id not in allowed:
            continue
        stats["seen_domains"] += 1
        if args.max_domains and len(metadata_rows) >= args.max_domains:
            break
        seq_len = len(sequence)
        if seq_len < args.min_len or seq_len > args.max_len:
            stats["skipped_length"] += 1
            continue
        if unknown_fraction(sequence) > args.max_unknown_fraction:
            stats["skipped_unknown"] += 1
            continue
        metadata_rows.append(
            {
                "accession": domain_id,
                "protein_id": domain_id,
                "sequence": sequence,
                "length": seq_len,
                "dataset": "cath_s40",
                "source_header": header,
            }
        )
        segments, core_start, core_end, terminal_end, c_terminal_start = build_segments(seq_len, lengths, args.core_margin)
        for start, end, source in segments:
            segment_rows.append(
                convert_segment(domain_id, sequence, start, end, source, core_start, core_end, terminal_end, c_terminal_start, args)
            )

    segment_fields = [
        "accession", "protein_length", "seg_start", "seg_end", "seg_len",
        "proposal_source", "biological_rationale", "n_protected_overlap",
        "protected_overlap_fraction", "n_shadow_overlap", "shadow_overlap_fraction",
        "min_distance_to_protected", "mean_distance_to_protected",
        "mean_contact_density_8A", "max_contact_density_8A",
        "mean_motif_contact_count_8A", "max_motif_contact_count_8A",
        "mean_pLDDT", "min_pLDDT", "low_pLDDT_fraction", "high_pLDDT_fraction",
        "terminal_overlap_fraction", "boundary_ca_distance", "closure_type",
        "closure_friendly_8A", "terminal_tail_score", "surface_flexible_loop_score",
        "disorder_like_score", "linker_compressibility_score",
        "functional_core_risk_score", "structural_core_risk_score",
        "motif_shadow_risk_score", "geometric_closure_score", "final_bioprior_score",
        "hard_reject", "reject_reason", "domain_core_start", "domain_core_end",
        "domain_core_overlap_fraction", "domain_boundary_crossing",
        "domain_terminal_overlap_fraction",
    ]
    metadata_fields = ["accession", "protein_id", "sequence", "length", "dataset", "source_header"]
    write_csv(args.out_core_csv, metadata_rows, metadata_fields)
    write_csv(args.out_segments_csv, segment_rows, segment_fields)

    hard = sum(1 for row in segment_rows if row["hard_reject"] == "True")
    boundary = sum(1 for row in segment_rows if int(row["domain_boundary_crossing"]) > 0)
    terminal = sum(1 for row in segment_rows if float(row["domain_terminal_overlap_fraction"]) > 0)
    ensure_parent(args.summary_txt)
    with open(args.summary_txt, "w") as handle:
        handle.write("CATH S40 BioDel domain benchmark summary\n\n")
        handle.write("s40_fasta: {}\n".format(args.s40_fasta))
        handle.write("s40_list: {}\n".format(args.s40_list))
        handle.write("out_core_csv: {}\n".format(args.out_core_csv))
        handle.write("out_segments_csv: {}\n".format(args.out_segments_csv))
        handle.write("domains: {}\n".format(len(metadata_rows)))
        handle.write("segments: {}\n".format(len(segment_rows)))
        handle.write("hard_reject_domain_core_segments: {}\n".format(hard))
        handle.write("boundary_crossing_segments: {}\n".format(boundary))
        handle.write("terminal_edge_segments: {}\n".format(terminal))
        handle.write("skipped_length: {}\n".format(stats["skipped_length"]))
        handle.write("skipped_unknown: {}\n".format(stats["skipped_unknown"]))
        handle.write("\nCATH_S40_BIODEL_BENCHMARK_PASS\n")
    print("Wrote {}".format(args.out_core_csv))
    print("Wrote {}".format(args.out_segments_csv))
    print("Wrote {}".format(args.summary_txt))


def parse_args():
    parser = argparse.ArgumentParser(description="Build CATH S40 domain-aware BioDel benchmark.")
    parser.add_argument("--s40_fasta", required=True)
    parser.add_argument("--s40_list", default="")
    parser.add_argument("--out_core_csv", required=True)
    parser.add_argument("--out_segments_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--max_domains", type=int, default=5000)
    parser.add_argument("--min_len", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=1200)
    parser.add_argument("--segment_lengths", default="5,10,15,20,30")
    parser.add_argument("--core_margin", type=float, default=0.20)
    parser.add_argument("--hard_reject_core_fraction", type=float, default=0.50)
    parser.add_argument("--max_unknown_fraction", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
