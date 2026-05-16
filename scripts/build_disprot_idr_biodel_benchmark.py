#!/usr/bin/env python3
"""Build a DisProt IDR BioDel benchmark.

This benchmark uses full DisProt protein sequences and annotated regions from
the JSON release. It asks whether BioDel is permissive in non-functional IDR
segments while remaining conservative on functional disorder annotations.
"""

import argparse
import csv
import json
import os
from collections import Counter


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def parse_lengths(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def unknown_fraction(sequence):
    if not sequence:
        return 1.0
    return sum(1 for aa in sequence if aa not in AA20) / float(len(sequence))


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def overlap_len(a0, a1, b0, b1):
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo + 1)


def total_overlap(start, end, intervals):
    return sum(overlap_len(start, end, lo, hi) for lo, hi in intervals)


def complement_intervals(length, intervals):
    intervals = merge_intervals(intervals)
    out = []
    pos = 0
    for start, end in intervals:
        if pos < start:
            out.append((pos, start - 1))
        pos = end + 1
    if pos < length:
        out.append((pos, length - 1))
    return out


def clamp_segment(start, seg_len, length):
    start = max(0, min(length - seg_len, start))
    return start, start + seg_len - 1


def add_segment(segments, seen, start, end, source):
    if start < 0 or end < start:
        return
    key = (start, end, source)
    if key in seen:
        return
    seen.add(key)
    segments.append((start, end, source))


def is_disorder_region(region):
    return (
        str(region.get("term_namespace", "")).lower() == "structural state"
        and str(region.get("term_name", "")).lower() == "disorder"
    )


def is_functional_region(region):
    if is_disorder_region(region):
        return False
    namespace = str(region.get("term_namespace", "")).lower()
    term_name = str(region.get("term_name", "")).lower()
    if namespace in ("molecular function", "biological process", "cellular component"):
        return True
    if "binding" in term_name or "transition" in term_name or "order" in term_name:
        return True
    if str(region.get("term_is_binding", "")).lower() in ("true", "1", "yes"):
        return True
    return False


def region_interval(region, length):
    start = int(region.get("start", 1)) - 1
    end = int(region.get("end", 0)) - 1
    start = max(0, min(length - 1, start))
    end = max(0, min(length - 1, end))
    if end < start:
        return None
    return start, end


def build_segments(length, idr_intervals, functional_intervals, lengths):
    segments = []
    seen = set()
    ordered_intervals = complement_intervals(length, idr_intervals)

    for seg_len in lengths:
        if seg_len > length:
            continue
        add_segment(segments, seen, 0, seg_len - 1, "n_terminal_context")
        add_segment(segments, seen, length - seg_len, length - 1, "c_terminal_context")

        for lo, hi in idr_intervals:
            if hi - lo + 1 < seg_len:
                continue
            for frac in (0.25, 0.50, 0.75):
                center = lo + int(round((hi - lo) * frac))
                start, end = clamp_segment(center - seg_len // 2, seg_len, length)
                add_segment(segments, seen, start, end, "idr_candidate")

        for lo, hi in functional_intervals:
            if hi - lo + 1 < min(seg_len, 5):
                continue
            center = (lo + hi) // 2
            start, end = clamp_segment(center - seg_len // 2, seg_len, length)
            add_segment(segments, seen, start, end, "functional_idr_challenge")

        for lo, hi in ordered_intervals[:4]:
            if hi - lo + 1 < seg_len:
                continue
            center = (lo + hi) // 2
            start, end = clamp_segment(center - seg_len // 2, seg_len, length)
            add_segment(segments, seen, start, end, "ordered_control")

    return segments


def fmt(value):
    return "{:.6f}".format(float(value))


def convert_segment(protein, start, end, source, idr_intervals, functional_intervals):
    length = int(protein["length"])
    seg_len = end - start + 1
    idr_overlap = total_overlap(start, end, idr_intervals)
    functional_overlap = total_overlap(start, end, functional_intervals)
    idr_fraction = idr_overlap / float(seg_len)
    functional_fraction = functional_overlap / float(seg_len)
    ordered_fraction = max(0.0, 1.0 - idr_fraction)
    terminal_fraction = 0.0
    terminal_len = max(1, int(round(length * 0.10)))
    terminal_fraction += overlap_len(start, end, 0, terminal_len - 1) / float(seg_len)
    terminal_fraction += overlap_len(start, end, length - terminal_len, length - 1) / float(seg_len)
    terminal_fraction = min(1.0, terminal_fraction)
    hard_reject = functional_overlap > 0
    closure_friendly = idr_fraction >= 0.80 or terminal_fraction > 0.0
    bioprior = (
        1.0 * idr_fraction
        + 0.25 * terminal_fraction
        - 1.4 * functional_fraction
        - 0.45 * ordered_fraction
    )
    return {
        "accession": protein["acc"],
        "protein_length": length,
        "seg_start": start,
        "seg_end": end,
        "seg_len": seg_len,
        "proposal_source": source,
        "biological_rationale": "DisProt IDR permissiveness and functional-IDR protection benchmark",
        "n_protected_overlap": functional_overlap,
        "protected_overlap_fraction": fmt(functional_fraction),
        "n_shadow_overlap": 0,
        "shadow_overlap_fraction": fmt(0.0),
        "min_distance_to_protected": 0 if functional_overlap else "",
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
        "closure_type": "",
        "closure_friendly_8A": str(closure_friendly),
        "terminal_tail_score": fmt(terminal_fraction),
        "surface_flexible_loop_score": fmt(idr_fraction),
        "disorder_like_score": fmt(idr_fraction),
        "linker_compressibility_score": fmt(max(idr_fraction, terminal_fraction)),
        "functional_core_risk_score": fmt(functional_fraction),
        "structural_core_risk_score": "",
        "motif_shadow_risk_score": fmt(0.0),
        "geometric_closure_score": "",
        "final_bioprior_score": fmt(bioprior),
        "hard_reject": str(hard_reject),
        "reject_reason": "functional_disorder_overlap" if hard_reject else "",
        "idr_overlap_fraction": fmt(idr_fraction),
        "functional_idr_overlap_fraction": fmt(functional_fraction),
        "ordered_overlap_fraction": fmt(ordered_fraction),
        "disorder_content": fmt(protein.get("disorder_content", 0.0)),
    }


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    with open(args.disprot_json) as handle:
        payload = json.load(handle)
    proteins = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    lengths = parse_lengths(args.segment_lengths)
    metadata_rows = []
    segment_rows = []
    stats = Counter()

    for protein in proteins:
        if args.max_proteins and len(metadata_rows) >= args.max_proteins:
            break
        sequence = protein.get("sequence") or ""
        seq_len = len(sequence)
        stats["seen_proteins"] += 1
        if seq_len < args.min_len or seq_len > args.max_len:
            stats["skipped_length"] += 1
            continue
        if unknown_fraction(sequence) > args.max_unknown_fraction:
            stats["skipped_unknown"] += 1
            continue
        idr_intervals = []
        functional_intervals = []
        for region in protein.get("regions", []):
            interval = region_interval(region, seq_len)
            if interval is None:
                continue
            if is_disorder_region(region):
                idr_intervals.append(interval)
            elif is_functional_region(region):
                functional_intervals.append(interval)
        idr_intervals = merge_intervals(idr_intervals)
        functional_intervals = merge_intervals(functional_intervals)
        if not idr_intervals:
            stats["skipped_no_idr"] += 1
            continue
        metadata_rows.append(
            {
                "accession": protein["acc"],
                "protein_id": protein["acc"],
                "sequence": sequence,
                "length": seq_len,
                "dataset": "disprot",
                "disprot_id": protein.get("disprot_id", ""),
                "name": protein.get("name", ""),
                "organism": protein.get("organism", ""),
                "disorder_content": fmt(protein.get("disorder_content", 0.0)),
                "n_idr_regions": len(idr_intervals),
                "n_functional_regions": len(functional_intervals),
            }
        )
        for start, end, source in build_segments(seq_len, idr_intervals, functional_intervals, lengths):
            segment_rows.append(convert_segment(protein, start, end, source, idr_intervals, functional_intervals))

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
        "hard_reject", "reject_reason", "idr_overlap_fraction",
        "functional_idr_overlap_fraction", "ordered_overlap_fraction", "disorder_content",
    ]
    metadata_fields = [
        "accession", "protein_id", "sequence", "length", "dataset", "disprot_id",
        "name", "organism", "disorder_content", "n_idr_regions", "n_functional_regions",
    ]
    write_csv(args.out_core_csv, metadata_rows, metadata_fields)
    write_csv(args.out_segments_csv, segment_rows, segment_fields)

    hard = sum(1 for row in segment_rows if row["hard_reject"] == "True")
    idr_candidates = sum(1 for row in segment_rows if float(row["idr_overlap_fraction"]) >= 0.8)
    functional = sum(1 for row in segment_rows if float(row["functional_idr_overlap_fraction"]) > 0)
    ensure_parent(args.summary_txt)
    with open(args.summary_txt, "w") as handle:
        handle.write("DisProt BioDel IDR benchmark summary\n\n")
        handle.write("disprot_json: {}\n".format(args.disprot_json))
        handle.write("out_core_csv: {}\n".format(args.out_core_csv))
        handle.write("out_segments_csv: {}\n".format(args.out_segments_csv))
        handle.write("proteins: {}\n".format(len(metadata_rows)))
        handle.write("segments: {}\n".format(len(segment_rows)))
        handle.write("idr_candidate_segments: {}\n".format(idr_candidates))
        handle.write("functional_idr_overlap_segments: {}\n".format(functional))
        handle.write("hard_reject_functional_segments: {}\n".format(hard))
        handle.write("skipped_length: {}\n".format(stats["skipped_length"]))
        handle.write("skipped_unknown: {}\n".format(stats["skipped_unknown"]))
        handle.write("skipped_no_idr: {}\n".format(stats["skipped_no_idr"]))
        handle.write("\nDISPROT_IDR_BIODEL_BENCHMARK_PASS\n")
    print("Wrote {}".format(args.out_core_csv))
    print("Wrote {}".format(args.out_segments_csv))
    print("Wrote {}".format(args.summary_txt))


def parse_args():
    parser = argparse.ArgumentParser(description="Build DisProt IDR BioDel benchmark.")
    parser.add_argument("--disprot_json", required=True)
    parser.add_argument("--out_core_csv", required=True)
    parser.add_argument("--out_segments_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--max_proteins", type=int, default=2500)
    parser.add_argument("--min_len", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=1200)
    parser.add_argument("--segment_lengths", default="5,10,15,20,30")
    parser.add_argument("--max_unknown_fraction", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
