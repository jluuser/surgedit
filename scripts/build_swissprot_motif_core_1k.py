#!/usr/bin/env python3
"""Build a Core-1K Swiss-Prot motif-constrained protein dataset."""

from __future__ import print_function

import argparse
import gzip
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
FEATURE_PRIORITY = [
    "active site",
    "binding site",
    "site",
    "short sequence motif",
]
FEATURE_SET = set(FEATURE_PRIORITY)
DEFAULT_QUOTAS = {
    "active site": 300,
    "binding site": 300,
    "site": 200,
    "short sequence motif": 200,
}
LENGTH_BINS = [
    ("100-200", 100, 200),
    ("201-400", 201, 400),
    ("401-600", 401, 600),
    ("601-800", 601, 800),
]


try:
    basestring
except NameError:
    basestring = str


def local_name(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def direct_child(element, name):
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def clean_sequence(value):
    if value is None:
        return ""
    return "".join(value.split())


def first_accession(entry):
    for child in list(entry):
        if local_name(child.tag) == "accession":
            return (child.text or "").strip()
    return ""


def entry_name(entry):
    child = direct_child(entry, "name")
    return (child.text or "").strip() if child is not None else ""


def recommended_protein_name(entry, fallback):
    protein = direct_child(entry, "protein")
    if protein is None:
        return fallback
    for child in protein.iter():
        if local_name(child.tag) != "recommendedName":
            continue
        for item in child.iter():
            if local_name(item.tag) == "fullName":
                text = item.text or ""
                if text.strip():
                    return text.strip()
    return fallback


def organism_name(entry):
    organism = direct_child(entry, "organism")
    if organism is None:
        return ""
    first_name = ""
    for child in list(organism):
        if local_name(child.tag) != "name":
            continue
        text = child.text or ""
        if not first_name and text.strip():
            first_name = text.strip()
        if child.get("type") == "scientific" and text.strip():
            return text.strip()
    return first_name


def gene_name(entry):
    gene = direct_child(entry, "gene")
    if gene is None:
        return ""
    first_name = ""
    for child in list(gene):
        if local_name(child.tag) != "name":
            continue
        text = child.text or ""
        if not first_name and text.strip():
            first_name = text.strip()
        if child.get("type") == "primary" and text.strip():
            return text.strip()
    return first_name


def entry_sequence(entry):
    sequence = direct_child(entry, "sequence")
    if sequence is None:
        return ""
    return clean_sequence(sequence.text)


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_location(feature):
    location = direct_child(feature, "location")
    if location is None:
        return {"kind": "fuzzy", "start": None, "end": None, "label": "missing"}

    position = direct_child(location, "position")
    if position is not None:
        pos = position.get("position")
        status = position.get("status")
        parsed = safe_int(pos)
        if parsed is not None and not status:
            return {
                "kind": "single",
                "start": parsed,
                "end": parsed,
                "label": "position:{0}".format(parsed),
            }
        return {
            "kind": "fuzzy",
            "start": None,
            "end": None,
            "label": "position:{0};status:{1}".format(pos or "", status or ""),
        }

    begin = direct_child(location, "begin")
    end = direct_child(location, "end")
    if begin is not None and end is not None:
        begin_pos = begin.get("position")
        end_pos = end.get("position")
        begin_status = begin.get("status")
        end_status = end.get("status")
        parsed_begin = safe_int(begin_pos)
        parsed_end = safe_int(end_pos)
        if (
            parsed_begin is not None
            and parsed_end is not None
            and parsed_begin <= parsed_end
            and not begin_status
            and not end_status
        ):
            return {
                "kind": "range",
                "start": parsed_begin,
                "end": parsed_end,
                "label": "range:{0}-{1}".format(parsed_begin, parsed_end),
            }
        return {
            "kind": "fuzzy",
            "start": None,
            "end": None,
            "label": "range:{0}-{1};status:{2}/{3}".format(
                begin_pos or "", end_pos or "", begin_status or "", end_status or ""
            ),
        }

    return {"kind": "fuzzy", "start": None, "end": None, "label": "unrecognized"}


def feature_description(feature_type, loc, description):
    return "{0}:{1}:{2}".format(feature_type, loc["label"], description or "")


def usable_feature_positions(feature, seq_len, stats):
    feature_type = feature.get("type", "")
    if feature_type not in FEATURE_SET:
        return [], None

    loc = parse_location(feature)
    description = feature.get("description", "")
    if loc["kind"] == "fuzzy":
        stats["skipped_fuzzy_features"] += 1
        return [], None

    if loc["kind"] == "single":
        pos0 = loc["start"] - 1
        if 0 <= pos0 < seq_len:
            return [pos0], feature_description(feature_type, loc, description)
        stats["skipped_out_of_bounds_features"] += 1
        return [], None

    range_len = loc["end"] - loc["start"] + 1
    if range_len > 20:
        stats["skipped_long_range_features"] += 1
        return [], None

    start0 = loc["start"] - 1
    end0 = loc["end"] - 1
    if start0 < 0 or end0 >= seq_len:
        stats["skipped_out_of_bounds_features"] += 1
        return [], None
    return list(range(start0, end0 + 1)), feature_description(
        feature_type, loc, description
    )


def primary_type(protected_types):
    for feature_type in FEATURE_PRIORITY:
        if feature_type in protected_types:
            return feature_type
    return ""


def length_bin(length):
    for label, lo, hi in LENGTH_BINS:
        if lo <= length <= hi:
            return label
    return "outside"


def is_cterminal_short_motif(candidate):
    if candidate["primary_protected_type"] != "short sequence motif":
        return False
    positions = candidate["protected_positions_list"]
    return bool(positions) and all(pos >= candidate["length"] - 10 for pos in positions)


def csv_escape(value):
    if value is None:
        value = ""
    if not isinstance(value, basestring):
        value = str(value)
    value = value.replace("\r", " ").replace("\n", " ")
    if '"' in value:
        value = value.replace('"', '""')
    if "," in value or '"' in value:
        value = '"{0}"'.format(value)
    return value


def encode_utf8(value):
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8")


def write_csv(path, fieldnames, rows):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "wb") as handle:
        handle.write(encode_utf8(",".join(fieldnames) + "\n"))
        for row in rows:
            line = ",".join(csv_escape(row.get(field, "")) for field in fieldnames)
            handle.write(encode_utf8(line + "\n"))


def write_fasta(path, rows):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "wb") as handle:
        for row in rows:
            header = "{0}|primary={1}".format(
                row["accession"], row["primary_protected_type"].replace(" ", "_")
            )
            handle.write(encode_utf8(">{0}\n".format(header)))
            seq = row["sequence"]
            for i in range(0, len(seq), 80):
                handle.write(encode_utf8(seq[i : i + 80] + "\n"))


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def collect_candidates(args):
    candidates = []
    stats = defaultdict(int)
    seen_accessions = set()

    with gzip.open(args.xml, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for event, elem in context:
            if local_name(elem.tag) != "entry":
                continue

            stats["scanned_entries"] += 1
            accession = first_accession(elem)
            if not accession or accession in seen_accessions:
                stats["skipped_duplicate_accession"] += int(bool(accession))
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue
            seen_accessions.add(accession)

            sequence = entry_sequence(elem)
            seq_len = len(sequence)
            if args.min_len <= seq_len <= args.max_len:
                stats["length_in_range"] += 1
            else:
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue

            if sequence and set(sequence).issubset(STANDARD_AA):
                stats["length_and_standard_20aa"] += 1
            else:
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue

            protected_positions = set()
            protected_types_seen = set()
            descriptions = []
            feature_type_counts = defaultdict(int)

            for feature in list(elem):
                if local_name(feature.tag) != "feature":
                    continue
                feature_type = feature.get("type", "")
                if feature_type not in FEATURE_SET:
                    continue
                positions, description = usable_feature_positions(feature, seq_len, stats)
                if not positions:
                    continue
                protected_positions.update(positions)
                protected_types_seen.add(feature_type)
                feature_type_counts[feature_type] += 1
                if description is not None:
                    descriptions.append(description)

            if not protected_positions:
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue
            stats["entries_with_usable_protected_features"] += 1

            sorted_positions = sorted(protected_positions)
            if len(sorted_positions) > args.max_protected:
                stats["skipped_too_many_protected_positions"] += 1
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue
            if len(sorted_positions) > int(seq_len * args.max_protected_fraction):
                stats["skipped_too_high_protected_fraction"] += 1
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue

            ptype = primary_type(protected_types_seen)
            protected_types = ";".join(
                item for item in FEATURE_PRIORITY if item in protected_types_seen
            )
            feature_counts = ";".join(
                "{0}:{1}".format(item, feature_type_counts[item])
                for item in FEATURE_PRIORITY
                if feature_type_counts[item] > 0
            )
            candidate = {
                "protein_id": accession,
                "accession": accession,
                "entry_name": entry_name(elem),
                "sequence": sequence,
                "length": seq_len,
                "protected_positions_list": sorted_positions,
                "protected_positions": ";".join(str(pos) for pos in sorted_positions),
                "protected_types": protected_types,
                "protected_feature_descriptions": "|".join(descriptions),
                "primary_protected_type": ptype,
                "n_protected": len(sorted_positions),
                "protected_fraction": float(len(sorted_positions)) / seq_len,
                "feature_counts": feature_counts,
                "organism": organism_name(elem),
                "protein_name": recommended_protein_name(elem, accession),
                "gene_name": gene_name(elem),
                "length_bin": length_bin(seq_len),
                "feature_type_counts": dict(feature_type_counts),
            }
            candidate["is_cterminal_short_motif"] = is_cterminal_short_motif(candidate)
            candidates.append(candidate)

            elem.clear()
            if stats["scanned_entries"] >= args.max_entries:
                break

    return candidates, stats


def select_round_robin(candidates, quota, selected, selected_accessions, selected_sequences, counters, max_cterminal):
    by_bin = defaultdict(list)
    for candidate in candidates:
        by_bin[candidate["length_bin"]].append(candidate)

    labels = [item[0] for item in LENGTH_BINS]
    progress = True
    while progress and quota > 0:
        progress = False
        for label in labels:
            bucket = by_bin[label]
            while bucket:
                candidate = bucket.pop(0)
                if candidate["accession"] in selected_accessions:
                    continue
                if candidate["sequence"] in selected_sequences:
                    counters["duplicate_sequence_removed"] += 1
                    continue
                if (
                    candidate["is_cterminal_short_motif"]
                    and counters["cterminal_short_motif"] >= max_cterminal
                ):
                    counters["cterminal_short_motif_skipped"] += 1
                    continue
                selected.append(candidate)
                selected_accessions.add(candidate["accession"])
                selected_sequences.add(candidate["sequence"])
                counters["selected_{0}".format(candidate["primary_protected_type"])] += 1
                if candidate["is_cterminal_short_motif"]:
                    counters["cterminal_short_motif"] += 1
                quota -= 1
                progress = True
                break
            if quota <= 0:
                break
    return quota


def sample_core(candidates, args):
    rng = random.Random(args.seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    by_type = defaultdict(list)
    for candidate in shuffled:
        by_type[candidate["primary_protected_type"]].append(candidate)
    for feature_type in by_type:
        rng.shuffle(by_type[feature_type])

    selected = []
    selected_accessions = set()
    selected_sequences = set()
    counters = defaultdict(int)

    quotas = dict(DEFAULT_QUOTAS)
    if args.target_size != 1000:
        scale = float(args.target_size) / 1000.0
        quotas = {key: int(round(value * scale)) for key, value in quotas.items()}
        delta = args.target_size - sum(quotas.values())
        for feature_type in FEATURE_PRIORITY:
            if delta == 0:
                break
            quotas[feature_type] += 1 if delta > 0 else -1
            delta += -1 if delta > 0 else 1

    for feature_type in FEATURE_PRIORITY:
        before = len(selected)
        remaining = select_round_robin(
            by_type[feature_type],
            quotas[feature_type],
            selected,
            selected_accessions,
            selected_sequences,
            counters,
            args.max_cterminal_short_motif,
        )
        counters["quota_requested_{0}".format(feature_type)] = quotas[feature_type]
        counters["quota_filled_{0}".format(feature_type)] = len(selected) - before
        counters["quota_unfilled_{0}".format(feature_type)] = remaining

    if len(selected) < args.target_size:
        remaining_candidates = [
            candidate
            for candidate in shuffled
            if candidate["accession"] not in selected_accessions
        ]
        rng.shuffle(remaining_candidates)
        select_round_robin(
            remaining_candidates,
            args.target_size - len(selected),
            selected,
            selected_accessions,
            selected_sequences,
            counters,
            args.max_cterminal_short_motif,
        )

    return selected[: args.target_size], counters, quotas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Core-1K Swiss-Prot motif-constrained dataset."
    )
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_fasta", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--target_size", type=int, default=1000)
    parser.add_argument("--max_entries", type=int, default=200000)
    parser.add_argument("--min_len", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--max_protected", type=int, default=30)
    parser.add_argument("--max_protected_fraction", type=float, default=0.20)
    parser.add_argument("--max_cterminal_short_motif", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_summary(path, args, candidates, selected, stats, counters, quotas):
    primary_dist = defaultdict(int)
    protected_type_dist = defaultdict(int)
    length_bin_dist = defaultdict(int)
    feature_annotation_counts = defaultdict(int)
    cterminal_short = 0

    for row in selected:
        primary_dist[row["primary_protected_type"]] += 1
        protected_type_dist[row["protected_types"]] += 1
        length_bin_dist[row["length_bin"]] += 1
        cterminal_short += int(row["is_cterminal_short_motif"])
        for feature_type, count in row["feature_type_counts"].items():
            feature_annotation_counts[feature_type] += count

    lengths = [row["length"] for row in selected]
    protected_counts = [row["n_protected"] for row in selected]
    protected_fractions = [row["protected_fraction"] for row in selected]

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    lines = []
    lines.append("Swiss-Prot Core-1K motif-constrained dataset summary")
    lines.append("")
    lines.append("Input XML: {0}".format(args.xml))
    lines.append("Output CSV: {0}".format(args.out_csv))
    lines.append("Output FASTA: {0}".format(args.out_fasta))
    lines.append("Seed: {0}".format(args.seed))
    lines.append("Target size: {0}".format(args.target_size))
    lines.append("Length range: {0}-{1}".format(args.min_len, args.max_len))
    lines.append("")
    lines.append("Scanned Swiss-Prot entries: {0}".format(stats["scanned_entries"]))
    lines.append("Length in range: {0}".format(stats["length_in_range"]))
    lines.append("Length in range and standard 20AA: {0}".format(stats["length_and_standard_20aa"]))
    lines.append("Entries with usable protected features: {0}".format(stats["entries_with_usable_protected_features"]))
    lines.append("Candidate pool count after protected-position filters: {0}".format(len(candidates)))
    lines.append("Final output count: {0}".format(len(selected)))
    lines.append("")
    lines.append("Quota requested / filled / unfilled:")
    for feature_type in FEATURE_PRIORITY:
        lines.append(
            "- {0}: requested={1}, filled={2}, unfilled={3}".format(
                feature_type,
                quotas[feature_type],
                counters["quota_filled_{0}".format(feature_type)],
                counters["quota_unfilled_{0}".format(feature_type)],
            )
        )
    lines.append("")
    lines.append("Primary protected type distribution:")
    for feature_type in FEATURE_PRIORITY:
        lines.append("- {0}: {1}".format(feature_type, primary_dist[feature_type]))
    lines.append("")
    lines.append("Protected type combination distribution:")
    for key in sorted(protected_type_dist):
        lines.append("- {0}: {1}".format(key, protected_type_dist[key]))
    lines.append("")
    if lengths:
        lines.append("Length min / max / mean: {0} / {1} / {2:.2f}".format(min(lengths), max(lengths), mean(lengths)))
    lines.append("Length bin distribution:")
    for label, lo, hi in LENGTH_BINS:
        lines.append("- {0}: {1}".format(label, length_bin_dist[label]))
    if protected_counts:
        lines.append(
            "Protected count min / max / mean: {0} / {1} / {2:.2f}".format(
                min(protected_counts), max(protected_counts), mean(protected_counts)
            )
        )
        lines.append(
            "Protected fraction min / max / mean: {0:.4f} / {1:.4f} / {2:.4f}".format(
                min(protected_fractions), max(protected_fractions), mean(protected_fractions)
            )
        )
    lines.append("")
    lines.append("Selected feature annotation counts:")
    for feature_type in FEATURE_PRIORITY:
        lines.append("- {0}: {1}".format(feature_type, feature_annotation_counts[feature_type]))
    lines.append("")
    lines.append("C-terminal short sequence motif samples: {0}".format(cterminal_short))
    lines.append("C-terminal short sequence motif cap: {0}".format(args.max_cterminal_short_motif))
    lines.append("Skipped by C-terminal motif cap: {0}".format(counters["cterminal_short_motif_skipped"]))
    lines.append("Removed exact duplicate sequences during selection: {0}".format(counters["duplicate_sequence_removed"]))
    lines.append("90% sequence identity filtering: not applied for Core-1K prototype; exact duplicate sequence removal was applied.")
    lines.append("")
    lines.append("Skipped fuzzy features: {0}".format(stats["skipped_fuzzy_features"]))
    lines.append("Skipped range features longer than 20 aa: {0}".format(stats["skipped_long_range_features"]))
    lines.append("Skipped out-of-bounds features: {0}".format(stats["skipped_out_of_bounds_features"]))
    lines.append("Skipped entries with too many protected positions: {0}".format(stats["skipped_too_many_protected_positions"]))
    lines.append("Skipped entries with too high protected fraction: {0}".format(stats["skipped_too_high_protected_fraction"]))
    lines.append("")
    lines.append("First 20 samples:")
    for row in selected[:20]:
        letters = ";".join(
            "{0}{1}".format(row["sequence"][pos], pos)
            for pos in row["protected_positions_list"]
        )
        lines.append(
            "- {0}\tentry={1}\tlength={2}\tprimary={3}\tn_protected={4}\tpositions={5}\tletters={6}".format(
                row["accession"],
                row["entry_name"],
                row["length"],
                row["primary_protected_type"],
                row["n_protected"],
                row["protected_positions"],
                letters,
            )
        )
    lines.append("")
    lines.append("CORE_1K_BUILD_PASS" if selected else "CORE_1K_BUILD_FAIL")

    with open(path, "wb") as handle:
        handle.write(encode_utf8("\n".join(lines) + "\n"))


def main():
    args = parse_args()
    candidates, stats = collect_candidates(args)
    selected, counters, quotas = sample_core(candidates, args)

    output_rows = []
    for row in selected:
        output_rows.append(
            {
                "protein_id": row["protein_id"],
                "accession": row["accession"],
                "entry_name": row["entry_name"],
                "sequence": row["sequence"],
                "length": row["length"],
                "protected_positions": row["protected_positions"],
                "protected_types": row["protected_types"],
                "protected_feature_descriptions": row["protected_feature_descriptions"],
                "primary_protected_type": row["primary_protected_type"],
                "n_protected": row["n_protected"],
                "protected_fraction": "{0:.6f}".format(row["protected_fraction"]),
                "feature_counts": row["feature_counts"],
                "organism": row["organism"],
                "protein_name": row["protein_name"],
                "gene_name": row["gene_name"],
            }
        )

    fieldnames = [
        "protein_id",
        "accession",
        "entry_name",
        "sequence",
        "length",
        "protected_positions",
        "protected_types",
        "protected_feature_descriptions",
        "primary_protected_type",
        "n_protected",
        "protected_fraction",
        "feature_counts",
        "organism",
        "protein_name",
        "gene_name",
    ]
    write_csv(args.out_csv, fieldnames, output_rows)
    write_fasta(args.out_fasta, output_rows)
    write_summary(args.summary, args, candidates, selected, stats, counters, quotas)

    print("Scanned {0} entries".format(stats["scanned_entries"]))
    print("Candidate pool count: {0}".format(len(candidates)))
    print("Wrote {0} rows".format(len(output_rows)))
    print("Wrote {0}".format(args.out_csv))
    print("Wrote {0}".format(args.out_fasta))
    print("Wrote {0}".format(args.summary))


if __name__ == "__main__":
    main()
