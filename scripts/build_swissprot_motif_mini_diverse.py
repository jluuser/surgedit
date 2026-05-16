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
SUGGESTED_QUOTAS = {
    "active site": 25,
    "binding site": 25,
    "site": 15,
    "short sequence motif": 15,
}
LENGTH_BINS = [
    ("100-200", 100, 200),
    ("200-400", 201, 400),
    ("400-600", 401, 600),
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
    if not positions:
        return False
    return all(pos >= candidate["length"] - 10 for pos in positions)


def sequence_identity_too_high(seq_a, seq_b):
    len_a = len(seq_a)
    len_b = len(seq_b)
    max_len = max(len_a, len_b)
    if max_len == 0:
        return True
    if float(abs(len_a - len_b)) / max_len > 0.10:
        return False
    min_len = min(len_a, len_b)
    matches = 0
    for i in range(min_len):
        if seq_a[i] == seq_b[i]:
            matches += 1
    return float(matches) / min_len > 0.90


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
            handle.write(encode_utf8(">{0}\n".format(row["protein_id"])))
            seq = row["sequence"]
            for i in range(0, len(seq), 80):
                handle.write(encode_utf8(seq[i : i + 80] + "\n"))


def mean(values):
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def identity_ok(candidate, selected, identity_rejected):
    for chosen in selected:
        if sequence_identity_too_high(candidate["sequence"], chosen["sequence"]):
            identity_rejected.add(candidate["accession"])
            return False
    return True


def select_candidate(candidate, selected, selected_accessions, identity_rejected, counters):
    if candidate["accession"] in selected_accessions:
        return False
    if candidate["is_cterminal_short_motif"] and counters["cterminal_short_motif"] >= 5:
        return False
    if not identity_ok(candidate, selected, identity_rejected):
        return False
    selected.append(candidate)
    selected_accessions.add(candidate["accession"])
    if candidate["is_cterminal_short_motif"]:
        counters["cterminal_short_motif"] += 1
    return True


def round_robin_select(candidates, quota, selected, selected_accessions, identity_rejected, counters):
    by_bin = defaultdict(list)
    for candidate in candidates:
        by_bin[candidate["length_bin"]].append(candidate)

    bin_labels = [item[0] for item in LENGTH_BINS]
    progress = True
    while progress and quota > 0:
        progress = False
        for label in bin_labels:
            bucket = by_bin[label]
            while bucket:
                candidate = bucket.pop(0)
                if select_candidate(
                    candidate, selected, selected_accessions, identity_rejected, counters
                ):
                    quota -= 1
                    progress = True
                    break
            if quota <= 0:
                break
    return quota


def diverse_sample(candidates, target_size, seed):
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    seen_sequences = set()
    deduped = []
    duplicate_sequence_removed = 0
    for candidate in shuffled:
        seq = candidate["sequence"]
        if seq in seen_sequences:
            duplicate_sequence_removed += 1
            continue
        seen_sequences.add(seq)
        deduped.append(candidate)

    by_type = defaultdict(list)
    for candidate in deduped:
        by_type[candidate["primary_protected_type"]].append(candidate)
    for feature_type in by_type:
        rng.shuffle(by_type[feature_type])

    selected = []
    selected_accessions = set()
    identity_rejected = set()
    counters = defaultdict(int)

    for feature_type in FEATURE_PRIORITY:
        round_robin_select(
            by_type[feature_type],
            SUGGESTED_QUOTAS[feature_type],
            selected,
            selected_accessions,
            identity_rejected,
            counters,
        )
        if len(selected) >= target_size:
            break

    if len(selected) < target_size:
        remaining = [
            candidate
            for candidate in deduped
            if candidate["accession"] not in selected_accessions
        ]
        rng.shuffle(remaining)
        round_robin_select(
            remaining,
            target_size - len(selected),
            selected,
            selected_accessions,
            identity_rejected,
            counters,
        )

    return {
        "selected": selected[:target_size],
        "duplicate_sequence_removed": duplicate_sequence_removed,
        "identity_removed": len(identity_rejected),
        "deduped_candidate_count": len(deduped),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a diverse Swiss-Prot motif/protected-site mini-set."
    )
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_fasta", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max_entries", type=int, default=200000)
    parser.add_argument("--target_size", type=int, default=80)
    parser.add_argument("--min_len", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_candidates(args):
    candidates = []
    stats = defaultdict(int)

    with gzip.open(args.xml, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for event, elem in context:
            if local_name(elem.tag) != "entry":
                continue

            stats["scanned_entries"] += 1
            accession = first_accession(elem)
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

            if protected_positions:
                stats["entries_with_usable_protected_features"] += 1
            else:
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue

            sorted_positions = sorted(protected_positions)
            if len(sorted_positions) > 30 or len(sorted_positions) > int(seq_len * 0.2):
                stats["skipped_too_many_protected_positions"] += 1
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue

            ptype = primary_type(protected_types_seen)
            protected_type = ";".join(
                item for item in FEATURE_PRIORITY if item in protected_types_seen
            )
            candidate = {
                "protein_id": accession,
                "accession": accession,
                "protein_name": recommended_protein_name(elem, accession),
                "organism": organism_name(elem),
                "sequence": sequence,
                "length": seq_len,
                "protected_positions_list": sorted_positions,
                "protected_positions": ";".join(str(pos) for pos in sorted_positions),
                "protected_type": protected_type,
                "primary_protected_type": ptype,
                "feature_descriptions": "|".join(descriptions),
                "structure_path": "",
                "length_bin": length_bin(seq_len),
                "feature_type_counts": dict(feature_type_counts),
            }
            candidate["is_cterminal_short_motif"] = is_cterminal_short_motif(candidate)
            candidates.append(candidate)

            elem.clear()
            if stats["scanned_entries"] >= args.max_entries:
                break

    return candidates, stats


def summarize(args, selected, selection_stats, collect_stats):
    selected_feature_counts = defaultdict(int)
    primary_dist = defaultdict(int)
    protected_type_dist = defaultdict(int)
    length_bin_dist = defaultdict(int)
    cterminal_short_motif_count = 0

    for row in selected:
        primary_dist[row["primary_protected_type"]] += 1
        protected_type_dist[row["protected_type"]] += 1
        length_bin_dist[row["length_bin"]] += 1
        if row["is_cterminal_short_motif"]:
            cterminal_short_motif_count += 1
        for feature_type, count in row["feature_type_counts"].items():
            selected_feature_counts[feature_type] += count

    lengths = [int(row["length"]) for row in selected]
    protected_counts = [len(row["protected_positions_list"]) for row in selected]

    parent = os.path.dirname(args.summary)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.summary, "wb") as handle:
        lines = []
        lines.append("Swiss-Prot diverse motif mini-set summary")
        lines.append("")
        lines.append("Input XML: {0}".format(args.xml))
        lines.append("Output CSV: {0}".format(args.out_csv))
        lines.append("Output FASTA: {0}".format(args.out_fasta))
        lines.append("Seed: {0}".format(args.seed))
        lines.append("")
        lines.append("Scanned Swiss-Prot entries: {0}".format(collect_stats["scanned_entries"]))
        lines.append("Candidate pool count: {0}".format(selection_stats["candidate_pool_count"]))
        lines.append(
            "Length {0}-{1} and standard 20AA count: {2}".format(
                args.min_len,
                args.max_len,
                collect_stats["length_and_standard_20aa"],
            )
        )
        lines.append(
            "Entries with usable protected features: {0}".format(
                collect_stats["entries_with_usable_protected_features"]
            )
        )
        lines.append(
            "Removed exact duplicate sequences: {0}".format(
                selection_stats["duplicate_sequence_removed"]
            )
        )
        lines.append(
            "Removed/rejected by >90% identity filter: {0}".format(
                selection_stats["identity_removed"]
            )
        )
        lines.append("Final output count: {0}".format(len(selected)))
        lines.append("")
        lines.append("Primary_protected_type distribution:")
        for feature_type in FEATURE_PRIORITY:
            lines.append("- {0}: {1}".format(feature_type, primary_dist[feature_type]))
        lines.append("")
        lines.append("Protected_type distribution:")
        for key in sorted(protected_type_dist):
            lines.append("- {0}: {1}".format(key, protected_type_dist[key]))
        lines.append("")
        if lengths:
            lines.append(
                "Length min / max / mean: {0} / {1} / {2:.2f}".format(
                    min(lengths), max(lengths), mean(lengths)
                )
            )
        else:
            lines.append("Length min / max / mean: NA / NA / NA")
        lines.append("Length bin distribution:")
        for label, lo, hi in LENGTH_BINS:
            lines.append("- {0}: {1}".format(label, length_bin_dist[label]))
        if protected_counts:
            lines.append(
                "Protected position count min / max / mean: {0} / {1} / {2:.2f}".format(
                    min(protected_counts),
                    max(protected_counts),
                    mean(protected_counts),
                )
            )
        else:
            lines.append("Protected position count min / max / mean: NA / NA / NA")
        lines.append(
            "C-terminal short sequence motif sample count: {0}".format(
                cterminal_short_motif_count
            )
        )
        lines.append("")
        lines.append("Selected feature annotation counts:")
        for feature_type in FEATURE_PRIORITY:
            lines.append("- {0}: {1}".format(feature_type, selected_feature_counts[feature_type]))
        lines.append("")
        lines.append("Skipped fuzzy features: {0}".format(collect_stats["skipped_fuzzy_features"]))
        lines.append(
            "Skipped range features longer than 20 aa: {0}".format(
                collect_stats["skipped_long_range_features"]
            )
        )
        lines.append(
            "Skipped out-of-bounds features: {0}".format(
                collect_stats["skipped_out_of_bounds_features"]
            )
        )
        lines.append(
            "Skipped entries with too many protected positions: {0}".format(
                collect_stats["skipped_too_many_protected_positions"]
            )
        )
        lines.append("")
        lines.append("First 15 samples:")
        for row in selected[:15]:
            letters = ";".join(
                "{0}{1}".format(row["sequence"][pos], pos)
                for pos in row["protected_positions_list"]
            )
            lines.append(
                "- {0}\tlength={1}\tprimary={2}\tprotected_positions={3}\tletters={4}".format(
                    row["accession"],
                    row["length"],
                    row["primary_protected_type"],
                    row["protected_positions"],
                    letters,
                )
            )
        handle.write(encode_utf8("\n".join(lines) + "\n"))


def main():
    args = parse_args()
    candidates, collect_stats = collect_candidates(args)
    sample_result = diverse_sample(candidates, args.target_size, args.seed)
    selected = sample_result["selected"]
    sample_result["candidate_pool_count"] = len(candidates)

    output_rows = []
    for row in selected:
        output_rows.append(
            {
                "protein_id": row["protein_id"],
                "accession": row["accession"],
                "protein_name": row["protein_name"],
                "organism": row["organism"],
                "sequence": row["sequence"],
                "length": row["length"],
                "protected_positions": row["protected_positions"],
                "protected_type": row["protected_type"],
                "primary_protected_type": row["primary_protected_type"],
                "feature_descriptions": row["feature_descriptions"],
                "structure_path": "",
            }
        )

    write_csv(
        args.out_csv,
        [
            "protein_id",
            "accession",
            "protein_name",
            "organism",
            "sequence",
            "length",
            "protected_positions",
            "protected_type",
            "primary_protected_type",
            "feature_descriptions",
            "structure_path",
        ],
        output_rows,
    )
    write_fasta(args.out_fasta, output_rows)
    summarize(args, selected, sample_result, collect_stats)

    print("Scanned {0} entries".format(collect_stats["scanned_entries"]))
    print("Candidate pool count: {0}".format(len(candidates)))
    print("Wrote {0} rows".format(len(output_rows)))
    print("Wrote {0}".format(args.out_csv))
    print("Wrote {0}".format(args.out_fasta))
    print("Wrote {0}".format(args.summary))


if __name__ == "__main__":
    main()
