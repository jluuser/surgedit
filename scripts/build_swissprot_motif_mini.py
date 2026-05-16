from __future__ import print_function

import argparse
import gzip
import os
import xml.etree.ElementTree as ET
from collections import defaultdict


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
ALLOWED_FEATURE_TYPES = [
    "active site",
    "binding site",
    "site",
    "short sequence motif",
]
ALLOWED_FEATURE_SET = set(ALLOWED_FEATURE_TYPES)


try:
    basestring
except NameError:
    basestring = str


def local_name(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def clean_sequence(value):
    if value is None:
        return ""
    return "".join(value.split())


def direct_child(element, name):
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


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


def entry_sequence(entry):
    sequence = direct_child(entry, "sequence")
    if sequence is None:
        return ""
    return clean_sequence(sequence.text)


def parse_location(feature):
    location = direct_child(feature, "location")
    if location is None:
        return {"kind": "fuzzy", "start": None, "end": None, "label": "missing"}

    position = direct_child(location, "position")
    if position is not None:
        pos = position.get("position")
        status = position.get("status")
        if pos and not status:
            pos = int(pos)
            return {
                "kind": "single",
                "start": pos,
                "end": pos,
                "label": "position:{0}".format(pos),
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
        if begin_pos and end_pos and not begin_status and not end_status:
            begin_pos = int(begin_pos)
            end_pos = int(end_pos)
            if begin_pos <= end_pos:
                return {
                    "kind": "range",
                    "start": begin_pos,
                    "end": end_pos,
                    "label": "range:{0}-{1}".format(begin_pos, end_pos),
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
    desc = description or ""
    return "{0}:{1}:{2}".format(feature_type, loc["label"], desc)


def usable_feature_positions(feature, seq_len, stats):
    feature_type = feature.get("type", "")
    if feature_type not in ALLOWED_FEATURE_SET:
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a small Swiss-Prot motif/protected-site mini-set."
    )
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_fasta", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--target_size", type=int, default=50)
    parser.add_argument("--max_entries", type=int, default=200000)
    parser.add_argument("--min_len", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    stats = defaultdict(int)
    used_feature_type_counts = defaultdict(int)
    protected_type_distribution = defaultdict(int)
    skipped_too_many_positions = 0
    stopped_after_target = False

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
                stats["standard_20aa"] += 1
            else:
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue

            protected_positions = set()
            protected_types_seen = set()
            used_descriptions = []

            for feature in list(elem):
                if local_name(feature.tag) != "feature":
                    continue
                feature_type = feature.get("type", "")
                if feature_type not in ALLOWED_FEATURE_SET:
                    continue
                positions, description = usable_feature_positions(feature, seq_len, stats)
                if not positions:
                    continue
                protected_positions.update(positions)
                protected_types_seen.add(feature_type)
                used_feature_type_counts[feature_type] += 1
                if description is not None:
                    used_descriptions.append(description)

            if protected_positions:
                stats["entries_with_usable_protected_features"] += 1
            else:
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue

            sorted_positions = sorted(protected_positions)
            if len(sorted_positions) > 30 or len(sorted_positions) > int(seq_len * 0.2):
                skipped_too_many_positions += 1
                elem.clear()
                if stats["scanned_entries"] >= args.max_entries:
                    break
                continue

            protein_name = recommended_protein_name(elem, accession)
            protected_type = ";".join(
                item for item in ALLOWED_FEATURE_TYPES if item in protected_types_seen
            )
            protected_type_distribution[protected_type] += 1

            rows.append(
                {
                    "protein_id": accession,
                    "accession": accession,
                    "protein_name": protein_name,
                    "sequence": sequence,
                    "length": seq_len,
                    "protected_positions": ";".join(
                        str(pos) for pos in sorted_positions
                    ),
                    "protected_type": protected_type,
                    "feature_descriptions": "|".join(used_descriptions),
                    "structure_path": "",
                }
            )

            elem.clear()
            if len(rows) >= args.target_size:
                stopped_after_target = True
                break
            if stats["scanned_entries"] >= args.max_entries:
                break

    write_csv(
        args.out_csv,
        [
            "protein_id",
            "accession",
            "protein_name",
            "sequence",
            "length",
            "protected_positions",
            "protected_type",
            "feature_descriptions",
            "structure_path",
        ],
        rows,
    )
    write_fasta(args.out_fasta, rows)

    lengths = [int(row["length"]) for row in rows]
    protected_counts = [
        len(row["protected_positions"].split(";"))
        for row in rows
        if row["protected_positions"]
    ]

    summary_parent = os.path.dirname(args.summary)
    if summary_parent and not os.path.isdir(summary_parent):
        os.makedirs(summary_parent)
    with open(args.summary, "wb") as handle:
        lines = []
        lines.append("Swiss-Prot motif mini-set summary")
        lines.append("")
        lines.append("Input XML: {0}".format(args.xml))
        lines.append("Output CSV: {0}".format(args.out_csv))
        lines.append("Output FASTA: {0}".format(args.out_fasta))
        lines.append("Target size: {0}".format(args.target_size))
        lines.append("Max entries: {0}".format(args.max_entries))
        lines.append("Stopped after reaching target size: {0}".format(stopped_after_target))
        lines.append("")
        lines.append("Scanned Swiss-Prot entries: {0}".format(stats["scanned_entries"]))
        lines.append(
            "Length {0}-{1} count: {2}".format(
                args.min_len, args.max_len, stats["length_in_range"]
            )
        )
        lines.append("Standard 20AA count: {0}".format(stats["standard_20aa"]))
        lines.append(
            "Entries with usable protected features: {0}".format(
                stats["entries_with_usable_protected_features"]
            )
        )
        lines.append("Final output count: {0}".format(len(rows)))
        lines.append(
            "Skipped entries with too many protected positions: {0}".format(
                skipped_too_many_positions
            )
        )
        lines.append("")
        lines.append("Protected_type distribution:")
        for key in sorted(protected_type_distribution):
            lines.append("- {0}: {1}".format(key, protected_type_distribution[key]))
        lines.append("")
        if lengths:
            lines.append(
                "Length min / max / mean: {0} / {1} / {2:.2f}".format(
                    min(lengths), max(lengths), mean(lengths)
                )
            )
        else:
            lines.append("Length min / max / mean: NA / NA / NA")
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
        lines.append("")
        lines.append("Used feature annotation counts:")
        for feature_type in ALLOWED_FEATURE_TYPES:
            lines.append(
                "- {0}: {1}".format(
                    feature_type, used_feature_type_counts[feature_type]
                )
            )
        lines.append("")
        lines.append("Skipped fuzzy features: {0}".format(stats["skipped_fuzzy_features"]))
        lines.append(
            "Skipped range features longer than 20 aa: {0}".format(
                stats["skipped_long_range_features"]
            )
        )
        lines.append(
            "Skipped out-of-bounds features: {0}".format(
                stats["skipped_out_of_bounds_features"]
            )
        )
        lines.append("")
        lines.append("First 10 samples:")
        for row in rows[:10]:
            positions = [int(pos) for pos in row["protected_positions"].split(";") if pos]
            letters = ";".join("{0}{1}".format(row["sequence"][pos], pos) for pos in positions)
            lines.append(
                "- {0}\tlength={1}\tprotected_type={2}\tprotected_positions={3}\tletters={4}".format(
                    row["accession"],
                    row["length"],
                    row["protected_type"],
                    row["protected_positions"],
                    letters,
                )
            )
        handle.write(encode_utf8("\n".join(lines) + "\n"))

    print("Wrote {0} rows".format(len(rows)))
    print("Wrote {0}".format(args.out_csv))
    print("Wrote {0}".format(args.out_fasta))
    print("Wrote {0}".format(args.summary))


if __name__ == "__main__":
    main()
