#!/usr/bin/env python3
"""Annotate mapped ProteinGym deletions with Swiss-Prot protected-site features."""

import argparse
import csv
import gzip
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


FEATURE_PRIORITY = ["active site", "binding site", "site", "short sequence motif"]
FEATURE_SET = set(FEATURE_PRIORITY)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


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
            return {"kind": "single", "start": parsed, "end": parsed, "label": "position:{}".format(parsed)}
        return {"kind": "fuzzy", "start": None, "end": None, "label": "position:{};status:{}".format(pos or "", status or "")}
    begin = direct_child(location, "begin")
    end = direct_child(location, "end")
    if begin is not None and end is not None:
        begin_pos = begin.get("position")
        end_pos = end.get("position")
        begin_status = begin.get("status")
        end_status = end.get("status")
        parsed_begin = safe_int(begin_pos)
        parsed_end = safe_int(end_pos)
        if parsed_begin is not None and parsed_end is not None and parsed_begin <= parsed_end and not begin_status and not end_status:
            return {"kind": "range", "start": parsed_begin, "end": parsed_end, "label": "range:{}-{}".format(parsed_begin, parsed_end)}
        return {
            "kind": "fuzzy",
            "start": None,
            "end": None,
            "label": "range:{}-{};status:{}/{}".format(begin_pos or "", end_pos or "", begin_status or "", end_status or ""),
        }
    return {"kind": "fuzzy", "start": None, "end": None, "label": "unrecognized"}


def feature_positions(feature, seq_len, max_range_len, stats):
    feature_type = feature.get("type", "")
    if feature_type not in FEATURE_SET:
        return [], ""
    loc = parse_location(feature)
    description = feature.get("description", "")
    if loc["kind"] == "fuzzy":
        stats["skipped_fuzzy_features"] += 1
        return [], ""
    if loc["kind"] == "single":
        pos0 = loc["start"] - 1
        if 0 <= pos0 < seq_len:
            return [pos0], "{}:{}:{}".format(feature_type, loc["label"], description or "")
        stats["skipped_out_of_bounds_features"] += 1
        return [], ""
    range_len = loc["end"] - loc["start"] + 1
    if range_len > max_range_len:
        stats["skipped_long_range_features"] += 1
        return [], ""
    start0 = loc["start"] - 1
    end0 = loc["end"] - 1
    if start0 < 0 or end0 >= seq_len:
        stats["skipped_out_of_bounds_features"] += 1
        return [], ""
    return list(range(start0, end0 + 1)), "{}:{}:{}".format(feature_type, loc["label"], description or "")


def load_target_mapping(path):
    by_sequence_hash = {}
    accessions = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_accessions = [item for item in row.get("swissprot_accessions", "").split(";") if item]
            for accession in row_accessions:
                accessions.add(accession)
            by_sequence_hash[row["target_sha1"]] = row
    return by_sequence_hash, accessions


def load_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def extract_features(xml_path, wanted_accessions, max_range_len):
    features = {}
    stats = Counter()
    opener = gzip.open if xml_path.endswith(".gz") else open
    with opener(xml_path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, elem in context:
            if local_name(elem.tag) != "entry":
                continue
            stats["entries_scanned"] += 1
            accession = first_accession(elem)
            if accession not in wanted_accessions:
                elem.clear()
                continue
            sequence = entry_sequence(elem)
            seq_len = len(sequence)
            positions = set()
            types = set()
            descriptions = []
            for feature in elem.iter():
                if local_name(feature.tag) != "feature":
                    continue
                used_positions, desc = feature_positions(feature, seq_len, max_range_len, stats)
                if not used_positions:
                    continue
                positions.update(used_positions)
                types.add(feature.get("type", ""))
                if desc:
                    descriptions.append(desc)
            features[accession] = {
                "accession": accession,
                "entry_name": entry_name(elem),
                "sequence": sequence,
                "length": seq_len,
                "protected_positions": sorted(positions),
                "protected_types": ";".join(item for item in FEATURE_PRIORITY if item in types),
                "protected_feature_descriptions": "|".join(descriptions[:50]),
            }
            stats["matched_entries"] += 1
            elem.clear()
    return features, stats


def parse_positions(value):
    if not value:
        return []
    return [int(item) for item in value.split(";") if item != ""]


def join_positions(values):
    return ";".join(str(value) for value in sorted(values))


def annotate_rows(rows, feature_by_accession, shadow_radius):
    out_rows = []
    summary = Counter()
    per_source = defaultdict(Counter)
    for row in rows:
        accessions = [item for item in row.get("swissprot_accessions", "").split(";") if item]
        matched_features = [feature_by_accession[acc] for acc in accessions if acc in feature_by_accession]
        protected_positions = set()
        protected_types = set()
        descriptions = []
        for item in matched_features:
            protected_positions.update(item["protected_positions"])
            protected_types.update(t for t in item["protected_types"].split(";") if t)
            if item["protected_feature_descriptions"]:
                descriptions.extend(item["protected_feature_descriptions"].split("|"))

        start = int(row["deletion_start"])
        end = int(row["deletion_end"])
        target_len = int(row["target_length"])
        deleted = set(range(start, end + 1))
        window_start = max(0, start - shadow_radius)
        window_end = min(target_len - 1, end + shadow_radius)
        window = set(range(window_start, window_end + 1))
        direct = protected_positions & deleted
        shadow = (protected_positions & window) - direct

        out = dict(row)
        out["swissprot_feature_accessions"] = ";".join(item["accession"] for item in matched_features)
        out["swissprot_feature_entry_names"] = ";".join(item["entry_name"] for item in matched_features)
        out["protected_positions"] = join_positions(protected_positions)
        out["protected_types"] = ";".join(item for item in FEATURE_PRIORITY if item in protected_types)
        out["n_protected_positions"] = len(protected_positions)
        out["protected_overlap_count"] = len(direct)
        out["protected_overlap_fraction"] = len(direct) / float(max(1, end - start + 1))
        out["protected_overlap_positions"] = join_positions(direct)
        out["sequence_shadow_radius"] = shadow_radius
        out["sequence_shadow_count"] = len(shadow)
        out["sequence_shadow_positions"] = join_positions(shadow)
        out["sequence_shadow_overlap_fraction"] = len(shadow) / float(max(1, end - start + 1))
        out["has_swissprot_features"] = bool(matched_features)
        out["has_protected_overlap"] = bool(direct)
        out["has_sequence_shadow"] = bool(shadow)
        out["protected_feature_descriptions"] = "|".join(descriptions[:50])
        out_rows.append(out)

        source = row.get("source", "")
        summary["rows"] += 1
        per_source[source]["rows"] += 1
        if matched_features:
            summary["rows_with_swissprot_features"] += 1
            per_source[source]["rows_with_swissprot_features"] += 1
        if direct:
            summary["rows_with_protected_overlap"] += 1
            per_source[source]["rows_with_protected_overlap"] += 1
        if shadow:
            summary["rows_with_sequence_shadow"] += 1
            per_source[source]["rows_with_sequence_shadow"] += 1
    return out_rows, summary, per_source


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, args, extraction_stats, annotation_summary, per_source):
    ensure_parent(path)
    with open(path, "w") as handle:
        handle.write("ProteinGym Swiss-Prot feature annotation summary\n\n")
        handle.write("input_csv: {}\n".format(args.input_csv))
        handle.write("mapping_csv: {}\n".format(args.mapping_csv))
        handle.write("swissprot_xml: {}\n".format(args.swissprot_xml))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("shadow_radius_residues: {}\n".format(args.shadow_radius))
        handle.write("max_range_feature_len: {}\n\n".format(args.max_range_feature_len))
        handle.write("XML extraction:\n")
        for key in sorted(extraction_stats):
            handle.write("- {}: {}\n".format(key, extraction_stats[key]))
        handle.write("\nAnnotation rows:\n")
        for key in sorted(annotation_summary):
            handle.write("- {}: {}\n".format(key, annotation_summary[key]))
        handle.write("\nBy source:\n")
        for source in sorted(per_source):
            handle.write("- {}: {}\n".format(source, dict(per_source[source])))
        handle.write("\nNotes:\n")
        handle.write("- Protected features use the same Swiss-Prot feature types as the internal BioPrior build: active site, binding site, site, short sequence motif.\n")
        handle.write("- sequence_shadow_* is a sequence-neighborhood proxy around protected residues, not a structural 8A contact metric.\n")
        handle.write("- Rows without exact Swiss-Prot feature accessions remain unannotated by this script.\n\n")
        handle.write("PROTEINGYM_SWISSPROT_FEATURE_ANNOTATION_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Annotate mapped ProteinGym deletions with Swiss-Prot protected features.")
    parser.add_argument("--input_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_mapped.csv")
    parser.add_argument("--mapping_csv", default="results/proteingym_deletion_benchmark/proteingym_target_swissprot_mapping.csv")
    parser.add_argument("--swissprot_xml", default="data/raw/uniprot/uniprot_sprot.xml.gz")
    parser.add_argument("--out_csv", default="results/proteingym_deletion_benchmark/proteingym_single_segment_deletions_stage1_scored_mapped_features.csv")
    parser.add_argument("--summary_txt", default="results/proteingym_deletion_benchmark/proteingym_swissprot_feature_annotation_summary.txt")
    parser.add_argument("--shadow_radius", type=int, default=8)
    parser.add_argument("--max_range_feature_len", type=int, default=20)
    args = parser.parse_args()

    _, wanted_accessions = load_target_mapping(args.mapping_csv)
    feature_by_accession, extraction_stats = extract_features(args.swissprot_xml, wanted_accessions, args.max_range_feature_len)
    rows, input_fields = load_rows(args.input_csv)
    out_rows, annotation_summary, per_source = annotate_rows(rows, feature_by_accession, args.shadow_radius)
    added_fields = [
        "swissprot_feature_accessions",
        "swissprot_feature_entry_names",
        "protected_positions",
        "protected_types",
        "n_protected_positions",
        "protected_overlap_count",
        "protected_overlap_fraction",
        "protected_overlap_positions",
        "sequence_shadow_radius",
        "sequence_shadow_count",
        "sequence_shadow_positions",
        "sequence_shadow_overlap_fraction",
        "has_swissprot_features",
        "has_protected_overlap",
        "has_sequence_shadow",
        "protected_feature_descriptions",
    ]
    write_csv(args.out_csv, out_rows, input_fields + [field for field in added_fields if field not in input_fields])
    write_summary(args.summary_txt, args, extraction_stats, annotation_summary, per_source)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
