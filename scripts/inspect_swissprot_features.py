from __future__ import print_function

import argparse
import gzip
import os
import xml.etree.ElementTree as ET
from collections import defaultdict


FOCUS_TYPES = set(
    [
        "active site",
        "binding site",
        "metal ion-binding site",
        "site",
        "motif",
        "short sequence motif",
        "domain",
        "region of interest",
        "transmembrane region",
    ]
)


try:
    basestring
except NameError:
    basestring = str


def local_name(tag):
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def child_text(element, name):
    for child in list(element):
        if local_name(child.tag) == name:
            return clean_text(child.text)
    return ""


def clean_text(value):
    if value is None:
        return ""
    return "".join(value.split())


def protein_name(entry):
    protein = None
    for child in list(entry):
        if local_name(child.tag) == "protein":
            protein = child
            break
    if protein is None:
        return ""

    preferred_blocks = ["recommendedName", "submittedName", "allergenName"]
    for block_name in preferred_blocks:
        for block in protein.iter():
            if local_name(block.tag) == block_name:
                for item in block.iter():
                    if local_name(item.tag) == "fullName":
                        text = item.text or ""
                        if text.strip():
                            return text.strip()
    for item in protein.iter():
        if local_name(item.tag) == "fullName":
            text = item.text or ""
            if text.strip():
                return text.strip()
    return ""


def first_accession(entry):
    for child in list(entry):
        if local_name(child.tag) == "accession":
            return (child.text or "").strip()
    return ""


def entry_sequence(entry):
    for child in list(entry):
        if local_name(child.tag) == "sequence":
            return clean_text(child.text)
    return ""


def feature_location(feature):
    location = None
    for child in list(feature):
        if local_name(child.tag) == "location":
            location = child
            break
    if location is None:
        return {
            "kind": "uncertain_fuzzy",
            "display": "",
            "start": "",
            "end": "",
            "status": "missing location",
        }

    position = None
    begin = None
    end = None
    for child in list(location):
        name = local_name(child.tag)
        if name == "position":
            position = child
        elif name == "begin":
            begin = child
        elif name == "end":
            end = child

    if position is not None:
        pos = position.get("position", "")
        status = position.get("status", "")
        if pos and not status:
            return {
                "kind": "exact_single",
                "display": "position:{0}".format(pos),
                "start": pos,
                "end": pos,
                "status": "",
            }
        return {
            "kind": "uncertain_fuzzy",
            "display": "position:{0};status:{1}".format(pos, status),
            "start": pos,
            "end": pos,
            "status": status or "fuzzy position",
        }

    if begin is not None or end is not None:
        begin_pos = begin.get("position", "") if begin is not None else ""
        end_pos = end.get("position", "") if end is not None else ""
        begin_status = begin.get("status", "") if begin is not None else ""
        end_status = end.get("status", "") if end is not None else ""
        if begin_pos and end_pos and not begin_status and not end_status:
            return {
                "kind": "exact_range",
                "display": "range:{0}-{1}".format(begin_pos, end_pos),
                "start": begin_pos,
                "end": end_pos,
                "status": "",
            }
        status = ";".join(
            item
            for item in [
                "begin:{0}".format(begin_status) if begin_status else "",
                "end:{0}".format(end_status) if end_status else "",
            ]
            if item
        )
        return {
            "kind": "uncertain_fuzzy",
            "display": "range:{0}-{1};status:{2}".format(begin_pos, end_pos, status),
            "start": begin_pos,
            "end": end_pos,
            "status": status or "fuzzy range",
        }

    return {
        "kind": "uncertain_fuzzy",
        "display": "",
        "start": "",
        "end": "",
        "status": "unrecognized location",
    }


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


def write_csv(path, fieldnames, rows):
    with open(path, "wb") as handle:
        header = ",".join(fieldnames) + "\n"
        handle.write(header.encode("utf-8"))
        for row in rows:
            line = ",".join(csv_escape(row.get(field, "")) for field in fieldnames)
            handle.write((line + "\n").encode("utf-8"))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect Swiss-Prot XML feature annotations with streaming XML parsing."
    )
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=50000)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    stats = defaultdict(
        lambda: {
            "total_features": 0,
            "exact_single_position": 0,
            "exact_range": 0,
            "uncertain_fuzzy_position": 0,
        }
    )
    examples = defaultdict(list)

    records_seen = 0
    features_seen = 0

    with gzip.open(args.xml, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for event, elem in context:
            if local_name(elem.tag) != "entry":
                continue

            records_seen += 1
            accession = first_accession(elem)
            name = protein_name(elem)
            sequence = entry_sequence(elem)
            seq_len = len(sequence)

            for feature in elem.iter():
                if local_name(feature.tag) != "feature":
                    continue
                feature_type = feature.get("type", "")
                description = feature.get("description", "")
                loc = feature_location(feature)

                stats[feature_type]["total_features"] += 1
                stats[feature_type][
                    {
                        "exact_single": "exact_single_position",
                        "exact_range": "exact_range",
                    }.get(loc["kind"], "uncertain_fuzzy_position")
                ] += 1
                features_seen += 1

                if len(examples[feature_type]) < 20:
                    examples[feature_type].append(
                        {
                            "accession": accession,
                            "protein_name": name,
                            "sequence": sequence,
                            "sequence_length": seq_len,
                            "feature_type": feature_type,
                            "feature_location": loc["display"],
                            "location_kind": loc["kind"],
                            "location_start_1based": loc["start"],
                            "location_end_1based": loc["end"],
                            "location_status": loc["status"],
                            "feature_description": description,
                        }
                    )

            elem.clear()
            if records_seen >= args.max_records:
                break

    count_rows = []
    for feature_type in sorted(stats):
        row = {"feature_type": feature_type, "is_focus_type": feature_type in FOCUS_TYPES}
        row.update(stats[feature_type])
        count_rows.append(row)

    example_rows = []
    for feature_type in sorted(examples):
        example_rows.extend(examples[feature_type])

    counts_path = os.path.join(args.out_dir, "feature_type_counts.csv")
    examples_path = os.path.join(args.out_dir, "example_features.csv")
    report_path = os.path.join(args.out_dir, "inspect_report.txt")

    write_csv(
        counts_path,
        [
            "feature_type",
            "total_features",
            "exact_single_position",
            "exact_range",
            "uncertain_fuzzy_position",
            "is_focus_type",
        ],
        count_rows,
    )
    write_csv(
        examples_path,
        [
            "accession",
            "protein_name",
            "sequence",
            "sequence_length",
            "feature_type",
            "feature_location",
            "location_kind",
            "location_start_1based",
            "location_end_1based",
            "location_status",
            "feature_description",
        ],
        example_rows,
    )

    with open(report_path, "wb") as handle:
        lines = []
        lines.append("Swiss-Prot feature inspection report")
        lines.append("")
        lines.append("Input XML: {0}".format(args.xml))
        lines.append("Entries inspected: {0}".format(records_seen))
        lines.append("Features inspected: {0}".format(features_seen))
        lines.append("Output counts: {0}".format(counts_path))
        lines.append("Output examples: {0}".format(examples_path))
        lines.append("")
        lines.append("Focused feature type counts:")
        for feature_type in sorted(FOCUS_TYPES):
            s = stats.get(
                feature_type,
                {
                    "total_features": 0,
                    "exact_single_position": 0,
                    "exact_range": 0,
                    "uncertain_fuzzy_position": 0,
                },
            )
            lines.append(
                "- {0}: total={1}, exact_single={2}, exact_range={3}, fuzzy={4}".format(
                    feature_type,
                    s["total_features"],
                    s["exact_single_position"],
                    s["exact_range"],
                    s["uncertain_fuzzy_position"],
                )
            )
        lines.append("")
        lines.append("Best protected motif candidates:")
        lines.append(
            "- active site, binding site, metal ion-binding site, and site are the best residue-level protected-site candidates when their locations are exact single positions."
        )
        lines.append(
            "- motif/short sequence motif can be useful as a protected motif span when locations are exact ranges and descriptions are biologically meaningful."
        )
        lines.append(
            "- domain and region of interest can be useful range-level protected regions, but they may be broad and should usually be filtered by length."
        )
        lines.append("")
        lines.append("Residue-level feature types:")
        lines.append(
            "- active site, binding site, metal ion-binding site, and many site annotations are primarily residue-level exact single-position annotations."
        )
        lines.append("")
        lines.append("Range-level feature types:")
        lines.append(
            "- motif/short sequence motif, domain, region of interest, and transmembrane region are primarily range-level annotations."
        )
        lines.append("")
        lines.append("Temporarily not recommended:")
        lines.append(
            "- transmembrane region is not recommended as a motif mask by default because it often marks broad topology segments rather than compact functional residues."
        )
        lines.append(
            "- domain is not recommended without length filtering because protecting whole domains can leave too little deletable sequence."
        )
        lines.append(
            "- uncertain or fuzzy positions should be excluded from the first protected-site prototype."
        )
        lines.append("")
        lines.append(
            "Note: UniProt XML feature coordinates are reported as 1-based positions in the CSV examples."
        )
        handle.write(("\n".join(lines) + "\n").encode("utf-8"))

    print("Inspected {0} entries and {1} features".format(records_seen, features_seen))
    print("Wrote {0}".format(counts_path))
    print("Wrote {0}".format(examples_path))
    print("Wrote {0}".format(report_path))


if __name__ == "__main__":
    main()
