#!/usr/bin/env python3
"""Extract Core-1K AlphaFold structures from local AFDB v4 tar archives."""

from __future__ import print_function

import argparse
import csv
import gzip
import os
import shutil
import sys
import tarfile


SOURCE_PDB = "local_afdb_swissprot_pdb_v4_tar"
SOURCE_CIF = "local_afdb_swissprot_cif_v4_tar"


def open_csv_read(path):
    if sys.version_info[0] >= 3:
        return open(path, "r", newline="")
    return open(path, "rb")


def open_csv_write(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    if sys.version_info[0] >= 3:
        return open(path, "w", newline="")
    return open(path, "wb")


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def is_nonempty_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_targets(input_csv, limit=None):
    targets = []
    seen = set()
    with open_csv_read(input_csv) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = (row.get("accession") or row.get("protein_id") or "").strip()
            if not accession or accession in seen:
                continue
            sequence = row.get("sequence", "")
            length = row.get("length", "")
            if not length and sequence:
                length = str(len(sequence))
            targets.append(
                {
                    "accession": accession,
                    "sequence_length": int(length) if str(length).isdigit() else "",
                }
            )
            seen.add(accession)
            if limit is not None and len(targets) >= limit:
                break
    return targets


def choose_tar(afdb_root):
    pdb_tar = os.path.join(afdb_root, "swissprot_pdb_v4.tar")
    cif_tar = os.path.join(afdb_root, "swissprot_cif_v4.tar")
    if os.path.exists(pdb_tar):
        return {
            "path": pdb_tar,
            "kind": "pdb",
            "source": SOURCE_PDB,
            "member_suffix": ".pdb.gz",
            "output_suffix": ".pdb",
        }
    if os.path.exists(cif_tar):
        return {
            "path": cif_tar,
            "kind": "cif",
            "source": SOURCE_CIF,
            "member_suffix": ".cif.gz",
            "output_suffix": ".cif",
        }
    return None


def expected_member_name(accession, kind):
    return "AF-{0}-F1-model_v4.{1}.gz".format(accession, kind)


def output_structure_path(out_dir, accession, suffix):
    return os.path.join(out_dir, "{0}{1}".format(accession, suffix))


def index_tar_members(tar, wanted_names):
    found = {}
    remaining = set(wanted_names)
    while remaining:
        member = tar.next()
        if member is None:
            break
        if member.name in remaining:
            found[member.name] = member
            remaining.remove(member.name)
    return found


def extract_member_to_file(tar, member, out_path):
    tmp_path = out_path + ".tmp"
    source = tar.extractfile(member)
    if source is None:
        raise RuntimeError("tar member could not be opened")
    try:
        with gzip.GzipFile(fileobj=source, mode="rb") as gz_handle:
            with open(tmp_path, "wb") as out_handle:
                shutil.copyfileobj(gz_handle, out_handle)
        if not is_nonempty_file(tmp_path):
            raise RuntimeError("extracted structure is empty")
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def parse_pdb_stats(path):
    ca_count = 0
    b_factors = []
    has_atom = False
    with open(path, "rb") as handle:
        for raw_line in handle:
            if not raw_line.startswith(b"ATOM"):
                continue
            has_atom = True
            line = raw_line.decode("ascii", errors="ignore")
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            ca_count += 1
            try:
                b_factors.append(float(line[60:66]))
            except ValueError:
                pass
    if b_factors:
        mean_plddt = sum(b_factors) / float(len(b_factors))
        plddt_readable = True
    else:
        mean_plddt = None
        plddt_readable = False
    return {
        "structure_length": ca_count if ca_count else "",
        "has_atom": has_atom,
        "plddt_readable": plddt_readable,
        "mean_plddt": mean_plddt,
    }


def inspect_structure(path, kind):
    if not is_nonempty_file(path):
        return {"structure_length": "", "notes": "missing_or_empty"}
    if kind != "pdb":
        return {
            "structure_length": "",
            "notes": "cif_extracted;structure_length_not_parsed",
        }
    stats = parse_pdb_stats(path)
    notes = [
        "contains_atom={0}".format(stats["has_atom"]),
        "plddt_bfactor_readable={0}".format(stats["plddt_readable"]),
    ]
    if stats["mean_plddt"] is not None:
        notes.append("mean_plddt={0:.2f}".format(stats["mean_plddt"]))
    return {
        "structure_length": stats["structure_length"],
        "notes": ";".join(notes),
    }


def length_match_note(sequence_length, structure_length):
    if sequence_length == "" or structure_length == "":
        return "length_check=unavailable"
    diff = abs(int(sequence_length) - int(structure_length))
    if diff == 0:
        return "length_match=exact"
    if diff <= 5:
        return "length_match=near;diff={0}".format(diff)
    return "length_mismatch;diff={0}".format(diff)


def write_mapping(path, rows):
    fields = [
        "accession",
        "expected_afdb_name",
        "structure_path",
        "found",
        "error_reason",
        "sequence_length",
        "structure_length",
        "notes",
    ]
    with open_csv_write(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path, args, tar_info, rows):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)

    total = len(rows)
    found_rows = [row for row in rows if row["found"] == "True"]
    failed_rows = [row for row in rows if row["found"] != "True"]
    mismatch_rows = [
        row
        for row in found_rows
        if "length_mismatch" in row.get("notes", "")
    ]
    plddt_rows = [
        row
        for row in found_rows
        if "plddt_bfactor_readable=True" in row.get("notes", "")
    ]
    success_rate = float(len(found_rows)) / total if total else 0.0

    lines = []
    lines.append("AFDB Core-1K structure extraction summary")
    lines.append("")
    lines.append("input_csv: {0}".format(args.input_csv))
    lines.append("afdb_root: {0}".format(args.afdb_root))
    lines.append("out_dir: {0}".format(args.out_dir))
    lines.append("mapping_csv: {0}".format(args.mapping_csv))
    lines.append("summary_txt: {0}".format(args.summary_txt))
    if tar_info is None:
        lines.append("selected_tar: NONE")
        lines.append("ERROR: neither swissprot_pdb_v4.tar nor swissprot_cif_v4.tar found")
    else:
        lines.append("selected_tar: {0}".format(tar_info["path"]))
        lines.append("selected_tar_kind: {0}".format(tar_info["kind"]))
    lines.append("")
    lines.append("targets: {0}".format(total))
    lines.append("found: {0}".format(len(found_rows)))
    lines.append("failed: {0}".format(len(failed_rows)))
    lines.append("success_rate: {0:.4f}".format(success_rate))
    lines.append("pLDDT readable from PDB B-factor: {0}".format(len(plddt_rows)))
    lines.append("length mismatch count: {0}".format(len(mismatch_rows)))
    lines.append("")
    lines.append("Failed accessions:")
    if failed_rows:
        for row in failed_rows[:200]:
            lines.append("- {0}: {1}".format(row["accession"], row["error_reason"]))
        if len(failed_rows) > 200:
            lines.append("- ... {0} more".format(len(failed_rows) - 200))
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Length mismatch accessions:")
    if mismatch_rows:
        for row in mismatch_rows[:200]:
            lines.append(
                "- {0}: sequence_length={1}, structure_length={2}, notes={3}".format(
                    row["accession"],
                    row["sequence_length"],
                    row["structure_length"],
                    row["notes"],
                )
            )
        if len(mismatch_rows) > 200:
            lines.append("- ... {0} more".format(len(mismatch_rows) - 200))
    else:
        lines.append("- none")
    lines.append("")
    if tar_info is None:
        conclusion = "AFDB_CORE_1K_WARN"
    elif len(found_rows) >= max(1, int(total * 0.5)):
        conclusion = "AFDB_CORE_1K_PASS"
    else:
        conclusion = "AFDB_CORE_1K_WARN"
    lines.append(conclusion)

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Core-1K AlphaFold structures from local AFDB v4 tar."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--afdb_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mapping_csv", required=True)
    parser.add_argument("--summary_txt", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    targets = read_targets(args.input_csv, args.limit)
    ensure_dir(args.out_dir)

    tar_info = choose_tar(args.afdb_root)
    rows = []
    pending = []

    for target in targets:
        if tar_info is None:
            expected = "AF-{0}-F1-model_v4.pdb.gz".format(target["accession"])
            out_path = output_structure_path(args.out_dir, target["accession"], ".pdb")
        else:
            expected = expected_member_name(target["accession"], tar_info["kind"])
            out_path = output_structure_path(
                args.out_dir, target["accession"], tar_info["output_suffix"]
            )
        if tar_info is not None and is_nonempty_file(out_path):
            stats = inspect_structure(out_path, tar_info["kind"])
            notes = stats["notes"] + ";" + length_match_note(
                target["sequence_length"], stats["structure_length"]
            )
            rows.append(
                {
                    "accession": target["accession"],
                    "expected_afdb_name": expected,
                    "structure_path": out_path,
                    "found": "True",
                    "error_reason": "",
                    "sequence_length": target["sequence_length"],
                    "structure_length": stats["structure_length"],
                    "notes": "exists;" + notes,
                }
            )
        elif tar_info is None:
            rows.append(
                {
                    "accession": target["accession"],
                    "expected_afdb_name": expected,
                    "structure_path": out_path,
                    "found": "False",
                    "error_reason": "AFDB tar not found under {0}".format(args.afdb_root),
                    "sequence_length": target["sequence_length"],
                    "structure_length": "",
                    "notes": "path_check_failed",
                }
            )
        else:
            pending.append((target, expected, out_path))

    if tar_info is not None and pending:
        wanted = dict((expected, (target, out_path)) for target, expected, out_path in pending)
        print("Scanning {0} once for {1} requested structures".format(tar_info["path"], len(wanted)))
        with tarfile.open(tar_info["path"], "r:") as tar:
            found_members = index_tar_members(tar, set(wanted.keys()))
            print("Found {0}/{1} requested tar members".format(len(found_members), len(wanted)))

            for target, expected, out_path in pending:
                member = found_members.get(expected)
                if member is None:
                    rows.append(
                        {
                            "accession": target["accession"],
                            "expected_afdb_name": expected,
                            "structure_path": out_path,
                            "found": "False",
                            "error_reason": "tar member not found",
                            "sequence_length": target["sequence_length"],
                            "structure_length": "",
                            "notes": "",
                        }
                    )
                    continue

                try:
                    extract_member_to_file(tar, member, out_path)
                    stats = inspect_structure(out_path, tar_info["kind"])
                    notes = stats["notes"] + ";" + length_match_note(
                        target["sequence_length"], stats["structure_length"]
                    )
                    rows.append(
                        {
                            "accession": target["accession"],
                            "expected_afdb_name": expected,
                            "structure_path": out_path,
                            "found": "True",
                            "error_reason": "",
                            "sequence_length": target["sequence_length"],
                            "structure_length": stats["structure_length"],
                            "notes": "extracted;" + notes,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "accession": target["accession"],
                            "expected_afdb_name": expected,
                            "structure_path": out_path,
                            "found": "False",
                            "error_reason": str(exc),
                            "sequence_length": target["sequence_length"],
                            "structure_length": "",
                            "notes": "",
                        }
                    )

    order = dict((target["accession"], idx) for idx, target in enumerate(targets))
    rows.sort(key=lambda row: order.get(row["accession"], 10**9))
    write_mapping(args.mapping_csv, rows)
    write_summary(args.summary_txt, args, tar_info, rows)

    found_count = sum(1 for row in rows if row["found"] == "True")
    print("Targets: {0}".format(len(rows)))
    print("Found: {0}".format(found_count))
    print("Failed: {0}".format(len(rows) - found_count))
    print("Wrote {0}".format(args.mapping_csv))
    print("Wrote {0}".format(args.summary_txt))


if __name__ == "__main__":
    main()
