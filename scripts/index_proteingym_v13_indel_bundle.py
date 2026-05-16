#!/usr/bin/env python3
"""Index the local ProteinGym v1.3 indel bundle.

The script does not require unpacking the large MSA archives. It records which
per-assay DMS, AF2 structure, MSA, weight, and zero-shot files are present so
downstream Step-3 validation can fail early on missing assets.
"""

import argparse
import csv
import os
import zipfile
from collections import Counter


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def zip_members(path):
    if not os.path.exists(path):
        return set(), Counter()
    members = set()
    suffix_counts = Counter()
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            members.add(name)
            suffix_counts[os.path.splitext(name)[1].lower() or "<none>"] += 1
    return members, suffix_counts


def member_by_basename(members):
    out = {}
    for name in members:
        out[os.path.basename(name)] = name
    return out


def present(mapping, basename):
    if not basename:
        return "", "False"
    path = mapping.get(basename, "")
    return path, str(bool(path))


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, args, rows, archive_counts):
    ensure_parent(path)
    status_counts = Counter(row["asset_status"] for row in rows)
    with open(path, "w") as handle:
        handle.write("ProteinGym v1.3 indel bundle index\n\n")
        handle.write("bundle_dir: {}\n".format(args.bundle_dir))
        handle.write("DMS metadata rows: {}\n".format(len(rows)))
        handle.write("asset_status_counts: {}\n\n".format(dict(status_counts)))
        handle.write("Archive member counts:\n")
        for name, counts in archive_counts.items():
            handle.write("- {}: {}\n".format(name, dict(counts)))
        handle.write("\nCoverage:\n")
        for key in [
            "has_dms_file",
            "has_af2_structure",
            "has_msa_file",
            "has_msa_weight",
            "has_zero_shot_scores",
        ]:
            handle.write("- {}: {}/{}\n".format(key, sum(1 for row in rows if row[key] == "True"), len(rows)))
        handle.write("\nPROTEINGYM_V13_INDEL_INDEX_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Index ProteinGym v1.3 indel bundle assets.")
    parser.add_argument("--bundle_dir", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/external/proteingym_v13")
    parser.add_argument("--out_csv", default="results/proteingym_v13_indel/metadata_asset_index.csv")
    parser.add_argument("--summary_txt", default="results/proteingym_v13_indel/metadata_asset_index_summary.txt")
    args = parser.parse_args()

    dms_metadata_path = os.path.join(args.bundle_dir, "DMS_indels.csv")
    rows, _ = read_csv(dms_metadata_path)

    archives = {
        "DMS_ProteinGym_indels.zip": os.path.join(args.bundle_dir, "DMS_ProteinGym_indels.zip"),
        "ProteinGym_AF2_structures.zip": os.path.join(args.bundle_dir, "ProteinGym_AF2_structures.zip"),
        "DMS_msa_files.zip": os.path.join(args.bundle_dir, "DMS_msa_files.zip"),
        "DMS_msa_weights.zip": os.path.join(args.bundle_dir, "DMS_msa_weights.zip"),
        "zero_shot_indels_scores.zip": os.path.join(args.bundle_dir, "zero_shot_indels_scores.zip"),
    }
    archive_members = {}
    archive_counts = {}
    basename_maps = {}
    for name, path in archives.items():
        members, counts = zip_members(path)
        archive_members[name] = members
        archive_counts[name] = counts
        basename_maps[name] = member_by_basename(members)

    out_rows = []
    for row in rows:
        dms_file = row.get("DMS_filename", "")
        uniprot_id = row.get("UniProt_ID", "")
        msa_file = row.get("MSA_filename", "")
        weight_file = row.get("weight_file_name", "")
        af2_file = "{}.pdb".format(uniprot_id) if uniprot_id else ""

        dms_member, has_dms = present(basename_maps["DMS_ProteinGym_indels.zip"], dms_file)
        af2_member, has_af2 = present(basename_maps["ProteinGym_AF2_structures.zip"], af2_file)
        msa_member, has_msa = present(basename_maps["DMS_msa_files.zip"], msa_file)
        weight_member, has_weight = present(basename_maps["DMS_msa_weights.zip"], weight_file)
        zero_member, has_zero = present(basename_maps["zero_shot_indels_scores.zip"], dms_file)
        required = [has_dms, has_af2, has_zero]
        full = required + [has_msa, has_weight]
        if all(value == "True" for value in full):
            status = "complete"
        elif all(value == "True" for value in required):
            status = "validation_ready_missing_msa_asset"
        else:
            status = "incomplete"

        out = {
            "DMS_index": row.get("DMS_index", ""),
            "DMS_id": row.get("DMS_id", ""),
            "DMS_filename": dms_file,
            "UniProt_ID": uniprot_id,
            "target_seq_len": len(row.get("target_seq", "")),
            "seq_len_metadata": row.get("seq_len", ""),
            "MSA_filename": msa_file,
            "weight_file_name": weight_file,
            "has_dms_file": has_dms,
            "dms_zip_member": dms_member,
            "has_af2_structure": has_af2,
            "af2_zip_member": af2_member,
            "has_msa_file": has_msa,
            "msa_zip_member": msa_member,
            "has_msa_weight": has_weight,
            "msa_weight_zip_member": weight_member,
            "has_zero_shot_scores": has_zero,
            "zero_shot_zip_member": zero_member,
            "asset_status": status,
        }
        out_rows.append(out)

    fields = [
        "DMS_index",
        "DMS_id",
        "DMS_filename",
        "UniProt_ID",
        "target_seq_len",
        "seq_len_metadata",
        "MSA_filename",
        "weight_file_name",
        "has_dms_file",
        "dms_zip_member",
        "has_af2_structure",
        "af2_zip_member",
        "has_msa_file",
        "msa_zip_member",
        "has_msa_weight",
        "msa_weight_zip_member",
        "has_zero_shot_scores",
        "zero_shot_zip_member",
        "asset_status",
    ]
    write_csv(args.out_csv, out_rows, fields)
    write_report(args.summary_txt, args, out_rows, archive_counts)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
