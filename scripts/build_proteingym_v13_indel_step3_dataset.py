#!/usr/bin/env python3
"""Build ProteinGym v1.3 single-segment deletion rows with zero-shot scores."""

import argparse
import csv
import hashlib
import os
import zipfile
from collections import Counter
from io import TextIOWrapper


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def is_standard(sequence):
    return isinstance(sequence, str) and bool(sequence) and set(sequence) <= STANDARD_AA


def find_single_deletion(target, mutated):
    if not isinstance(target, str) or not isinstance(mutated, str):
        return None
    if len(mutated) >= len(target):
        return None
    i = 0
    while i < len(mutated) and target[i] == mutated[i]:
        i += 1
    suffix = 0
    while suffix < (len(mutated) - i) and target[len(target) - 1 - suffix] == mutated[len(mutated) - 1 - suffix]:
        suffix += 1
    if i + suffix != len(mutated):
        return None
    deletion_len = len(target) - len(mutated)
    start = i
    end = i + deletion_len - 1
    if target[:start] + target[end + 1 :] != mutated:
        return None
    return start, end


def read_metadata(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def zip_basename_map(path):
    out = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith("/"):
                out[os.path.basename(name)] = name
    return out


def read_zip_csv_rows(zip_path, member):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as raw:
            wrapper = TextIOWrapper(raw, encoding="utf-8")
            reader = csv.DictReader(wrapper)
            return list(reader), list(reader.fieldnames or [])


def fmt_float(value):
    try:
        return "{:.12g}".format(float(value))
    except (TypeError, ValueError):
        return ""


def write_csv(path, rows, fields):
    ensure_parent(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, args, rows, stats, zero_shot_columns):
    ensure_parent(path)
    assay_counts = Counter(row["DMS_id"] for row in rows)
    with open(path, "w") as handle:
        handle.write("ProteinGym v1.3 Step-3 single-segment deletion dataset\n\n")
        handle.write("bundle_dir: {}\n".format(args.bundle_dir))
        handle.write("out_csv: {}\n".format(args.out_csv))
        handle.write("max_target_len: {}\n\n".format(args.max_target_len))
        handle.write("stats: {}\n".format(dict(stats)))
        handle.write("single_segment_deletions: {}\n".format(len(rows)))
        handle.write("assays_with_single_segment_deletions: {}\n".format(len(assay_counts)))
        handle.write("zero_shot_score_columns: {}\n\n".format(",".join(zero_shot_columns)))
        handle.write("Top assays:\n")
        for assay, count in assay_counts.most_common(20):
            handle.write("- {}: {}\n".format(assay, count))
        handle.write("\nPROTEINGYM_V13_STEP3_DATASET_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Build ProteinGym v1.3 single-segment deletion Step-3 dataset.")
    parser.add_argument("--bundle_dir", default="/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/external/proteingym_v13")
    parser.add_argument("--out_csv", default="results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_zero_shot.csv")
    parser.add_argument("--summary_txt", default="results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_zero_shot_summary.txt")
    parser.add_argument("--max_target_len", type=int, default=1200)
    args = parser.parse_args()

    metadata = read_metadata(os.path.join(args.bundle_dir, "DMS_indels.csv"))
    dms_zip = os.path.join(args.bundle_dir, "DMS_ProteinGym_indels.zip")
    zero_zip = os.path.join(args.bundle_dir, "zero_shot_indels_scores.zip")
    dms_members = zip_basename_map(dms_zip)
    zero_members = zip_basename_map(zero_zip)

    rows = []
    stats = Counter()
    zero_shot_columns = []
    for meta in metadata:
        dms_filename = meta["DMS_filename"]
        target = meta["target_seq"]
        stats["metadata_rows"] += 1
        if not is_standard(target):
            stats["nonstandard_target"] += 1
            continue
        if len(target) > args.max_target_len:
            stats["target_too_long"] += 1
            continue
        dms_member = dms_members.get(dms_filename)
        if not dms_member:
            stats["missing_dms_file"] += 1
            continue
        dms_rows, _ = read_zip_csv_rows(dms_zip, dms_member)

        zero_by_mutated = {}
        zero_member = zero_members.get(dms_filename)
        zero_fields = []
        if zero_member:
            zero_rows, zero_fields = read_zip_csv_rows(zero_zip, zero_member)
            for zrow in zero_rows:
                zero_by_mutated[zrow.get("mutated_sequence", "")] = zrow
            for field in zero_fields:
                if field not in ("mutated_sequence", "DMS_score", "DMS_score_bin") and field not in zero_shot_columns:
                    zero_shot_columns.append(field)
        else:
            stats["missing_zero_shot_file"] += 1

        for idx, row in enumerate(dms_rows):
            stats["input_mutants"] += 1
            mutated = row.get("mutated_sequence", "")
            if not is_standard(mutated):
                stats["nonstandard_mutated"] += 1
                continue
            interval = find_single_deletion(target, mutated)
            if interval is None:
                stats["not_single_segment_deletion"] += 1
                continue
            start, end = interval
            deleted = target[start : end + 1]
            zrow = zero_by_mutated.get(mutated, {})
            out = {
                "example_id": "ProteinGym_v1.3:{}:{}".format(meta["DMS_id"], idx),
                "source": "ProteinGym_v1.3_DMS_indels",
                "DMS_index": meta.get("DMS_index", ""),
                "DMS_id": meta["DMS_id"],
                "DMS_filename": dms_filename,
                "UniProt_ID": meta.get("UniProt_ID", ""),
                "target_sequence": target,
                "target_sha1": hashlib.sha1(target.encode("ascii")).hexdigest(),
                "mutated_sequence": mutated,
                "target_length": len(target),
                "mutated_length": len(mutated),
                "deletion_start": start,
                "deletion_end": end,
                "deletion_len": end - start + 1,
                "deletion_sequence": deleted,
                "normalized_start": start / float(len(target)),
                "normalized_end": end / float(len(target)),
                "normalized_midpoint": ((start + end) / 2.0) / float(len(target)),
                "deletion_fraction": (end - start + 1) / float(len(target)),
                "is_terminal_deletion": start == 0 or end == len(target) - 1,
                "DMS_score": fmt_float(row.get("DMS_score")),
                "DMS_score_bin": fmt_float(row.get("DMS_score_bin")),
                "MSA_filename": meta.get("MSA_filename", ""),
                "weight_file_name": meta.get("weight_file_name", ""),
                "MSA_start": meta.get("MSA_start", ""),
                "MSA_end": meta.get("MSA_end", ""),
                "MSA_N_eff": meta.get("MSA_N_eff", ""),
                "MSA_Neff_L": meta.get("MSA_Neff_L", ""),
                "zero_shot_scores_found": bool(zrow),
            }
            for field in zero_fields:
                if field in ("mutated_sequence", "DMS_score", "DMS_score_bin"):
                    continue
                out["zero_shot_{}".format(field)] = fmt_float(zrow.get(field, ""))
            rows.append(out)
            stats["single_segment_deletion"] += 1

    base_fields = [
        "example_id",
        "source",
        "DMS_index",
        "DMS_id",
        "DMS_filename",
        "UniProt_ID",
        "target_sequence",
        "target_sha1",
        "mutated_sequence",
        "target_length",
        "mutated_length",
        "deletion_start",
        "deletion_end",
        "deletion_len",
        "deletion_sequence",
        "normalized_start",
        "normalized_end",
        "normalized_midpoint",
        "deletion_fraction",
        "is_terminal_deletion",
        "DMS_score",
        "DMS_score_bin",
        "MSA_filename",
        "weight_file_name",
        "MSA_start",
        "MSA_end",
        "MSA_N_eff",
        "MSA_Neff_L",
        "zero_shot_scores_found",
    ]
    zero_fields = ["zero_shot_{}".format(field) for field in zero_shot_columns]
    write_csv(args.out_csv, rows, base_fields + zero_fields)
    write_summary(args.summary_txt, args, rows, stats, zero_shot_columns)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
