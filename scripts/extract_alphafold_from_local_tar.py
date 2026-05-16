from __future__ import print_function

import argparse
import csv
import gzip
import os
import shutil
import sys
import tarfile


SOURCE = "local_afdb_swissprot_pdb_tar"
AF_VERSION = "v4"


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


def read_targets(motif_csv):
    rows = []
    seen = set()
    with open_csv_read(motif_csv) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accession = row.get("accession") or row.get("protein_id")
            protein_id = row.get("protein_id") or accession
            accession = (accession or "").strip()
            protein_id = (protein_id or "").strip()
            if not accession or accession in seen:
                continue
            rows.append({"protein_id": protein_id, "accession": accession})
            seen.add(accession)
    return rows


def member_name(accession):
    return "AF-{0}-F1-model_v4.pdb.gz".format(accession)


def output_pdb_path(out_dir, accession):
    return os.path.join(out_dir, "{0}.pdb".format(accession))


def is_nonempty_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


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


def extract_member_to_pdb(tar, member, out_path):
    tmp_path = out_path + ".tmp"
    source = tar.extractfile(member)
    if source is None:
        raise RuntimeError("tar member could not be opened")

    try:
        with gzip.GzipFile(fileobj=source, mode="rb") as gz_handle:
            with open(tmp_path, "wb") as out_handle:
                shutil.copyfileobj(gz_handle, out_handle)
        if not is_nonempty_file(tmp_path):
            raise RuntimeError("extracted PDB is empty")
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_index(index_csv, rows):
    fields = [
        "protein_id",
        "accession",
        "structure_path",
        "source",
        "status",
        "af_version",
        "error",
    ]
    with open_csv_write(index_csv) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_failed(failed_txt, failed_rows):
    parent = os.path.dirname(failed_txt)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(failed_txt, "w") as handle:
        for row in failed_rows:
            handle.write("{0}\t{1}\n".format(row["accession"], row["error"]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract selected AlphaFold Swiss-Prot PDBs from a local AFDB tar."
    )
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--afdb_dir", required=True)
    parser.add_argument("--tar_name", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--failed_txt", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    targets = read_targets(args.motif_csv)
    ensure_dir(args.out_dir)

    tar_path = os.path.join(args.afdb_dir, args.tar_name)
    index_rows = []
    failed_rows = []
    pending = []
    for target in targets:
        out_path = output_pdb_path(args.out_dir, target["accession"])
        if is_nonempty_file(out_path):
            index_rows.append(
                {
                    "protein_id": target["protein_id"],
                    "accession": target["accession"],
                    "structure_path": out_path,
                    "source": SOURCE,
                    "status": "exists",
                    "af_version": AF_VERSION,
                    "error": "",
                }
            )
        else:
            pending.append(target)

    if pending and not os.path.exists(tar_path):
        for target in pending:
            error = "tar not found: {0}".format(tar_path)
            row = {
                "protein_id": target["protein_id"],
                "accession": target["accession"],
                "structure_path": output_pdb_path(args.out_dir, target["accession"]),
                "source": SOURCE,
                "status": "failed",
                "af_version": AF_VERSION,
                "error": error,
            }
            index_rows.append(row)
            failed_rows.append(row)
    elif pending:
        wanted = dict((member_name(target["accession"]), target) for target in pending)
        print("Scanning tar members once for {0} requested structures".format(len(pending)))
        with tarfile.open(tar_path, "r:") as tar:
            found = index_tar_members(tar, set(wanted.keys()))
            print("Found {0}/{1} requested tar members".format(len(found), len(wanted)))

            for target in pending:
                accession = target["accession"]
                name = member_name(accession)
                out_path = output_pdb_path(args.out_dir, accession)
                if name not in found:
                    error = "tar member not found: {0}".format(name)
                    row = {
                        "protein_id": target["protein_id"],
                        "accession": accession,
                        "structure_path": out_path,
                        "source": SOURCE,
                        "status": "failed",
                        "af_version": AF_VERSION,
                        "error": error,
                    }
                    index_rows.append(row)
                    failed_rows.append(row)
                    continue

                try:
                    extract_member_to_pdb(tar, found[name], out_path)
                    status = "success"
                    error = ""
                except Exception as exc:
                    status = "failed"
                    error = str(exc)

                row = {
                    "protein_id": target["protein_id"],
                    "accession": accession,
                    "structure_path": out_path,
                    "source": SOURCE,
                    "status": status,
                    "af_version": AF_VERSION,
                    "error": error,
                }
                index_rows.append(row)
                if status == "failed":
                    failed_rows.append(row)

    order = dict((target["accession"], i) for i, target in enumerate(targets))
    index_rows.sort(key=lambda row: order.get(row["accession"], 10**9))
    write_index(args.index_csv, index_rows)
    write_failed(args.failed_txt, failed_rows)

    counts = {"success": 0, "exists": 0, "failed": 0}
    for row in index_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("Targets: {0}".format(len(targets)))
    print("success: {0}".format(counts.get("success", 0)))
    print("exists: {0}".format(counts.get("exists", 0)))
    print("failed: {0}".format(counts.get("failed", 0)))
    print("Wrote {0}".format(args.index_csv))
    print("Wrote {0}".format(args.failed_txt))


if __name__ == "__main__":
    main()
