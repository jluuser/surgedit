from __future__ import print_function

import argparse
import csv
import os
import sys


def open_csv_read(path):
    if sys.version_info[0] >= 3:
        return open(path, "r", newline="")
    return open(path, "rb")


def read_csv_rows(path):
    with open_csv_read(path) as handle:
        return list(csv.DictReader(handle))


def has_atom_line(path):
    with open(path, "rb") as handle:
        for raw_line in handle:
            if raw_line.startswith(b"ATOM"):
                return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect local AlphaFold structure index.")
    parser.add_argument("--motif_csv", required=True)
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--out_report", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    motif_rows = read_csv_rows(args.motif_csv) if os.path.exists(args.motif_csv) else []
    index_rows = read_csv_rows(args.index_csv) if os.path.exists(args.index_csv) else []

    counts = {"success": 0, "exists": 0, "failed": 0}
    for row in index_rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1

    usable_rows = [
        row for row in index_rows if row.get("status") in ("success", "exists")
    ]
    first_five = usable_rows[:5]
    pdb_checks = []
    for row in first_five:
        path = row.get("structure_path", "")
        exists_nonempty = os.path.exists(path) and os.path.getsize(path) > 0
        atom_ok = has_atom_line(path) if exists_nonempty else False
        pdb_checks.append(
            {
                "accession": row.get("accession", ""),
                "path": path,
                "exists_nonempty": exists_nonempty,
                "contains_atom": atom_ok,
            }
        )

    failed = [
        "{0}:{1}".format(row.get("accession", ""), row.get("error", ""))
        for row in index_rows
        if row.get("status") == "failed"
    ]
    usable_count = counts.get("success", 0) + counts.get("exists", 0)
    total = len(index_rows)
    success_rate = float(usable_count) / total if total else 0.0
    conclusion = (
        "STRUCTURE_LOCAL_AFDB_PASS"
        if usable_count >= 50
        else "STRUCTURE_LOCAL_AFDB_WARN"
    )

    parent = os.path.dirname(args.out_report)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.out_report, "w") as handle:
        handle.write("Local AlphaFold DB structure extraction report\n\n")
        handle.write("motif_csv: {0}\n".format(args.motif_csv))
        handle.write("index_csv: {0}\n".format(args.index_csv))
        handle.write("motif protein count: {0}\n".format(len(motif_rows)))
        handle.write("index record count: {0}\n".format(len(index_rows)))
        handle.write("success: {0}\n".format(counts.get("success", 0)))
        handle.write("exists: {0}\n".format(counts.get("exists", 0)))
        handle.write("failed: {0}\n".format(counts.get("failed", 0)))
        handle.write("success_plus_exists: {0}\n".format(usable_count))
        handle.write("success_rate: {0:.4f}\n\n".format(success_rate))
        handle.write("First 5 success/existing PDB checks:\n")
        for check in pdb_checks:
            handle.write(
                "- {0}\t{1}\texists_nonempty={2}\tcontains_atom={3}\n".format(
                    check["accession"],
                    check["path"],
                    check["exists_nonempty"],
                    check["contains_atom"],
                )
            )
        if not pdb_checks:
            handle.write("- none\n")
        handle.write("\nFailed accessions:\n")
        if failed:
            for item in failed:
                handle.write("- {0}\n".format(item))
        else:
            handle.write("- none\n")
        handle.write("\n{0}\n".format(conclusion))

    print("Wrote {0}".format(args.out_report))


if __name__ == "__main__":
    main()
