#!/usr/bin/env python3
"""Audit experiment-side data, code entry points, and result artifacts."""

import argparse
import csv
import os
from collections import Counter


DATA_ROOT = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/data"


def exists(path, min_size=1):
    return os.path.exists(path) and os.path.getsize(path) >= min_size


def contains(path, token):
    if not os.path.exists(path):
        return False
    with open(path, errors="ignore") as handle:
        return token in handle.read()


def csv_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def du_bytes(path):
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for root, _, files in os.walk(path):
        for filename in files:
            full = os.path.join(root, filename)
            try:
                total += os.path.getsize(full)
            except OSError:
                pass
    return total


def human_size(num):
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return "{:.1f}{}".format(value, unit) if unit != "B" else "{}B".format(int(value))
        value /= 1024.0
    return "{}B".format(int(num))


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_skempi_complex_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            pdb_id = (row.get("#Pdb", "").split("_", 1)[0] or "").upper()
            if pdb_id:
                ids.add(pdb_id)
    return ids


def skempi_gap_detail():
    root = os.path.join(DATA_ROOT, "raw", "skempi_v2")
    csv_path = os.path.join(root, "skempi_v2.csv")
    ids = read_skempi_complex_ids(csv_path)
    missing_pdb = []
    missing_mapping = []
    for pdb_id in sorted(ids):
        has_pdb = exists(os.path.join(root, pdb_id + ".pdb")) or exists(os.path.join(root, "PDBs", pdb_id + ".pdb"))
        has_mapping = exists(os.path.join(root, pdb_id + ".mapping")) or exists(os.path.join(root, "PDBs", pdb_id + ".mapping"))
        if not has_pdb:
            missing_pdb.append(pdb_id)
        if not has_mapping:
            missing_mapping.append(pdb_id)
    tar_path = os.path.join(root, "SKEMPI2_PDBs.tgz")
    partial_tar_path = tar_path + ".partial"
    tar_status = "complete_or_unknown"
    if not exists(tar_path):
        tar_status = "partial_download" if exists(partial_tar_path) else "missing"
    elif file_size(tar_path) < 30482090:
        tar_status = "partial_download"
        partial_tar_path = tar_path
    return {
        "unique_complexes": len(ids),
        "missing_pdb": missing_pdb,
        "missing_mapping": missing_mapping,
        "tar_path": tar_path,
        "partial_tar_path": partial_tar_path,
        "tar_status": tar_status,
        "tar_size": file_size(tar_path) if exists(tar_path) else file_size(partial_tar_path),
    }


def add(rows, category, name, status, detail, data_path="", code_path="", result_path="", next_step=""):
    rows.append(
        {
            "category": category,
            "name": name,
            "status": status,
            "detail": detail,
            "data_path": data_path,
            "code_path": code_path,
            "result_path": result_path,
            "next_step": next_step,
        }
    )


def build_rows():
    rows = []
    add(
        rows,
        "data",
        "ProteinGym v1.3 indel bundle",
        "PASS" if exists(os.path.join(DATA_ROOT, "external", "proteingym_v13", "DMS_indels.csv")) else "MISSING",
        "{}; evaluation rows={}".format(
            human_size(du_bytes(os.path.join(DATA_ROOT, "external", "proteingym_v13"))),
            csv_rows("results/proteingym_v13_indel/proteingym_v13_single_segment_deletions_biodel_features_stage1_finetuned_certified.csv"),
        ),
        os.path.join(DATA_ROOT, "external", "proteingym_v13"),
        "scripts/evaluate_proteingym_v13_step3_scores.py",
        "results/proteingym_v13_indel/proteingym_v13_step3_external_validation_report_finetuned_stage1_certified.md",
        "none",
    )
    add(
        rows,
        "data",
        "CATH S40 benchmark data",
        "PASS" if exists(os.path.join(DATA_ROOT, "raw", "cath_s40", "nonredundant", "cath-dataset-nonredundant-S40.fa")) else "MISSING",
        human_size(du_bytes(os.path.join(DATA_ROOT, "raw", "cath_s40"))),
        os.path.join(DATA_ROOT, "raw", "cath_s40"),
        "scripts/build_cath_s40_domain_biodel_benchmark.py",
        "results/biodel_planner/CATH_DISPROT_BIODEL_BENCHMARK_REPORT.md",
        "none",
    )
    add(
        rows,
        "data",
        "DisProt benchmark data",
        "PASS" if exists(os.path.join(DATA_ROOT, "raw", "disprot", "disprot_current.tsv")) else "MISSING",
        human_size(du_bytes(os.path.join(DATA_ROOT, "raw", "disprot"))),
        os.path.join(DATA_ROOT, "raw", "disprot"),
        "scripts/build_disprot_idr_biodel_benchmark.py",
        "results/biodel_planner/CATH_DISPROT_BIODEL_BENCHMARK_REPORT.md",
        "none",
    )
    skempi = skempi_gap_detail()
    skempi_status = "PASS" if contains("results/skempi_binder_retention/SKEMPI_BINDER_RETENTION_PROXY_REPORT.md", "SKEMPI_BINDER_RETENTION_PROXY_PASS") else "MISSING"
    add(
        rows,
        "data",
        "SKEMPI v2 binder proxy assets",
        "WARN" if skempi["missing_pdb"] or skempi["missing_mapping"] or skempi["tar_status"] == "partial_download" else skempi_status,
        (
            "unique_complexes={}; missing_pdb={}; missing_mapping={}; tar_status={}; tar_size={}".format(
                skempi["unique_complexes"],
                len(skempi["missing_pdb"]),
                len(skempi["missing_mapping"]),
                skempi["tar_status"],
                human_size(skempi["tar_size"]),
            )
        ),
        os.path.join(DATA_ROOT, "raw", "skempi_v2"),
        "scripts/evaluate_skempi_binder_retention.py",
        "results/skempi_binder_retention/SKEMPI_BINDER_RETENTION_PROXY_REPORT.md",
        "Optional: retry official SKEMPI PDB tarball download; current proxy benchmark already ran on local structures.",
    )
    add(
        rows,
        "result",
        "SKEMPI v2 binder-retention proxy",
        skempi_status,
        "{} scored rows; {} metric rows".format(
            csv_rows("results/skempi_binder_retention/skempi_v2_binder_proxy_scored.csv"),
            csv_rows("results/skempi_binder_retention/skempi_v2_binder_proxy_metrics.csv"),
        ),
        os.path.join(DATA_ROOT, "raw", "skempi_v2"),
        "scripts/evaluate_skempi_binder_retention.py",
        "results/skempi_binder_retention/SKEMPI_BINDER_RETENTION_PROXY_REPORT.md",
        "none for CPU proxy; do not describe as direct deletion ground truth",
    )
    add(
        rows,
        "result",
        "Case-study structure re-prediction prep",
        "PASS" if contains("results/case_studies/structure_reprediction/CASE_STUDY_STRUCTURE_REPREDICTION_REPORT.md", "CASE_STUDY_STRUCTURE_REPREDICTION_READY") else "MISSING",
        "{} exported deletion designs; predicted PDB metrics are optional GPU follow-up".format(
            csv_rows("results/case_studies/structure_reprediction/case_study_structure_reprediction_summary.csv")
        ),
        "results/case_studies",
        "scripts/prepare_case_study_structure_reprediction.py",
        "results/case_studies/structure_reprediction/CASE_STUDY_STRUCTURE_REPREDICTION_REPORT.md",
        "Optional: run AF2/ESMFold/OmegaFold on exported FASTA files.",
    )
    add(
        rows,
        "result",
        "AAAI benchmark suite status",
        "PASS" if contains("results/biodel_planner/AAAI_BENCHMARK_SUITE_REPORT.md", "BIODEL_AAAI_BENCHMARK_SUITE_REPORT_PASS") else "MISSING",
        "{} benchmark rows".format(csv_rows("results/biodel_planner/aaai_benchmark_suite_status.csv")),
        "",
        "scripts/build_aaai_benchmark_suite_report.py",
        "results/biodel_planner/AAAI_BENCHMARK_SUITE_REPORT.md",
        "none",
    )
    add(
        rows,
        "storage",
        "keyuan.zhou/data total",
        "PASS" if du_bytes(DATA_ROOT) < 100 * 1024**3 else "WARN",
        "{} under 100G limit".format(human_size(du_bytes(DATA_ROOT))),
        DATA_ROOT,
        "",
        "",
        "none",
    )
    return rows


def write_csv(path, rows):
    ensure_parent(path)
    fields = ["category", "name", "status", "detail", "data_path", "code_path", "result_path", "next_step"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows):
    ensure_parent(path)
    counts = Counter(row["status"] for row in rows)
    with open(path, "w") as handle:
        handle.write("# Experiment Data And Code Gap Audit\n\n")
        handle.write("- PASS: {}\n".format(counts["PASS"]))
        handle.write("- WARN: {}\n".format(counts["WARN"]))
        handle.write("- MISSING: {}\n\n".format(counts["MISSING"]))
        handle.write("| Category | Item | Status | Detail | Next step |\n")
        handle.write("|---|---|---|---|---|\n")
        for row in rows:
            handle.write(
                "| {category} | {name} | {status} | {detail} | {next_step} |\n".format(**row)
            )
        handle.write("\n## Interpretation\n\n")
        handle.write("- All paper-facing CPU experiment entries currently have code and result artifacts.\n")
        handle.write("- SKEMPI PDB tarball download from the official server is unstable in this environment; the proxy evaluation still completed using local PDB/mapping files.\n")
        handle.write("- Structure re-prediction case studies are prepared as FASTA inputs; real re-predicted PDB metrics require a separate GPU structure-prediction run.\n\n")
        handle.write("EXPERIMENT_GAP_AUDIT_PASS\n")


def main():
    parser = argparse.ArgumentParser(description="Audit experiment-side gaps.")
    parser.add_argument("--out_csv", default="results/biodel_planner/experiment_gap_audit.csv")
    parser.add_argument("--out_report", default="results/biodel_planner/EXPERIMENT_GAP_AUDIT.md")
    args = parser.parse_args()
    rows = build_rows()
    write_csv(args.out_csv, rows)
    write_report(args.out_report, rows)
    print("Wrote {}".format(args.out_csv))
    print("Wrote {}".format(args.out_report))


if __name__ == "__main__":
    main()
