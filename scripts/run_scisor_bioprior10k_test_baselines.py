#!/usr/bin/env python3
"""Run SCISOR-family baselines on the BioPrior-10K held-out test split.

The script is intentionally resumable: existing non-empty deletions.json files
are skipped unless --force is provided. By default it prints commands only;
pass --execute to run them.
"""

import argparse
import os
import subprocess
import sys

import torch


PYTHON = "/public/home/zhangyangroup/chengshiz/anaconda3/envs/surgedit/bin/python"


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def subprocess_env(args):
    env = os.environ.copy()
    if args.offline_hf:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
    if args.clear_proxy:
        for key in [
            "HF_ENDPOINT",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        ]:
            env.pop(key, None)
    return env


def run_cmd(cmd, execute, env=None):
    print(" ".join(cmd))
    if execute:
        subprocess.run(cmd, check=True, env=env)


def output_complete(out_dir):
    path = os.path.join(out_dir, "deletions.json")
    return os.path.exists(path) and os.path.getsize(path) > 0


def count_fasta_records(path):
    count = 0
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def write_fasta_prefix(input_path, output_path, max_records):
    ensure_dir(os.path.dirname(output_path))
    written = 0
    with open(input_path) as in_handle, open(output_path, "w") as out_handle:
        keep = False
        for line in in_handle:
            if line.startswith(">"):
                written += 1
                keep = written <= max_records
            if keep:
                out_handle.write(line)
            if written > max_records and line.startswith(">"):
                break
    return output_path


def baseline_specs():
    return [
        {
            "name": "nomask",
            "extra": [],
        },
        {
            "name": "hardmask",
            "extra": [
                "--motif_csv",
                "data/processed/bioprior_10k_splits/test.csv",
                "--protect_motif",
            ],
        },
        {
            "name": "hardmask_shadow02",
            "extra": [
                "--motif_csv",
                "data/processed/bioprior_10k_splits/test.csv",
                "--protect_motif",
                "--structure_priors",
                "data/features/bioprior_10k_test_residue_biopriors.csv",
                "--protect_shadow",
                "--shadow_penalty",
                "0.2",
                "--shadow_mode",
                "multiply",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Run BioPrior-10K test SCISOR baselines.")
    parser.add_argument("--input_fasta", default="data/processed/bioprior_10k_splits/test.fasta")
    parser.add_argument("--out_root", default="results/scisor_bioprior_10k_test")
    parser.add_argument("--budgets", default="10,20,30")
    parser.add_argument("--max_sequences", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--offline_hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set HuggingFace/Transformers offline mode for subprocesses.",
    )
    parser.add_argument(
        "--clear_proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove proxy and HF_ENDPOINT variables for subprocesses.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_root)
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    total_input_records = count_fasta_records(args.input_fasta)
    validation_input_fasta = args.input_fasta
    if args.max_sequences < total_input_records:
        validation_input_fasta = os.path.join(
            args.out_root, "validation_first{}_input.fasta".format(args.max_sequences)
        )
        write_fasta_prefix(args.input_fasta, validation_input_fasta, args.max_sequences)
        print("Using subset validation FASTA {}".format(validation_input_fasta))
    device_note = "cuda_available={}".format(torch.cuda.is_available())
    print(device_note)
    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA is not available. Full 1000-protein SCISOR baselines "
            "may be slow on CPU; consider running this script inside an srun GPU session.",
            file=sys.stderr,
        )
    child_env = subprocess_env(args)
    print("offline_hf={}".format(args.offline_hf))
    print("clear_proxy={}".format(args.clear_proxy))

    for budget in budgets:
        for spec in baseline_specs():
            out_dir = os.path.join(args.out_root, "run{:02d}_{}".format(budget, spec["name"]))
            ensure_dir(out_dir)
            if output_complete(out_dir) and not args.force:
                print("SKIP existing {}".format(out_dir))
            else:
                cmd = [
                    PYTHON,
                    "scripts/run_scisor_shrink.py",
                    "--input",
                    args.input_fasta,
                    "--output-dir",
                    out_dir,
                    "--shrink-pct",
                    str(budget),
                    "--temperature",
                    str(args.temperature),
                    "--disable-fa",
                    "--max-sequences",
                    str(args.max_sequences),
                ] + spec["extra"]
                run_cmd(cmd, args.execute, env=child_env)

            if args.validate:
                validation_path = os.path.join(out_dir, "validation_report.txt")
                cmd = [
                    PYTHON,
                    "scripts/validate_deletions.py",
                    "--input_fasta",
                    validation_input_fasta,
                    "--shrunk_fasta",
                    os.path.join(out_dir, "shrunk_sequences.fasta"),
                    "--deletions_json",
                    os.path.join(out_dir, "deletions.json"),
                    "--out_report",
                    validation_path,
                ]
                if output_complete(out_dir):
                    run_cmd(cmd, args.execute, env=child_env)
                else:
                    print("SKIP validation missing output {}".format(out_dir))

    print("Done. execute={}".format(args.execute))


if __name__ == "__main__":
    main()
