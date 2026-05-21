#!/usr/bin/env bash
set -euo pipefail

# Run this script inside a GPU allocation, for example:
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=48G --pty bash
#   bash scripts/run_family_scisor_gpu_benchmark.sh

PYTHON="${PYTHON:-/public/home/zhangyangroup/chengshiz/anaconda3/envs/surgedit/bin/python}"

"${PYTHON}" scripts/run_scisor_bioprior10k_test_baselines.py \
  --input_fasta data/processed/bioprior_10k_family_splits/test.fasta \
  --split_csv data/processed/bioprior_10k_family_splits/test.csv \
  --residue_priors data/features/bioprior_10k_family_test_residue_biopriors.csv \
  --out_root results/scisor_bioprior_10k_family_test \
  --budgets 10,20,30 \
  --max_sequences 1000 \
  --execute \
  --validate

"${PYTHON}" scripts/summarize_scisor_bioprior10k_test_baselines.py \
  --test_csv data/processed/bioprior_10k_family_splits/test.csv \
  --residue_priors data/features/bioprior_10k_family_test_residue_biopriors.csv \
  --scisor_root results/scisor_bioprior_10k_family_test \
  --budgets 10,20,30 \
  --out_csv results/scisor_bioprior_10k_family_test/scisor_bioprior10k_family_test_baseline_summary.csv \
  --out_report results/scisor_bioprior_10k_family_test/scisor_bioprior10k_family_test_baseline_summary.txt

"${PYTHON}" scripts/build_biodel_final_comparison_report.py \
  --biodel_dir results/biodel_planner/family_split \
  --split_csv data/processed/bioprior_10k_family_splits/test.csv \
  --split_name family_test \
  --segments_csv data/features/bioprior_10k_bioprior_segments_with_stage1_utility_certified.csv \
  --selected_segments_csv results/biodel_planner/family_split/bioprior_10k_test_v2_selected_segments.csv \
  --scisor_summary_csv results/scisor_bioprior_10k_family_test/scisor_bioprior10k_family_test_baseline_summary.csv \
  --out_dir results/biodel_planner/family_split/final_test_family_comparison

"${PYTHON}" scripts/build_aaai_benchmark_suite_report.py
"${PYTHON}" scripts/check_biodel_pipeline_status.py
