#!/usr/bin/env bash
set -euo pipefail

cd /public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit

mkdir -p logs/step4
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/step4/full_uniref50_finetune_2gpu_${timestamp}.log"
latest_log="logs/step4/full_uniref50_finetune_2gpu_latest.log"

input_fasta="${INPUT_FASTA:-/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/raw/uniref50/uniref50.fasta.gz}"
init_checkpoint="${INIT_CHECKPOINT:-checkpoints/stage1_deletion_prior_100k/best_model.pt}"
out_dir="${OUT_DIR:-results/stage1_deletion_prior_full_uniref50_finetune_2gpu}"
ckpt_dir="${CKPT_DIR:-checkpoints/stage1_deletion_prior_full_uniref50_finetune_2gpu}"
lr="${LR:-3e-5}"
steps_per_epoch="${STEPS_PER_EPOCH:-20000}"
eval_every_steps="${EVAL_EVERY_STEPS:-2000}"
periodic_max_val_samples="${PERIODIC_MAX_VAL_SAMPLES:-10000}"
max_val_samples="${MAX_VAL_SAMPLES:-30000}"
max_test_samples="${MAX_TEST_SAMPLES:-30000}"
log_every="${LOG_EVERY:-100}"

echo "Logging to ${log_file}"
echo "Latest log symlink: ${latest_log}"

ln -sfn "$(basename "${log_file}")" "${latest_log}"

{
  echo "===== Step 4 full UniRef50 2-GPU fine-tuning ====="
  echo "start_time: $(date '+%F %T %Z')"
  echo "host: $(hostname)"
  echo "workdir: $(pwd)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}"
  echo "INPUT_FASTA: ${input_fasta}"
  echo "INIT_CHECKPOINT: ${init_checkpoint}"
  echo "OUT_DIR: ${out_dir}"
  echo "CKPT_DIR: ${ckpt_dir}"
  echo "LR: ${lr}"
  echo "STEPS_PER_EPOCH: ${steps_per_epoch}"
  echo "EVAL_EVERY_STEPS: ${eval_every_steps}"
  echo "PERIODIC_MAX_VAL_SAMPLES: ${periodic_max_val_samples}"
  echo "MAX_VAL_SAMPLES: ${max_val_samples}"
  echo "MAX_TEST_SAMPLES: ${max_test_samples}"
  echo "LOG_EVERY: ${log_every}"
  echo
  echo "Python:"
  python3 --version
  echo
  echo "Torch:"
  python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print("device_{} {}".format(idx, torch.cuda.get_device_name(idx)))
PY
  echo
  echo "nvidia-smi:"
  nvidia-smi || true
  echo
  echo "Command:"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} torchrun --standalone --nproc_per_node=2 scripts/train_stage1_deletion_prior_streaming.py ..."
  echo

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  PYTHONUNBUFFERED=1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  torchrun --standalone --nproc_per_node=2 \
    scripts/train_stage1_deletion_prior_streaming.py \
    --input_fasta "${input_fasta}" \
    --config configs/stage1_corruption.yaml \
    --device cuda \
    --epochs 1 \
    --batch_size 64 \
    --lr "${lr}" \
    --init_checkpoint "${init_checkpoint}" \
    --steps_per_epoch "${steps_per_epoch}" \
    --eval_every_steps "${eval_every_steps}" \
    --periodic_max_val_samples "${periodic_max_val_samples}" \
    --max_val_samples "${max_val_samples}" \
    --max_test_samples "${max_test_samples}" \
    --out_dir "${out_dir}" \
    --ckpt_dir "${ckpt_dir}" \
    --log_every "${log_every}" \
    --save_last_checkpoint

  echo
  echo "end_time: $(date '+%F %T %Z')"
  echo "STEP4_FULL_UNIREF50_2GPU_FINETUNE_FINISHED"
} 2>&1 | tee "${log_file}"
