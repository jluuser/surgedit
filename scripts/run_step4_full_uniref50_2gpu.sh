#!/usr/bin/env bash
set -euo pipefail

cd /public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit

mkdir -p logs/step4
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/step4/full_uniref50_2gpu_${timestamp}.log"
latest_log="logs/step4/full_uniref50_2gpu_latest.log"
log_every="${LOG_EVERY:-100}"

echo "Logging to ${log_file}"
echo "Latest log symlink: ${latest_log}"

ln -sfn "$(basename "${log_file}")" "${latest_log}"

{
  echo "===== Step 4 full UniRef50 2-GPU training ====="
  echo "start_time: $(date '+%F %T %Z')"
  echo "host: $(hostname)"
  echo "workdir: $(pwd)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}"
  echo "LOG_EVERY: ${log_every}"
  echo "ETA: enabled by --estimate_steps_before_train"
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
  echo "CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_stage1_deletion_prior_streaming.py ..."
  echo

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  PYTHONUNBUFFERED=1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  torchrun --standalone --nproc_per_node=2 \
    scripts/train_stage1_deletion_prior_streaming.py \
    --input_fasta /public/home/zhangyangroup/chengshiz/keyuan.zhou/data/raw/uniref50/uniref50.fasta.gz \
    --config configs/stage1_corruption.yaml \
    --device cuda \
    --epochs 1 \
    --batch_size 64 \
    --out_dir results/stage1_deletion_prior_full_uniref50_stream_ddp_2gpu \
    --ckpt_dir checkpoints/stage1_deletion_prior_full_uniref50_stream_ddp_2gpu \
    --log_every "${log_every}" \
    --estimate_steps_before_train

  echo
  echo "end_time: $(date '+%F %T %Z')"
  echo "STEP4_FULL_UNIREF50_2GPU_TRAINING_FINISHED"
} 2>&1 | tee "${log_file}"
