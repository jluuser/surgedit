#!/usr/bin/env bash
set -euo pipefail

cd /public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/stage1_scisor_style

input_fasta="${INPUT_FASTA:-/public/home/zhangyangroup/chengshiz/keyuan.zhou/data/raw/uniref50/uniref50.fasta.gz}"
checkpoint="${SCISOR_CHECKPOINT:-/public/home/zhangyangroup/chengshiz/.cache/shortening_scud/SCISOR_U90_S.ckpt}"
p0_path="${P0_PATH:-/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit/p0.pt}"
out_dir="${OUT_DIR:-results/stage1_scisor_style_uniref50_capped_2gpu_len800}"
ckpt_dir="${CKPT_DIR:-checkpoints/stage1_scisor_style_uniref50_capped_2gpu_len800}"
steps_per_epoch="${STEPS_PER_EPOCH:-200000}"
eval_every_steps="${EVAL_EVERY_STEPS:-1000}"
periodic_max_val_samples="${PERIODIC_MAX_VAL_SAMPLES:-5000}"
max_val_samples="${MAX_VAL_SAMPLES:-10000}"
max_test_samples="${MAX_TEST_SAMPLES:-10000}"
batch_size="${BATCH_SIZE:-64}"
lr="${LR:-3e-5}"
max_original_len="${MAX_ORIGINAL_LEN:-800}"
max_len="${MAX_LEN:-1200}"
log_every="${LOG_EVERY:-25}"
disable_fa="${DISABLE_FA:-1}"
estimate_steps="${ESTIMATE_STEPS:-0}"
num_workers="${NUM_WORKERS:-2}"
eval_num_workers="${EVAL_NUM_WORKERS:-0}"
prefetch_factor="${PREFETCH_FACTOR:-2}"
persistent_workers="${PERSISTENT_WORKERS:-1}"
pin_memory="${PIN_MEMORY:-1}"
length_bucket_size="${LENGTH_BUCKET_SIZE:-1024}"
eval_length_bucket_size="${EVAL_LENGTH_BUCKET_SIZE:-0}"
tokenizer_cache_size="${TOKENIZER_CACHE_SIZE:-200000}"
disable_fast_tokenizer="${DISABLE_FAST_TOKENIZER:-0}"
disable_batch_alignments="${DISABLE_BATCH_ALIGNMENTS:-0}"
python_bin="${PYTHON_BIN:-/public/home/zhangyangroup/chengshiz/anaconda3/envs/surgedit/bin/python}"
torchrun_bin="${TORCHRUN_BIN:-/public/home/zhangyangroup/chengshiz/anaconda3/envs/surgedit/bin/torchrun}"

log_file="logs/stage1_scisor_style/uniref50_capped_len${max_original_len}_${timestamp}.log"
latest_log="logs/stage1_scisor_style/uniref50_capped_latest.log"
ln -sfn "$(basename "${log_file}")" "${latest_log}"

extra_args=()
if [[ "${disable_fa}" == "auto" ]]; then
  if "${python_bin}" -c "import flash_attn" >/dev/null 2>&1; then
    disable_fa="0"
  else
    disable_fa="1"
  fi
fi
if [[ "${disable_fa}" == "1" || "${disable_fa}" == "true" ]]; then
  extra_args+=(--disable-fa)
fi
if [[ "${estimate_steps}" == "1" || "${estimate_steps}" == "true" ]]; then
  extra_args+=(--estimate_steps_before_train)
fi
if [[ "${persistent_workers}" == "1" || "${persistent_workers}" == "true" ]]; then
  extra_args+=(--persistent_workers)
fi
if [[ "${pin_memory}" == "1" || "${pin_memory}" == "true" ]]; then
  extra_args+=(--pin_memory)
fi
if [[ "${disable_fast_tokenizer}" == "1" || "${disable_fast_tokenizer}" == "true" ]]; then
  extra_args+=(--disable_fast_tokenizer)
fi
if [[ "${disable_batch_alignments}" == "1" || "${disable_batch_alignments}" == "true" ]]; then
  extra_args+=(--disable_batch_alignments)
fi

{
  echo "===== SCISOR-style Stage-1 UniRef50 capped 2-GPU training ====="
  echo "start_time: $(date '+%F %T %Z')"
  echo "host: $(hostname)"
  echo "workdir: $(pwd)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0,1}"
  echo "INPUT_FASTA: ${input_fasta}"
  echo "SCISOR_CHECKPOINT: ${checkpoint}"
  echo "OUT_DIR: ${out_dir}"
  echo "CKPT_DIR: ${ckpt_dir}"
  echo "STEPS_PER_EPOCH: ${steps_per_epoch}"
  echo "EVAL_EVERY_STEPS: ${eval_every_steps}"
  echo "MAX_ORIGINAL_LEN: ${max_original_len}"
  echo "MAX_LEN: ${max_len}"
  echo "BATCH_SIZE: ${batch_size}"
  echo "LR: ${lr}"
  echo "DISABLE_FA: ${disable_fa}"
  echo "NUM_WORKERS: ${num_workers}"
  echo "EVAL_NUM_WORKERS: ${eval_num_workers}"
  echo "LENGTH_BUCKET_SIZE: ${length_bucket_size}"
  echo "PIN_MEMORY: ${pin_memory}"
  echo "FAST_TOKENIZER: $([[ "${disable_fast_tokenizer}" == "1" || "${disable_fast_tokenizer}" == "true" ]] && echo 0 || echo 1)"
  echo "BATCH_ALIGNMENTS: $([[ "${disable_batch_alignments}" == "1" || "${disable_batch_alignments}" == "true" ]] && echo 0 || echo 1)"
  echo

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  PYTHONUNBUFFERED=1 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  "${torchrun_bin}" --standalone --nproc_per_node=2 \
    scripts/train_stage1_scisor_style_streaming.py \
    --input_fasta "${input_fasta}" \
    --config configs/stage1_corruption.yaml \
    --checkpoint "${checkpoint}" \
    --p0 "${p0_path}" \
    --device cuda \
    --epochs 1 \
    --batch_size "${batch_size}" \
    --lr "${lr}" \
    --out_dir "${out_dir}" \
    --ckpt_dir "${ckpt_dir}" \
    --steps_per_epoch "${steps_per_epoch}" \
    --eval_every_steps "${eval_every_steps}" \
    --periodic_max_val_samples "${periodic_max_val_samples}" \
    --max_val_samples "${max_val_samples}" \
    --max_test_samples "${max_test_samples}" \
    --max_original_len "${max_original_len}" \
    --max_len "${max_len}" \
    --log_every "${log_every}" \
    --num_workers "${num_workers}" \
    --eval_num_workers "${eval_num_workers}" \
    --prefetch_factor "${prefetch_factor}" \
    --length_bucket_size "${length_bucket_size}" \
    --eval_length_bucket_size "${eval_length_bucket_size}" \
    --tokenizer_cache_size "${tokenizer_cache_size}" \
    --save_last_checkpoint \
    ${extra_args[@]+"${extra_args[@]}"}

  echo
  echo "end_time: $(date '+%F %T %Z')"
  echo "STAGE1_SCISOR_STYLE_CAPPED_TRAINING_FINISHED"
} 2>&1 | tee "${log_file}"
