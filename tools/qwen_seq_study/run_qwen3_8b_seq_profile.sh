#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/qwen8b_seq_study}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-8B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
IDLE_THRESHOLD_MB="${IDLE_THRESHOLD_MB:-512}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
TP_SIZE="${TP_SIZE:-2}"
STOP_ON_OOM="${STOP_ON_OOM:-0}"
AUTO_SEQ_UNTIL_OOM="${AUTO_SEQ_UNTIL_OOM:-0}"
SEQ_LINEAR_STEP="${SEQ_LINEAR_STEP:-1024}"
SEQ_LENGTHS="${SEQ_LENGTHS:-1024 2048 4096 8192}"

pick_idle_gpus() {
  local selected=""
  mapfile -t idle_gpus < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' -v threshold="$IDLE_THRESHOLD_MB" '$2 + 0 <= threshold {print $1}'
  )

  if [[ ${#idle_gpus[@]} -lt ${TP_SIZE} ]]; then
    echo "Unable to find ${TP_SIZE} idle GPUs with memory.used <= ${IDLE_THRESHOLD_MB} MB" >&2
    exit 1
  fi

  selected=$(IFS=,; echo "${idle_gpus[*]:0:${TP_SIZE}}")
  echo "${selected}"
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=$(pick_idle_gpus)
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_ENDPOINT

mkdir -p "${OUT_ROOT}"
printf 'selected_gpus=%s\n' "${CUDA_VISIBLE_DEVICES}" > "${OUT_ROOT}/run_metadata.env"
printf 'tokenizer_model=%s\n' "${TOKENIZER_MODEL}" >> "${OUT_ROOT}/run_metadata.env"
printf 'hf_endpoint=%s\n' "${HF_ENDPOINT}" >> "${OUT_ROOT}/run_metadata.env"
printf 'seq_lengths=%s\n' "${SEQ_LENGTHS}" >> "${OUT_ROOT}/run_metadata.env"
printf 'tp_size=%s\n' "${TP_SIZE}" >> "${OUT_ROOT}/run_metadata.env"
printf 'auto_seq_until_oom=%s\n' "${AUTO_SEQ_UNTIL_OOM}" >> "${OUT_ROOT}/run_metadata.env"

RUNTIME_ARGS=(
  --use-mcore-models
  --transformer-impl transformer_engine
  --use-cpu-initialization
  --bf16
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size 1
)

BATCH_ARGS=(
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --eval-iters 0
)

OPTIMIZER_ARGS=(
  --lr 3e-4
  --min-lr 3e-5
  --lr-decay-style cosine
  --lr-warmup-iters 2
  --weight-decay 0.1
  --clip-grad 1.0
  --adam-beta1 0.9
  --adam-beta2 0.95
  --adam-eps 1e-8
)

MODEL_ARGS=(
  --num-layers 36
  --hidden-size 4096
  --ffn-hidden-size 12288
  --num-attention-heads 32
  --group-query-attention
  --num-query-groups 8
  --kv-channels 128
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --qk-layernorm
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 1000000
  --max-position-embeddings 40960
  --swiglu
  --disable-bias-linear
  --attention-softmax-in-fp32
)

TOKENIZER_AND_DATA_ARGS=(
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER_MODEL}"
  --make-vocab-size-divisible-by 128
  --mock-data
)

LOGGING_ARGS=(
  --log-interval 1
  --save-interval 100000
  --eval-interval 100000
  --rerun-mode disabled
  --async-strategy nvrx
)

PROFILE_ARGS=(
  --use-pytorch-profiler
  --profile
  --nvtx-ranges
  --profile-step-start "${PROFILE_STEP_START}"
  --profile-step-end "${PROFILE_STEP_END}"
  --pytorch-profiler-collect-shapes
)

run_seq() {
  local seq="$1"
  run_dir="${OUT_ROOT}/prof_seq${seq}"
  trace_path="${run_dir}/torch_profile/rank-0.json.gz"

  if [[ -f "${trace_path}" ]]; then
    echo "Skipping seq=${seq}; trace already exists at ${trace_path}"
    return 0
  fi

  mkdir -p "${run_dir}/tb"
  printf 'seq_length=%s\n' "${seq}" > "${run_dir}/run_metadata.env"
  printf 'selected_gpus=%s\n' "${CUDA_VISIBLE_DEVICES}" >> "${run_dir}/run_metadata.env"
  printf 'tp_size=%s\n' "${TP_SIZE}" >> "${run_dir}/run_metadata.env"

  log_path="${run_dir}/train.log"
  echo "Running seq=${seq} on GPUs ${CUDA_VISIBLE_DEVICES}"

  set +e
  torchrun --nproc_per_node="${TP_SIZE}" pretrain_gpt.py \
    "${RUNTIME_ARGS[@]}" \
    "${BATCH_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    --seq-length "${seq}" \
    "${TOKENIZER_AND_DATA_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    --tensorboard-dir "${run_dir}/tb" \
    > "${log_path}" 2>&1
  status=$?
  set -e

  printf 'exit_code=%s\n' "${status}" >> "${run_dir}/run_metadata.env"

  if [[ ${status} -eq 0 ]]; then
    printf 'status=success\n' >> "${run_dir}/run_metadata.env"
    echo "Finished seq=${seq}"
    return 0
  fi

  if rg -n "out of memory|CUDA error: out of memory|torch.OutOfMemoryError" "${log_path}" >/dev/null 2>&1; then
    printf 'status=oom\n' >> "${run_dir}/run_metadata.env"
    echo "seq=${seq} OOM; recorded"
    return 2
  else
    printf 'status=failed\n' >> "${run_dir}/run_metadata.env"
    echo "seq=${seq} failed; see ${log_path}" >&2
    return 1
  fi
}

if [[ "${AUTO_SEQ_UNTIL_OOM}" == "1" ]]; then
  seq_values=(512 1024)
  next_seq=2048
  while true; do
    seq_values+=("${next_seq}")
    next_seq=$((next_seq + SEQ_LINEAR_STEP))
    if [[ ${#seq_values[@]} -ge 32 ]]; then
      break
    fi
  done
else
  read -r -a seq_values <<< "${SEQ_LENGTHS}"
fi

for seq in "${seq_values[@]}"; do
  run_seq "${seq}"
  status=$?
  if [[ ${status} -eq 0 ]]; then
    continue
  fi
  if [[ ${status} -eq 2 && "${STOP_ON_OOM}" == "1" ]]; then
    break
  fi
done
