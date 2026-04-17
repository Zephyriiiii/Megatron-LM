#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/qwen8b_seq_study_tp4_nsys}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-0.6B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
IDLE_THRESHOLD_MB="${IDLE_THRESHOLD_MB:-512}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
TP_SIZE="${TP_SIZE:-4}"
STOP_ON_OOM="${STOP_ON_OOM:-1}"
AUTO_SEQ_UNTIL_OOM="${AUTO_SEQ_UNTIL_OOM:-1}"
SEQ_LINEAR_STEP="${SEQ_LINEAR_STEP:-1024}"
SEQ_LENGTHS="${SEQ_LENGTHS:-512 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360}"

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export HF_ENDPOINT
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

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

mkdir -p "${OUT_ROOT}"
printf 'selected_gpus=%s\n' "${CUDA_VISIBLE_DEVICES}" > "${OUT_ROOT}/run_metadata.env"
printf 'tokenizer_model=%s\n' "${TOKENIZER_MODEL}" >> "${OUT_ROOT}/run_metadata.env"
printf 'hf_endpoint=%s\n' "${HF_ENDPOINT}" >> "${OUT_ROOT}/run_metadata.env"
printf 'seq_lengths=%s\n' "${SEQ_LENGTHS}" >> "${OUT_ROOT}/run_metadata.env"
printf 'tp_size=%s\n' "${TP_SIZE}" >> "${OUT_ROOT}/run_metadata.env"
printf 'auto_seq_until_oom=%s\n' "${AUTO_SEQ_UNTIL_OOM}" >> "${OUT_ROOT}/run_metadata.env"

echo "Warming tokenizer cache for ${TOKENIZER_MODEL} via ${HF_ENDPOINT}"
python - <<PY
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("${TOKENIZER_MODEL}")
print("tokenizer cache ready")
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
  --profile
  --nvtx-ranges
  --profile-step-start "${PROFILE_STEP_START}"
  --profile-step-end "${PROFILE_STEP_END}"
)

read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS:-}"

run_seq() {
  local seq="$1"
  local run_dir="${OUT_ROOT}/prof_seq${seq}"
  local rep_path="${run_dir}/seq${seq}.nsys-rep"
  local log_path="${run_dir}/train.log"

  if [[ -f "${rep_path}" ]]; then
    echo "Skipping seq=${seq}; report already exists at ${rep_path}"
    return 0
  fi

  mkdir -p "${run_dir}"
  printf 'seq_length=%s\n' "${seq}" > "${run_dir}/run_metadata.env"
  printf 'selected_gpus=%s\n' "${CUDA_VISIBLE_DEVICES}" >> "${run_dir}/run_metadata.env"
  printf 'tp_size=%s\n' "${TP_SIZE}" >> "${run_dir}/run_metadata.env"

  echo "Running seq=${seq} on GPUs ${CUDA_VISIBLE_DEVICES}"

  set +e
  nsys profile \
    -s none \
    -t cuda,nvtx \
    -o "${run_dir}/seq${seq}" \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    torchrun --nproc_per_node="${TP_SIZE}" pretrain_gpt.py \
      "${RUNTIME_ARGS[@]}" \
      "${BATCH_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${MODEL_ARGS[@]}" \
      --seq-length "${seq}" \
      "${TOKENIZER_AND_DATA_ARGS[@]}" \
      "${LOGGING_ARGS[@]}" \
      "${PROFILE_ARGS[@]}" \
      "${EXTRA_ARGS_ARRAY[@]}" \
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
  fi

  printf 'status=failed\n' >> "${run_dir}/run_metadata.env"
  echo "seq=${seq} failed; see ${log_path}" >&2
  return 1
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
  set +e
  run_seq "${seq}"
  status=$?
  set -e
  if [[ ${status} -eq 0 ]]; then
    continue
  fi
  if [[ ${status} -eq 2 && "${STOP_ON_OOM}" == "1" ]]; then
    break
  fi
done
