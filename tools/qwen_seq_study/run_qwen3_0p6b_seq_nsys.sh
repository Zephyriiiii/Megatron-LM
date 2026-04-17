#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_VARIANT="${MODEL_VARIANT:-0p6b_base}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/qwen0p6b_seq_study_pp4_nsys_${MODEL_VARIANT}}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-0.6B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
IDLE_THRESHOLD_MB="${IDLE_THRESHOLD_MB:-512}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
PARALLEL_MODE="${PARALLEL_MODE:-mp}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-1}"
CP_COMM_TYPE="${CP_COMM_TYPE:-p2p}"
ENABLE_SEQUENCE_PARALLEL="${ENABLE_SEQUENCE_PARALLEL:-0}"
FSDP_WORLD_SIZE="${FSDP_WORLD_SIZE:-4}"
FSDP_IMPL="${FSDP_IMPL:-megatron_fsdp}"
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-optim_grads_params}"
FSDP_ENABLE_DOUBLE_BUFFER="${FSDP_ENABLE_DOUBLE_BUFFER:-0}"
FSDP_ENABLE_NCCL_UB="${FSDP_ENABLE_NCCL_UB:-0}"
ENABLE_MIN_CHECKPOINT="${ENABLE_MIN_CHECKPOINT:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
STOP_ON_OOM="${STOP_ON_OOM:-1}"
AUTO_SEQ_UNTIL_OOM="${AUTO_SEQ_UNTIL_OOM:-0}"
SEQ_LENGTHS="${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072 262144 393216 524288 655360 786432 917504}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-1048576}"
if [[ "${PARALLEL_MODE}" == "fsdp" ]]; then
  WORLD_SIZE="${FSDP_WORLD_SIZE}"
else
  WORLD_SIZE=$((TP_SIZE * PP_SIZE * CP_SIZE))
fi

case "${MODEL_VARIANT}" in
  0p6b_base)
    NUM_LAYERS="${NUM_LAYERS:-28}"
    ;;
  0p6b_1layer)
    NUM_LAYERS="${NUM_LAYERS:-1}"
    ;;
  0p6b_12layers)
    NUM_LAYERS="${NUM_LAYERS:-12}"
    ;;
  0p6b_4layers)
    NUM_LAYERS="${NUM_LAYERS:-4}"
    ;;
  0p6b_2layers)
    NUM_LAYERS="${NUM_LAYERS:-2}"
    ;;
  *)
    echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}" >&2
    exit 1
    ;;
esac

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export HF_ENDPOINT
export HF_HUB_DISABLE_TELEMETRY=1
if [[ "${PARALLEL_MODE}" == "fsdp" ]]; then
  unset CUDA_DEVICE_MAX_CONNECTIONS
else
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi

pick_idle_gpus() {
  local selected=""
  mapfile -t idle_gpus < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F', ' -v threshold="$IDLE_THRESHOLD_MB" '$2 + 0 <= threshold {print $1}'
  )

  if [[ ${#idle_gpus[@]} -lt ${WORLD_SIZE} ]]; then
    echo "Unable to find ${WORLD_SIZE} idle GPUs with memory.used <= ${IDLE_THRESHOLD_MB} MB" >&2
    exit 1
  fi

  selected=$(IFS=,; echo "${idle_gpus[*]:0:${WORLD_SIZE}}")
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
printf 'pp_size=%s\n' "${PP_SIZE}" >> "${OUT_ROOT}/run_metadata.env"
printf 'cp_size=%s\n' "${CP_SIZE}" >> "${OUT_ROOT}/run_metadata.env"
printf 'sequence_parallel=%s\n' "${ENABLE_SEQUENCE_PARALLEL}" >> "${OUT_ROOT}/run_metadata.env"
printf 'world_size=%s\n' "${WORLD_SIZE}" >> "${OUT_ROOT}/run_metadata.env"
printf 'model_variant=%s\n' "${MODEL_VARIANT}" >> "${OUT_ROOT}/run_metadata.env"
printf 'num_layers=%s\n' "${NUM_LAYERS}" >> "${OUT_ROOT}/run_metadata.env"
printf 'parallel_mode=%s\n' "${PARALLEL_MODE}" >> "${OUT_ROOT}/run_metadata.env"
printf 'enable_min_checkpoint=%s\n' "${ENABLE_MIN_CHECKPOINT}" >> "${OUT_ROOT}/run_metadata.env"
printf 'save_interval=%s\n' "${SAVE_INTERVAL}" >> "${OUT_ROOT}/run_metadata.env"
if [[ "${PARALLEL_MODE}" == "fsdp" ]]; then
  printf 'fsdp_impl=%s\n' "${FSDP_IMPL}" >> "${OUT_ROOT}/run_metadata.env"
  printf 'data_parallel_sharding_strategy=%s\n' "${FSDP_SHARDING_STRATEGY}" >> "${OUT_ROOT}/run_metadata.env"
fi

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
  --bf16
)

if [[ "${PARALLEL_MODE}" == "fsdp" ]]; then
  RUNTIME_ARGS+=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-megatron-fsdp
    --data-parallel-sharding-strategy "${FSDP_SHARDING_STRATEGY}"
    --use-distributed-optimizer
    --no-gradient-accumulation-fusion
    --ckpt-format fsdp_dtensor
    --calculate-per-token-loss
    --init-model-with-meta-device
    --grad-reduce-in-bf16
  )
  if [[ "${FSDP_ENABLE_DOUBLE_BUFFER}" == "1" ]]; then
    RUNTIME_ARGS+=(--fsdp-double-buffer)
  fi
  if [[ "${FSDP_ENABLE_NCCL_UB}" == "1" ]]; then
    RUNTIME_ARGS+=(--use-nccl-ub)
  fi
else
  RUNTIME_ARGS+=(
    --use-cpu-initialization
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
  )

  if [[ "${CP_SIZE}" -gt 1 ]]; then
    RUNTIME_ARGS+=(
      --context-parallel-size "${CP_SIZE}"
      --cp-comm-type "${CP_COMM_TYPE}"
    )
  fi

  if [[ "${ENABLE_SEQUENCE_PARALLEL}" == "1" ]]; then
    RUNTIME_ARGS+=(--sequence-parallel)
  fi
fi

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
  --num-layers "${NUM_LAYERS}"
  --hidden-size 1024
  --ffn-hidden-size 3072
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 8
  --kv-channels 128
  --normalization RMSNorm
  --norm-epsilon 1e-6
  --qk-layernorm
  --position-embedding-type rope
  --rotary-percent 1.0
  --rotary-base 1000000
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
  --swiglu
  --disable-bias-linear
  --attention-softmax-in-fp32
  --no-masked-softmax-fusion
  --cross-entropy-loss-fusion
  --cross-entropy-fusion-impl te
)

TOKENIZER_AND_DATA_ARGS=(
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model "${TOKENIZER_MODEL}"
  --make-vocab-size-divisible-by 1187
  --mock-data
  --no-create-attention-mask-in-dataloader
)

LOGGING_ARGS=(
  --log-interval 1
  --save-interval "${SAVE_INTERVAL}"
  --eval-interval 100000
  --num-workers 0
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
  printf 'pp_size=%s\n' "${PP_SIZE}" >> "${run_dir}/run_metadata.env"
  printf 'cp_size=%s\n' "${CP_SIZE}" >> "${run_dir}/run_metadata.env"
  printf 'sequence_parallel=%s\n' "${ENABLE_SEQUENCE_PARALLEL}" >> "${run_dir}/run_metadata.env"
  printf 'world_size=%s\n' "${WORLD_SIZE}" >> "${run_dir}/run_metadata.env"
  printf 'model_variant=%s\n' "${MODEL_VARIANT}" >> "${run_dir}/run_metadata.env"
  printf 'num_layers=%s\n' "${NUM_LAYERS}" >> "${run_dir}/run_metadata.env"
  printf 'parallel_mode=%s\n' "${PARALLEL_MODE}" >> "${run_dir}/run_metadata.env"
  printf 'enable_min_checkpoint=%s\n' "${ENABLE_MIN_CHECKPOINT}" >> "${run_dir}/run_metadata.env"
  if [[ "${PARALLEL_MODE}" == "fsdp" ]]; then
    printf 'fsdp_impl=%s\n' "${FSDP_IMPL}" >> "${run_dir}/run_metadata.env"
    printf 'data_parallel_sharding_strategy=%s\n' "${FSDP_SHARDING_STRATEGY}" >> "${run_dir}/run_metadata.env"
  fi

  echo "Running seq=${seq} on GPUs ${CUDA_VISIBLE_DEVICES}"

  local save_args=()
  if [[ "${ENABLE_MIN_CHECKPOINT}" == "1" ]]; then
    local save_dir="${run_dir}/checkpoints"
    mkdir -p "${save_dir}"
    save_args+=(--save "${save_dir}")
    printf 'checkpoint_dir=%s\n' "${save_dir}" >> "${run_dir}/run_metadata.env"
  fi

  set +e
  nsys profile \
    -s none \
    -t cuda,nvtx \
    -o "${run_dir}/seq${seq}" \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    torchrun --nproc_per_node="${WORLD_SIZE}" pretrain_gpt.py \
      "${RUNTIME_ARGS[@]}" \
      "${BATCH_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${MODEL_ARGS[@]}" \
      --seq-length "${seq}" \
      "${TOKENIZER_AND_DATA_ARGS[@]}" \
      "${LOGGING_ARGS[@]}" \
      "${save_args[@]}" \
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
  echo "AUTO_SEQ_UNTIL_OOM=1 is not supported for this script; use explicit exponential SEQ_LENGTHS" >&2
  exit 1
fi

read -r -a seq_values <<< "${SEQ_LENGTHS}"

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
