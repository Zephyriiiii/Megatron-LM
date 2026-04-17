#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOCAL_TOKENIZER_DIR="${ROOT_DIR}/assets/tokenizers/Qwen3-0.6B"

if [[ ! -f "${LOCAL_TOKENIZER_DIR}/tokenizer_config.json" ]]; then
  echo "Local tokenizer not found at ${LOCAL_TOKENIZER_DIR}" >&2
  exit 1
fi

export PARALLEL_MODE="${PARALLEL_MODE:-fsdp}"
export FSDP_IMPL="${FSDP_IMPL:-megatron_fsdp}"
export FSDP_WORLD_SIZE="${FSDP_WORLD_SIZE:-4}"
export FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-optim_grads_params}"
export FSDP_ENABLE_DOUBLE_BUFFER="${FSDP_ENABLE_DOUBLE_BUFFER:-0}"
export FSDP_ENABLE_NCCL_UB="${FSDP_ENABLE_NCCL_UB:-0}"

export MODEL_VARIANT="${MODEL_VARIANT:-0p6b_1layer}"
export OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/qwen0p6b_seq_study_1layer_fsdp4_nsys}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$LOCAL_TOKENIZER_DIR}"
export TP_SIZE=1
export PP_SIZE=1
export CP_SIZE=1
export ENABLE_SEQUENCE_PARALLEL=0
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export TRAIN_ITERS="${TRAIN_ITERS:-5}"
export PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
export PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
export ENABLE_MIN_CHECKPOINT="${ENABLE_MIN_CHECKPOINT:-1}"
export SEQ_LENGTHS="${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072 262144 524288 1048576}"
export MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-1048576}"
export AUTO_SEQ_UNTIL_OOM=0
export STOP_ON_OOM="${STOP_ON_OOM:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec bash "${ROOT_DIR}/tools/qwen_seq_study/run_qwen3_0p6b_seq_nsys.sh"
