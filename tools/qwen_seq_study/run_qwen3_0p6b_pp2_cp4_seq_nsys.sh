#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOCAL_TOKENIZER_DIR="${ROOT_DIR}/assets/tokenizers/Qwen3-0.6B"

if [[ ! -f "${LOCAL_TOKENIZER_DIR}/tokenizer_config.json" ]]; then
  echo "Local tokenizer not found at ${LOCAL_TOKENIZER_DIR}" >&2
  exit 1
fi

export MODEL_VARIANT="${MODEL_VARIANT:-0p6b_2layers}"
export OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/runs/qwen0p6b_seq_study_pp2_cp4_nsys_2layers}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-$LOCAL_TOKENIZER_DIR}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-2}"
export CP_SIZE="${CP_SIZE:-4}"
export CP_COMM_TYPE="${CP_COMM_TYPE:-p2p}"
export ENABLE_SEQUENCE_PARALLEL="${ENABLE_SEQUENCE_PARALLEL:-0}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export TRAIN_ITERS="${TRAIN_ITERS:-5}"
export PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
export PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
export SEQ_LENGTHS="${SEQ_LENGTHS:-4096 8192 16384 32768 65536 131072 262144 393216 524288 655360 786432 917504}"
export MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-1048576}"
export AUTO_SEQ_UNTIL_OOM=0
export STOP_ON_OOM="${STOP_ON_OOM:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec bash "${ROOT_DIR}/tools/qwen_seq_study/run_qwen3_0p6b_seq_nsys.sh"
