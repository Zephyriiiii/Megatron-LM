#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

CONDA_HOME=${CONDA_HOME:-/home/zsh/miniconda3}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-llm}

if [[ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_HOME}/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV_NAME}" >/dev/null 2>&1 || conda activate "${CONDA_ENV_NAME}"
fi

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
USE_SP=${USE_SP:-0}
USE_DIST_OPT=${USE_DIST_OPT:-1}

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
TRAIN_ITERS=${TRAIN_ITERS:-10}
SEQ_LENGTH=${SEQ_LENGTH:-2048}

LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-5}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}

LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
EVAL_ITERS=${EVAL_ITERS:-0}

DRY_RUN=${DRY_RUN:-0}

print_command() {
  printf 'Running command:\n'
  printf ' %q' "$@"
  printf '\n'
}

validate_dense_parallelism() {
  local parallel_product=$(( TP_SIZE * PP_SIZE * CP_SIZE ))
  if (( parallel_product <= 0 )); then
    echo "Invalid dense parallel product: ${parallel_product}" >&2
    exit 1
  fi

  if (( NPROC_PER_NODE % parallel_product != 0 )); then
    echo "NPROC_PER_NODE=${NPROC_PER_NODE} must be divisible by TP*PP*CP=${parallel_product}" >&2
    exit 1
  fi

  if (( USE_SP == 1 && TP_SIZE == 1 )); then
    echo "Sequence parallelism requires TP_SIZE > 1" >&2
    exit 1
  fi
}

validate_moe_parallelism() {
  local parallel_product=$(( TP_SIZE * PP_SIZE * CP_SIZE * EP_SIZE ))
  if (( parallel_product <= 0 )); then
    echo "Invalid MoE parallel product: ${parallel_product}" >&2
    exit 1
  fi

  if (( NPROC_PER_NODE % parallel_product != 0 )); then
    echo "NPROC_PER_NODE=${NPROC_PER_NODE} must be divisible by TP*PP*CP*EP=${parallel_product}" >&2
    exit 1
  fi

  if (( USE_SP == 1 && TP_SIZE == 1 )); then
    echo "Sequence parallelism requires TP_SIZE > 1" >&2
    exit 1
  fi
}

run_qwen3_8b_dense() {
  validate_dense_parallelism

  local run_dir="${REPO_ROOT}/${OUT_DIR}"
  mkdir -p "${run_dir}/ckpt" "${run_dir}/tb"

  local -a cmd=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    pretrain_gpt.py
    --use-mcore-models
    --transformer-impl transformer_engine
    --use-cpu-initialization
    --bf16
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --seq-length "${SEQ_LENGTH}"
    --lr "${LR}"
    --min-lr "${MIN_LR}"
    --lr-decay-style cosine
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --weight-decay "${WEIGHT_DECAY}"
    --clip-grad "${CLIP_GRAD}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --untie-embeddings-and-output-weights
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
    --no-bias-swiglu-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-8B
    --vocab-size 151936
    --make-vocab-size-divisible-by 128
    --mock-data
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --rerun-mode disabled
    --async-strategy nvrx
    --save "${run_dir}/ckpt"
    --tensorboard-dir "${run_dir}/tb"
  )

  if (( CP_SIZE > 1 )); then
    cmd+=(--context-parallel-size "${CP_SIZE}" --cp-comm-type p2p)
  fi

  if (( USE_SP == 1 )); then
    cmd+=(--sequence-parallel)
  fi

  if (( USE_DIST_OPT == 1 && NPROC_PER_NODE > 1 )); then
    cmd+=(--use-distributed-optimizer)
  fi

  print_command "${cmd[@]}"
  if (( DRY_RUN == 1 )); then
    return 0
  fi

  (
    cd "${REPO_ROOT}"
    "${cmd[@]}"
  )
}

run_toy_moe_ep_demo() {
  validate_moe_parallelism

  local run_dir="${REPO_ROOT}/${OUT_DIR}"
  mkdir -p "${run_dir}/ckpt" "${run_dir}/tb"

  local -a cmd=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    pretrain_gpt.py
    --use-mcore-models
    --transformer-impl transformer_engine
    --use-cpu-initialization
    --bf16
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --seq-length "${SEQ_LENGTH}"
    --lr "${LR}"
    --min-lr "${MIN_LR}"
    --lr-decay-style cosine
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --weight-decay "${WEIGHT_DECAY}"
    --clip-grad "${CLIP_GRAD}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --num-layers 12
    --hidden-size 1536
    --ffn-hidden-size 4096
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 4
    --kv-channels 96
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --qk-layernorm
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --max-position-embeddings 32768
    --swiglu
    --disable-bias-linear
    --no-bias-swiglu-fusion
    --attention-softmax-in-fp32
    --num-experts 8
    --moe-layer-freq 1
    --moe-router-topk 2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-8B
    --vocab-size 151936
    --make-vocab-size-divisible-by 128
    --mock-data
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --rerun-mode disabled
    --async-strategy nvrx
    --save "${run_dir}/ckpt"
    --tensorboard-dir "${run_dir}/tb"
  )

  if (( CP_SIZE > 1 )); then
    cmd+=(--context-parallel-size "${CP_SIZE}" --cp-comm-type p2p)
  fi

  if (( USE_SP == 1 )); then
    cmd+=(--sequence-parallel)
  fi

  if (( USE_DIST_OPT == 1 && NPROC_PER_NODE > 1 )); then
    cmd+=(--use-distributed-optimizer)
  fi

  print_command "${cmd[@]}"
  if (( DRY_RUN == 1 )); then
    return 0
  fi

  (
    cd "${REPO_ROOT}"
    "${cmd[@]}"
  )
}
