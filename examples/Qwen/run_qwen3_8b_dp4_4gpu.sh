#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
USE_SP=${USE_SP:-0}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
OUT_DIR=${OUT_DIR:-runs/qwen_parallel/qwen3_8b_dp4_4gpu}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_qwen3_8b.sh"

run_qwen3_8b_dense
