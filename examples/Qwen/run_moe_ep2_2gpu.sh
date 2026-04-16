#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
EP_SIZE=${EP_SIZE:-2}
USE_SP=${USE_SP:-0}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
OUT_DIR=${OUT_DIR:-runs/qwen_parallel/toy_moe_ep2_2gpu}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_qwen3_8b.sh"

run_toy_moe_ep_demo
