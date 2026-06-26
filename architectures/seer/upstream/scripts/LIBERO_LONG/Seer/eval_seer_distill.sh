#!/bin/bash

set -euo pipefail

# Evaluate Seer-only teacher-distillation control checkpoints at K=1.
# This intentionally uses eval.sh because LR-NODE is disabled for this control.

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
distill_seer_env="${SEER_DISTILL_ENV:-${protocol_root}/train/_latest/distill_seer.env}"

if [[ ! -f "${distill_seer_env}" ]]; then
    echo "[ERROR] Seer distill env not found: ${distill_seer_env}" >&2
    echo "[ERROR] Run scripts/LIBERO_LONG/Seer/distill_seer.sh first, or set SEER_DISTILL_ENV." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${distill_seer_env}"

export BASELINE_RUN_NAME="${LRNODE_RUN_NAME}"
export BASELINE_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
export DATASET="${DATASET:-${LRNODE_DATASET}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-12443}"

experiment_tag="${EXPERIMENT_TAG:-seer_distill_eval_$(date +%Y%m%d_%H%M%S)}"
export EVAL_RESULT_ROOT="${EVAL_RESULT_ROOT:-${protocol_root}/eval/seer_distill_control_${LRNODE_RUN_NAME}_${experiment_tag}}"
export CKPT_IDS="${CKPT_IDS:-31 32 33 34 35 36 37 38 39}"

echo "[EVAL SEER DISTILL] env=${distill_seer_env}"
echo "[EVAL SEER DISTILL] run=${BASELINE_RUN_NAME}"
echo "[EVAL SEER DISTILL] ckpt_root=${BASELINE_CKPT_ROOT}"
echo "[EVAL SEER DISTILL] ckpts=${CKPT_IDS}"
echo "[EVAL SEER DISTILL] result=${EVAL_RESULT_ROOT}"

bash scripts/LIBERO_LONG/Seer/eval.sh
