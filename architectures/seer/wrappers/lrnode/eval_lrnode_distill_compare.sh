#!/bin/bash

set -euo pipefail

# Evaluation wrapper for frozen-baseline LR-NODE distill protocol.
# Training script:
#   distill_node.sh
#
# Runs:
#   1) original Seer baseline full-forward
#   2) LR-NODE distill full-forward K=1
#   3) LR-NODE distill skip-forward K sweep

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
latest_baseline="${protocol_root}/train/_latest/scratch.env"
latest_distill="${protocol_root}/train/_latest/distill_node.env"
latest_finetune_compat="${protocol_root}/train/_latest/finetune_node.env"

if [[ -z "${BASELINE_CKPT:-}" && -z "${BASELINE_RUN_NAME:-}" && -z "${BASELINE_CKPT_ROOT:-}" && -f "${latest_baseline}" ]]; then
    # shellcheck source=/dev/null
    source "${latest_baseline}"
    BASELINE_RUN_NAME="${LRNODE_RUN_NAME}"
    BASELINE_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
    echo "[EVAL INFO] loaded latest baseline run from ${latest_baseline}"
fi
BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-33}"
if [[ -z "${BASELINE_CKPT:-}" && ( -z "${BASELINE_RUN_NAME:-}" || -z "${BASELINE_CKPT_ROOT:-}" ) ]]; then
    echo "[ERROR] Baseline checkpoint is not configured." >&2
    echo "[ERROR] Run scratch.sh first or set BASELINE_CKPT=/path/to/baseline.pth." >&2
    exit 1
fi

METHOD_TAG="${METHOD_TAG:-lrnode_distill_from_scratch_baseline_ckpt${BASELINE_CKPT_ID}_lronly_v1_lw05_aw01_g4}"
if [[ -f "${latest_distill}" && -z "${OURS_RUN_NAME:-}" && -z "${OURS_CKPT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${latest_distill}"
    OURS_RUN_NAME="${LRNODE_RUN_NAME}"
    OURS_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
    echo "[EVAL INFO] loaded latest distill run from ${latest_distill}"
elif [[ -f "${latest_finetune_compat}" && -z "${OURS_RUN_NAME:-}" && -z "${OURS_CKPT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${latest_finetune_compat}"
    OURS_RUN_NAME="${LRNODE_RUN_NAME}"
    OURS_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
    echo "[EVAL INFO] loaded compatibility finetune pointer from ${latest_finetune_compat}"
fi
OURS_RUN_NAME="${OURS_RUN_NAME:-sd1_distill_node_${METHOD_TAG}}"
OURS_CKPT_ROOT="${OURS_CKPT_ROOT:-${protocol_root}/train/distill_node}"
OURS_CKPT_ID="${OURS_CKPT_ID:-39}"

export BASELINE_NAME="${BASELINE_NAME:-seer_scratch_baseline}"
export BASELINE_RUN_NAME
export BASELINE_CKPT_ID
export BASELINE_CKPT_ROOT
export BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_CKPT_ROOT}/${BASELINE_RUN_NAME}/${BASELINE_CKPT_ID}.pth}"

export METHOD_TAG
export OURS_NAME="${OURS_NAME:-${METHOD_TAG}}"
export OURS_RUN_NAME
export OURS_CKPT_ID
export OURS_CKPT_ROOT
export OURS_CKPT="${OURS_CKPT:-${OURS_CKPT_ROOT}/${OURS_RUN_NAME}/${OURS_CKPT_ID}.pth}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lrnode_distill_compare}"
export LRNODE_PROTOCOL_ROOT="${protocol_root}"
export LRNODE_TRAIN_PROTOCOL="adapter"
export LRNODE_FREEZE_SEER_FOR_ADAPTER="1"
export LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE="1"
export LRNODE_EVAL_BASE_CKPT="${LRNODE_EVAL_BASE_CKPT:-${BASELINE_CKPT}}"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR-2 3 4 5 6 8}"

# Keep efficiency logging on. Shadow full-forward is off by default because it
# changes measured policy latency by adding extra full forward calls.
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
export LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh
