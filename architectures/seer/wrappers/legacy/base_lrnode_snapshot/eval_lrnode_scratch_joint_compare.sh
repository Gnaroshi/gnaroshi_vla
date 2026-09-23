#!/bin/bash

set -euo pipefail

# Evaluation wrapper for scratch joint LR-NODE protocol.
# Intended fair comparison:
#   scratch_node.sh
#   vs scratch_node_joint.sh
#
# If a scratch baseline control checkpoint is not available yet, override
# BASELINE_CKPT or BASELINE_CKPT_ROOT/BASELINE_RUN_NAME/BASELINE_CKPT_ID.

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
latest_scratch="${protocol_root}/train/_latest/scratch_node.env"
latest_joint="${protocol_root}/train/_latest/scratch_node_joint.env"

BASELINE_NAME="${BASELINE_NAME:-lrnode_scratch_teacher_student}"
if [[ -f "${latest_scratch}" && -z "${BASELINE_RUN_NAME:-}" && -z "${BASELINE_CKPT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${latest_scratch}"
    BASELINE_RUN_NAME="${LRNODE_RUN_NAME}"
    BASELINE_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
    echo "[EVAL INFO] loaded latest scratch baseline run from ${latest_scratch}"
fi
if [[ -z "${BASELINE_CKPT:-}" && ( -z "${BASELINE_RUN_NAME:-}" || -z "${BASELINE_CKPT_ROOT:-}" ) ]]; then
    echo "[ERROR] LR-NODE teacher-student checkpoint is not configured." >&2
    echo "[ERROR] Run scratch_node.sh first or set BASELINE_CKPT=/path/to/scratch_node.pth." >&2
    exit 1
fi
BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-39}"

METHOD_TAG="${METHOD_TAG:-lrnode_scratch_coupled_v1_lw05_aw01_g4}"
if [[ -f "${latest_joint}" && -z "${OURS_RUN_NAME:-}" && -z "${OURS_CKPT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${latest_joint}"
    OURS_RUN_NAME="${LRNODE_RUN_NAME}"
    OURS_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
    echo "[EVAL INFO] loaded latest scratch joint run from ${latest_joint}"
fi
if [[ -z "${OURS_CKPT:-}" && ( -z "${OURS_RUN_NAME:-}" || -z "${OURS_CKPT_ROOT:-}" ) ]]; then
    echo "[ERROR] LR-NODE coupled-joint checkpoint is not configured." >&2
    echo "[ERROR] Run scratch_node_joint.sh first or set OURS_CKPT=/path/to/scratch_node_joint.pth." >&2
    exit 1
fi
OURS_CKPT_ID="${OURS_CKPT_ID:-39}"

export BASELINE_NAME
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

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lrnode_scratch_joint_compare}"
export LRNODE_PROTOCOL_ROOT="${protocol_root}"
export LRNODE_TRAIN_PROTOCOL="joint"
export LRNODE_FREEZE_SEER_FOR_ADAPTER="0"
export LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE="0"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR:-2 3 4 5 6 8}"

export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
export LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh
