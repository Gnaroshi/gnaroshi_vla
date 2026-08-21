#!/bin/bash

set -euo pipefail

# DISTILL-QRED20-K4-REMAINING-CKPT-SWEEP
#
# Purpose:
#   Fast screening for remaining LR-NODE distill adapter checkpoints using
#   QRED20 K=4 only. This follows the interrupted/full ckpt sweep after
#   ckpt31-33 K=4 were already evaluated.
#
# Default target:
#   ckpts: 34 35 36 37 38
#   rows per ckpt: K=4 skip only
#
# Default speed settings:
#   no video, no per-step log. Re-enable with SAVE_VIDEO=1 or
#   LRNODE_EVAL_STEP_LOG=1 when official artifacts are needed.

export CKPT_IDS_STR="${CKPT_IDS_STR:-34 35 36 37 38}"
export RUN_BASELINE="${RUN_BASELINE:-0}"
export RUN_OURS_FULL="${RUN_OURS_FULL:-0}"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR:-4}"
export SAVE_VIDEO="${SAVE_VIDEO:-0}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-0}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-0}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-0}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lrnode_distill_qred20_k4_ckpt_sweep}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-k4_remaining_$(date +%Y%m%d_%H%M%S)}"
export MASTER_PORT="${MASTER_PORT:-12840}"

conda_env_name="${CONDA_ENV_NAME:-seer_libero}"
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    if [[ -z "${CONDA_PREFIX:-}" || "$(basename "${CONDA_PREFIX}")" != "${conda_env_name}" ]]; then
        if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
            # Fresh tmux/non-interactive shells do not inherit the training env.
            source "${HOME}/miniconda3/etc/profile.d/conda.sh"
            conda activate "${conda_env_name}"
        fi
    fi
fi

echo "[K4 REMAINING CKPT SWEEP] CONDA_PREFIX=${CONDA_PREFIX:-none}"
echo "[K4 REMAINING CKPT SWEEP] PYTHON=$(command -v python)"
echo "[K4 REMAINING CKPT SWEEP] CKPT_IDS_STR=${CKPT_IDS_STR}"
echo "[K4 REMAINING CKPT SWEEP] LRNODE_QUERY_INTERVALS_STR=${LRNODE_QUERY_INTERVALS_STR}"
echo "[K4 REMAINING CKPT SWEEP] SAVE_VIDEO=${SAVE_VIDEO}, LRNODE_EVAL_STEP_LOG=${LRNODE_EVAL_STEP_LOG}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20_ckpt_sweep.sh
