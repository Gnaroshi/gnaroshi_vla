#!/bin/bash

set -euo pipefail

# DISTILL-QRED20-K123-REMAINING-CKPT-SWEEP
#
# Purpose:
#   Complete the interrupted checkpoint sweep without re-running K=4.
#   K=4 for ckpt34-38 was already evaluated by:
#     eval_lrnode_distill_qred20_k4_remaining_ckpt_sweep.sh
#
# Default target:
#   ckpts: 34 35 36 37 38
#   rows per ckpt:
#     K=1 full-forward parity row
#     K=2 skip
#     K=3 skip
#
# This gives the missing rows needed to combine with the existing K=4-only
# results and reconstruct a full K=1/2/3/4 checkpoint sweep.

export CKPT_IDS_STR="${CKPT_IDS_STR:-34 35 36 37 38}"
export RUN_BASELINE="${RUN_BASELINE:-0}"
export RUN_OURS_FULL="${RUN_OURS_FULL:-1}"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR:-2 3}"
export SAVE_VIDEO="${SAVE_VIDEO:-0}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-0}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-0}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-0}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-0}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lrnode_distill_qred20_k123_ckpt_sweep}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-k123_remaining_$(date +%Y%m%d_%H%M%S)}"
export MASTER_PORT="${MASTER_PORT:-12940}"

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

echo "[K123 REMAINING CKPT SWEEP] CONDA_PREFIX=${CONDA_PREFIX:-none}"
echo "[K123 REMAINING CKPT SWEEP] PYTHON=$(command -v python)"
echo "[K123 REMAINING CKPT SWEEP] CKPT_IDS_STR=${CKPT_IDS_STR}"
echo "[K123 REMAINING CKPT SWEEP] RUN_OURS_FULL=${RUN_OURS_FULL}"
echo "[K123 REMAINING CKPT SWEEP] LRNODE_QUERY_INTERVALS_STR=${LRNODE_QUERY_INTERVALS_STR}"
echo "[K123 REMAINING CKPT SWEEP] SAVE_VIDEO=${SAVE_VIDEO}, LRNODE_EVAL_STEP_LOG=${LRNODE_EVAL_STEP_LOG}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20_ckpt_sweep.sh
