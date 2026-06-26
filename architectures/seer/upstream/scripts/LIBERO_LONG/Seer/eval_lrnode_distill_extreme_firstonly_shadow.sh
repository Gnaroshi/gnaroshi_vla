#!/bin/bash

set -euo pipefail

# DISTILL-EXTREME-FIRSTONLY-SHADOW
#
# Purpose:
#   Diagnostic-only version of first_only. The executed action still comes from
#   LR-NODE after the first full-Seer query, but each skipped step also runs a
#   shadow full-Seer forward to log latent/action drift.
#
# Do not use this run for latency claims because shadow full-forward adds extra
# compute that is not part of the deployed LR-NODE policy.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lrnode_distill_extreme_firstonly_shadow}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-distill_extreme_firstonly_shadow_$(date +%Y%m%d_%H%M%S)}"
export RUN_BASELINE="${RUN_BASELINE:-0}"
export RUN_OURS_FULL="${RUN_OURS_FULL:-0}"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR:-1}"
export LRNODE_EVAL_REFRESH_POLICY="first_only"
export LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE="1"
export LRNODE_EVAL_SHADOW_FULL_FORWARD="1"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
export MASTER_PORT="${MASTER_PORT:-12690}"
export NODE_NUM="${NODE_NUM:-4}"

export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
