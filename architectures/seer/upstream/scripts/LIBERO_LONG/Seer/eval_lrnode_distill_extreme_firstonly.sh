#!/bin/bash

set -euo pipefail

# DISTILL-EXTREME-FIRSTONLY
#
# Purpose:
#   Stress-test the frozen-baseline LR-NODE adapter with the most aggressive
#   query-reduction policy at normal LIBERO control_freq=20:
#     full Seer at the first policy step only, then LR-NODE for all later steps.
#
# Rows:
#   baseline full K=1
#   adapter-composed full K=1
#   adapter-composed first_only rollout

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lrnode_distill_extreme_firstonly}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-distill_extreme_firstonly_$(date +%Y%m%d_%H%M%S)}"
export RUN_BASELINE="${RUN_BASELINE:-1}"
export RUN_OURS_FULL="${RUN_OURS_FULL:-1}"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR:-1}"
export LRNODE_EVAL_REFRESH_POLICY="first_only"
export LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE="1"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
export MASTER_PORT="${MASTER_PORT:-12660}"
export NODE_NUM="${NODE_NUM:-4}"

export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
export LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
