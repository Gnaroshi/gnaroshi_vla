#!/bin/bash

set -euo pipefail

# DISTILL-LOADPARITY
#
# Purpose:
#   Verify that a frozen-baseline LR-NODE adapter checkpoint is evaluated with
#   the correct two-stage load:
#     1) full Seer baseline checkpoint
#     2) adapter-only LR-NODE checkpoint
#
# Expected result:
#   ours K=1 full-forward should match the baseline checkpoint success rate.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lrnode_distill_loadparity}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-distill_loadparity_$(date +%Y%m%d_%H%M%S)}"
export RUN_BASELINE="${RUN_BASELINE:-1}"
export RUN_OURS_FULL="${RUN_OURS_FULL:-1}"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR-}"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
export MASTER_PORT="${MASTER_PORT:-12630}"
export NODE_NUM="${NODE_NUM:-4}"

export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
