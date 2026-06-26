#!/bin/bash

set -euo pipefail

# DISTILL-QRED20
#
# Purpose:
#   Evaluate query reduction at LIBERO control_freq=20 for the frozen-baseline
#   LR-NODE adapter protocol.
#
# Rows:
#   baseline full K=1
#   adapter-composed full K=1
#   adapter-composed skip K=2,3,4

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lrnode_distill_qred20}"
export EXPERIMENT_TAG="${EXPERIMENT_TAG:-distill_qred20_$(date +%Y%m%d_%H%M%S)}"
export RUN_BASELINE="${RUN_BASELINE:-1}"
export RUN_OURS_FULL="${RUN_OURS_FULL:-1}"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR:-2 3 4}"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
export MASTER_PORT="${MASTER_PORT:-12640}"
export NODE_NUM="${NODE_NUM:-4}"

export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
