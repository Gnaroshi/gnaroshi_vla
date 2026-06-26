#!/bin/bash

set -euo pipefail

# Experiment: HZUP20Q
#
# Question:
#   Can LR-NODE raise the actual LIBERO control_freq while keeping the expensive
#   full Seer query rate near the original 20 Hz budget?
#
# Key rows:
#   40:2 -> control_freq 40 Hz, full Seer query 20 Hz, LR-NODE 20 Hz
#   60:3 -> control_freq 60 Hz, full Seer query 20 Hz, LR-NODE 40 Hz
#   80:4 -> control_freq 80 Hz, full Seer query 20 Hz, LR-NODE 60 Hz
#
# K=1 rows are expensive full-query upper-bound references at the same Hz.

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-hzup20q}"
export HZ_K_PAIRS_STR="${HZ_K_PAIRS_STR:-20:1 40:1 40:2 60:1 60:3 80:1 80:4}"
export MASTER_PORT="${MASTER_PORT:-12540}"
export EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ:-20}"
export EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ:-1}"
export EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ:-1}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
