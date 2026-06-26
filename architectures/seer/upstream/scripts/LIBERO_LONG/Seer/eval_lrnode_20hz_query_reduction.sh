#!/bin/bash

set -euo pipefail

# Experiment: QRED20
#
# Question:
#   At the original LIBERO 20 Hz control rate, how much can LR-NODE replace
#   full Seer forwarding?
#
# Interpretation:
#   K=1 is full-query Seer for the same scratch_node checkpoint.
#   K>1 uses full Seer once every K env steps and LR-NODE for skipped steps.

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qred20}"
export HZ_K_PAIRS_STR="${HZ_K_PAIRS_STR:-20:1 20:2 20:3 20:4 20:5 20:6 20:8}"
export MASTER_PORT="${MASTER_PORT:-12520}"
export EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ:-20}"
export EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ:-1}"
export EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ:-1}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
