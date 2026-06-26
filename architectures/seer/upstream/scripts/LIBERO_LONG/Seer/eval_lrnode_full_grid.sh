#!/bin/bash

set -euo pipefail

# Experiment: GRID
#
# Diagnostic grid combining 20 Hz query-reduction rows and high-Hz rows.
# Use this after QRED20/HZUP20Q when a broader K sweep is needed.

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grid}"
export HZ_K_PAIRS_STR="${HZ_K_PAIRS_STR:-20:1 20:2 20:3 20:4 20:5 20:6 20:8 40:1 40:2 40:4 60:1 60:2 60:3 60:6 80:1 80:2 80:4 80:8}"
export MASTER_PORT="${MASTER_PORT:-12580}"
export EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ:-20}"
export EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ:-1}"
export EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ:-1}"

bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
