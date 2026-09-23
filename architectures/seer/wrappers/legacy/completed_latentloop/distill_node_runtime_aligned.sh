#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"

: "${BASELINE_CKPT:?Set BASELINE_CKPT to the frozen Seer teacher checkpoint}"
: "${LRNODE_INIT_ADAPTER_CKPT:?Set LRNODE_INIT_ADAPTER_CKPT to the existing V0 adapter}"

[[ -f "${BASELINE_CKPT}" ]] || {
    echo "[ERROR] missing Seer teacher: ${BASELINE_CKPT}" >&2
    exit 1
}
[[ -f "${LRNODE_INIT_ADAPTER_CKPT}" ]] || {
    echo "[ERROR] missing initial LatentLoop adapter: ${LRNODE_INIT_ADAPTER_CKPT}" >&2
    exit 1
}

# This is an exploratory recovery post-training protocol. It preserves the V0
# architecture and existing evaluation path, while matching the recurrent K=4
# rollout and temporal ensemble during training.
export METHOD_TAG="${METHOD_TAG:-latentloop_runtime_aligned_k4_v1}"
export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT:-0}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
export LRNODE_TEACHER_TARGET_MODE=shifted_context
export LRNODE_CONTEXT_SELECTED_STEP=-1
export LRNODE_MULTISTEP_TRAIN=0
export LRNODE_RUNTIME_ALIGNED_TRAIN=1
export LRNODE_RUNTIME_HORIZON=3
export LRNODE_RUNTIME_AGE3_WEIGHT="${LRNODE_RUNTIME_AGE3_WEIGHT:-2.0}"
export LRNODE_RUNTIME_ENSEMBLE_TEMP="${LRNODE_RUNTIME_ENSEMBLE_TEMP:-0.01}"
export LRNODE_FROZEN_TEACHER_EVAL_MODE=1

# Retain the original latent/action objectives, reduce the update-suppression
# prior, and add losses for the failure mode observed in Spatial traces.
export LRNODE_LATENT_WEIGHT="${LRNODE_LATENT_WEIGHT:-0.05}"
export LRNODE_ACTION_DISTILL_WEIGHT="${LRNODE_ACTION_DISTILL_WEIGHT:-0.1}"
export LRNODE_SMOOTH_WEIGHT="${LRNODE_SMOOTH_WEIGHT:-0.0001}"
export LRNODE_GRIPPER_DISTILL_WEIGHT="${LRNODE_GRIPPER_DISTILL_WEIGHT:-0.1}"
export LRNODE_OVERLAP_WEIGHT="${LRNODE_OVERLAP_WEIGHT:-0.05}"
export LRNODE_ENSEMBLE_WEIGHT="${LRNODE_ENSEMBLE_WEIGHT:-0.1}"
export LRNODE_GRIPPER_SWITCH_WEIGHT="${LRNODE_GRIPPER_SWITCH_WEIGHT:-0.05}"

echo "[RUNTIME-ALIGNED] teacher=${BASELINE_CKPT}"
echo "[RUNTIME-ALIGNED] init_adapter=${LRNODE_INIT_ADAPTER_CKPT}"
echo "[RUNTIME-ALIGNED] horizon=3 ages=1,2,3 action_tokens=3"
echo "[RUNTIME-ALIGNED] this wrapper does not change the evaluator or K=1 path"

exec bash "${SCRIPT_DIR}/distill_node.sh"
