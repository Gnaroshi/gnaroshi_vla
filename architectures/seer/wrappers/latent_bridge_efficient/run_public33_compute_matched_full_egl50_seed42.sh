#!/usr/bin/env bash

set -euo pipefail
wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BRIDGE_PRESET=full
export LATENT_BRIDGE_RENDERER=egl
export EVAL_SEEDS=42
export EVAL_EPISODES_PER_TASK=50
export EVAL_NUM_TASKS=10
export REFRESH_PERIODS="2 3 4"
export NODE_NUM=4
export R0_OPTIMIZER_STEPS="${R0_OPTIMIZER_STEPS:-50200}"
export R1_OPTIMIZER_STEPS="${R1_OPTIMIZER_STEPS:-44000}"
export VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-32}"
export VALIDATION_EVERY_DATA_EPOCHS="${VALIDATION_EVERY_DATA_EPOCHS:-4}"
export CHECKPOINT_EVERY_DATA_EPOCHS="${CHECKPOINT_EVERY_DATA_EPOCHS:-4}"
export LOG_EVERY_UPDATES="${LOG_EVERY_UPDATES:-100}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-18200}"
export LATENT_BRIDGE_RESULT_ROOT="${LATENT_BRIDGE_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/latent_bridge/public33_compute_matched_full_egl50_seed42}"
export LATENT_BRIDGE_EFFICIENT_SYNC_STAGE="${LATENT_BRIDGE_EFFICIENT_SYNC_STAGE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/latent_bridge/public33_full_egl50_seed42/continuity_and_sync}"

source "${wrapper_dir}/common.sh"
latent_bridge_require_runtime

cat <<EOF
[SEER LATENT BRIDGE COMPUTE-MATCHED RUN]
repo=${LATENT_BRIDGE_REPO_ROOT}
result_root=${LATENT_BRIDGE_RESULT_ROOT}
sync_source=${LATENT_BRIDGE_EFFICIENT_SYNC_STAGE} (read-only reuse)
base_checkpoint=${LATENT_BRIDGE_PUBLIC33}
renderer=${LATENT_BRIDGE_RENDERER}
bridge_preset=${BRIDGE_PRESET}
R0_optimizer_steps=${R0_OPTIMIZER_STEPS}
R1_optimizer_steps=${R1_OPTIMIZER_STEPS}
effective_training_batch=64
evaluation_seed=${EVAL_SEEDS}
evaluation_episodes=$((EVAL_EPISODES_PER_TASK * EVAL_NUM_TASKS))
refresh_periods=${REFRESH_PERIODS}
EOF

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[PREFLIGHT_ONLY] static configuration and required paths passed"
    exit 0
fi

exec bash "${wrapper_dir}/run_pipeline.sh"
