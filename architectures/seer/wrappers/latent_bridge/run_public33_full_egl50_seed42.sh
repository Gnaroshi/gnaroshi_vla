#!/usr/bin/env bash

set -euo pipefail
wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This is the paper-comparison run: one trained bridge, one execution seed,
# 50 initial states per LIBERO-Long task, and the same EGL protocol used by
# the current Seer/LatentLoop result table.
export BRIDGE_PRESET=full
export LATENT_BRIDGE_RENDERER=egl
export EVAL_SEEDS="42"
export EVAL_EPISODES_PER_TASK=50
export EVAL_NUM_TASKS=10
export REFRESH_PERIODS="2 3 4"
export NODE_NUM=4
export TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
export TRAIN_CHECKPOINT_EVERY_EPOCHS="${TRAIN_CHECKPOINT_EVERY_EPOCHS:-1}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-18100}"
export LATENT_BRIDGE_RETRY_PARTIAL=1
export LATENT_BRIDGE_AUTO_RETRY_ONCE=1
export LATENT_BRIDGE_RESULT_ROOT="${LATENT_BRIDGE_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/latent_bridge/public33_full_egl50_seed42}"

source "${wrapper_dir}/common.sh"
latent_bridge_require_runtime

cat <<EOF
[SEER LATENT BRIDGE RUN]
repo=${LATENT_BRIDGE_REPO_ROOT}
result_root=${LATENT_BRIDGE_RESULT_ROOT}
base_checkpoint=${LATENT_BRIDGE_PUBLIC33}
renderer=${LATENT_BRIDGE_RENDERER}
bridge_preset=${BRIDGE_PRESET}
training_seed=42
evaluation_seed=${EVAL_SEEDS}
evaluation_episodes=$((EVAL_EPISODES_PER_TASK * EVAL_NUM_TASKS))
refresh_periods=${REFRESH_PERIODS}
effective_training_batch=$((2 * NODE_NUM * 8))
EOF

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[PREFLIGHT_ONLY] configuration and required paths passed"
    exit 0
fi

exec bash "${wrapper_dir}/run_pipeline.sh"
