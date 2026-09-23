#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SIMVLA_STABILITY_V3_RUN=1
export SIMVLA_STABILITY_V3_GPU_POOL="${SIMVLA_STABILITY_V3_GPU_POOL:-2,3}"
export SIMVLA_STABILITY_V3_ENABLE_R150=0
export SIMVLA_STABILITY_V3_RESULT_ROOT="${SIMVLA_STABILITY_V3_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/stability_alignment/recurrence_v3_bounded_pilot_v3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RB2_V3_BUNDLE_DESTINATION="${RB2_V3_BUNDLE_DESTINATION:-rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_stability_v3_selected/}"

printf '%s\n' "SimVLA stability V3 R50 bounded pipeline"
printf 'GPU pool: %s\n' "${SIMVLA_STABILITY_V3_GPU_POOL}"
printf 'Result root: %s\n' "${SIMVLA_STABILITY_V3_RESULT_ROOT}"
printf '%s\n' "R150 control: disabled"

bash "${SCRIPT_DIR}/run_sd1_simvla_stability_v3.sh"
exit 0
