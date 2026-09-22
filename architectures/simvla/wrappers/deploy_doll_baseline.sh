#!/usr/bin/env bash
# Run from a monitor-connected terminal. No hardware is used by --check.
set -euo pipefail

# Edit deployment settings here. CLI arguments override these defaults.
max_steps="${SIMVLA_DOLL_MAX_STEPS:-5000}"
control_hz="${SIMVLA_DOLL_CONTROL_HZ:-60}"
camera_fps="${SIMVLA_DOLL_CAMERA_FPS:-60}"
num_rollouts="${SIMVLA_DOLL_NUM_ROLLOUTS:-15}"
warmup_steps="${SIMVLA_DOLL_WARMUP_STEPS:-3}"

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${root}"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${root}"
export SIMVLA_REAL_RUNTIME_ROOT="${SIMVLA_REAL_RUNTIME_ROOT:-${root}/runtime}"
export SIMVLA_REAL_PYTHON="${SIMVLA_REAL_PYTHON:-${SIMVLA_REAL_RUNTIME_ROOT}/envs/simvla_real/bin/python}"
export SIMVLA_REAL_LOG_ROOT="${SIMVLA_REAL_LOG_ROOT:-${SIMVLA_REAL_RUNTIME_ROOT}/results/simvla/real_deploy}"
export PATH="$(dirname -- "${SIMVLA_REAL_PYTHON}"):${PATH}"
"${SIMVLA_REAL_PYTHON}" -m tools.simvla.launch_doll_baseline \
    --site-profile seer_doll --max-steps "${max_steps}" \
    --control-hz "${control_hz}" --camera-fps "${camera_fps}" \
    --num-rollouts "${num_rollouts}" --warmup-steps "${warmup_steps}" "$@"
