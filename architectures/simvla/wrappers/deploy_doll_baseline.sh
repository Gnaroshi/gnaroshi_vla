#!/usr/bin/env bash
# Run from a monitor-connected terminal. No hardware is used by --check.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${root}"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="${root}"
export SIMVLA_REAL_PYTHON="${SIMVLA_REAL_PYTHON:-${HOME}/gnaroshi_vla_runtime/envs/simvla_real/bin/python}"
export SIMVLA_REAL_LOG_ROOT="${SIMVLA_REAL_LOG_ROOT:-${HOME}/gnaroshi_vla_runtime/results/simvla/real_deploy}"
export PATH="$(dirname -- "${SIMVLA_REAL_PYTHON}"):${PATH}"
"${SIMVLA_REAL_PYTHON}" -m tools.simvla.launch_doll_baseline "$@"
