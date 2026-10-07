#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false
cd "${ROOT}" || exit 1
"${PY}" -u -m tools.simvla.prefix_residual "${@:-all}"
status=$?
printf 'PREFIX_RESIDUAL_EXIT status=%s\n' "${status}"
exit "${status}"
