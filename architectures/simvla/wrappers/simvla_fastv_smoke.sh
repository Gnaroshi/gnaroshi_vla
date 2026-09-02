#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export SIMVLA_UPSTREAM_ROOT=${SIMVLA_UPSTREAM_ROOT:-"${ROOT}/architectures/simvla/upstream"}
export FASTV_UPSTREAM_ROOT=${FASTV_UPSTREAM_ROOT:-"${ROOT}/architectures/fastv/upstream"}
export PYTHONPATH="${ROOT}:${SIMVLA_UPSTREAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ${SIMVLA_FASTV_SMOKE_RUN:-0} != 1 ]]; then
  echo "Set SIMVLA_FASTV_SMOKE_RUN=1 after reviewing the bounded GPU smoke." >&2
  exit 2
fi

exec "${PYTHON:-python}" -m architectures.simvla.adapters.fastv.smoke "$@"
