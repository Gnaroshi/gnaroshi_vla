#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export SIMVLA_UPSTREAM_ROOT=${SIMVLA_UPSTREAM_ROOT:-"${ROOT}/architectures/simvla/upstream"}
export FASTV_UPSTREAM_ROOT=${FASTV_UPSTREAM_ROOT:-"${ROOT}/architectures/fastv/upstream"}
export LIBERO_ROOT=${LIBERO_ROOT:-"${SIMVLA_UPSTREAM_ROOT}/evaluation/libero/LIBERO"}
export PYTHONPATH="${ROOT}:${SIMVLA_UPSTREAM_ROOT}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONTRACT_ONLY=0
for argument in "$@"; do
  if [[ ${argument} == --contract-only ]]; then
    CONTRACT_ONLY=1
  fi
done

if [[ ${CONTRACT_ONLY} != 1 && ${SIMVLA_FASTV_EVAL_RUN:-0} != 1 ]]; then
  echo "Refusing LIBERO evaluation. Set SIMVLA_FASTV_EVAL_RUN=1 after reviewing arguments." >&2
  exit 2
fi

exec "${PYTHON:-python}" -m architectures.simvla.adapters.fastv.eval "$@"
