#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export SIMVLA_UPSTREAM_ROOT=${SIMVLA_UPSTREAM_ROOT:-"${ROOT}/architectures/simvla/upstream"}
export LATENT_BRIDGE_UPSTREAM_ROOT=${LATENT_BRIDGE_UPSTREAM_ROOT:-"${ROOT}/architectures/latent_bridge/upstream"}
export PYTHONPATH="${ROOT}:${SIMVLA_UPSTREAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ${SIMVLA_LATENT_BRIDGE_RUN:-0} != 1 ]]; then
  echo "Refusing bridge training. Set SIMVLA_LATENT_BRIDGE_RUN=1 after reviewing arguments." >&2
  exit 2
fi

exec "${PYTHON:-python}" -m architectures.simvla.adapters.latent_bridge.train "$@"
