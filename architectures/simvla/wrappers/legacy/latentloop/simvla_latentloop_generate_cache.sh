#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_CACHE_RUN:-0}" != "1" ]]; then
  echo "Refusing cache operation: set SIMVLA_LATENTLOOP_CACHE_RUN=1 explicitly." >&2
  exit 2
fi

exec python -m architectures.simvla.adapters.latentloop.query_cache_generator "$@"
