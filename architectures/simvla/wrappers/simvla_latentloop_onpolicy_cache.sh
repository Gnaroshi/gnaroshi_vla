#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_ONPOLICY_RUN:-0}" != "1" ]]; then
  echo "Refusing on-policy cache generation: set SIMVLA_LATENTLOOP_ONPOLICY_RUN=1 explicitly." >&2
  exit 2
fi

exec python -m architectures.simvla.adapters.latentloop.on_policy_cache_generator "$@"
