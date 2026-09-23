#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_EXACT_Q2_OFFLINE_RUN:-0}" != "1" ]]; then
  echo "Refusing exact-q2 offline validation: set SIMVLA_EXACT_Q2_OFFLINE_RUN=1." >&2
  exit 2
fi

exec python -m architectures.simvla.adapters.exact_q2.offline_evaluator "$@"
