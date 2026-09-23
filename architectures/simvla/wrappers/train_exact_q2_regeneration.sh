#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_EXACT_Q2_TRAIN_RUN:-0}" != "1" ]]; then
  echo "Refusing exact-q2 calibration/training: set SIMVLA_EXACT_Q2_TRAIN_RUN=1." >&2
  exit 2
fi

exec python -m architectures.simvla.adapters.exact_q2.trainer "$@"
