#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_TRAIN_RUN:-0}" != "1" ]]; then
  echo "Refusing training/calibration: set SIMVLA_LATENTLOOP_TRAIN_RUN=1 explicitly." >&2
  exit 2
fi

exec python -m architectures.simvla.adapters.latentloop.trainer "$@"
