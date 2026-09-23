#!/usr/bin/env bash
set -euo pipefail

ROOT="${GNAROSHI_VLA_ROOT:-/home/mingyujung/private/gnaroshi_vla}"
cd "${ROOT}"
OUT="${SIMVLA_DCLD_EVAL_OUTPUT:-${ROOT}/results/simvla/dcld/eval/qred20/dry_run_$(date +%Y%m%d_%H%M%S)}"

MODE_FLAG="--dry-run"
for arg in "$@"; do
  if [[ "${arg}" == "--run" ]]; then
    MODE_FLAG=""
  fi
  if [[ "${arg}" == "--dry-run" ]]; then
    MODE_FLAG=""
  fi
done
if [[ "${SIMVLA_DCLD_EVAL_RUN:-0}" == "1" ]]; then
  MODE_FLAG=""
fi

exec python architectures/simvla/wrappers/simvla_dcld_eval.py \
  --protocol qred20 \
  --output "${OUT}" \
  ${MODE_FLAG} \
  "$@"
