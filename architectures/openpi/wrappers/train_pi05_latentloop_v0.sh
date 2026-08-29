#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

if [[ "${OPENPI_LATENTLOOP_TRAIN_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_LATENTLOOP_TRAIN_RUN=1 to enable training." >&2
  exit 2
fi
exec "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/train_pi05_latentloop.py" \
  --run --variant v0 --checkpoint "${OPENPI_LL_CHECKPOINT}" "$@"
