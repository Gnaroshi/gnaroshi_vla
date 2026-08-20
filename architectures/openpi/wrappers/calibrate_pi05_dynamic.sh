#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

exec "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/calibrate_pi05_dynamic.py" "$@"
