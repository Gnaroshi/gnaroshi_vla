#!/usr/bin/env bash
set -euo pipefail

ROOT="${GNAROSHI_VLA_ROOT:-/home/mingyujung/private/gnaroshi_vla}"
cd "${ROOT}"

exec python tools/count_dcld_params.py "$@"
