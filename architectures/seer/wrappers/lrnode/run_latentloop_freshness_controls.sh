#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
PYTHON=/home/mingyujung/miniconda3/envs/seer_libero/bin/python
if [[ ! -x "$PYTHON" ]]; then
    printf '[ERROR] missing Seer interpreter: %s\n' "$PYTHON" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
exec "$PYTHON" tools/seer/run_latentloop_freshness_controls.py "${1:-run}"
