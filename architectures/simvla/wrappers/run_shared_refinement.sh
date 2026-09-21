#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
cd "$ROOT"
exec "$PY" -u tools/simvla/shared_refinement_pipeline.py "${1:-wait-and-train}"
