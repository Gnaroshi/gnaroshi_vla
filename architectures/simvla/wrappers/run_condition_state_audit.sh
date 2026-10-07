#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_state_audit/frozen_observed_k8_seed01_v1
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT" USE_TF=0 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"
"$PY" -u -m tools.simvla.condition_state_audit "$@" 2>&1 | tee -a "$OUT/logs/launcher.log"
status=${PIPESTATUS[0]}
printf 'CONDITION_STATE_AUDIT_EXIT status=%s\n' "$status"
printf '%s\n' "$status" > "$OUT/logs/launcher.status"
exit "$status"
