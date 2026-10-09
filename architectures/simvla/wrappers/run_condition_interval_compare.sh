#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
if [[ $(hostname) != jbr-TRX50 ]]; then
  printf 'This evaluation runs on rb2.\n'
  exit 0
fi
PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
OUT=/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/condition_output_split/bridge_interval_nfe1_compiled_seed01_v1
mkdir -p "$OUT/logs" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -u -m tools.simvla.condition_interval_compare "$@" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
printf 'CONDITION_INTERVAL_COMPARE_EXIT=%s status=%s\n' "$rc" "$OUT/logs/launcher.status"
exit 0
