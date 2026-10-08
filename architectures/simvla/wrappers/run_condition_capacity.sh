#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
if [[ $(hostname) != jbrserver1 ]]; then
  printf 'This bounded diagnostic uses sd1 GPUs 4,5,6,7.\n'
  exit 0
fi
PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_capacity/matched_input_fit_v1
mkdir -p "$OUT/logs" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -u -m tools.simvla.condition_capacity "$@" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
printf 'CONDITION_CAPACITY_EXIT=%s status=%s\n' "$rc" "$OUT/logs/launcher.status"
exit 0
