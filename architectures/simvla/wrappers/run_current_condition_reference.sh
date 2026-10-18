#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
STORE=/home/mingyujung/private/gnaroshi_vla_storage
OUT="$STORE/results/simvla/current_condition/original_reference_v1"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$OUT/logs"
/usr/bin/python3 "$STORE/tools/nvidia_user_runtime.py" \
  "$STORE/envs/simvla/libero_mujoco237/bin/python" -u -m tools.simvla.current_condition_reference \
  "${@:-all}" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
echo "CURRENT_CONDITION_REFERENCE_EXIT=$rc (pane retained)"
exit 0
