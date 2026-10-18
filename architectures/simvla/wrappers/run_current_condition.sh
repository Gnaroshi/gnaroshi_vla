#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ $(hostname) == jbr-TRX50 ]]; then
  STORE=/home/mingyujung/private/gnaroshi_vla_storage
  OUT="$STORE/results/simvla/current_condition/compiled_confirmation_v1"
  PY=(/usr/bin/python3 "$STORE/tools/nvidia_user_runtime.py" "$STORE/envs/simvla/libero_mujoco237/bin/python")
else
  STORE=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla
  OUT="$STORE/results/simvla/current_condition/matched_targets_v1"
  PY=(/home/mingyujung/miniconda3/envs/simvla_libero/bin/python)
fi
mkdir -p "$OUT/logs"
"${PY[@]}" -u -m tools.simvla.current_condition_campaign "${@:-all}" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
echo "CURRENT_CONDITION_EXIT=$rc (pane retained)"
exit 0
