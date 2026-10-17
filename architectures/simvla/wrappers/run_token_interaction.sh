#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ $(hostname) == jbr-TRX50 ]]; then
  STORE=/home/mingyujung/private/gnaroshi_vla_storage
  PY=(/usr/bin/python3 "$STORE/tools/nvidia_user_runtime.py" "$STORE/envs/simvla/libero_mujoco237/bin/python")
  MODULE=tools.simvla.token_interaction_rb2
elif [[ $(hostname) == jbrserver1 ]]; then
  STORE=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla
  PY=(/home/mingyujung/miniconda3/envs/simvla_libero/bin/python)
  MODULE=tools.simvla.token_interaction_campaign
else
  echo 'Unsupported host; no work launched.'
  exit 0
fi
OUT="$STORE/results/simvla/token_interaction/matched_nfe1_v1"
mkdir -p "$OUT/logs"
"${PY[@]}" -u -m "$MODULE" "${@:-all}" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
echo "TOKEN_INTERACTION_EXIT=$rc (pane retained)"
exit 0
