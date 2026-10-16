#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
STORE=/home/mingyujung/private/gnaroshi_vla_storage
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUT="$STORE/results/simvla/geometry_target_compiled/matched_k8_nfe1_v1"
mkdir -p "$OUT/logs"
/usr/bin/python3 "$STORE/tools/nvidia_user_runtime.py" "$STORE/envs/simvla/libero_mujoco237/bin/python" -u -m tools.simvla.geometry_comparison_rb2 "${@:-all}" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
echo "GEOMETRY_COMPARISON_EXIT=$rc (pane retained)"
exit 0
