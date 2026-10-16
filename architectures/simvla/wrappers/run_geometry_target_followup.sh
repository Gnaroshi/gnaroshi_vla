#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
[[ $(hostname) == jbrserver1 ]] || { echo 'sd1 only'; exit 0; }
OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/geometry_target_followup/matched_k8_nfe1_v1
mkdir -p "$OUT/logs"
cd "$ROOT" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -u -m tools.simvla.geometry_target_campaign "${@:-all}" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
echo "GEOMETRY_TARGET_EXIT=$rc"
exit 0
