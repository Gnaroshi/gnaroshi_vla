#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/action_coverage/fresh_nfe1_v1
mkdir -p "$OUT/logs"
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -u -m tools.simvla.action_coverage_campaign "${@:-all}" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
echo "ACTION_COVERAGE_EXIT=$rc (pane retained)"
exit 0
