#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export WANDB_MODE=offline
PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
"$PY" -u -m tools.simvla.error_compensation_kc_transfer "$@"
rc=$?
printf '\nSimVLA longer-condition comparison exit code: %s\n' "$rc"
printf 'Final status: shared results/simvla/error_compensation/long_kc34_transfer_seed01_v1/campaign_complete.json\n'
if [[ "$rc" != 0 ]]; then
  printf 'Execution failed. Check logs; this child shell leaves the tmux pane open.\n'
fi
exit 0
