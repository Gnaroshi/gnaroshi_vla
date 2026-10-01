#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export WANDB_MODE="${WANDB_MODE:-offline}"
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -m tools.simvla.action_aligned_campaign "$@"
rc=$?
printf '\nSimVLA action-aligned campaign exit code: %s\n' "$rc"
printf 'Real status: shared results/simvla/action_aligned_joint/long_kc4_seed01_v1/status.json\n'
exit 0
