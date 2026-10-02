#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -m tools.simvla.trend_followup_campaign "$@"
rc=$?
printf '\nSimVLA trend followup exit code: %s\n' "$rc"
printf 'Real status: shared results/simvla/trend_condition/long_followup_seed01_v1/pipeline_status.json\n'
exit 0
