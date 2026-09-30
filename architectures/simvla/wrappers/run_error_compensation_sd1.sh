#!/usr/bin/env bash
# Run as a child shell, never close the user's tmux pane on failure.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 1
PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
"$PY" -m tools.simvla.error_compensation_campaign "$@"
rc=$?
printf '\nSimVLA campaign exit code: %s\n' "$rc"
if [[ "$rc" != 0 ]]; then
  printf 'Run failed; see shared results/simvla/error_compensation/long_seed01_v1/logs. Pane stays open.\n'
fi
exit 0
