#!/usr/bin/env bash

set +e
set -o pipefail

WORKTREE=/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment
UPSTREAM=/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream
PYTHON=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
RESULT_ROOT=/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/stability_alignment/s50_10k_diagnostic_rb2

mkdir -p "${RESULT_ROOT}/logs"
export SIMVLA_STABILITY_WORKTREE="${WORKTREE}"
export PYTHONPATH="${WORKTREE}:${UPSTREAM}:${PYTHONPATH:-}"

"${PYTHON}" -u -m architectures.simvla.adapters.latentloop.stability_alignment.s50_diagnostic_pipeline \
  2>&1 | tee "${RESULT_ROOT}/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "${rc}" > "${RESULT_ROOT}/launcher.status"

if [[ ${rc} -eq 0 ]]; then
  echo "S50_10K_RB2_DIAGNOSTIC_COMPLETE"
  echo "Summary: ${RESULT_ROOT}/diagnostic_summary.json"
else
  echo "S50_10K_RB2_DIAGNOSTIC_FAILED rc=${rc}"
  echo "Inspect: ${RESULT_ROOT}/failure.json"
fi

echo "Launcher status: ${RESULT_ROOT}/launcher.status"
echo "The wrapper returns 0 so an attached tmux shell remains available."
exit 0
