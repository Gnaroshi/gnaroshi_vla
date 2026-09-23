#!/usr/bin/env bash

set -u -o pipefail

WORKTREE="${SIMVLA_STABILITY_V3_WORKTREE:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_v3}"
PYTHON="${SIMVLA_STABILITY_V3_PYTHON:-/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python}"
RESULT_ROOT="${SIMVLA_STABILITY_V3_TRAJECTORY_RESULT_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/stability_alignment/recurrence_v3_trajectory_rb2}"
LOG_ROOT="${RESULT_ROOT}/logs"
STATUS="${RESULT_ROOT}/launcher.status"

mkdir -p "${LOG_ROOT}"
if ! cd "${WORKTREE}"; then
  printf '%s\n' "FAILED worktree_not_found=${WORKTREE}" > "${STATUS}"
  printf '%s\n' "STABILITY_V3_TRAJECTORY_FAILED: worktree not found"
  printf '%s\n' "The launcher returns 0 so this tmux pane remains available."
  exit 0
fi

if [[ "${SIMVLA_STABILITY_V3_TRAJECTORY_RUN:-0}" != "1" ]]; then
  printf '%s\n' "Set SIMVLA_STABILITY_V3_TRAJECTORY_RUN=1 to approve rb2 evaluation."
  printf '%s\n' "NOT_APPROVED" > "${STATUS}"
  exit 0
fi

LOG="${LOG_ROOT}/rb2_stability_v3_trajectory.log"
MAX_ATTEMPTS="${SIMVLA_STABILITY_V3_MAX_ATTEMPTS:-3}"
printf '%s\n' "V3 trajectory queue: steps 500,2K,5K,10K; K_C=3,4 learned N_G=3; 500 episodes/row."

rc=1
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  printf 'TRAJECTORY_ATTEMPT=%d/%d\n' "${attempt}" "${MAX_ATTEMPTS}" | tee -a "${LOG}"
  "${PYTHON}" -m architectures.simvla.adapters.latentloop.stability_alignment.v3_trajectory_rb2 \
    2>&1 | tee -a "${LOG}"
  rc=${PIPESTATUS[0]}
  if [[ ${rc} -eq 0 ]]; then
    break
  fi
  printf 'Attempt %d failed rc=%d; episode-resume retry in 60s.\n' "${attempt}" "${rc}" | tee -a "${LOG}"
  sleep 60
done

if [[ ${rc} -eq 0 ]]; then
  printf '%s\n' "STABILITY_V3_TRAJECTORY_RB2_COMPLETE" > "${STATUS}"
  printf '%s\n' "STABILITY_V3_TRAJECTORY_RB2_COMPLETE"
else
  printf 'FAILED rc=%d\n' "${rc}" > "${STATUS}"
  printf 'STABILITY_V3_TRAJECTORY_RB2_FAILED rc=%d\n' "${rc}"
  printf 'Inspect %s and %s\n' "${LOG}" "${RESULT_ROOT}/pipeline_failure.json"
fi

printf '%s\n' "The launcher returns 0 so this tmux pane remains available."
exit 0
