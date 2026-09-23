#!/usr/bin/env bash

set -u -o pipefail

WORKTREE="${SIMVLA_STABILITY_V3_WORKTREE:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment}"
PYTHON="${SIMVLA_STABILITY_V3_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}"
RESULT_ROOT="${SIMVLA_STABILITY_V3_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/stability_alignment/recurrence_v3_r150_continuous_v1}"
LOG_ROOT="${RESULT_ROOT}/logs"
STATUS="${RESULT_ROOT}/launcher.status"

mkdir -p "${LOG_ROOT}"
if ! cd "${WORKTREE}"; then
  printf '%s\n' "FAILED worktree_not_found=${WORKTREE}" > "${STATUS}"
  printf '%s\n' "R150_CONTINUOUS_FAILED: worktree not found"
  exit 0
fi
if [[ "${SIMVLA_STABILITY_V3_R150_RUN:-0}" != "1" ]]; then
  printf '%s\n' "Set SIMVLA_STABILITY_V3_R150_RUN=1 to approve R150 control."
  printf '%s\n' "NOT_APPROVED" > "${STATUS}"
  exit 0
fi

export SIMVLA_STABILITY_V3_GPU_POOL="${SIMVLA_STABILITY_V3_GPU_POOL:-6,7}"
export SIMVLA_STABILITY_V3_RESULT_ROOT="${RESULT_ROOT}"
export WANDB_MODE="${WANDB_MODE:-online}"
LOG="${LOG_ROOT}/sd1_stability_v3_r150_continuous.log"
MAX_ATTEMPTS="${SIMVLA_STABILITY_V3_MAX_ATTEMPTS:-3}"

rc=1
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  printf 'R150_ATTEMPT=%d/%d\n' "${attempt}" "${MAX_ATTEMPTS}" | tee -a "${LOG}"
  "${PYTHON}" -m architectures.simvla.adapters.latentloop.stability_alignment.v3_r150_continuous \
    2>&1 | tee -a "${LOG}"
  rc=${PIPESTATUS[0]}
  if [[ ${rc} -eq 0 ]]; then
    break
  fi
  printf 'Attempt %d failed rc=%d; checkpoint-resume retry in 60s.\n' "${attempt}" "${rc}" | tee -a "${LOG}"
  sleep 60
done
if [[ ${rc} -eq 0 ]]; then
  verdict="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "${RESULT_ROOT}/pipeline_summary.json")"
  printf '%s\n' "${verdict}" > "${STATUS}"
  printf 'R150_CONTINUOUS_COMPLETE verdict=%s\n' "${verdict}"
else
  printf 'FAILED rc=%d\n' "${rc}" > "${STATUS}"
  printf 'R150_CONTINUOUS_FAILED rc=%d\n' "${rc}"
  printf 'Inspect %s and %s\n' "${LOG}" "${RESULT_ROOT}/pipeline_failure.json"
fi
printf '%s\n' "The launcher returns 0 so this tmux pane remains available."
exit 0
