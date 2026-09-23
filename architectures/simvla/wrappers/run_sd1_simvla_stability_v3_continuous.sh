#!/usr/bin/env bash

set -u -o pipefail

WORKTREE="${SIMVLA_STABILITY_V3_WORKTREE:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment}"
PYTHON="${SIMVLA_STABILITY_V3_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}"
RESULT_ROOT="${SIMVLA_STABILITY_V3_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/stability_alignment/recurrence_v3_bounded_pilot_v3}"
LOG_ROOT="${RESULT_ROOT}/logs"
STATUS="${RESULT_ROOT}/continuation_launcher.status"

mkdir -p "${LOG_ROOT}"
if ! cd "${WORKTREE}"; then
  printf '%s\n' "FAILED worktree_not_found=${WORKTREE}" > "${STATUS}"
  printf '%s\n' "STABILITY_V3_CONTINUATION_FAILED: worktree not found"
  printf '%s\n' "The launcher returns 0 so this tmux pane remains available."
  exit 0
fi

if [[ "${SIMVLA_STABILITY_V3_CONTINUATION_RUN:-0}" != "1" ]]; then
  printf '%s\n' "Set SIMVLA_STABILITY_V3_CONTINUATION_RUN=1 to approve exact continuation."
  printf '%s\n' "NOT_APPROVED" > "${STATUS}"
  exit 0
fi

export SIMVLA_STABILITY_V3_GPU_POOL="${SIMVLA_STABILITY_V3_GPU_POOL:-2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
LOG="${LOG_ROOT}/sd1_stability_v3_continuation.log"
MAX_ATTEMPTS="${SIMVLA_STABILITY_V3_MAX_ATTEMPTS:-3}"

printf 'V3 exact continuation GPU pool: %s\n' "${SIMVLA_STABILITY_V3_GPU_POOL}"
printf '%s\n' "Policy: record soft gates; continue 500 -> 2K -> 5K -> 10K."
printf '%s\n' "Hard stop: runtime/nonfinite, numerical safety, freeze, hash, or source-lock failure."

rc=1
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  printf 'CONTINUATION_ATTEMPT=%d/%d\n' "${attempt}" "${MAX_ATTEMPTS}" | tee -a "${LOG}"
  "${PYTHON}" -m architectures.simvla.adapters.latentloop.stability_alignment.v3_continuation \
    2>&1 | tee -a "${LOG}"
  rc=${PIPESTATUS[0]}
  if [[ ${rc} -eq 0 ]]; then
    break
  fi
  printf 'Attempt %d failed rc=%d; exact-resume retry in 60s.\n' "${attempt}" "${rc}" | tee -a "${LOG}"
  sleep 60
done

if [[ ${rc} -eq 0 ]]; then
  verdict="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "${RESULT_ROOT}/continuation_summary.json")"
  printf '%s\n' "${verdict}" > "${STATUS}"
  printf 'STABILITY_V3_CONTINUATION_COMPLETE verdict=%s\n' "${verdict}"
else
  printf 'FAILED rc=%d\n' "${rc}" > "${STATUS}"
  printf 'STABILITY_V3_CONTINUATION_FAILED rc=%d\n' "${rc}"
  printf 'Inspect %s and %s\n' "${LOG}" "${RESULT_ROOT}/continuation_failure.json"
fi

printf '%s\n' "The launcher returns 0 so this tmux pane remains available."
exit 0
