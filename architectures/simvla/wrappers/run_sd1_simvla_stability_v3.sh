#!/usr/bin/env bash

set -u -o pipefail

WORKTREE="${SIMVLA_STABILITY_V3_WORKTREE:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment}"
PYTHON="${SIMVLA_STABILITY_V3_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}"
RESULT_ROOT="${SIMVLA_STABILITY_V3_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/stability_alignment/recurrence_v3_bounded_pilot_v3}"
LOG_ROOT="${RESULT_ROOT}/logs"
STATUS="${RESULT_ROOT}/launcher.status"

mkdir -p "${LOG_ROOT}"
if ! cd "${WORKTREE}"; then
  printf '%s\n' "FAILED worktree_not_found=${WORKTREE}" > "${STATUS}"
  printf '%s\n' "STABILITY_V3_PIPELINE_FAILED: worktree not found: ${WORKTREE}"
  printf '%s\n' "The launcher returns 0 so this tmux pane remains available."
  exit 0
fi

if [[ "${SIMVLA_STABILITY_V3_RUN:-0}" != "1" ]]; then
  printf '%s\n' "Set SIMVLA_STABILITY_V3_RUN=1 to approve the bounded V3 pipeline."
  printf '%s\n' "NOT_APPROVED" > "${STATUS}"
  exit 0
fi

LOG="${LOG_ROOT}/sd1_stability_v3_pipeline.log"
"${PYTHON}" -m architectures.simvla.adapters.latentloop.stability_alignment.v3_pipeline \
  2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}

if [[ ${rc} -eq 0 ]]; then
  verdict="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "${RESULT_ROOT}/pipeline_summary.json")"
  printf '%s\n' "${verdict}" > "${STATUS}"
  printf '%s\n' "STABILITY_V3_PIPELINE_COMPLETE verdict=${verdict}"
else
  printf 'FAILED rc=%d\n' "${rc}" > "${STATUS}"
  printf '%s\n' "STABILITY_V3_PIPELINE_FAILED rc=${rc}"
  printf '%s\n' "Inspect ${LOG} and ${RESULT_ROOT}/pipeline_failure.json"
fi

printf '%s\n' "The launcher returns 0 so this tmux pane remains available."
exit 0
