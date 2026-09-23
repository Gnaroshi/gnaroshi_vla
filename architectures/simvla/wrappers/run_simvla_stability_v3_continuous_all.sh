#!/usr/bin/env bash

set -u -o pipefail

WORKTREE="${SIMVLA_STABILITY_V3_WORKTREE:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment}"
RB2_WORKTREE="${SIMVLA_STABILITY_V3_RB2_WORKTREE:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_v3}"
CONTROL_ROOT="${SIMVLA_STABILITY_V3_CONTROL_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/stability_alignment/recurrence_v3_bounded_pilot_v3/control}"
RB2_SESSION="${SIMVLA_STABILITY_V3_RB2_SESSION:-simvla_v3_trajectory}"
LOCAL_WRAPPER="${WORKTREE}/architectures/simvla/wrappers/run_sd1_simvla_stability_v3_continuous.sh"
R150_WRAPPER="${WORKTREE}/architectures/simvla/wrappers/run_sd1_simvla_stability_v3_r150_continuous.sh"
REMOTE_WRAPPER="${RB2_WORKTREE}/architectures/simvla/wrappers/run_rb2_simvla_stability_v3_trajectory.sh"
SOURCE_LOCK="${SIMVLA_STABILITY_V3_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/stability_alignment/recurrence_v3_bounded_pilot_v3}/training/r50/source_lock.json"
SOURCE_LIST="${CONTROL_ROOT}/source_locked_files.txt"

mkdir -p "${CONTROL_ROOT}"
cd "${WORKTREE}" || {
  printf '%s\n' "FAILED sd1_worktree_not_found=${WORKTREE}"
  exit 0
}

if [[ ! -f "${SOURCE_LOCK}" ]]; then
  printf '%s\n' "FAILED source_lock_not_found=${SOURCE_LOCK}"
  exit 0
fi
python -c 'import json,sys; print("\n".join(sorted(json.load(open(sys.argv[1]))["source_files"])))' \
  "${SOURCE_LOCK}" > "${SOURCE_LIST}"

start_rb2_queue() {
  local attempt
  for attempt in $(seq 1 10); do
    printf '[rb2-control] attempt=%d sync and launch\n' "${attempt}"
    if rsync -aR --files-from="${SOURCE_LIST}" ./ "rb2:${RB2_WORKTREE}/" && \
      rsync -aR \
      ./architectures/simvla/adapters/latentloop/stability_alignment/v3_trajectory_rb2.py \
      ./architectures/simvla/wrappers/run_rb2_simvla_stability_v3_trajectory.sh \
      "rb2:${RB2_WORKTREE}/" && \
      ssh rb2 "RB2_SESSION='${RB2_SESSION}' REMOTE_WRAPPER='${REMOTE_WRAPPER}' bash -s" <<'REMOTE'
set -u
tmux set-option -g remain-on-exit on
if tmux has-session -t "${RB2_SESSION}" 2>/dev/null; then
  if tmux list-panes -t "${RB2_SESSION}" -F '#{pane_dead}' | grep -qx '0'; then
    printf '%s\n' "RB2_QUEUE_ALREADY_RUNNING session=${RB2_SESSION}"
    exit 0
  fi
  tmux kill-session -t "${RB2_SESSION}"
fi
tmux new-session -d -s "${RB2_SESSION}" \
  "export SIMVLA_STABILITY_V3_TRAJECTORY_RUN=1; bash '${REMOTE_WRAPPER}'; exec bash"
printf '%s\n' "RB2_QUEUE_LAUNCHED session=${RB2_SESSION}"
REMOTE
    then
      printf '%s\n' "RB2_QUEUE_CONTROL_PASS" > "${CONTROL_ROOT}/rb2_queue_launch.status"
      return 0
    fi
    sleep 60
  done
  printf '%s\n' "RB2_QUEUE_CONTROL_FAILED_AFTER_RETRIES" > "${CONTROL_ROOT}/rb2_queue_launch.status"
  return 1
}

start_rb2_queue > "${CONTROL_ROOT}/rb2_queue_launch.log" 2>&1 &
rb2_control_pid=$!
printf 'rb2 queue controller pid=%d log=%s\n' "${rb2_control_pid}" "${CONTROL_ROOT}/rb2_queue_launch.log"

SIMVLA_STABILITY_V3_R150_RUN=1 \
SIMVLA_STABILITY_V3_GPU_POOL="${SIMVLA_STABILITY_V3_R150_GPU_POOL:-6,7}" \
SIMVLA_STABILITY_V3_RESULT_ROOT="${SIMVLA_STABILITY_V3_R150_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/stability_alignment/recurrence_v3_r150_continuous_v1}" \
WANDB_MODE="${WANDB_MODE:-online}" \
bash "${R150_WRAPPER}" &
r150_pid=$!
printf 'R150 control pid=%d GPU pool=%s\n' \
  "${r150_pid}" "${SIMVLA_STABILITY_V3_R150_GPU_POOL:-6,7}"

export SIMVLA_STABILITY_V3_CONTINUATION_RUN=1
export SIMVLA_STABILITY_V3_GPU_POOL="${SIMVLA_STABILITY_V3_GPU_POOL:-4,5}"
export RB2_V3_TRAJECTORY_DESTINATION="${RB2_V3_TRAJECTORY_DESTINATION:-rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_stability_v3_trajectory}"
export WANDB_MODE="${WANDB_MODE:-online}"

bash "${LOCAL_WRAPPER}"
local_rc=$?
wait "${r150_pid}"
r150_rc=$?
wait "${rb2_control_pid}"
rb2_rc=$?

printf 'R50_WRAPPER_RC=%d R150_WRAPPER_RC=%d RB2_CONTROL_RC=%d\n' \
  "${local_rc}" "${r150_rc}" "${rb2_rc}" \
  > "${CONTROL_ROOT}/coordinator.status"
printf 'Coordinator finished: R50 rc=%d, R150 rc=%d, rb2 controller rc=%d\n' \
  "${local_rc}" "${r150_rc}" "${rb2_rc}"
printf '%s\n' "The coordinator returns 0 so this tmux pane remains available."
exit 0
