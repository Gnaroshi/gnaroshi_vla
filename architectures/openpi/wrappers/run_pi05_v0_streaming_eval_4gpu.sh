#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

CONTRACT=${OPENPI_LL_RESULTS}/contracts/pi05_v0_streaming_reference
TRAIN_RUN=${OPENPI_LL_RESULTS}/cacheless_streaming/train/v0_streaming_seed42_r1
OUTPUT=${OPENPI_LL_RESULTS}/eval/v0_streaming_seed42_best_r1
GPUS=(4 5 6 7)
SUITES=(libero_spatial libero_object libero_goal libero_10)
PORTS=(8160 8161 8162 8163)
SAVE_VIDEO=1
PREFLIGHT_ONLY=0

while (($#)); do
  case "$1" in
    --output) OUTPUT=$2; shift 2 ;;
    --no-video) SAVE_VIDEO=0; shift ;;
    --preflight) PREFLIGHT_ONLY=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "${OPENPI_PI05_V0_EVAL_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_PI05_V0_EVAL_RUN=1 to enable paired four-suite evaluation." >&2
  exit 2
fi

SOURCE_LOCK=${CONTRACT}/source_lock_v2.json
FINAL_MANIFEST=${CONTRACT}/protocol/pi05_final_evaluation_manifest_v2.json
K1_TENSOR_REPORT=${CONTRACT}/k1_tensor/pi05_k1_equivalence.json
K1_EPISODE_GATE=${CONTRACT}/k1_episode/combined_gate/pi05_k1_equivalence.json
FREEZE_GATE=${CONTRACT}/freeze/freeze_gate.json
TRAINING_RUN_SUMMARY=${TRAIN_RUN}/run_summary.json
ADAPTER_CHECKPOINT=${TRAIN_RUN}/checkpoints/best.pt

for path in \
  "${SOURCE_LOCK}" "${FINAL_MANIFEST}" "${K1_TENSOR_REPORT}" \
  "${K1_EPISODE_GATE}" "${FREEZE_GATE}" "${TRAINING_RUN_SUMMARY}" \
  "${ADAPTER_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing required evaluation input: ${path}" >&2; exit 1; }
done

if ((PREFLIGHT_ONLY)); then
  "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/verify_pi05_v0_streaming_eval.py" \
    --source-lock "${SOURCE_LOCK}" \
    --checkpoint "${OPENPI_LL_CHECKPOINT}" \
    --norm-stats "${OPENPI_LL_NORM}" \
    --final-manifest "${FINAL_MANIFEST}" \
    --suite libero_10 \
    --k1-tensor-report "${K1_TENSOR_REPORT}" \
    --k1-episode-gate "${K1_EPISODE_GATE}" \
    --freeze-gate "${FREEZE_GATE}" \
    --training-run-summary "${TRAINING_RUN_SUMMARY}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" >/dev/null
  echo "V0_STREAMING_EVALUATION_PREFLIGHT_PASS checkpoint=best.pt step=7000 suites=4 episodes_per_row=2000"
  exit 0
fi

if [[ "${OPENPI_PI05_V0_EVAL_SKIP_GPU_IDLE_CHECK:-0}" != "1" ]]; then
  command -v nvidia-smi >/dev/null
  declare -A gpu_used=()
  while IFS=, read -r index used; do
    index=${index//[[:space:]]/}
    used=${used//[[:space:]]/}
    gpu_used["${index}"]=${used}
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  for gpu in "${GPUS[@]}"; do
    used=${gpu_used[${gpu}]:-}
    if [[ ! "${used}" =~ ^[0-9]+$ ]]; then
      echo "Unable to read GPU ${gpu} memory state." >&2
      exit 1
    fi
    if ((used > 1024)); then
      echo "GPU ${gpu} is not idle: ${used} MiB already used." >&2
      exit 1
    fi
  done
fi

mkdir -p "${OUTPUT}"
pids=()
monitor_pid=
cleanup() {
  local pid
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
  fi
  for pid in "${pids[@]:-}"; do
    kill -- "-${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for index in "${!SUITES[@]}"; do
  suite=${SUITES[${index}]}
  gpu=${GPUS[${index}]}
  port=${PORTS[${index}]}
  args=(
    --suite "${suite}"
    --output "${OUTPUT}/${suite}"
    --source-lock "${SOURCE_LOCK}"
    --final-manifest "${FINAL_MANIFEST}"
    --k1-tensor-report "${K1_TENSOR_REPORT}"
    --k1-episode-gate "${K1_EPISODE_GATE}"
    --freeze-gate "${FREEZE_GATE}"
    --training-run-summary "${TRAINING_RUN_SUMMARY}"
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}"
    --gpu "${gpu}"
    --port "${port}"
  )
  if ((SAVE_VIDEO)); then
    args+=(--save-video)
  fi
  setsid bash "${SCRIPT_DIR}/eval_pi05_v0_streaming_suite.sh" "${args[@]}" \
    > "${OUTPUT}/${suite}_launcher.log" 2>&1 &
  pid=$!
  pids+=("${pid}")
  echo "LAUNCHED suite=${suite} gpu=${gpu} port=${port} pid=${pid}"
done

progress_counts() {
  local log=$1
  local completed successes
  if [[ -f "${log}" ]]; then
    completed=$(grep -c '"success":' "${log}" 2>/dev/null || true)
    successes=$(grep -c '"success": true' "${log}" 2>/dev/null || true)
    printf '%s %s\n' "${successes:-0}" "${completed:-0}"
  else
    printf '0 0\n'
  fi
}
monitor_progress() {
  local active index suite
  local v0_success v0_count original_success original_count
  while :; do
    active=0
    for index in "${!pids[@]}"; do
      kill -0 "${pids[${index}]}" 2>/dev/null && active=1
      suite=${SUITES[${index}]}
      read -r v0_success v0_count \
        < <(progress_counts "${OUTPUT}/${suite}/v0_client.log")
      read -r original_success original_count \
        < <(progress_counts "${OUTPUT}/${suite}/original_client.log")
      printf '%s\n' \
        "PROGRESS suite=${suite} gpu=${GPUS[${index}]}" \
        "  v0_success=${v0_success}/${v0_count} v0_progress=${v0_count}/500" \
        "  original_success=${original_success}/${original_count} original_progress=${original_count}/500"
    done
    ((active)) || return 0
    sleep 60
  done
}
monitor_progress &
monitor_pid=$!

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[${index}]}"; then
    echo "FINISHED suite=${SUITES[${index}]} gpu=${GPUS[${index}]}"
  else
    status=$?
    echo "FAILED suite=${SUITES[${index}]} gpu=${GPUS[${index}]} status=${status}" >&2
    failed=1
  fi
done
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
monitor_pid=
trap - EXIT INT TERM
if ((failed)); then
  echo "At least one suite failed. Inspect ${OUTPUT}/*_launcher.log; completed rows are resumable." >&2
  exit 1
fi

"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/aggregate_pi05_v0_streaming_eval.py" \
  --root "${OUTPUT}" | tee "${OUTPUT}/aggregate.log"
echo "PAIRED_FOUR_SUITE_EVALUATION_COMPLETE output=${OUTPUT}/combined_summary.json"
