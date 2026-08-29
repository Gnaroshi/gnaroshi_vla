#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

SUITE= OUTPUT= SOURCE_LOCK= FINAL_MANIFEST= K1_TENSOR_REPORT=
K1_EPISODE_GATE= FREEZE_GATE= TRAINING_RUN_SUMMARY= ADAPTER_CHECKPOINT=
GPU= PORT= SAVE_VIDEO=0

while (($#)); do
  case "$1" in
    --suite) SUITE=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --final-manifest) FINAL_MANIFEST=$2; shift 2 ;;
    --k1-tensor-report) K1_TENSOR_REPORT=$2; shift 2 ;;
    --k1-episode-gate) K1_EPISODE_GATE=$2; shift 2 ;;
    --freeze-gate) FREEZE_GATE=$2; shift 2 ;;
    --training-run-summary) TRAINING_RUN_SUMMARY=$2; shift 2 ;;
    --adapter-checkpoint) ADAPTER_CHECKPOINT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --save-video) SAVE_VIDEO=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${SUITE:?--suite is required}"
: "${OUTPUT:?--output is required}"
: "${SOURCE_LOCK:?--source-lock is required}"
: "${FINAL_MANIFEST:?--final-manifest is required}"
: "${K1_TENSOR_REPORT:?--k1-tensor-report is required}"
: "${K1_EPISODE_GATE:?--k1-episode-gate is required}"
: "${FREEZE_GATE:?--freeze-gate is required}"
: "${TRAINING_RUN_SUMMARY:?--training-run-summary is required}"
: "${ADAPTER_CHECKPOINT:?--adapter-checkpoint is required}"
: "${GPU:?--gpu is required}"
: "${PORT:?--port is required}"

if [[ "${OPENPI_PI05_V0_EVAL_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_PI05_V0_EVAL_RUN=1 to enable evaluation." >&2
  exit 2
fi
export PYTHONHASHSEED=7

OUTPUT=$(realpath -m -- "${OUTPUT}")
preserve_alias_abspath() {
  if [[ $1 == /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$(pwd -P)" "$1"
  fi
}
SOURCE_LOCK=$(preserve_alias_abspath "${SOURCE_LOCK}")
FINAL_MANIFEST=$(preserve_alias_abspath "${FINAL_MANIFEST}")
K1_TENSOR_REPORT=$(preserve_alias_abspath "${K1_TENSOR_REPORT}")
K1_EPISODE_GATE=$(preserve_alias_abspath "${K1_EPISODE_GATE}")
FREEZE_GATE=$(preserve_alias_abspath "${FREEZE_GATE}")
TRAINING_RUN_SUMMARY=$(preserve_alias_abspath "${TRAINING_RUN_SUMMARY}")
ADAPTER_CHECKPOINT=$(preserve_alias_abspath "${ADAPTER_CHECKPOINT}")

for path in \
  "${SOURCE_LOCK}" "${FINAL_MANIFEST}" "${K1_TENSOR_REPORT}" \
  "${K1_EPISODE_GATE}" "${FREEZE_GATE}" "${TRAINING_RUN_SUMMARY}" \
  "${ADAPTER_CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 1; }
done

is_complete() {
  local summary=$1
  [[ -f "${summary}" ]] || return 1
  "${OPENPI_LL_MAIN_PY}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("complete") and d.get("rollouts")==500 else 1)' \
    "${summary}"
}

for row in v0 original; do
  if [[ -e "${OUTPUT}/${row}" ]] && ! is_complete "${OUTPUT}/${row}/summary.json"; then
    echo "Refusing incomplete existing row: ${OUTPUT}/${row}" >&2
    exit 1
  fi
done
if is_complete "${OUTPUT}/v0/summary.json" && is_complete "${OUTPUT}/original/summary.json"; then
  echo "SUITE_ALREADY_COMPLETE suite=${SUITE} output=${OUTPUT}"
  exit 0
fi

preflight_tmp=$(mktemp /tmp/pi05_v0_eval_preflight.XXXXXX.json)
server_pid=
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -f "${preflight_tmp}"
}
trap cleanup EXIT INT TERM

"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/verify_pi05_v0_streaming_eval.py" \
  --source-lock "${SOURCE_LOCK}" \
  --checkpoint "${OPENPI_LL_CHECKPOINT}" \
  --norm-stats "${OPENPI_LL_NORM}" \
  --final-manifest "${FINAL_MANIFEST}" \
  --suite "${SUITE}" \
  --k1-tensor-report "${K1_TENSOR_REPORT}" \
  --k1-episode-gate "${K1_EPISODE_GATE}" \
  --freeze-gate "${FREEZE_GATE}" \
  --training-run-summary "${TRAINING_RUN_SUMMARY}" \
  --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
  --output "${preflight_tmp}" >/dev/null

mkdir -p "${OUTPUT}/protocol/libero_config"
cp "${preflight_tmp}" "${OUTPUT}/protocol/evaluation_preflight.json"
printf '%s\n' \
  "assets: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/assets" \
  "bddl_files: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/bddl_files" \
  "benchmark_root: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero" \
  "datasets: ${HF_LEROBOT_HOME}" \
  "init_states: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/init_files" \
  > "${OUTPUT}/protocol/libero_config/config.yaml"

cd "${OUTPUT}"
CUDA_VISIBLE_DEVICES=${GPU} "${OPENPI_LL_MAIN_PY}" \
  "${OPENPI_LL_ROOT}/tools/openpi/serve_pi05_v0_streaming.py" \
  --run \
  --checkpoint "${OPENPI_LL_CHECKPOINT}" \
  --norm-stats "${OPENPI_LL_NORM}" \
  --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
  --training-run-summary "${TRAINING_RUN_SUMMARY}" \
  --source-lock "${SOURCE_LOCK}" \
  --final-manifest "${FINAL_MANIFEST}" \
  --k1-tensor-report "${K1_TENSOR_REPORT}" \
  --k1-episode-gate "${K1_EPISODE_GATE}" \
  --freeze-gate "${FREEZE_GATE}" \
  --suite "${SUITE}" \
  --k-q 4 --flow-steps 10 --noise-seed-base 7 \
  --port "${PORT}" --device cuda \
  >> "${OUTPUT}/server.log" 2>&1 &
server_pid=$!
openpi_ll_wait_server "${PORT}"

run_row() {
  local row=$1
  if is_complete "${OUTPUT}/${row}/summary.json"; then
    echo "ROW_ALREADY_COMPLETE suite=${SUITE} row=${row}"
    return 0
  fi
  local client_args=(
    --output "${OUTPUT}/${row}"
    --suite "${SUITE}"
    --host 127.0.0.1 --port "${PORT}"
    --seed 7 --noise-seed-base 7
    --num-trials 50 --max-tasks 10
    --wait-steps 10 --replan-steps 5 --resize-size 224
    --policy-path "${row}"
    --final-evaluation-manifest "${FINAL_MANIFEST}"
  )
  if ((SAVE_VIDEO)); then
    client_args+=(--save-video --video-fps 30)
  fi
  CUDA_VISIBLE_DEVICES=${GPU} \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  NUMBA_CACHE_DIR=${OPENPI_LL_SHARED}/cache/openpi/numba/v0_streaming_${SUITE} \
  MPLCONFIGDIR=${OPENPI_LL_SHARED}/cache/openpi/matplotlib/v0_streaming_${SUITE} \
  LIBERO_CONFIG_PATH=${OUTPUT}/protocol/libero_config \
  PYTHONPATH=${OPENPI_LL_ROOT}:${OPENPI_LL_UPSTREAM}/packages/openpi-client/src:${OPENPI_LL_UPSTREAM}/third_party/libero \
    "${OPENPI_LL_CLIENT_PY}" "${OPENPI_LL_ROOT}/tools/openpi/evaluate_pi05_latentloop_client.py" \
    "${client_args[@]}" 2>&1 | tee "${OUTPUT}/${row}_client.log"
  is_complete "${OUTPUT}/${row}/summary.json"
  echo "ROW_COMPLETE suite=${SUITE} row=${row} episodes=500"
}

# Return the requested V0 result first; its exact paired original row follows.
run_row v0
run_row original

cleanup
server_pid=
trap - EXIT INT TERM
echo "SUITE_COMPLETE suite=${SUITE} paired_episodes=500 output=${OUTPUT}"
