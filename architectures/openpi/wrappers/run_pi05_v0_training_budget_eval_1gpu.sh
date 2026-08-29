#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OPENPI_LL_ROOT=${OPENPI_LL_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
if [[ -z ${OPENPI_LL_SHARED:-} ]]; then
  if [[ -d /home/mingyujung/private/gnaroshi_vla_storage ]]; then
    export OPENPI_LL_SHARED=/home/mingyujung/private/gnaroshi_vla_storage
  fi
fi
if [[ -z ${OPENPI_LL_CHECKPOINT:-} && -d ${OPENPI_LL_SHARED:-}/checkpoints/finetuned/pi05_libero_lora_pytorch/pi05_base_lora_r16_eb16_1x5090_seed42_30k/30000 ]]; then
  export OPENPI_LL_CHECKPOINT=${OPENPI_LL_SHARED}/checkpoints/finetuned/pi05_libero_lora_pytorch/pi05_base_lora_r16_eb16_1x5090_seed42_30k/30000
fi
if [[ -z ${OPENPI_LL_NORM:-} && -f ${OPENPI_LL_SHARED:-}/assets/pi05_libero_lora_pytorch/physical-intelligence/libero/norm_stats.json ]]; then
  export OPENPI_LL_NORM=${OPENPI_LL_SHARED}/assets/pi05_libero_lora_pytorch/physical-intelligence/libero/norm_stats.json
fi
source "${SCRIPT_DIR}/latentloop_common.sh"

RUN_NAME=${OPENPI_PI05_V0_BUDGET_RUN_NAME:-pi05_v0_mode_b_rb2_seed42_30k}
CONTRACT=${OPENPI_LL_RESULTS}/contracts/${RUN_NAME}
TRAIN_RUN=${OPENPI_LL_RESULTS}/cacheless_streaming/train/${RUN_NAME}
OUTPUT=${OPENPI_LL_RESULTS}/eval/${RUN_NAME}_best24k_seed7
ADAPTER_CHECKPOINT=${TRAIN_RUN}/checkpoints/best.pt
EXPECTED_TRAINING_STEPS=30000
GPU=0
PORT=8160
MIN_FREE_MIB=18000
POLL_SECONDS=60
WAIT_FOR_GPU=1
SAVE_VIDEO=0

while (($#)); do
  case "$1" in
    --contract) CONTRACT=$2; shift 2 ;;
    --training-run) TRAIN_RUN=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --adapter-checkpoint) ADAPTER_CHECKPOINT=$2; shift 2 ;;
    --expected-training-steps) EXPECTED_TRAINING_STEPS=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB=$2; shift 2 ;;
    --poll-seconds) POLL_SECONDS=$2; shift 2 ;;
    --no-wait) WAIT_FOR_GPU=0; shift ;;
    --save-video) SAVE_VIDEO=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ${OPENPI_PI05_V0_BUDGET_EVAL_RUN:-0} != 1 ]]; then
  echo "Set OPENPI_PI05_V0_BUDGET_EVAL_RUN=1 to enable evaluation." >&2
  exit 2
fi
for value in EXPECTED_TRAINING_STEPS GPU PORT MIN_FREE_MIB POLL_SECONDS; do
  [[ ${!value} =~ ^[0-9]+$ ]] || { echo "${value} must be numeric" >&2; exit 2; }
done

CONTRACT=$(realpath -- "${CONTRACT}")
TRAIN_RUN=$(realpath -- "${TRAIN_RUN}")
ADAPTER_CHECKPOINT=$(realpath -- "${ADAPTER_CHECKPOINT}")
OUTPUT=$(realpath -m -- "${OUTPUT}")
SOURCE_LOCK=${CONTRACT}/source_lock_v2.json
FINAL_MANIFEST=${CONTRACT}/protocol/pi05_final_evaluation_manifest_v2.json
K1_TENSOR_REPORT=${CONTRACT}/k1_tensor/pi05_k1_equivalence.json
K1_EPISODE_GATE=${CONTRACT}/k1_episode/combined_gate/pi05_k1_equivalence.json
FREEZE_GATE=${CONTRACT}/freeze/freeze_gate.json
TRAINING_RUN_SUMMARY=${TRAIN_RUN}/run_summary.json

for path in \
  "${SOURCE_LOCK}" "${FINAL_MANIFEST}" "${K1_TENSOR_REPORT}" \
  "${K1_EPISODE_GATE}" "${FREEZE_GATE}" "${TRAINING_RUN_SUMMARY}" \
  "${ADAPTER_CHECKPOINT}"; do
  [[ -f ${path} ]] || { echo "Missing required input: ${path}" >&2; exit 1; }
done

is_complete() {
  local summary=$1
  [[ -f ${summary} ]] || return 1
  "${OPENPI_LL_MAIN_PY}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("complete") and d.get("rollouts")==500 else 1)' \
    "${summary}"
}

if [[ -f ${OUTPUT}/combined_summary.json ]]; then
  "${OPENPI_LL_MAIN_PY}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("complete") is True' \
    "${OUTPUT}/combined_summary.json"
  echo "EVALUATION_ALREADY_COMPLETE output=${OUTPUT}/combined_summary.json"
  exit 0
fi

while :; do
  free_mib=$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [[ ${free_mib} =~ ^[0-9]+$ ]] || { echo "Unable to read GPU ${GPU}" >&2; exit 1; }
  if ((free_mib >= MIN_FREE_MIB)); then
    echo "GPU_READY gpu=${GPU} free_mib=${free_mib} required_mib=${MIN_FREE_MIB}"
    break
  fi
  if ((WAIT_FOR_GPU == 0)); then
    echo "GPU ${GPU} has ${free_mib} MiB free; require ${MIN_FREE_MIB} MiB." >&2
    exit 1
  fi
  echo "GPU_WAIT gpu=${GPU} free_mib=${free_mib} required_mib=${MIN_FREE_MIB} poll_seconds=${POLL_SECONDS}"
  sleep "${POLL_SECONDS}"
done

SUITES=(libero_spatial libero_object libero_goal libero_10)
mkdir -p "${OUTPUT}"
server_pid=
cleanup() {
  if [[ -n ${server_pid} ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for suite_index in "${!SUITES[@]}"; do
  suite=${SUITES[${suite_index}]}
  suite_output=${OUTPUT}/${suite}
  suite_port=$((PORT + suite_index))
  if is_complete "${suite_output}/v0/summary.json" && is_complete "${suite_output}/original/summary.json"; then
    echo "SUITE_ALREADY_COMPLETE suite=${suite}"
    continue
  fi
  for row in v0 original; do
    if [[ -e ${suite_output}/${row} ]] && ! is_complete "${suite_output}/${row}/summary.json"; then
      echo "Refusing incomplete existing row: ${suite_output}/${row}" >&2
      exit 1
    fi
  done

  preflight_tmp=$(mktemp /tmp/pi05_v0_budget_eval.XXXXXX.json)
  "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/verify_pi05_v0_training_budget_eval.py" \
    --source-lock "${SOURCE_LOCK}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
    --norm-stats "${OPENPI_LL_NORM}" --final-manifest "${FINAL_MANIFEST}" \
    --suite "${suite}" --k1-tensor-report "${K1_TENSOR_REPORT}" \
    --k1-episode-gate "${K1_EPISODE_GATE}" --freeze-gate "${FREEZE_GATE}" \
    --training-run-summary "${TRAINING_RUN_SUMMARY}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --expected-training-steps "${EXPECTED_TRAINING_STEPS}" \
    --output "${preflight_tmp}" >/dev/null

  mkdir -p "${suite_output}/protocol/libero_config"
  cp "${preflight_tmp}" "${suite_output}/protocol/evaluation_preflight.json"
  rm -f "${preflight_tmp}"
  printf '%s\n' \
    "assets: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/assets" \
    "bddl_files: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/bddl_files" \
    "benchmark_root: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero" \
    "datasets: ${HF_LEROBOT_HOME}" \
    "init_states: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/init_files" \
    > "${suite_output}/protocol/libero_config/config.yaml"

  cd "${suite_output}"
  OPENPI_PI05_V0_EVAL_RUN=1 \
  OPENPI_PI05_V0_EXPECTED_TRAINING_STEPS=${EXPECTED_TRAINING_STEPS} \
  CUDA_VISIBLE_DEVICES=${GPU} "${OPENPI_LL_MAIN_PY}" \
    "${OPENPI_LL_ROOT}/tools/openpi/serve_pi05_v0_training_budget.py" \
    --run --checkpoint "${OPENPI_LL_CHECKPOINT}" --norm-stats "${OPENPI_LL_NORM}" \
    --adapter-checkpoint "${ADAPTER_CHECKPOINT}" \
    --training-run-summary "${TRAINING_RUN_SUMMARY}" \
    --source-lock "${SOURCE_LOCK}" --final-manifest "${FINAL_MANIFEST}" \
    --k1-tensor-report "${K1_TENSOR_REPORT}" --k1-episode-gate "${K1_EPISODE_GATE}" \
    --freeze-gate "${FREEZE_GATE}" --suite "${suite}" \
    --k-q 4 --flow-steps 10 --noise-seed-base 7 \
    --port "${suite_port}" --device cuda > "${suite_output}/server.log" 2>&1 &
  server_pid=$!
  openpi_ll_wait_server "${suite_port}"

  for row in v0 original; do
    if is_complete "${suite_output}/${row}/summary.json"; then
      echo "ROW_ALREADY_COMPLETE suite=${suite} row=${row}"
      continue
    fi
    client_args=(
      --output "${suite_output}/${row}" --suite "${suite}"
      --host 127.0.0.1 --port "${suite_port}"
      --seed 7 --noise-seed-base 7 --num-trials 50 --max-tasks 10
      --wait-steps 10 --replan-steps 5 --resize-size 224
      --policy-path "${row}" --final-evaluation-manifest "${FINAL_MANIFEST}"
    )
    if ((SAVE_VIDEO)); then client_args+=(--save-video --video-fps 30); fi
    CUDA_VISIBLE_DEVICES=${GPU} \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    NUMBA_CACHE_DIR=${OPENPI_LL_SHARED}/cache/openpi/numba/v0_budget_${suite} \
    MPLCONFIGDIR=${OPENPI_LL_SHARED}/cache/openpi/matplotlib/v0_budget_${suite} \
    LIBERO_CONFIG_PATH=${suite_output}/protocol/libero_config \
    PYTHONPATH=${OPENPI_LL_ROOT}:${OPENPI_LL_UPSTREAM}/packages/openpi-client/src:${OPENPI_LL_UPSTREAM}/third_party/libero \
      "${OPENPI_LL_CLIENT_PY}" "${OPENPI_LL_ROOT}/tools/openpi/evaluate_pi05_latentloop_client.py" \
      "${client_args[@]}" 2>&1 | tee "${suite_output}/${row}_client.log"
    is_complete "${suite_output}/${row}/summary.json"
    successes=$("${OPENPI_LL_MAIN_PY}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["successes"])' \
      "${suite_output}/${row}/summary.json")
    echo "ROW_COMPLETE suite=${suite} row=${row} successes=${successes}/500"
  done

  cleanup
  server_pid=
  echo "SUITE_COMPLETE suite=${suite} paired_episodes=1000"
done

trap - EXIT INT TERM
cd "${OPENPI_LL_ROOT}"
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/aggregate_pi05_v0_streaming_eval.py" \
  --root "${OUTPUT}" | tee "${OUTPUT}/aggregate.log"
echo "PAIRED_FOUR_SUITE_EVALUATION_COMPLETE output=${OUTPUT}/combined_summary.json"
