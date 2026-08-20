#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

MODE= SUITE= OUTPUT= SOURCE_LOCK= FINAL_MANIFEST=
K1_TENSOR_REPORT= FREEZE_GATE= CACHE_GATE= ADAPTER= OFFLINE_GATE=
DYNAMIC_THRESHOLD_LOCK=
GPU=${CUDA_VISIBLE_DEVICES:-4}
PORT=8150
SAVE_VIDEO=0

while (($#)); do
  case "$1" in
    --mode) MODE=$2; shift 2 ;;
    --suite) SUITE=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --final-evaluation-manifest) FINAL_MANIFEST=$2; shift 2 ;;
    --k1-tensor-report) K1_TENSOR_REPORT=$2; shift 2 ;;
    --freeze-gate) FREEZE_GATE=$2; shift 2 ;;
    --cache-gate) CACHE_GATE=$2; shift 2 ;;
    --adapter-checkpoint) ADAPTER=$2; shift 2 ;;
    --offline-gate) OFFLINE_GATE=$2; shift 2 ;;
    --dynamic-threshold-lock) DYNAMIC_THRESHOLD_LOCK=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --save-video) SAVE_VIDEO=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${MODE:?--mode is required}"
: "${SUITE:?--suite is required}"
: "${OUTPUT:?--output is required}"
: "${SOURCE_LOCK:?--source-lock is required}"
: "${K1_TENSOR_REPORT:?--k1-tensor-report is required}"
: "${FREEZE_GATE:?--freeze-gate is required}"
OUTPUT=$(realpath -m -- "${OUTPUT}")
SOURCE_LOCK=$(realpath -- "${SOURCE_LOCK}")
K1_TENSOR_REPORT=$(realpath -- "${K1_TENSOR_REPORT}")
FREEZE_GATE=$(realpath -- "${FREEZE_GATE}")
for path_name in FINAL_MANIFEST CACHE_GATE ADAPTER OFFLINE_GATE DYNAMIC_THRESHOLD_LOCK; do
  if [[ -n ${!path_name} ]]; then
    printf -v "${path_name}" '%s' "$(realpath -- "${!path_name}")"
  fi
done
if [[ "${OPENPI_LATENTLOOP_EVAL_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_LATENTLOOP_EVAL_RUN=1 to enable this user-run command." >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite or reuse output path: ${OUTPUT}" >&2
  exit 1
fi
for required_file in "${SOURCE_LOCK}" "${K1_TENSOR_REPORT}" "${FREEZE_GATE}"; do
  [[ -f "${required_file}" ]] || { echo "Missing required gate file: ${required_file}" >&2; exit 1; }
done

server_mode=${MODE}
policy_path=${MODE}
stage=
stage_artifacts=("${K1_TENSOR_REPORT}" "${FREEZE_GATE}")
case "${MODE}" in
  k1-smoke)
    [[ "${SUITE}" == "libero_10" ]] || { echo "K1 smoke suite must be libero_10" >&2; exit 2; }
    server_mode=k1
    policy_path=k1
    stage=stage1_episode_smoke
    ;;
  paired-full)
    : "${FINAL_MANIFEST:?paired-full requires --final-evaluation-manifest}"
    server_mode=original
    policy_path=original
    stage=paired_full_baseline
    ;;
  hold)
    : "${FINAL_MANIFEST:?hold requires --final-evaluation-manifest}"
    stage=paired_full_baseline
    ;;
  v0)
    : "${FINAL_MANIFEST:?v0 requires --final-evaluation-manifest}"
    : "${CACHE_GATE:?v0 requires --cache-gate}"
    : "${ADAPTER:?v0 requires --adapter-checkpoint}"
    : "${OFFLINE_GATE:?v0 requires --offline-gate}"
    stage=stage5_v0_paired_eval
    stage_artifacts=("${OFFLINE_GATE}")
    ;;
  v1)
    : "${FINAL_MANIFEST:?v1 requires --final-evaluation-manifest}"
    : "${CACHE_GATE:?v1 requires --cache-gate}"
    : "${ADAPTER:?v1 requires --adapter-checkpoint}"
    : "${OFFLINE_GATE:?v1 requires --offline-gate}"
    stage=stage8_v1_paired_eval
    stage_artifacts=("${OFFLINE_GATE}")
    ;;
  v2)
    : "${FINAL_MANIFEST:?v2 requires --final-evaluation-manifest}"
    : "${CACHE_GATE:?v2 requires --cache-gate}"
    : "${ADAPTER:?v2 requires --adapter-checkpoint}"
    : "${DYNAMIC_THRESHOLD_LOCK:?v2 requires --dynamic-threshold-lock}"
    stage=stage12_v2_paired_eval
    stage_artifacts=("${DYNAMIC_THRESHOLD_LOCK}")
    ;;
  latent_bridge|*)
    echo "Mode ${MODE} is disabled or unknown. Latent Bridge remains style-only." >&2
    exit 2
    ;;
esac

gate_args=(--stage "${stage}" --source-lock "${SOURCE_LOCK}" --output-candidate "${OUTPUT}")
server_gate_args=(--stage "${stage}" --source-lock "${SOURCE_LOCK}")
for artifact in "${stage_artifacts[@]}"; do
  gate_args+=(--artifact "${artifact}")
  server_gate_args+=(--stage-artifact "${artifact}")
done
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/pi05_stage_gate_v2.py" "${gate_args[@]}" >/dev/null

if [[ "${MODE}" != "k1-smoke" ]]; then
  "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/pi05_stage_gate_v2.py" \
    --stage paired_full_baseline --source-lock "${SOURCE_LOCK}" \
    --artifact "${K1_TENSOR_REPORT}" --artifact "${FREEZE_GATE}" >/dev/null
  "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/verify_pi05_final_manifest_v2.py" \
    --manifest "${FINAL_MANIFEST}" --source-lock "${SOURCE_LOCK}" --suite "${SUITE}" >/dev/null
fi
if [[ "${MODE}" == v0 || "${MODE}" == v1 || "${MODE}" == v2 ]]; then
  [[ -f "${ADAPTER}" ]] || { echo "Missing adapter checkpoint: ${ADAPTER}" >&2; exit 1; }
  [[ -f "${CACHE_GATE}" ]] || { echo "Missing cache gate: ${CACHE_GATE}" >&2; exit 1; }
  "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/pi05_stage_gate_v2.py" \
    --stage cache_artifact_current --source-lock "${SOURCE_LOCK}" --artifact "${CACHE_GATE}" >/dev/null
fi

# No OUTPUT path exists until every source/checkpoint/norm/stage/manifest gate passes.
mkdir -p "${OUTPUT}/protocol/libero_config"
printf '%s\n' \
  "assets: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/assets" \
  "bddl_files: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/bddl_files" \
  "benchmark_root: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero" \
  "datasets: ${HF_LEROBOT_HOME}" \
  "init_states: ${OPENPI_LL_UPSTREAM}/third_party/libero/libero/libero/init_files" \
  > "${OUTPUT}/protocol/libero_config/config.yaml"

# Keep MuJoCo and client runtime residue inside the already-gated result root.
cd "${OUTPUT}"

server_args=(
  --run --checkpoint "${OPENPI_LL_CHECKPOINT}" --mode "${server_mode}"
  --k-q 4 --flow-steps 10 --noise-seed-base 7 --suite "${SUITE}"
  --device cuda --port "${PORT}" "${server_gate_args[@]}"
)
if [[ -n "${FINAL_MANIFEST}" ]]; then server_args+=(--final-evaluation-manifest "${FINAL_MANIFEST}"); fi
if [[ -n "${ADAPTER}" ]]; then server_args+=(--adapter-checkpoint "${ADAPTER}"); fi
if [[ -n "${DYNAMIC_THRESHOLD_LOCK}" ]]; then server_args+=(--dynamic-threshold-lock "${DYNAMIC_THRESHOLD_LOCK}"); fi
if [[ "${MODE}" == "k1-smoke" ]]; then server_args+=(--k1-audit); fi

CUDA_VISIBLE_DEVICES=${GPU} "${OPENPI_LL_MAIN_PY}" \
  "${OPENPI_LL_ROOT}/tools/openpi/serve_pi05_latentloop.py" "${server_args[@]}" \
  > "${OUTPUT}/server.log" 2>&1 &
server_pid=$!
cleanup() { kill "${server_pid}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
openpi_ll_wait_server "${PORT}"

client_args=(
  --output "${OUTPUT}/client" --suite "${SUITE}" --host 127.0.0.1 --port "${PORT}"
  --seed 7 --noise-seed-base 7 --wait-steps 10 --replan-steps 5 --resize-size 224
  --policy-path "${policy_path}"
)
if [[ "${MODE}" == "k1-smoke" ]]; then
  client_args+=(--num-trials 2 --max-tasks 1 --paired-k1-smoke)
else
  client_args+=(--num-trials 50 --max-tasks 10 --final-evaluation-manifest "${FINAL_MANIFEST}")
fi
if ((SAVE_VIDEO)); then client_args+=(--save-video); fi

CUDA_VISIBLE_DEVICES=${GPU} \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
NUMBA_CACHE_DIR=${OPENPI_LL_SHARED}/cache/openpi/numba/${MODE}_${SUITE} \
MPLCONFIGDIR=${OPENPI_LL_SHARED}/cache/openpi/matplotlib/${MODE}_${SUITE} \
LIBERO_CONFIG_PATH=${OUTPUT}/protocol/libero_config \
PYTHONPATH=${OPENPI_LL_ROOT}:${OPENPI_LL_UPSTREAM}/packages/openpi-client/src:${OPENPI_LL_UPSTREAM}/third_party/libero \
  "${OPENPI_LL_CLIENT_PY}" "${OPENPI_LL_ROOT}/tools/openpi/evaluate_pi05_latentloop_client.py" \
  "${client_args[@]}" 2>&1 | tee "${OUTPUT}/client.log"

cleanup
wait "${server_pid}" 2>/dev/null || true
trap - EXIT INT TERM

if [[ "${MODE}" == "k1-smoke" ]]; then
  "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/audit_pi05_k1_equivalence.py" \
    --merge-only --tensor-report "${K1_TENSOR_REPORT}" \
    --episode-smoke-json "${OUTPUT}/client/k1_episode_smoke.json" \
    --output "${OUTPUT}/combined_gate"
else
  "${OPENPI_LL_MAIN_PY}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["rollouts"] == 500 and d["complete"]' \
    "${OUTPUT}/client/summary.json"
  printf 'ONLINE_SUITE_SHARD_COMPLETE mode=%s suite=%s episodes=500 output=%s\n' "${MODE}" "${SUITE}" "${OUTPUT}"
fi
