#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/mingyujung/miniconda3/envs/seer_libero/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
RUN_ID=${RUN_ID:-paper_r1}
WARMUP_CYCLES=${WARMUP_CYCLES:-10}
MEASURED_CYCLES=${MEASURED_CYCLES:-100}
SAMPLE_START=${SAMPLE_START:-0}
BARRIER_TIMEOUT_SECONDS=${BARRIER_TIMEOUT_SECONDS:-3600}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

UNIFIED_SOURCE_COMMIT=3bc2e0ec2a3a5aeb3373fc006d2ca905aa666216
BRIDGE_SOURCE_ROOT=${BRIDGE_SOURCE_ROOT:-/home/mingyujung/private/gnaroshi_vla_latent_bridge_paper}
BRIDGE_SOURCE_COMMIT=3bedd4bad0cfc0646518165b43a0232b08587d38

PAPER_CHECKPOINT_ROOT=${PAPER_CHECKPOINT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/artifacts/checkpoints/seer/paper}
SEER_CHECKPOINT=${SEER_CHECKPOINT:-${PAPER_CHECKPOINT_ROOT}/libero_long/teacher_public33.pth}
SEER_SHA256=a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646
LATENTLOOP_CHECKPOINT=${LATENTLOOP_CHECKPOINT:-${PAPER_CHECKPOINT_ROOT}/libero_long/latentloop_adapter39.pth}
LATENTLOOP_SHA256=3f70179ab9b1bae64fc772d71c57a93592b9f82e53b5fcaf1a6beb319c280462
BRIDGE_CHECKPOINT=${BRIDGE_CHECKPOINT:-${PAPER_CHECKPOINT_ROOT}/libero_long/latent_bridge_large_best.pt}
BRIDGE_SHA256=2fc92f989721b555709fe6ba45d7a3f47dbf0f4a2763e708d807e49211cce7fb
VIT_CHECKPOINT=${VIT_CHECKPOINT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}
VIT_SHA256=aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-/home/mingyujung/.cache/clip/ViT-B-32.pt}
CLIP_SHA256=40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
DATASET_ROOT=${DATASET_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/LIBERO_DATASETS/libero_10_converted}
DATASET_NAME=libero_10_converted
LIBERO_PATH=${LIBERO_PATH:-/home/mingyujung/private/LIBERO}
RESULT_BASE=${RESULT_BASE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/seer/latency/unified_method}
RESULT_ROOT=${RESULT_ROOT:-${RESULT_BASE}/${RUN_ID}}

BENCHMARK=${REPO_ROOT}/tools/seer/unified_policy_latency.py
AGGREGATOR=${REPO_ROOT}/tools/seer/aggregate_unified_policy_latency.py
CORE_FILES=(
    architectures/seer/upstream/models/seer_model.py
    architectures/seer/upstream/models/gpt2.py
    architectures/seer/upstream/models/lrnode_modules.py
    architectures/seer/upstream/models/vla_cache.py
)
BRIDGE_CORE_FILES=(
    architectures/seer/adapters/latent_bridge/bridge.py
    architectures/seer/adapters/latent_bridge/checkpoint.py
    architectures/seer/adapters/latent_bridge/hooks.py
    architectures/seer/adapters/latent_bridge/layout.py
)

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

verify_hash() {
    local label=$1 path=$2 expected=$3 actual
    require_file "${path}"
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ "${actual}" == "${expected}" ]] || fail \
        "${label} SHA256 mismatch: expected=${expected}, actual=${actual}, path=${path}"
    echo "[VERIFY][OK] ${label}: ${actual}"
}

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "invalid RUN_ID: ${RUN_ID}"
case "${CUDA_VISIBLE_DEVICES}" in
    0,1,2,3|4,5,6,7) ;;
    *) fail "use one complete sd1 RTX 3090 partition: 0,1,2,3 or 4,5,6,7; got ${CUDA_VISIBLE_DEVICES}" ;;
esac
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#GPU_IDS[@]}" -eq 4 ]] || fail "exactly four physical GPUs are required"
[[ "${WARMUP_CYCLES}" =~ ^[0-9]+$ ]] && (( WARMUP_CYCLES >= 1 )) || fail \
    "WARMUP_CYCLES must be an integer >= 1"
[[ "${MEASURED_CYCLES}" =~ ^[0-9]+$ ]] && (( MEASURED_CYCLES >= 2 )) || fail \
    "MEASURED_CYCLES must be an integer >= 2"
[[ "${SAMPLE_START}" =~ ^[0-9]+$ ]] || fail "SAMPLE_START must be a non-negative integer"
[[ "${BARRIER_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] && (( BARRIER_TIMEOUT_SECONDS >= 60 )) || fail \
    "BARRIER_TIMEOUT_SECONDS must be an integer >= 60"
[[ "${PREFLIGHT_ONLY}" == "0" || "${PREFLIGHT_ONLY}" == "1" ]] || fail \
    "PREFLIGHT_ONLY must be 0 or 1"
[[ -x "${PYTHON_BIN}" ]] || fail "missing Python interpreter: ${PYTHON_BIN}"
require_file "${BENCHMARK}"
require_file "${AGGREGATOR}"
require_file "${DATASET_ROOT}/${DATASET_NAME}/meta_info.h5"
[[ -d "${LIBERO_PATH}/libero/libero" ]] || fail "invalid LIBERO_PATH: ${LIBERO_PATH}"
require_file "${REPO_ROOT}/architectures/seer/upstream/data_info/libero_10_converted.json"

verify_hash "public Seer checkpoint 33" "${SEER_CHECKPOINT}" "${SEER_SHA256}"
verify_hash "public33 LatentLoop adapter 39" "${LATENTLOOP_CHECKPOINT}" "${LATENTLOOP_SHA256}"
verify_hash "Latent Bridge Large R1" "${BRIDGE_CHECKPOINT}" "${BRIDGE_SHA256}"
verify_hash "ViT-MAE" "${VIT_CHECKPOINT}" "${VIT_SHA256}"
verify_hash "CLIP ViT-B/32" "${CLIP_CHECKPOINT}" "${CLIP_SHA256}"

actual_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${UNIFIED_SOURCE_COMMIT}" ]] || fail \
    "unified Seer source commit changed: expected=${UNIFIED_SOURCE_COMMIT}, actual=${actual_commit}"
git -C "${REPO_ROOT}" diff --quiet HEAD -- "${CORE_FILES[@]}" || fail \
    "core Seer timing source differs from locked commit"
bridge_commit=$(git -C "${BRIDGE_SOURCE_ROOT}" rev-parse HEAD)
[[ "${bridge_commit}" == "${BRIDGE_SOURCE_COMMIT}" ]] || fail \
    "Latent Bridge source commit changed: expected=${BRIDGE_SOURCE_COMMIT}, actual=${bridge_commit}"
git -C "${BRIDGE_SOURCE_ROOT}" diff --quiet HEAD -- "${BRIDGE_CORE_FILES[@]}" || fail \
    "Latent Bridge benchmark source differs from locked commit"

for gpu in "${GPU_IDS[@]}"; do
    if ! gpu_name=$(nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader 2>/dev/null \
        | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'); then
        fail "nvidia-smi could not inspect GPU ${gpu}"
    fi
    [[ "${gpu_name}" == *"RTX 3090"* ]] || fail "GPU ${gpu} is not RTX 3090: ${gpu_name}"
    busy_pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {print $1}' \
        | paste -sd, -)
    [[ -z "${busy_pids}" ]] || fail "GPU ${gpu} has active compute PIDs: ${busy_pids}"
    echo "[VERIFY][OK] GPU ${gpu}: ${gpu_name}; idle"
done

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PREFLIGHT][PASS] assets, hashes, source locks, runtime, and GPU partition"
    exit 0
fi

[[ ! -e "${RESULT_ROOT}" ]] || fail \
    "RESULT_ROOT already exists; use a new RUN_ID or inspect the preserved run: ${RESULT_ROOT}"
mkdir -p "${RESULT_ROOT}/logs"
mkdir -p "${RESULT_ROOT}/barrier"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/architectures/seer/upstream:${LIBERO_PATH}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m py_compile "${BENCHMARK}" "${AGGREGATOR}"
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider "${REPO_ROOT}/tests/seer_latency"

{
    echo "run_id=${RUN_ID}"
    echo "source_commit=${actual_commit}"
    echo "bridge_source_commit=${bridge_commit}"
    echo "physical_gpus=${CUDA_VISIBLE_DEVICES}"
    echo "warmup_cycles=${WARMUP_CYCLES}"
    echo "measured_cycles=${MEASURED_CYCLES}"
    echo "barrier_timeout_seconds=${BARRIER_TIMEOUT_SECONDS}"
    echo "timing_scope=model_path_only_no_simulator_no_preprocessing"
} > "${RESULT_ROOT}/run_contract.env"
git -C "${REPO_ROOT}" status --short > "${RESULT_ROOT}/source_status.txt"

worker_pids=()
telemetry_pid=""
running_marker=${RESULT_ROOT}/.running
touch "${running_marker}"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    rm -f "${running_marker}"
    if [[ -n "${telemetry_pid}" ]]; then
        kill "${telemetry_pid}" 2>/dev/null || true
        wait "${telemetry_pid}" 2>/dev/null || true
    fi
    if (( status != 0 )); then
        touch "${RESULT_ROOT}/barrier/abort"
        for pid in "${worker_pids[@]:-}"; do
            kill "${pid}" 2>/dev/null || true
        done
        echo "[ERROR] benchmark stopped with status ${status}; partial artifacts preserved: ${RESULT_ROOT}" >&2
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

(
    echo "timestamp,index,name,utilization_gpu_percent,memory_used_mib,power_draw_w,temperature_c"
    while [[ -e "${running_marker}" ]]; do
        nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
            --query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw,temperature.gpu \
            --format=csv,noheader,nounits
        sleep 1
    done
) > "${RESULT_ROOT}/gpu_telemetry.csv" &
telemetry_pid=$!

echo "[RUN] Four independent RTX 3090 replicates are starting."
echo "[RUN] Live logs are prefixed by physical GPU; output=${RESULT_ROOT}"
for gpu in "${GPU_IDS[@]}"; do
    (
        set -o pipefail
        CUDA_VISIBLE_DEVICES="${gpu}" \
        TORCHINDUCTOR_CACHE_DIR="/tmp/seer_unified_latency_${RUN_ID}_gpu${gpu}" \
        "${PYTHON_BIN}" "${BENCHMARK}" \
            --replicate-id "gpu${gpu}" \
            --checkpoint "${SEER_CHECKPOINT}" \
            --checkpoint-sha256 "${SEER_SHA256}" \
            --latentloop-checkpoint "${LATENTLOOP_CHECKPOINT}" \
            --latentloop-sha256 "${LATENTLOOP_SHA256}" \
            --bridge-checkpoint "${BRIDGE_CHECKPOINT}" \
            --bridge-sha256 "${BRIDGE_SHA256}" \
            --bridge-source-root "${BRIDGE_SOURCE_ROOT}" \
            --bridge-source-commit "${BRIDGE_SOURCE_COMMIT}" \
            --vit-checkpoint "${VIT_CHECKPOINT}" \
            --vit-sha256 "${VIT_SHA256}" \
            --clip-checkpoint "${CLIP_CHECKPOINT}" \
            --clip-sha256 "${CLIP_SHA256}" \
            --dataset-root "${DATASET_ROOT}" \
            --dataset-name "${DATASET_NAME}" \
            --libero-path "${LIBERO_PATH}" \
            --sample-start "${SAMPLE_START}" \
            --k-values 2,3,4,5,6,7,8 \
            --bridge-k 4 \
            --warmup-cycles "${WARMUP_CYCLES}" \
            --measured-cycles "${MEASURED_CYCLES}" \
            --barrier-dir "${RESULT_ROOT}/barrier" \
            --barrier-timeout-seconds "${BARRIER_TIMEOUT_SECONDS}" \
            --output "${RESULT_ROOT}/replicate_gpu${gpu}.json" 2>&1 \
            | sed -u "s/^/[gpu${gpu}] /" \
            | tee "${RESULT_ROOT}/logs/gpu${gpu}.log"
    ) &
    worker_pids+=("$!")
done

barrier_started=$(date +%s)
while :; do
    ready_count=$(find "${RESULT_ROOT}/barrier" -maxdepth 1 -name 'gpu*.ready' -type f | wc -l)
    if (( ready_count == 4 )); then
        touch "${RESULT_ROOT}/barrier/release"
        echo "[RUN] All four workers completed compile/warmup; synchronized measurement released."
        break
    fi
    for pid in "${worker_pids[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            touch "${RESULT_ROOT}/barrier/abort"
            fail "a worker exited before the four-GPU measurement barrier (${ready_count}/4 ready)"
        fi
    done
    if (( $(date +%s) - barrier_started > BARRIER_TIMEOUT_SECONDS )); then
        touch "${RESULT_ROOT}/barrier/abort"
        fail "timed out after ${BARRIER_TIMEOUT_SECONDS} seconds waiting for compile/warmup barrier"
    fi
    sleep 1
done

failed=0
set +e
for pid in "${worker_pids[@]}"; do
    wait "${pid}"
    status=$?
    if (( status != 0 )); then
        failed=1
    fi
done
set -e
(( failed == 0 )) || fail "one or more GPU replicates failed; inspect ${RESULT_ROOT}/logs"

rm -f "${running_marker}"
kill "${telemetry_pid}" 2>/dev/null || true
wait "${telemetry_pid}" 2>/dev/null || true
telemetry_pid=""

aggregate_args=()
for gpu in "${GPU_IDS[@]}"; do
    require_file "${RESULT_ROOT}/replicate_gpu${gpu}.json"
    aggregate_args+=(--input "${RESULT_ROOT}/replicate_gpu${gpu}.json")
done
"${PYTHON_BIN}" "${AGGREGATOR}" \
    "${aggregate_args[@]}" \
    --output-dir "${RESULT_ROOT}/analysis"

echo "[DONE] ${RESULT_ROOT}/analysis/unified_policy_latency.md"
echo "[DONE] ${RESULT_ROOT}/analysis/unified_policy_latency.csv"
echo "[DONE] ${RESULT_ROOT}/analysis/unified_policy_latency.json"
