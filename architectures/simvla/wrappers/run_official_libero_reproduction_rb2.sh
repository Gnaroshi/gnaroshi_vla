#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
UPSTREAM_ROOT="${REPO_ROOT}/architectures/simvla/upstream"
EVAL_ROOT="${UPSTREAM_ROOT}/evaluation/libero"

STORAGE_ROOT="${SIMVLA_RB2_STORAGE_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage}"
ENV_ROOT="${SIMVLA_RB2_ENV_ROOT:-${STORAGE_ROOT}/envs/simvla/libero_mujoco237}"
LIBERO_ROOT="${SIMVLA_RB2_LIBERO_ROOT:-${STORAGE_ROOT}/datasets/LIBERO}"
HF_HOME="${HF_HOME:-${STORAGE_ROOT}/cache/simvla/huggingface}"
RESULT_ROOT="${SIMVLA_RB2_RESULT_ROOT:-${STORAGE_ROOT}/results/simvla/reproduction}"

RUN_NAME="${SIMVLA_REPRO_RUN_NAME:-official_ckpt_mujoco237_seed7_n10_r1}"
OUT="${RESULT_ROOT}/${RUN_NAME}"
PYTHON="${ENV_ROOT}/bin/python"
CHECKPOINT="${SIMVLA_REPRO_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}"
SMOLVLM_MODEL="${SIMVLA_REPRO_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
NORM_STATS="${SIMVLA_REPRO_NORM_STATS:-${UPSTREAM_ROOT}/norm_stats/libero_norm.json}"
GPU_INDEX="${SIMVLA_REPRO_GPU:-0}"
PORT="${SIMVLA_REPRO_PORT:-8102}"
NUM_TRIALS="${SIMVLA_REPRO_NUM_TRIALS:-10}"
SEED="${SIMVLA_REPRO_SEED:-7}"
REPLAN_STEPS="${SIMVLA_REPRO_REPLAN_STEPS:-5}"
WAIT_MIN_FREE_MIB="${SIMVLA_REPRO_WAIT_MIN_FREE_MIB:-30000}"
RESERVE_MIB="${SIMVLA_REPRO_RESERVE_MIB:-2048}"
WAIT_POLL_SECONDS="${SIMVLA_REPRO_WAIT_POLL_SECONDS:-30}"
PROGRESS_SECONDS="${SIMVLA_REPRO_PROGRESS_SECONDS:-30}"

SERVER_PID=""
MONITOR_PID=""
CLIENT_PIDS=()
CLIENT_SUITES=()
MAIN_PID="$$"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM

    if [[ -n "${MONITOR_PID}" ]]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi
    for pid in "${CLIENT_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]]; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi

    wait "${MONITOR_PID}" 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
    if [[ -n "${SERVER_PID}" ]]; then
        wait "${SERVER_PID}" 2>/dev/null || true
    fi

    if (( rc != 0 )); then
        log "FAILED rc=${rc}; inspect ${OUT}/logs"
    fi
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

require_file() {
    [[ -f "$1" ]] || {
        printf 'Missing required file: %s\n' "$1" >&2
        exit 1
    }
}

require_dir() {
    [[ -d "$1" ]] || {
        printf 'Missing required directory: %s\n' "$1" >&2
        exit 1
    }
}

gpu_free_mib() {
    nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk 'NR == 1 {gsub(/ /, "", $0); print $0}'
}

gpu_compute_pids() {
    nvidia-smi --id="${GPU_INDEX}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF {gsub(/ /, "", $0); print $0}'
}

wait_for_exclusive_gpu() {
    local free_mib pids
    while true; do
        free_mib="$(gpu_free_mib)"
        pids="$(gpu_compute_pids || true)"
        if [[ -z "${pids}" ]] && (( free_mib >= WAIT_MIN_FREE_MIB )); then
            log "GPU ${GPU_INDEX} is ready: free=${free_mib} MiB, no compute process"
            return
        fi
        log "Waiting for GPU ${GPU_INDEX}: free=${free_mib} MiB, compute_pids=${pids:-none}"
        sleep "${WAIT_POLL_SECONDS}"
    done
}

reserve_monitor() {
    local free_mib
    while true; do
        sleep 15
        free_mib="$(gpu_free_mib)"
        if (( free_mib < RESERVE_MIB )); then
            printf '[%s] GPU reserve violation: free=%s MiB < reserve=%s MiB\n' \
                "$(date -Is)" "${free_mib}" "${RESERVE_MIB}" >&2
            kill -TERM "${MAIN_PID}"
            return
        fi
    done
}

write_metadata() {
    {
        printf 'run_name=%s\n' "${RUN_NAME}"
        printf 'host=%s\n' "$(hostname)"
        printf 'checkpoint=%s\n' "${CHECKPOINT}"
        printf 'smolvlm_model=%s\n' "${SMOLVLM_MODEL}"
        printf 'norm_stats=%s\n' "${NORM_STATS}"
        printf 'libero_root=%s\n' "${LIBERO_ROOT}"
        printf 'libero_demonstrations=%s\n' 'not_required_for_evaluation_not_transferred'
        printf 'env_root=%s\n' "${ENV_ROOT}"
        printf 'hf_home=%s\n' "${HF_HOME}"
        printf 'gpu_index=%s\n' "${GPU_INDEX}"
        printf 'num_trials=%s\n' "${NUM_TRIALS}"
        printf 'seed=%s\n' "${SEED}"
        printf 'replan_steps=%s\n' "${REPLAN_STEPS}"
        printf 'mujoco_gl=%s\n' "${MUJOCO_GL}"
        printf 'python=%s\n' "${PYTHON}"
        printf 'started_at=%s\n' "$(date -Is)"
    } > "${OUT}/metadata/run.env"

    git -C "${REPO_ROOT}" rev-parse HEAD > "${OUT}/metadata/repository_head.txt" 2>&1 || true
    git -C "${REPO_ROOT}" status --short > "${OUT}/metadata/repository_status.txt" 2>&1 || true
    git -C "${UPSTREAM_ROOT}" rev-parse HEAD > "${OUT}/metadata/upstream_head.txt" 2>&1 || true
    git -C "${UPSTREAM_ROOT}" status --short > "${OUT}/metadata/upstream_status.txt" 2>&1 || true
    sha256sum \
        "${EVAL_ROOT}/serve_smolvlm_libero.py" \
        "${EVAL_ROOT}/libero_client.py" \
        "${UPSTREAM_ROOT}/models/modeling_smolvlm_vla.py" \
        "${NORM_STATS}" \
        > "${OUT}/metadata/source_sha256.txt"
    nvidia-smi -q > "${OUT}/metadata/nvidia_smi_q.txt"
    "${PYTHON}" -m pip freeze > "${OUT}/metadata/pip_freeze.txt"
    "${PYTHON}" - <<'PY' > "${OUT}/metadata/runtime_versions.txt"
import sys

import accelerate
import mujoco
import numpy
import peft
import robosuite
import torch
import torchvision
import transformers

print(f"python={sys.version.replace(chr(10), ' ')}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torchvision={torchvision.__version__}")
print(f"transformers={transformers.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"peft={peft.__version__}")
print(f"numpy={numpy.__version__}")
print(f"mujoco={mujoco.__version__}")
print(f"robosuite={robosuite.__version__}")
PY
}

write_libero_config() {
    local package_root="${LIBERO_ROOT}/libero/libero"
    cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
assets: ${package_root}/assets
bddl_files: ${package_root}/bddl_files
benchmark_root: ${package_root}
datasets: ${OUT}/runtime/evaluation_unused_dataset_path
init_states: ${package_root}/init_files
EOF
}

wait_for_server() {
    local deadline=$((SECONDS + 900))
    while (( SECONDS < deadline )); do
        if grep -q 'SimVLA server listening' "${OUT}/logs/server.log" 2>/dev/null; then
            log "Policy server is ready on port ${PORT}"
            return
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            printf 'Policy server exited during startup.\n' >&2
            tail -100 "${OUT}/logs/server.log" >&2 || true
            exit 1
        fi
        sleep 5
    done
    printf 'Timed out waiting for policy server.\n' >&2
    tail -100 "${OUT}/logs/server.log" >&2 || true
    exit 1
}

print_progress() {
    local suite log_file completed successes
    for suite in "${CLIENT_SUITES[@]}"; do
        log_file="${OUT}/logs/${suite}.log"
        completed="$(grep -Ec '^  \[(OK|FAIL)\] Task [0-9]+ Ep [0-9]+:' "${log_file}" 2>/dev/null || true)"
        successes="$(grep -Ec '^  \[OK\] Task [0-9]+ Ep [0-9]+:' "${log_file}" 2>/dev/null || true)"
        printf 'PROGRESS suite=%s completed=%s/%s successes=%s\n' \
            "${suite}" "${completed}" "$((10 * NUM_TRIALS))" "${successes}"
    done
}

summarize() {
    local suite line successes episodes pct total_successes=0 total_episodes=0
    printf 'suite,successes,episodes,success_rate_pct\n' > "${OUT}/summary.csv"
    for suite in "${CLIENT_SUITES[@]}"; do
        line="$(grep -E 'Total success rate:' "${OUT}/logs/${suite}.log" | tail -1 || true)"
        if [[ ! "${line}" =~ ([0-9]+)/([0-9]+).*\(([0-9.]+)%\) ]]; then
            printf 'Missing final success line for %s\n' "${suite}" >&2
            return 1
        fi
        successes="${BASH_REMATCH[1]}"
        episodes="${BASH_REMATCH[2]}"
        pct="${BASH_REMATCH[3]}"
        printf '%s,%s,%s,%s\n' "${suite}" "${successes}" "${episodes}" "${pct}" \
            >> "${OUT}/summary.csv"
        total_successes=$((total_successes + successes))
        total_episodes=$((total_episodes + episodes))
    done

    awk -F, '
        NR > 1 { total_success += $2; total_episodes += $3 }
        END {
            printf "total_successes=%d\n", total_success
            printf "total_episodes=%d\n", total_episodes
            printf "micro_success_rate_pct=%.2f\n", 100 * total_success / total_episodes
        }
    ' "${OUT}/summary.csv" > "${OUT}/summary.txt"
    printf 'completed_at=%s\n' "$(date -Is)" >> "${OUT}/summary.txt"
    cat "${OUT}/summary.csv"
    cat "${OUT}/summary.txt"
}

if [[ "${SIMVLA_RB2_REPRO_RUN:-0}" != "1" ]]; then
    printf 'Refusing to run. Set SIMVLA_RB2_REPRO_RUN=1.\n' >&2
    exit 2
fi

if [[ "$(hostname)" != "jbr-TRX50" && "${SIMVLA_ALLOW_OTHER_HOST:-0}" != "1" ]]; then
    printf 'This launcher is for rb2 (jbr-TRX50); current host is %s.\n' "$(hostname)" >&2
    exit 2
fi

command -v nvidia-smi >/dev/null
command -v git >/dev/null
require_file "${PYTHON}"
require_dir "${LIBERO_ROOT}/libero/libero"
require_dir "${HF_HOME}/hub/models--YuankaiLuo--SimVLA-LIBERO"
require_dir "${HF_HOME}/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct"
require_file "${EVAL_ROOT}/serve_smolvlm_libero.py"
require_file "${EVAL_ROOT}/libero_client.py"
require_file "${NORM_STATS}"

if [[ -e "${OUT}" ]] && find "${OUT}" -mindepth 1 -print -quit | grep -q .; then
    printf 'Result directory is not empty: %s\n' "${OUT}" >&2
    printf 'Set SIMVLA_REPRO_RUN_NAME to a new semantic repeat name.\n' >&2
    exit 2
fi

mkdir -p \
    "${OUT}/logs" \
    "${OUT}/metadata" \
    "${OUT}/runtime/libero_config" \
    "${OUT}/runtime/evaluation_unused_dataset_path" \
    "${OUT}/videos" \
    "${STORAGE_ROOT}/cache/simvla/numba/${RUN_NAME}" \
    "${STORAGE_ROOT}/cache/simvla/matplotlib/${RUN_NAME}" \
    "${STORAGE_ROOT}/cache/simvla/xdg"

exec > >(tee -a "${OUT}/logs/launcher.log") 2>&1

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="${STORAGE_ROOT}/cache/simvla/numba/${RUN_NAME}"
export MPLCONFIGDIR="${STORAGE_ROOT}/cache/simvla/matplotlib/${RUN_NAME}"
export XDG_CACHE_HOME="${STORAGE_ROOT}/cache/simvla/xdg"
export LIBERO_CONFIG_PATH="${OUT}/runtime/libero_config"
export PYTHONPATH="${LIBERO_ROOT}:${UPSTREAM_ROOT}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2

write_libero_config
write_metadata

"${PYTHON}" - <<'PY'
import mujoco
import torch
import transformers

assert mujoco.__version__ == "2.3.7", mujoco.__version__
assert transformers.__version__ == "4.57.3", transformers.__version__
print("PREFLIGHT_PASS", torch.__version__, torch.version.cuda)
PY

wait_for_exclusive_gpu

"${PYTHON}" - <<'PY'
import torch

x = torch.ones(1, device="cuda")
assert x.item() == 1.0
major, minor = torch.cuda.get_device_capability()
assert (major, minor) == (12, 0), (major, minor)
print("CUDA_SMOKE_PASS", torch.cuda.get_device_name(), (major, minor))
PY

reserve_monitor &
MONITOR_PID=$!

log "Starting official SimVLA policy server"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON}" -u "${EVAL_ROOT}/serve_smolvlm_libero.py" \
    --checkpoint "${CHECKPOINT}" \
    --norm_stats "${NORM_STATS}" \
    --smolvlm_model "${SMOLVLM_MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    > "${OUT}/logs/server.log" 2>&1 &
SERVER_PID=$!
wait_for_server

CLIENT_SUITES=(libero_spatial libero_object libero_goal libero_10)
for suite in "${CLIENT_SUITES[@]}"; do
    log "Launching official client suite=${suite}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON}" -u "${EVAL_ROOT}/libero_client.py" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --client_type websocket \
        --task_suite "${suite}" \
        --num_trials "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --replan_steps "${REPLAN_STEPS}" \
        --video_out "${OUT}/videos" \
        > "${OUT}/logs/${suite}.log" 2>&1 &
    CLIENT_PIDS+=("$!")
done

while true; do
    any_running=0
    for pid in "${CLIENT_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            any_running=1
            break
        fi
    done
    print_progress
    (( any_running == 0 )) && break
    sleep "${PROGRESS_SECONDS}"
done

client_failure=0
for pid in "${CLIENT_PIDS[@]}"; do
    if ! wait "${pid}"; then
        client_failure=1
    fi
done
CLIENT_PIDS=()

if (( client_failure != 0 )); then
    printf 'At least one official client failed.\n' >&2
    exit 1
fi

summarize
log "COMPLETE result=${OUT}"
