#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: run_real_vla_cache_comparison.sh [--all|--preflight|--benchmark]

Runs the non-robot SimVLA comparison on rb2:
  baseline, Condition Loop, full LatentLoop, VLA-Cache eager control, VLA-Cache.

The benchmark uses 500 fixed validation queries, three repeats, H=10/R=5,
paired flow noise, synchronized CUDA latency, peak VRAM, and action fidelity.
It does not report real-robot task success.
EOF
}

mode="${1:---all}"
case "${mode}" in
    --all|--preflight|--benchmark) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin="${SIMVLA_VLA_CACHE_PYTHON:-/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python}"
manifest="${SIMVLA_VLA_CACHE_MANIFEST:-/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/real_world/stackcupanddoll_v1/deployment/deployment_manifest.local.json}"
dataset_manifest="${SIMVLA_VLA_CACHE_DATASET:-/home/mingyujung/private/gnaroshi_vla_storage/datasets/simvla_real/stackcupanddoll_hdf5_v2/manifest.json}"
result_root="${SIMVLA_VLA_CACHE_RESULT_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/vla_cache/real_doll_offline}"
run_id="${SIMVLA_VLA_CACHE_RUN_ID:-doll_validation_500q_3repeat_r1}"
output="${result_root}/${run_id}"
gpu="${SIMVLA_VLA_CACHE_GPU:-0}"
queries="${SIMVLA_VLA_CACHE_QUERIES:-500}"
repeats="${SIMVLA_VLA_CACHE_REPEATS:-3}"
warmup_queries="${SIMVLA_VLA_CACHE_WARMUP_QUERIES:-4}"
seed="${SIMVLA_VLA_CACHE_SEED:-20260905}"

if [[ "${mode}" != "--preflight" && "${SIMVLA_VLA_CACHE_COMPARISON_RUN:-0}" != "1" ]]; then
    echo "[ERROR] Set SIMVLA_VLA_CACHE_COMPARISON_RUN=1 for benchmark execution." >&2
    exit 2
fi
for value in "${queries}" "${repeats}" "${warmup_queries}" "${seed}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || { echo "[ERROR] numeric options must be integers" >&2; exit 2; }
done
[[ "${queries}" -gt 0 && "${repeats}" -gt 0 ]] || {
    echo "[ERROR] queries and repeats must be positive" >&2
    exit 2
}
[[ -x "${python_bin}" ]] || { echo "[ERROR] missing Python: ${python_bin}" >&2; exit 1; }
[[ -f "${manifest}" ]] || { echo "[ERROR] missing deployment manifest: ${manifest}" >&2; exit 1; }
[[ -f "${dataset_manifest}" ]] || { echo "[ERROR] missing dataset manifest: ${dataset_manifest}" >&2; exit 1; }
if [[ "$(hostname)" != "jbr-TRX50" ]]; then
    echo "[ERROR] this launcher is configured for rb2/jbr-TRX50, observed $(hostname)" >&2
    exit 1
fi
if [[ -n "$(git -C "${repo_root}" status --short)" ]]; then
    echo "[ERROR] source worktree is dirty: ${repo_root}" >&2
    git -C "${repo_root}" status --short >&2
    exit 1
fi

mkdir -p "${result_root}"
exec 9>"${result_root}/.comparison.lock"
flock -n 9 || { echo "[ERROR] another VLA-Cache comparison is active" >&2; exit 1; }
if [[ -e "${output}" ]]; then
    echo "[ERROR] output already exists: ${output}" >&2
    exit 1
fi
mkdir -p "${output}/logs" "${output}/metadata"
status_file="${output}/status.txt"
printf 'RUNNING\n' > "${status_file}"
exec > >(tee "${output}/logs/launcher.log") 2>&1

on_error() {
    local rc=$?
    printf 'FAILED rc=%s\n' "${rc}" > "${status_file}"
    echo "VLA_CACHE_COMPARISON_FAILED rc=${rc} log=${output}/logs/launcher.log" >&2
    exit "${rc}"
}
trap on_error ERR

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONHASHSEED="${seed}"

{
    echo "hostname=$(hostname)"
    echo "repo_root=${repo_root}"
    echo "git_commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "git_branch=$(git -C "${repo_root}" branch --show-current)"
    echo "python=${python_bin}"
    echo "manifest=${manifest}"
    echo "dataset_manifest=${dataset_manifest}"
    echo "physical_gpu=${gpu}"
    echo "queries=${queries}"
    echo "repeats=${repeats}"
    echo "warmup_queries=${warmup_queries}"
    echo "seed=${seed}"
    echo "robot_commands_possible=no"
} > "${output}/metadata/launch_config.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
    --format=csv,noheader > "${output}/metadata/gpu_before.csv"
cp "${BASH_SOURCE[0]}" "${output}/metadata/launcher_snapshot.sh"

run_preflight() {
    echo "[1/3] VLA-Cache unit and deployment contract tests"
    "${python_bin}" -m pytest -q \
        tests/simvla_real_deploy/test_vla_cache_runtime.py \
        tests/simvla_real_deploy/test_real_deployment_contract.py \
        tests/simvla_real_deploy/test_vla_cache_offline_benchmark.py \
        2>&1 | tee "${output}/logs/tests.log"

    echo "[2/3] immutable copied-source preflight"
    "${python_bin}" -m architectures.simvla.adapters.latentloop_real_deploy.cli \
        source-preflight \
        --output "${output}/source_preflight" \
        2>&1 | tee "${output}/logs/source_preflight.log"

    echo "[3/3] five-row artifact and schedule preflight"
    for method in baseline condition_loop latentloop vla_cache_full vla_cache; do
        echo "  method=${method}"
        "${python_bin}" -m architectures.simvla.adapters.latentloop_real_deploy.cli \
            artifact-preflight \
            --manifest "${manifest}" \
            --method "${method}" \
            --steps 11 \
            --output "${output}/artifact_preflight_${method}" \
            2>&1 | tee "${output}/logs/artifact_preflight_${method}.log"
    done
    printf 'PREFLIGHT_PASS\n' > "${output}/preflight_status.txt"
}

run_benchmark() {
    echo "[benchmark] 5 methods x ${queries} queries x ${repeats} repeats"
    "${python_bin}" -m architectures.simvla.adapters.vla_cache.offline_benchmark \
        --manifest "${manifest}" \
        --dataset-manifest "${dataset_manifest}" \
        --output "${output}/benchmark" \
        --queries "${queries}" \
        --repeats "${repeats}" \
        --warmup-queries "${warmup_queries}" \
        --seed "${seed}" \
        --device cuda \
        2>&1 | tee "${output}/logs/benchmark.log"
    "${python_bin}" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p["verdict"]=="VLA_CACHE_OFFLINE_COMPARISON_COMPLETE", p; print(p["verdict"])' \
        "${output}/benchmark/comparison_summary.json"
}

case "${mode}" in
    --preflight) run_preflight ;;
    --benchmark) run_benchmark ;;
    --all) run_preflight; run_benchmark ;;
esac

nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
    --format=csv,noheader > "${output}/metadata/gpu_after.csv"
printf 'COMPLETE\n' > "${status_file}"
trap - ERR
echo "VLA_CACHE_COMPARISON_COMPLETE output=${output}"
