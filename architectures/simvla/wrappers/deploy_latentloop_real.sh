#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  deploy_latentloop_real.sh source-preflight
  deploy_latentloop_real.sh artifact-preflight --manifest FILE [--method baseline|condition_loop|latentloop|vla_cache_full|vla_cache]
  deploy_latentloop_real.sh read-only-profile --manifest FILE [--method baseline|condition_loop|latentloop|vla_cache_full|vla_cache] [--steps N]
  deploy_latentloop_real.sh live --manifest FILE [--method baseline|condition_loop|latentloop|vla_cache_full|vla_cache]

The default path cannot command the robot. Live mode additionally requires the
manifest safety approval, SIMVLA_REAL_LIVE_RUN=1, and a matching
SIMVLA_REAL_DEPLOYMENT_ID.
EOF
}

mode="${1:-source-preflight}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "${mode}" in
    source-preflight|artifact-preflight|read-only-profile|live) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

method="latentloop"
manifest=""
steps=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            [[ $# -ge 2 ]] || { echo "[ERROR] --manifest requires a value" >&2; exit 2; }
            manifest="$2"
            shift 2
            ;;
        --method)
            [[ $# -ge 2 ]] || { echo "[ERROR] --method requires a value" >&2; exit 2; }
            method="$2"
            shift 2
            ;;
        --steps)
            [[ $# -ge 2 ]] || { echo "[ERROR] --steps requires a value" >&2; exit 2; }
            steps="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${method}" in
    baseline|condition_loop|latentloop|vla_cache_full|vla_cache) ;;
    *) echo "[ERROR] --method must be baseline, condition_loop, latentloop, vla_cache_full, or vla_cache" >&2; exit 2 ;;
esac
if ! [[ "${steps}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --steps must be a non-negative integer" >&2
    exit 2
fi
if [[ "${mode}" != "source-preflight" && ! -f "${manifest}" ]]; then
    echo "[ERROR] A populated --manifest FILE is required for ${mode}" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
python_bin="${SIMVLA_REAL_PYTHON:-python}"
cuda_device="${SIMVLA_REAL_CUDA_DEVICE:-0}"
run_stamp=$(date +%Y%m%d_%H%M%S)
launch_dir="${repo_root}/results/simvla/real_deploy/launch_logs/${mode}/${run_stamp}"
output_dir="${launch_dir}/output"
mkdir -p "${launch_dir}"

export CUDA_VISIBLE_DEVICES="${cuda_device}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

command=(
    "${python_bin}"
    -m architectures.simvla.adapters.latentloop_real_deploy.cli
    "${mode}"
    --method "${method}"
)
if [[ -n "${manifest}" ]]; then
    command+=(--manifest "${manifest}")
fi
if (( steps > 0 )); then
    command+=(--steps "${steps}")
fi
if [[ "${mode}" != "live" ]]; then
    command+=(--output "${output_dir}")
fi

{
    echo "timestamp=${run_stamp}"
    echo "hostname=$(hostname)"
    echo "repo_root=${repo_root}"
    echo "git_commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "python=${python_bin}"
    echo "mode=${mode}"
    echo "method=${method}"
    echo "manifest=${manifest:-not_used}"
    echo "cuda_device=${cuda_device}"
    echo "robot_commands_possible=$([[ "${mode}" == "live" ]] && echo gated || echo no)"
} > "${launch_dir}/launch_config.txt"
git -C "${repo_root}" status --short > "${launch_dir}/git_status.txt"
printf '%q ' "${command[@]}" > "${launch_dir}/command.txt"
printf '\n' >> "${launch_dir}/command.txt"

echo "[SimVLA real deploy] mode=${mode} method=${method}"
echo "[SimVLA real deploy] log=${launch_dir}/console.log"
echo "[SimVLA real deploy] robot commands: $([[ "${mode}" == "live" ]] && echo 'authorization-gated' || echo 'disabled')"

set +e
"${command[@]}" 2>&1 | tee "${launch_dir}/console.log"
exit_code=${PIPESTATUS[0]}
set -e
echo "${exit_code}" > "${launch_dir}/exit_code.txt"
exit "${exit_code}"
