#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
MODULE=architectures.simvla.adapters.latentloop.native_v0_prepare
GUARD=${ROOT}/architectures/simvla/wrappers/simvla_two_gpu_guard.py

if [[ "${SIMVLA_NATIVE_V0_RUN:-0}" != "1" ]]; then
  echo "Set SIMVLA_NATIVE_V0_RUN=1 to enable this new, default-off workflow." >&2
  exit 2
fi
if [[ -z "${SIMVLA_GPU_IDS:-}" ]]; then
  echo "SIMVLA_GPU_IDS=<gpu_a>,<gpu_b> is required." >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <audit|parity|token-analysis|parameters|mode-ab|calibrate> [arguments]" >&2
  exit 2
fi

stage=$1
shift
args=("$@")
output=""
for ((index=0; index<${#args[@]}; index++)); do
  if [[ "${args[$index]}" == "--output" && $((index + 1)) -lt ${#args[@]} ]]; then
    output=${args[$((index + 1))]}
    break
  fi
done
if [[ -z "${output}" ]]; then
  echo "--output is required." >&2
  exit 2
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/architectures/simvla/upstream${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=20260815
export TRANSFORMERS_OFFLINE=1

guard_json=/tmp/simvla_native_v0_guard_prepare_${$}.json
trap 'rm -f "${guard_json}"' EXIT
"${PYTHON}" "${GUARD}" \
  --gpu-ids "${SIMVLA_GPU_IDS}" \
  --output "${output}" \
  --require-empty-output \
  --json "${guard_json}" >/dev/null
rm -f "${guard_json}"
trap - EXIT

IFS=',' read -r gpu_a gpu_b <<<"${SIMVLA_GPU_IDS}"
gpu_a=${gpu_a//[[:space:]]/}
gpu_b=${gpu_b//[[:space:]]/}

if [[ "${stage}" != "mode-ab" ]]; then
  export CUDA_VISIBLE_DEVICES="${gpu_a},${gpu_b}"
  exec "${PYTHON}" -m "${MODULE}" "${stage}" "${args[@]}"
fi

mkdir -p "${output}"
child_args=()
skip_next=0
for ((index=0; index<${#args[@]}; index++)); do
  if [[ ${skip_next} -eq 1 ]]; then
    skip_next=0
    continue
  fi
  if [[ "${args[$index]}" == "--output" ]]; then
    skip_next=1
    continue
  fi
  child_args+=("${args[$index]}")
done

CUDA_VISIBLE_DEVICES="${gpu_a}" "${PYTHON}" -m "${MODULE}" mode-ab \
  --output "${output}/gpu_${gpu_a}" "${child_args[@]}" \
  >"${output}/gpu_${gpu_a}.log" 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES="${gpu_b}" "${PYTHON}" -m "${MODULE}" mode-ab \
  --output "${output}/gpu_${gpu_b}" "${child_args[@]}" \
  >"${output}/gpu_${gpu_b}.log" 2>&1 &
pid_b=$!

status=0
wait "${pid_a}" || status=1
wait "${pid_b}" || status=1
if [[ ${status} -ne 0 ]]; then
  echo "Mode A/B benchmark failed; inspect ${output}/gpu_*.log" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_a},${gpu_b}"
"${PYTHON}" -m "${MODULE}" mode-ab-decide \
  --output "${output}/decision" \
  --local-report "${output}/gpu_${gpu_a}/mode_ab_local.json" \
  --local-report "${output}/gpu_${gpu_b}/mode_ab_local.json"
