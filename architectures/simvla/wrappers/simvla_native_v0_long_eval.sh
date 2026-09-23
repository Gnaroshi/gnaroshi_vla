#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
GUARD=${ROOT}/architectures/simvla/wrappers/simvla_two_gpu_guard.py

if [[ "${SIMVLA_NATIVE_V0_RUN:-0}" != "1" || -z "${SIMVLA_GPU_IDS:-}" ]]; then
  echo "SIMVLA_NATIVE_V0_RUN=1 and SIMVLA_GPU_IDS=<gpu_a>,<gpu_b> are required." >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <manifest|audit-baseline|evaluate|merge-row|paired-decision> [arguments]" >&2
  exit 2
fi
stage=$1
shift
args=("$@")
output=""
for ((index=0; index<${#args[@]}; index++)); do
  if [[ "${args[$index]}" == "--output" && $((index + 1)) -lt ${#args[@]} ]]; then
    output=${args[$((index + 1))]}
  fi
done
[[ -n "${output}" ]] || { echo "--output is required." >&2; exit 2; }

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/architectures/simvla/upstream${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
"${PYTHON}" "${GUARD}" --gpu-ids "${SIMVLA_GPU_IDS}" --output "${output}" --require-empty-output --json /tmp/simvla_native_v0_guard_long_${$}.json >/dev/null
export CUDA_VISIBLE_DEVICES="${SIMVLA_GPU_IDS//[[:space:]]/}"

if [[ "${stage}" == "evaluate" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export GALLIUM_DRIVER=llvmpipe
  export HF_HUB_OFFLINE=1
  export LIBGL_ALWAYS_SOFTWARE=true
  export LP_NUM_THREADS=0
  export MKL_NUM_THREADS=1
  export MUJOCO_GL=osmesa
  export NUMEXPR_NUM_THREADS=1
  export NVIDIA_TF32_OVERRIDE=0
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export PYOPENGL_PLATFORM=osmesa
  export PYTHONHASHSEED=20260815
  export TRANSFORMERS_OFFLINE=1
  export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}
  export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  port=${SIMVLA_DDP_PORT:-29624}
  exec "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 --master-port="${port}" \
    -m architectures.simvla.adapters.latentloop.native_v0_long_eval \
    evaluate "${args[@]}"
fi
if [[ "${stage}" == "merge-row" || "${stage}" == "paired-decision" ]]; then
  exec "${PYTHON}" -m architectures.simvla.adapters.latentloop.native_v0_aggregate \
    "${stage}" "${args[@]}"
fi
exec "${PYTHON}" -m architectures.simvla.adapters.latentloop.native_v0_long_eval \
  "${stage}" "${args[@]}"
