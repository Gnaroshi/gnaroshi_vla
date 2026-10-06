#!/usr/bin/env bash
set +e
if [ "$(hostname -s)" != "jbr-TRX50" ]; then
  printf 'PRO_EXIT=1: rb2 only\n'
  exit 0
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
PY=${STORAGE}/envs/simvla/libero_mujoco237/bin/python
OUT=${STORAGE}/results/simvla/libero_pro/long_position_task_seed01_v1
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
unset GALLIUM_DRIVER LIBGL_ALWAYS_SOFTWARE
export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER}
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
export PYTHONPATH="${ROOT}:${STORAGE}/datasets/LIBERO-PRO/upstream"
export LIBERO_CONFIG_PATH="${STORAGE}/datasets/LIBERO-PRO/runtime_config"
export SIMVLA_LIBERO_ROOT="${STORAGE}/datasets/LIBERO-PRO/upstream"
export TORCHINDUCTOR_CACHE_DIR="${STORAGE}/results/simvla/compile_benchmark/paired_long_inputs_v1/compiler_cache"
cd "${ROOT}" || exit 0
mkdir -p "${OUT}/logs" || exit 0
mode=${1:-all}
mode=${mode#--}
printf 'RUNNING\n' > "${OUT}/logs/launcher.status"
set -o pipefail
"${PY}" -u -m tools.simvla.libero_pro_zero_shot "${mode}" 2>&1 | tee -a "${OUT}/logs/launcher.log"
rc=$?
printf '%s\n' "${rc}" > "${OUT}/logs/launcher.status"
printf '\nPRO_EXIT=%s (tmux pane remains open)\n' "${rc}"
exit 0
