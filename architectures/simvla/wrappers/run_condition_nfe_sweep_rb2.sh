#!/usr/bin/env bash
# Preserve the invoking tmux shell; the status file records the actual exit code.
set +e
if [ "$(hostname -s)" != "jbr-TRX50" ]; then
  printf 'CONDITION_NFE_EXIT=1: rb2 only\n'
  exit 0
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
PY=${STORAGE}/envs/simvla/libero_mujoco237/bin/python
OUT=${STORAGE}/results/simvla/condition_nfe/compiled_seed01_v1
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
export PYTHONPATH="${ROOT}:${STORAGE}/datasets/LIBERO"
export LIBERO_CONFIG_PATH="${STORAGE}/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config"
export SIMVLA_LIBERO_ROOT="${STORAGE}/datasets/LIBERO"
export TORCHINDUCTOR_CACHE_DIR="${STORAGE}/results/simvla/compile_benchmark/paired_long_inputs_v1/compiler_cache"
cd "${ROOT}" || exit 0
mkdir -p "${OUT}/logs" || exit 0
mode=${1:-all}
mode=${mode#--}
set -o pipefail
"${PY}" -u -m tools.simvla.condition_nfe_sweep "${mode}" 2>&1 | tee -a "${OUT}/logs/launcher.log"
rc=$?
printf '%s\n' "${rc}" > "${OUT}/logs/launcher.status"
printf '\nCONDITION_NFE_EXIT=%s (tmux pane remains open)\n' "${rc}"
exit 0
