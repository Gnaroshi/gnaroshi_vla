#!/usr/bin/env bash
set +e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
PY=${STORAGE}/envs/simvla/libero_mujoco237/bin/python
if [ "$(hostname -s)" != jbr-TRX50 ]; then
  printf 'RESIDUAL_EXIT=1: rb2 only\n'
  exit 0
fi
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 USE_TF=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=20260815
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
unset GALLIUM_DRIVER LIBGL_ALWAYS_SOFTWARE
export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER} MPLCONFIGDIR=/tmp/matplotlib-${USER}
export PYTHONPATH="${ROOT}:${STORAGE}/datasets/LIBERO"
export SIMVLA_LIBERO_ROOT="${STORAGE}/datasets/LIBERO"
export TORCHINDUCTOR_CACHE_DIR="${STORAGE}/results/simvla/compile_benchmark/paired_long_inputs_v1/compiler_cache"
export WANDB_MODE="${WANDB_MODE:-online}"
cd "${ROOT}" || exit 0
if [ "$#" -eq 0 ]; then set -- all; fi
"${PY}" -u -m tools.simvla.trend_residual_pipeline "$@"
rc=$?
printf '\nRESIDUAL_EXIT=%s (inspect pipeline_status.json; pane remains open)\n' "$rc"
exit 0
