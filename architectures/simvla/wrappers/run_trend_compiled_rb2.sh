#!/usr/bin/env bash
# Return to the caller's existing tmux pane, including on failure.
set +e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
PY=${STORAGE}/envs/simvla/libero_mujoco237/bin/python
if [ "$(hostname -s)" != "jbr-TRX50" ]; then
  printf 'TREND_EVAL_EXIT=1: rb2 only\n'
  exit 0
fi
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
unset GALLIUM_DRIVER LIBGL_ALWAYS_SOFTWARE
export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER} MPLCONFIGDIR=/tmp/matplotlib-${USER}
export PYTHONPATH="${ROOT}:${STORAGE}/datasets/LIBERO"
export SIMVLA_LIBERO_ROOT="${STORAGE}/datasets/LIBERO"
export LIBERO_CONFIG_PATH="${STORAGE}/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config"
export TORCHINDUCTOR_CACHE_DIR="${STORAGE}/results/simvla/compile_benchmark/paired_long_inputs_v1/compiler_cache"
cd "${ROOT}" || exit 0
if [ "$#" -eq 0 ]; then set -- all; fi
MODULE=tools.simvla.trend_compiled_rb2
if [ "${1:-}" = "--bridge-sweep" ]; then
  MODULE=tools.simvla.bridge_interval_sweep
  shift
  if [ "$#" -eq 0 ]; then set -- all; fi
fi
if [ "${1:-}" = "--rollout-round2" ]; then
  MODULE=tools.simvla.rollout_round2_rb2
  shift
  if [ "$#" -eq 0 ]; then set -- all; fi
fi
if [ "${1:-}" = "--rollout-repair" ]; then
  MODULE=tools.simvla.rollout_repair_rb2
  shift
  if [ "$#" -eq 0 ]; then set -- all; fi
fi
if [ "${1:-}" = "--controls" ]; then
  MODULE=tools.simvla.trend_controls_rb2
  shift
  if [ "$#" -eq 0 ]; then set -- all; fi
fi
"${PY}" -u -m "${MODULE}" "$@"
rc=$?
printf '\nTREND_EVAL_EXIT=%s (tmux pane remains open; inspect status.json)\n' "$rc"
exit 0
