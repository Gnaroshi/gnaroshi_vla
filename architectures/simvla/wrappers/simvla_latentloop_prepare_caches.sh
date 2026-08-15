#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
CONDA_ENV=${SIMVLA_LATENTLOOP_CONDA_ENV:-simvla_libero}
PIPELINE_TAG=${SIMVLA_LATENTLOOP_CACHE_TAG:-20260804_chunkaware_v3}
GPU_LIST=${SIMVLA_LATENTLOOP_GPUS:-4,5,6,7}
CHECKPOINT=${SIMVLA_LATENTLOOP_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}
NORM_STATS=${SIMVLA_LATENTLOOP_NORM_STATS:-${ROOT}/architectures/simvla/upstream/norm_stats/libero_norm.json}
SHARED_ROOT=${SIMVLA_LATENTLOOP_SHARED_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla}
CACHE_ROOT=${SIMVLA_LATENTLOOP_CACHE_ROOT:-${SHARED_ROOT}/results/simvla/latentloop/${PIPELINE_TAG}/cache}
RUN_ROOT=${SIMVLA_LATENTLOOP_RUN_ROOT:-${ROOT}/results/simvla/latentloop/${PIPELINE_TAG}/cache_pipeline}
RENDER_BACKEND=${SIMVLA_LATENTLOOP_RENDER_BACKEND:-egl}
EXPERIMENT_SEED=${SIMVLA_LATENTLOOP_EXPERIMENT_SEED:-}

case "${RENDER_BACKEND}" in
  osmesa|egl) ;;
  *)
    echo "SIMVLA_LATENTLOOP_RENDER_BACKEND must be osmesa or egl: ${RENDER_BACKEND}" >&2
    exit 2
    ;;
esac

export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=${RENDER_BACKEND}
export PYOPENGL_PLATFORM=${RENDER_BACKEND}
if [[ "${RENDER_BACKEND}" == "osmesa" ]]; then
  export GALLIUM_DRIVER=llvmpipe
  export LIBGL_ALWAYS_SOFTWARE=true
  export LP_NUM_THREADS=0
else
  unset GALLIUM_DRIVER LIBGL_ALWAYS_SOFTWARE LP_NUM_THREADS
fi
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export SIMVLA_LATENTLOOP_CACHE_RUN=1
export PYTHONUNBUFFERED=1

pipeline_args=(
  --execute
  --cache-root "${CACHE_ROOT}"
  --run-root "${RUN_ROOT}"
  --required-cache-prefix "${SHARED_ROOT}"
  --conda-env "${CONDA_ENV}"
  --gpus "${GPU_LIST}"
  --checkpoint "${CHECKPOINT}"
  --norm-stats "${NORM_STATS}"
  --suite libero_10
  --pilot-trials "${SIMVLA_LATENTLOOP_PILOT_TRIALS:-2}"
  --production-trials "${SIMVLA_LATENTLOOP_PRODUCTION_TRIALS:-20}"
  --max-tasks 10
  --r1-max-policy-queries 900
  --r5-max-policy-queries 180
  --num-wait-steps 10
  --flow-steps 10
  --client-resize-size 224
  --image-size 384
  --resolution 256
  --control-hz 20
  --seed 7
  --action-noise-seed-base 20260804
  --render-backend "${RENDER_BACKEND}"
  --task-order official_reverse
  --records-per-shard 128
  --max-production-cache-gib "${SIMVLA_LATENTLOOP_MAX_CACHE_GIB:-200}"
  --min-free-after-gib "${SIMVLA_LATENTLOOP_MIN_FREE_GIB:-200}"
  --projection-safety-factor "${SIMVLA_LATENTLOOP_PROJECTION_FACTOR:-1.5}"
  --max-initial-gpu-memory-mib "${SIMVLA_LATENTLOOP_MAX_GPU_MEMORY_MIB:-1024}"
)
if [[ -n "${EXPERIMENT_SEED}" ]]; then
  pipeline_args+=(--experiment-seed "${EXPERIMENT_SEED}")
fi
if [[ "${SIMVLA_LATENTLOOP_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  pipeline_args+=(--preflight-only)
fi

cd "${ROOT}"
if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  exec python -m architectures.simvla.adapters.latentloop.query_cache_pipeline "${pipeline_args[@]}"
fi

CONDA_BIN=${CONDA_EXE:-/home/mingyujung/miniconda3/bin/conda}
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 2
fi
exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -m architectures.simvla.adapters.latentloop.query_cache_pipeline "${pipeline_args[@]}"
