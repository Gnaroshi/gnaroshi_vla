#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${PYTHON:-python}
GPU_ID=${FASTV_GPU_ID:-0}
OUTPUT_ROOT=${FASTV_OUTPUT_ROOT:?Set FASTV_OUTPUT_ROOT to a non-overlay result directory.}
RUN_ID=${FASTV_RUN_ID:-fastv_k2_r50_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${OUTPUT_ROOT}/${RUN_ID}

export SIMVLA_UPSTREAM_ROOT=${SIMVLA_UPSTREAM_ROOT:-"${ROOT}/architectures/simvla/upstream"}
export FASTV_UPSTREAM_ROOT=${FASTV_UPSTREAM_ROOT:-"${ROOT}/architectures/fastv/upstream"}
export LIBERO_ROOT=${LIBERO_ROOT:-"${SIMVLA_UPSTREAM_ROOT}/evaluation/libero/LIBERO"}
export HF_HOME=${HF_HOME:-"${ROOT}/.cache/huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}
export PYTHONPATH="${ROOT}:${SIMVLA_UPSTREAM_ROOT}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CHECKPOINT=${FASTV_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}
NORM_STATS=${FASTV_NORM_STATS:-"${SIMVLA_UPSTREAM_ROOT}/norm_stats/libero_norm.json"}

if [[ ${SIMVLA_FASTV_LONG_RUN:-0} != 1 ]]; then
  echo "Set SIMVLA_FASTV_LONG_RUN=1 after reviewing this full LIBERO-Long run." >&2
  exit 2
fi
if [[ -e ${RUN_ROOT} ]]; then
  echo "Refusing existing run directory: ${RUN_ROOT}" >&2
  exit 2
fi

test -x "${PYTHON}"
test -f "${SIMVLA_UPSTREAM_ROOT}/models/modeling_smolvlm_vla.py"
test -f "${NORM_STATS}"
test -d "${FASTV_UPSTREAM_ROOT}/.git"
test -d "${LIBERO_ROOT}/libero"

mkdir -p "${RUN_ROOT}/logs" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"
LOG=${RUN_ROOT}/logs/launcher.log
STATUS=${RUN_ROOT}/run.status

on_exit() {
  local rc=$?
  if [[ ${rc} == 0 ]]; then
    printf 'FASTV_LIBERO_LONG_COMPLETE\n' > "${STATUS}"
  else
    printf 'FASTV_LIBERO_LONG_FAILED rc=%s\n' "${rc}" > "${STATUS}"
  fi
  echo "Status: ${STATUS}"
  echo "Log: ${LOG}"
}
trap on_exit EXIT
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date --iso-8601=seconds)] FastV LIBERO-Long pipeline start"
echo "run_root=${RUN_ROOT}"
echo "physical_gpu=${GPU_ID}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader -i "${GPU_ID}"
"${PYTHON}" -c \
  'import torch,mujoco,transformers; print("versions", torch.__version__, mujoco.__version__, transformers.__version__)'

echo "[1/4] Source and scientific-contract preflight"
PYTHON="${PYTHON}" \
bash "${ROOT}/architectures/simvla/wrappers/simvla_fastv_eval.sh" \
  --contract-only \
  --output "${RUN_ROOT}/contract"

echo "[2/4] Real-checkpoint condition and latency smoke"
SIMVLA_FASTV_SMOKE_RUN=1 PYTHON="${PYTHON}" \
bash "${ROOT}/architectures/simvla/wrappers/simvla_fastv_smoke.sh" \
  --output "${RUN_ROOT}/real_checkpoint_smoke" \
  --checkpoint "${CHECKPOINT}" \
  --iterations 10 \
  --device cuda

echo "[3/4] One-task, one-trial paired LIBERO smoke"
SIMVLA_FASTV_EVAL_RUN=1 PYTHON="${PYTHON}" \
bash "${ROOT}/architectures/simvla/wrappers/simvla_fastv_eval.sh" \
  --output "${RUN_ROOT}/bounded_smoke" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM_STATS}" \
  --suite libero_10 \
  --rows baseline_k1 fastv_k2_r50 \
  --max-tasks 1 \
  --num-trials 1 \
  --device cuda

echo "[4/4] Primary paired LIBERO-Long evaluation: 500 episodes per row"
SIMVLA_FASTV_EVAL_RUN=1 PYTHON="${PYTHON}" \
bash "${ROOT}/architectures/simvla/wrappers/simvla_fastv_eval.sh" \
  --output "${RUN_ROOT}/libero_10_50trials" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM_STATS}" \
  --suite libero_10 \
  --rows baseline_k1 fastv_k2_r50 \
  --num-trials 50 \
  --environment-seed 0 \
  --action-noise-seed-base 20260902 \
  --save-video \
  --video-failures-only \
  --device cuda

echo "FASTV_LIBERO_LONG_COMPLETE"
