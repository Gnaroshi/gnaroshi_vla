#!/usr/bin/env bash

OPENPI_LL_ROOT=${OPENPI_LL_ROOT:-/home/mingyujung/private/gnaroshi_vla}
OPENPI_LL_UPSTREAM=${OPENPI_LL_ROOT}/architectures/openpi/upstream
OPENPI_LL_MAIN_PY=${OPENPI_LL_UPSTREAM}/.venv/bin/python
OPENPI_LL_CLIENT_PY=${OPENPI_LL_UPSTREAM}/examples/libero/.venv/bin/python
OPENPI_LL_SHARED=${OPENPI_LL_SHARED:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla}
OPENPI_LL_RESULTS=${OPENPI_LL_SHARED}/results/openpi/latentloop
OPENPI_LL_CACHE=${OPENPI_LL_SHARED}/cache/openpi/latentloop
OPENPI_LL_CHECKPOINT=${OPENPI_LL_CHECKPOINT:-${OPENPI_LL_SHARED}/results/openpi/checkpoints/pi05_libero_lora_pytorch/pi05_base_lora_r16_b16_4gpu_seed42_30k/30000}
OPENPI_LL_NORM=${OPENPI_LL_NORM:-${OPENPI_LL_SHARED}/results/openpi/assets/pi05_libero_lora_pytorch/physical-intelligence/libero/norm_stats.json}

export HF_HOME=${HF_HOME:-${OPENPI_LL_SHARED}/cache/openpi/huggingface}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-${OPENPI_LL_SHARED}/cache/openpi/lerobot}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export UV_CACHE_DIR=${UV_CACHE_DIR:-${OPENPI_LL_SHARED}/cache/openpi/uv}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${OPENPI_LL_SHARED}/cache/openpi/xdg}
export WANDB_DIR=${WANDB_DIR:-${OPENPI_LL_SHARED}/results/openpi/wandb}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# The pinned vendored LIBERO init-state files are trusted legacy NumPy pickles.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=${OPENPI_LL_ROOT}:${OPENPI_LL_UPSTREAM}:${OPENPI_LL_UPSTREAM}/src:${OPENPI_LL_UPSTREAM}/packages/openpi-client/src:${OPENPI_LL_UPSTREAM}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}

test -x "${OPENPI_LL_MAIN_PY}"
test -x "${OPENPI_LL_CLIENT_PY}"
test -f "${OPENPI_LL_CHECKPOINT}/model.safetensors"
test -f "${OPENPI_LL_NORM}"

openpi_ll_refuse_nonempty() {
  local target=$1
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite or reuse output: ${target}" >&2
    return 1
  fi
}

openpi_ll_wait_server() {
  local port=$1
  local attempts=${2:-180}
  local index
  for ((index=0; index<attempts; index++)); do
    if "${OPENPI_LL_MAIN_PY}" -c "import socket; s=socket.socket(); s.settimeout(.2); raise SystemExit(0 if s.connect_ex(('127.0.0.1', ${port})) == 0 else 1)"; then
      return 0
    fi
    sleep 2
  done
  echo "Policy server on port ${port} did not become ready." >&2
  return 1
}
