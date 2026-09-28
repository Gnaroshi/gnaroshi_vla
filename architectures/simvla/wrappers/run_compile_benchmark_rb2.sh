#!/usr/bin/env bash
# Run in an existing pane; never exit/replace the caller's interactive shell.
set +e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${SIMVLA_PYTHON:-/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}" || { printf 'COMPILE_BENCHMARK_FAILED: repository missing\n'; exit 0; }
if [ "$#" -eq 0 ]; then set -- all; fi
"${PY}" -u tools/simvla/compile_benchmark.py "$@"
rc=$?
printf '\nCOMPILE_BENCHMARK_EXIT=%s (the existing pane remains open)\n' "$rc"
exit 0
