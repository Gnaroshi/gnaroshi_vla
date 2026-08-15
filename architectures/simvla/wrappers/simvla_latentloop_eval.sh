#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_EVAL_RUN:-0}" != "1" ]]; then
  echo "Refusing evaluation/aggregation: set SIMVLA_LATENTLOOP_EVAL_RUN=1 explicitly." >&2
  exit 2
fi

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 {offline|online|aggregate|verify-repeat} [arguments...]" >&2
  exit 2
fi

mode="$1"
shift

legacy_seed=7
experiment_seed=""
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[$index]}" in
    --seed)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--seed requires a value" >&2
        exit 2
      }
      legacy_seed="${arguments[$((index + 1))]}"
      ;;
    --seed=*)
      legacy_seed="${arguments[$index]#--seed=}"
      ;;
    --experiment-seed)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--experiment-seed requires a value" >&2
        exit 2
      }
      experiment_seed="${arguments[$((index + 1))]}"
      ;;
    --experiment-seed=*)
      experiment_seed="${arguments[$index]#--experiment-seed=}"
      ;;
  esac
done

process_seed="${experiment_seed:-$legacy_seed}"
[[ "$process_seed" =~ ^[0-9]+$ ]] && ((10#$process_seed <= 4294967295)) || {
  echo "evaluation seed must be an integer in [0, 4294967295]: $process_seed" >&2
  exit 2
}

# These values must exist before Python initializes hashing, cuBLAS, or CPU pools.
export PYTHONHASHSEED="$process_seed"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

case "$mode" in
  offline)
    exec python -m architectures.simvla.adapters.latentloop.offline_evaluator "$@"
    ;;
  online)
    exec python -m architectures.simvla.adapters.latentloop.online_evaluator "$@"
    ;;
  aggregate)
    exec python -m architectures.simvla.adapters.latentloop.result_aggregator "$@"
    ;;
  verify-repeat)
    exec python -m architectures.simvla.adapters.latentloop.repeat_verifier "$@"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac
