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
render_backend="${SIMVLA_LATENTLOOP_RENDER_BACKEND:-osmesa}"
arguments=("$@")
forwarded=()
for ((index = 0; index < ${#arguments[@]}; index++)); do
  case "${arguments[$index]}" in
    --seed)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--seed requires a value" >&2
        exit 2
      }
      legacy_seed="${arguments[$((index + 1))]}"
      forwarded+=("${arguments[$index]}" "${arguments[$((index + 1))]}")
      ((index += 1))
      ;;
    --seed=*)
      legacy_seed="${arguments[$index]#--seed=}"
      forwarded+=("${arguments[$index]}")
      ;;
    --experiment-seed)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--experiment-seed requires a value" >&2
        exit 2
      }
      experiment_seed="${arguments[$((index + 1))]}"
      forwarded+=("${arguments[$index]}" "${arguments[$((index + 1))]}")
      ((index += 1))
      ;;
    --experiment-seed=*)
      experiment_seed="${arguments[$index]#--experiment-seed=}"
      forwarded+=("${arguments[$index]}")
      ;;
    --render-backend)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--render-backend requires a value" >&2
        exit 2
      }
      render_backend="${arguments[$((index + 1))]}"
      ((index += 1))
      ;;
    --render-backend=*)
      render_backend="${arguments[$index]#--render-backend=}"
      ;;
    *)
      forwarded+=("${arguments[$index]}")
      ;;
  esac
done

case "$render_backend" in
  osmesa|egl) ;;
  *)
    echo "--render-backend must be osmesa or egl: $render_backend" >&2
    exit 2
    ;;
esac

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
export SIMVLA_LATENTLOOP_RENDER_BACKEND="$render_backend"
export MUJOCO_GL="$render_backend"
export PYOPENGL_PLATFORM="$render_backend"
if [[ "$render_backend" == "osmesa" ]]; then
  export GALLIUM_DRIVER=llvmpipe
  export LIBGL_ALWAYS_SOFTWARE=true
  export LP_NUM_THREADS=0
else
  unset GALLIUM_DRIVER LIBGL_ALWAYS_SOFTWARE LP_NUM_THREADS
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

case "$mode" in
  offline)
    exec python -m architectures.simvla.adapters.latentloop.offline_evaluator "${forwarded[@]}"
    ;;
  online)
    exec python -m architectures.simvla.adapters.latentloop.online_evaluator \
      --render-backend "$render_backend" "${forwarded[@]}"
    ;;
  aggregate)
    exec python -m architectures.simvla.adapters.latentloop.result_aggregator "${forwarded[@]}"
    ;;
  verify-repeat)
    exec python -m architectures.simvla.adapters.latentloop.repeat_verifier "${forwarded[@]}"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac
