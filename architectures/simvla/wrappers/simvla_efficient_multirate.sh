#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
GPU_GUARD=architectures.simvla.adapters.latentloop.efficient_multirate.gpu_contract

usage() {
  cat >&2 <<'EOF'
Usage: simvla_efficient_multirate.sh <stage-command> --output PATH [arguments]

Stage commands:
  source-lock       create a new efficient-campaign source lock
  cache-project     exact FP32 storage projection
  cache-pilot       bounded full-VLM/action cache identity pilot
  cache-generate    approved two-rank production cache generation
  cache-validate    offline shard/checksum validation
  mode-ab           adopt source-identical parent Mode A/B evidence after cache identity
  batch             select the only effective-global-batch-one configuration
  benchmark         bounded cache-backed Mode B or Mode D training benchmark
  mode-d            compare completed bounded Mode B and Mode D benchmarks
  mode-d-not-required  record that measured Mode B already meets 12 hours
  wallclock         make the measured 150K wall-clock decision
  train             cache-backed V0 smoke or approved 150K training
  offline           child-source-compatible unchanged native V0 offline gate
  long              child-source-compatible unchanged native V0 Long stage
  generation-audit  hidden-hook parity and naive NFE audit
EOF
  exit 2
}

if [[ "${SIMVLA_EFFICIENT_MULTIRATE_RUN:-0}" != "1" ]]; then
  echo "Set SIMVLA_EFFICIENT_MULTIRATE_RUN=1 to enable this default-off workflow." >&2
  exit 2
fi
if [[ -z "${SIMVLA_GPU_IDS:-}" ]]; then
  echo "SIMVLA_GPU_IDS=<gpu_a>,<gpu_b> is required." >&2
  exit 2
fi
[[ $# -ge 1 ]] || usage
command_name=$1
shift

args=("$@")
output=""
resume_requested=0
for ((index=0; index<${#args[@]}; index++)); do
  if [[ "${args[$index]}" == "--output" && $((index + 1)) -lt ${#args[@]} ]]; then
    output=${args[$((index + 1))]}
  fi
  if [[ "${args[$index]}" == "--resume" && $((index + 1)) -lt ${#args[@]} && -n "${args[$((index + 1))]}" ]]; then
    resume_requested=1
  fi
done
if [[ -z "${output}" ]]; then
  echo "Every stage command requires an explicit --output path." >&2
  exit 2
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/architectures/simvla/upstream${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=20260822

gpu_contract=$(mktemp /tmp/simvla_efficient_gpu_contract.XXXXXX.json)
trap 'rm -f "${gpu_contract}"' EXIT
guard_args=(--gpu-ids "${SIMVLA_GPU_IDS}" --output "${output}" --json "${gpu_contract}")
if [[ ${resume_requested} -eq 0 ]]; then
  guard_args+=(--require-absent-output)
fi
"${PYTHON}" -m "${GPU_GUARD}" "${guard_args[@]}" >/dev/null
export SIMVLA_GPU_CONTRACT_JSON=${gpu_contract}
export CUDA_VISIBLE_DEVICES=${SIMVLA_GPU_IDS//[[:space:]]/}

case "${command_name}" in
  source-lock)
    "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.decisions \
      source-lock "${args[@]}"
    ;;
  cache-project)
    "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache \
      project "${args[@]}"
    ;;
  cache-pilot)
    "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache \
      pilot "${args[@]}" --device cuda:0
    ;;
  cache-generate)
    if [[ "${SIMVLA_EXACT_CACHE_GENERATION_APPROVED:-0}" != "1" ]]; then
      echo "Set SIMVLA_EXACT_CACHE_GENERATION_APPROVED=1 after approving the cache pilot." >&2
      exit 2
    fi
    "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache \
      generate "${args[@]}"
    ;;
  cache-validate)
    "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache \
      validate "${args[@]}"
    ;;
  mode-ab|batch|mode-d|mode-d-not-required|wallclock)
    "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.decisions \
      "${command_name}" "${args[@]}"
    ;;
  benchmark|train)
    "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.efficient_v0_train \
      "${args[@]}" --gpu-contract "${gpu_contract}"
    ;;
  offline)
    "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.efficient_v0_offline \
      "${args[@]}"
    ;;
  long)
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export GALLIUM_DRIVER=llvmpipe
    export HF_HUB_OFFLINE=1
    export LIBGL_ALWAYS_SOFTWARE=true
    export LP_NUM_THREADS=0
    export MKL_NUM_THREADS=1
    export MUJOCO_GL=osmesa
    export NUMEXPR_NUM_THREADS=1
    export NVIDIA_TF32_OVERRIDE=0
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export PYOPENGL_PLATFORM=osmesa
    export PYTHONHASHSEED=20260815
    export TRANSFORMERS_OFFLINE=1
    export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}
    export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    if [[ "${args[0]:-}" == "evaluate" ]]; then
      "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
        -m architectures.simvla.adapters.latentloop.efficient_multirate.efficient_v0_long_eval \
        "${args[@]}"
    else
      "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.efficient_v0_long_eval \
        "${args[@]}"
    fi
    ;;
  generation-audit)
    "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_audit \
      "${args[@]}"
    ;;
  *)
    usage
    ;;
esac
