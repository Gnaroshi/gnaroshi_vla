#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
GUARD=${ROOT}/architectures/simvla/wrappers/simvla_two_gpu_guard.py

if [[ "${SIMVLA_NATIVE_V0_RUN:-0}" != "1" ]]; then
  echo "Set SIMVLA_NATIVE_V0_RUN=1 to enable this new, default-off workflow." >&2
  exit 2
fi
if [[ -z "${SIMVLA_GPU_IDS:-}" ]]; then
  echo "SIMVLA_GPU_IDS=<gpu_a>,<gpu_b> is required." >&2
  exit 2
fi
args=("$@")
output=""
resume=0
for ((index=0; index<${#args[@]}; index++)); do
  [[ "${args[$index]}" == "--resume" ]] && resume=1
  if [[ "${args[$index]}" == "--output" && $((index + 1)) -lt ${#args[@]} ]]; then
    output=${args[$((index + 1))]}
  fi
done
if [[ -z "${output}" ]]; then
  echo "--output is required." >&2
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
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=20260815
export TRANSFORMERS_OFFLINE=1

guard_args=(--gpu-ids "${SIMVLA_GPU_IDS}" --output "${output}")
if [[ ${resume} -eq 0 ]]; then
  guard_args+=(--require-empty-output)
fi
guard_json=/tmp/simvla_native_v0_guard_train_${$}.json
trap 'rm -f "${guard_json}"' EXIT
"${PYTHON}" "${GUARD}" "${guard_args[@]}" --json "${guard_json}" >/dev/null
rm -f "${guard_json}"
trap - EXIT
export CUDA_VISIBLE_DEVICES="${SIMVLA_GPU_IDS//[[:space:]]/}"
port=${SIMVLA_DDP_PORT:-29622}
exec "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  --master-port="${port}" \
  -m architectures.simvla.adapters.latentloop.native_v0_train \
  "${args[@]}"
