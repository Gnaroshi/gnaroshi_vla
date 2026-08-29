#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
RESULT_BASE=${SIMVLA_RESULT_BASE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/correct_native_v0_seed20260815_v1}
TRAIN_ROOT=${SIMVLA_NATIVE_V0_TRAIN_ROOT:-${RESULT_BASE}/08_train_150k}
CACHE=${SIMVLA_NATIVE_V0_CACHE:-${RESULT_BASE}/00_training_cache_libero10_r5}
PARITY_GATE=${SIMVLA_NATIVE_V0_PARITY_GATE:-${RESULT_BASE}/02_k1_parity/k1_parity.json}
NORM=${SIMVLA_NATIVE_V0_NORM:-${ROOT}/architectures/simvla/adapters/latentloop/assets/libero_norm_official_32700d0.json}
CHECKPOINT=${SIMVLA_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}
SMOLVLM=${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}
GPU_IDS=${SIMVLA_GPU_IDS:-6,7}

if [[ "${SIMVLA_INTERMEDIATE_EVAL_RUN:-0}" != "1" ]]; then
  echo "SIMVLA_INTERMEDIATE_EVAL_RUN=1 is required." >&2
  exit 2
fi
if [[ "${GPU_IDS//[[:space:]]/}" != "6,7" ]]; then
  echo "This sd1 diagnostic lane is reserved for physical GPUs 6,7; got ${GPU_IDS}." >&2
  exit 2
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/architectures/simvla/upstream${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false

LATEST_FILE=${TRAIN_ROOT}/latest_checkpoint.txt
[[ -s "${LATEST_FILE}" ]] || { echo "Missing latest checkpoint pointer: ${LATEST_FILE}" >&2; exit 1; }
V0_CHECKPOINT=$(<"${LATEST_FILE}")
[[ -f "${V0_CHECKPOINT}" ]] || { echo "Missing checkpoint: ${V0_CHECKPOINT}" >&2; exit 1; }
STEP=$("${PYTHON}" -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["global_optimizer_step"]))' "${V0_CHECKPOINT}")
if (( STEP <= 0 || STEP >= 150000 )); then
  echo "Expected a non-final intermediate checkpoint, got step=${STEP}." >&2
  exit 1
fi
OUT=${SIMVLA_INTERMEDIATE_EVAL_OUTPUT:-${RESULT_BASE}/intermediate_diagnostics/step_$(printf '%06d' "${STEP}")_long500}

"${PYTHON}" architectures/simvla/wrappers/simvla_two_gpu_guard.py \
  --gpu-ids "${GPU_IDS}" --output "${OUT}" --require-empty-output \
  --json "/tmp/simvla_condition_v0_intermediate_guard_${$}.json" >/dev/null

export SIMVLA_GPU_IDS="${GPU_IDS}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
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

mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

echo "INTERMEDIATE_DIAGNOSTIC_START step=${STEP} gpus=${GPU_IDS} output=${OUT}"
"${PYTHON}" -m tools.simvla.native_v0_intermediate_eval manifest \
  --output "${OUT}/episode_manifest.json" \
  --cache "${CACHE}" \
  --v0-checkpoint "${V0_CHECKPOINT}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM}" \
  --parity-gate "${PARITY_GATE}"

echo "PHASE_1_OURS_K4_START"
"${PYTHON}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --master-port=29635 \
  -m tools.simvla.native_v0_intermediate_eval evaluate \
  --row native_v0_k4 \
  --output "${OUT}/ours_k4_shards" \
  --manifest "${OUT}/episode_manifest.json" \
  --cache "${CACHE}" \
  --v0-checkpoint "${V0_CHECKPOINT}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM}" \
  --smolvlm-model "${SMOLVLM}" \
  --parity-gate "${PARITY_GATE}" \
  --save-video --video-failures-only --video-stride 2 --video-max-per-task 2
"${PYTHON}" -m architectures.simvla.adapters.latentloop.native_v0_aggregate merge-row \
  --output "${OUT}/ours_k4_merged" \
  --row-root "${OUT}/ours_k4_shards" \
  --manifest "${OUT}/episode_manifest.json"

echo "PHASE_2_MATCHED_BASELINE_K1_START"
"${PYTHON}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 --master-port=29636 \
  -m tools.simvla.native_v0_intermediate_eval evaluate \
  --row baseline_k1 \
  --output "${OUT}/baseline_k1_shards" \
  --manifest "${OUT}/episode_manifest.json" \
  --cache "${CACHE}" \
  --v0-checkpoint "${V0_CHECKPOINT}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM}" \
  --smolvlm-model "${SMOLVLM}" \
  --parity-gate "${PARITY_GATE}"
"${PYTHON}" -m architectures.simvla.adapters.latentloop.native_v0_aggregate merge-row \
  --output "${OUT}/baseline_k1_merged" \
  --row-root "${OUT}/baseline_k1_shards" \
  --manifest "${OUT}/episode_manifest.json"

"${PYTHON}" -m tools.simvla.native_v0_intermediate_eval compare \
  --output "${OUT}/comparison" \
  --manifest "${OUT}/episode_manifest.json" \
  --baseline-summary "${OUT}/baseline_k1_merged/row_summary.json" \
  --v0-summary "${OUT}/ours_k4_merged/row_summary.json"

echo "INTERMEDIATE_DIAGNOSTIC_COMPLETE step=${STEP} output=${OUT}"
