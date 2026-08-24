#!/usr/bin/env bash
set -euo pipefail

ROOT=${SIMVLA_ROOT:-/home/mingyujung/private/gnaroshi_vla}
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
RESULT_BASE=${SIMVLA_RESULT_BASE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/correct_native_v0_seed20260815_v1}
OUT=${SIMVLA_KC2_DIAGNOSTIC_OUTPUT:-${RESULT_BASE}/11_kc2_diagnostic_150k_long500_v1}
CACHE=${SIMVLA_NATIVE_V0_CACHE:-${RESULT_BASE}/00_training_cache_libero10_r5}
V0_CHECKPOINT=${SIMVLA_NATIVE_V0_CHECKPOINT:-${RESULT_BASE}/08_train_150k/checkpoints/native_v0_step_150000.pt}
OFFLINE_GATE=${SIMVLA_NATIVE_V0_OFFLINE_GATE:-${RESULT_BASE}/09_offline_k4/simvla_v0_offline_gate.json}
PARITY_GATE=${SIMVLA_NATIVE_V0_PARITY_GATE:-${RESULT_BASE}/02_k1_parity/k1_parity.json}
NORM=${SIMVLA_NATIVE_V0_NORM:-${ROOT}/architectures/simvla/adapters/latentloop/assets/libero_norm_official_32700d0.json}
CHECKPOINT=${SIMVLA_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}
SMOLVLM=${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}
CANDIDATE_GPUS=${SIMVLA_KC2_GPUS:-4,5}
BASELINE_GPUS=${SIMVLA_BASELINE_GPUS:-6,7}

if [[ "${SIMVLA_KC2_DIAGNOSTIC_RUN:-0}" != "1" ]]; then
  echo "SIMVLA_KC2_DIAGNOSTIC_RUN=1 is required." >&2
  exit 2
fi
if [[ "${CANDIDATE_GPUS//[[:space:]]/}" != "4,5" || "${BASELINE_GPUS//[[:space:]]/}" != "6,7" ]]; then
  echo "This sd1 lane requires candidate GPUs 4,5 and baseline GPUs 6,7." >&2
  exit 2
fi

for required in "${PYTHON}" "${V0_CHECKPOINT}" "${OFFLINE_GATE}" "${PARITY_GATE}" "${CACHE}/manifest.json" "${NORM}"; do
  [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 1; }
done

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${ROOT}/architectures/simvla/upstream${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
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

"${PYTHON}" architectures/simvla/wrappers/simvla_two_gpu_guard.py \
  --gpu-ids "${CANDIDATE_GPUS}" --output "${OUT}/native_v0_k2_shards" --require-empty-output \
  --json "/tmp/simvla_native_v0_kc2_guard_${$}.json" >/dev/null
"${PYTHON}" architectures/simvla/wrappers/simvla_two_gpu_guard.py \
  --gpu-ids "${BASELINE_GPUS}" --output "${OUT}/baseline_k1_shards" --require-empty-output \
  --json "/tmp/simvla_native_v0_baseline_guard_${$}.json" >/dev/null

mkdir -p "${OUT}/logs"
exec > >(tee -a "${OUT}/launcher.log") 2>&1

echo "KC2_DIAGNOSTIC_START output=${OUT} candidate_gpus=${CANDIDATE_GPUS} baseline_gpus=${BASELINE_GPUS}"
SIMVLA_GPU_IDS="${CANDIDATE_GPUS}" CUDA_VISIBLE_DEVICES="${CANDIDATE_GPUS}" \
  "${PYTHON}" -m tools.simvla.native_v0_intermediate_eval manifest \
    --output "${OUT}/episode_manifest.json" \
    --cache "${CACHE}" \
    --v0-checkpoint "${V0_CHECKPOINT}" \
    --checkpoint "${CHECKPOINT}" \
    --norm-stats "${NORM}" \
    --parity-gate "${PARITY_GATE}" \
    --allow-final-diagnostic \
    --offline-gate "${OFFLINE_GATE}" \
    --candidate-gpu-ids "${CANDIDATE_GPUS}" \
    --baseline-gpu-ids "${BASELINE_GPUS}" \
    --rows-run-concurrently

set +e
(
  SIMVLA_GPU_IDS="${CANDIDATE_GPUS}" CUDA_VISIBLE_DEVICES="${CANDIDATE_GPUS}" \
    "${PYTHON}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node=2 --master-port=29645 \
      -m tools.simvla.native_v0_intermediate_eval evaluate \
      --row native_v0_k2 \
      --output "${OUT}/native_v0_k2_shards" \
      --manifest "${OUT}/episode_manifest.json" \
      --cache "${CACHE}" \
      --v0-checkpoint "${V0_CHECKPOINT}" \
      --checkpoint "${CHECKPOINT}" \
      --norm-stats "${NORM}" \
      --smolvlm-model "${SMOLVLM}" \
      --parity-gate "${PARITY_GATE}" \
      --allow-final-diagnostic \
      --save-video --video-failures-only --video-stride 2 --video-max-per-task 2
) > >(tee -a "${OUT}/logs/native_v0_k2.log") 2>&1 &
candidate_pid=$!

(
  SIMVLA_GPU_IDS="${BASELINE_GPUS}" CUDA_VISIBLE_DEVICES="${BASELINE_GPUS}" \
    "${PYTHON}" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc-per-node=2 --master-port=29646 \
      -m tools.simvla.native_v0_intermediate_eval evaluate \
      --row baseline_k1 \
      --output "${OUT}/baseline_k1_shards" \
      --manifest "${OUT}/episode_manifest.json" \
      --cache "${CACHE}" \
      --v0-checkpoint "${V0_CHECKPOINT}" \
      --checkpoint "${CHECKPOINT}" \
      --norm-stats "${NORM}" \
      --smolvlm-model "${SMOLVLM}" \
      --parity-gate "${PARITY_GATE}" \
      --allow-final-diagnostic
) > >(tee -a "${OUT}/logs/baseline_k1.log") 2>&1 &
baseline_pid=$!

wait "${candidate_pid}"
candidate_rc=$?
wait "${baseline_pid}"
baseline_rc=$?
set -e
if (( candidate_rc != 0 || baseline_rc != 0 )); then
  echo "KC2_DIAGNOSTIC_EVAL_FAILED candidate_rc=${candidate_rc} baseline_rc=${baseline_rc}" >&2
  exit 1
fi

"${PYTHON}" -m architectures.simvla.adapters.latentloop.native_v0_aggregate merge-row \
  --output "${OUT}/native_v0_k2_merged" \
  --row-root "${OUT}/native_v0_k2_shards" \
  --manifest "${OUT}/episode_manifest.json"
"${PYTHON}" -m architectures.simvla.adapters.latentloop.native_v0_aggregate merge-row \
  --output "${OUT}/baseline_k1_merged" \
  --row-root "${OUT}/baseline_k1_shards" \
  --manifest "${OUT}/episode_manifest.json"
"${PYTHON}" -m tools.simvla.native_v0_intermediate_eval compare \
  --output "${OUT}/comparison" \
  --manifest "${OUT}/episode_manifest.json" \
  --baseline-summary "${OUT}/baseline_k1_merged/row_summary.json" \
  --v0-summary "${OUT}/native_v0_k2_merged/row_summary.json"

echo "KC2_DIAGNOSTIC_COMPLETE output=${OUT}"
