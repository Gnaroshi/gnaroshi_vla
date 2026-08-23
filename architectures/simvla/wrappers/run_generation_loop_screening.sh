#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_GENERATION_RUN:-0}" != "1" ]]; then
  echo "Set SIMVLA_GENERATION_RUN=1 to enable Generation Loop screening." >&2
  exit 2
fi
if [[ -z "${SIMVLA_GPU_IDS:-}" ]]; then
  echo "SIMVLA_GPU_IDS=<gpu_a>,<gpu_b> is required." >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-${ROOT}/architectures/simvla/upstream}
if [[ ! -d "${UPSTREAM}/models" ]]; then
  echo "Set SIMVLA_UPSTREAM_ROOT to the released SimVLA checkout." >&2
  exit 2
fi

MODE=${1:---all-10k}
EXP=${SIMVLA_GENERATION_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/generation_loop_ng2_v1}
CACHE=${SIMVLA_EXACT_CACHE_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/03_exact_teacher_cache}
NORM=${SIMVLA_NORM_STATS:-${ROOT}/architectures/simvla/adapters/latentloop/assets/libero_norm_official_32700d0.json}
CHECKPOINT=${SIMVLA_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}
REVISION=${SIMVLA_CHECKPOINT_REVISION:-93dc4d90b0596c652ad2840ad743c62b9c4473fb}
SMOLVLM=${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}
TRAIN=${EXP}/train/ng2_schedule30k
OFFLINE10=${EXP}/offline/step_010000_ng3_ng2
ONLINE10=${EXP}/online/step_010000_long100
PORT=${SIMVLA_GENERATION_PORT:-29723}

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${UPSTREAM}:${UPSTREAM}/evaluation/libero/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"
export SIMVLA_UPSTREAM_ROOT="${UPSTREAM}"
export CUDA_VISIBLE_DEVICES=${SIMVLA_GPU_IDS//[[:space:]]/}
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=20260823
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

guard() {
  local output=$1
  "${PYTHON}" -m architectures.simvla.wrappers.simvla_two_gpu_guard \
    --gpu-ids "${SIMVLA_GPU_IDS}" \
    --output "${output}" \
    --require-empty-output >/dev/null
}

train_10k() {
  guard "${TRAIN}"
  "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 --master-port="${PORT}" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_train \
    --output "${TRAIN}" \
    --cache "${CACHE}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-revision "${REVISION}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}" \
    --n-g 2 \
    --stop-step 10000 \
    --schedule-total-steps 30000 \
    --save-interval 5000 \
    --wandb-project "${SIMVLA_GENERATION_WANDB_PROJECT:-gnaroshi-simvla-generation-loop}" \
    --wandb-name "${SIMVLA_GENERATION_WANDB_NAME:-simvla_generation_ng2_10k}"
}

offline_10k() {
  local checkpoint=${TRAIN}/checkpoints/generation_step_010000.pt
  [[ -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}" >&2; exit 1; }
  guard "${OFFLINE10}"
  "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 --master-port="$((PORT + 1))" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_offline \
    --output "${OFFLINE10}" \
    --cache "${CACHE}" \
    --generation-checkpoint "${checkpoint}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-revision "${REVISION}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}" \
    --queries 512
}

resume_30k() {
  local checkpoint=${TRAIN}/checkpoints/generation_step_010000.pt
  [[ -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}" >&2; exit 1; }
  "${PYTHON}" -m architectures.simvla.wrappers.simvla_two_gpu_guard \
    --gpu-ids "${SIMVLA_GPU_IDS}" >/dev/null
  "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 --master-port="${PORT}" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_train \
    --output "${TRAIN}" \
    --cache "${CACHE}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-revision "${REVISION}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}" \
    --n-g 2 \
    --stop-step 30000 \
    --schedule-total-steps 30000 \
    --resume "${checkpoint}" \
    --save-interval 5000 \
    --wandb-project "${SIMVLA_GENERATION_WANDB_PROJECT:-gnaroshi-simvla-generation-loop}" \
    --wandb-name "${SIMVLA_GENERATION_WANDB_NAME:-simvla_generation_ng2_30k_resume}"
}

online_10k() {
  local checkpoint=${TRAIN}/checkpoints/generation_step_010000.pt
  local offline=${OFFLINE10}/offline_screen.json
  local manifest=${ONLINE10}/episode_manifest.json
  [[ -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}" >&2; exit 1; }
  [[ -f "${offline}" ]] || { echo "Missing ${offline}" >&2; exit 1; }
  local candidate
  candidate=$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_n_g"])' "${offline}")
  [[ "${candidate}" == "2" || "${candidate}" == "3" ]] || {
    echo "Invalid offline candidate N_G=${candidate}" >&2
    exit 1
  }

  export GALLIUM_DRIVER=llvmpipe
  export LIBGL_ALWAYS_SOFTWARE=true
  export MUJOCO_GL=osmesa
  export PYOPENGL_PLATFORM=osmesa
  export PYTHONHASHSEED=20260815
  export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}
  export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  mkdir -p "$(dirname "${ONLINE10}")"
  "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_eval \
    manifest \
    --output "${manifest}" \
    --cache "${CACHE}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-revision "${REVISION}" \
    --norm-stats "${NORM}" \
    --trials-per-task 10

  guard "${ONLINE10}/baseline_k1"
  "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 --master-port="$((PORT + 2))" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_eval \
    evaluate \
    --row baseline_k1 \
    --output "${ONLINE10}/baseline_k1" \
    --manifest "${manifest}" \
    --cache "${CACHE}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-revision "${REVISION}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}"

  guard "${ONLINE10}/generation_ng${candidate}"
  "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 --master-port="$((PORT + 3))" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_eval \
    evaluate \
    --row "generation_ng${candidate}" \
    --output "${ONLINE10}/generation_ng${candidate}" \
    --manifest "${manifest}" \
    --cache "${CACHE}" \
    --generation-checkpoint "${checkpoint}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-revision "${REVISION}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}" \
    --save-video --video-failures-only
}

case "${MODE}" in
  --train-10k) train_10k ;;
  --offline-10k) offline_10k ;;
  --all-10k) train_10k; offline_10k ;;
  --screen-10k) train_10k; offline_10k; online_10k ;;
  --online-10k) online_10k ;;
  --resume-30k) resume_30k ;;
  *)
    echo "Usage: $0 [--train-10k|--offline-10k|--all-10k|--online-10k|--screen-10k|--resume-30k]" >&2
    exit 2
    ;;
esac
