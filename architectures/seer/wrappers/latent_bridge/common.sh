#!/usr/bin/env bash

set -euo pipefail

LATENT_BRIDGE_WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATENT_BRIDGE_REPO_ROOT="$(cd "${LATENT_BRIDGE_WRAPPER_DIR}/../../../.." && pwd)"
LATENT_BRIDGE_SEER_UPSTREAM="${LATENT_BRIDGE_REPO_ROOT}/architectures/seer/upstream"
LATENT_BRIDGE_OFFICIAL_SOURCE="${LATENT_BRIDGE_REPO_ROOT}/architectures/latent_bridge/upstream"

LATENT_BRIDGE_PUBLIC33="${LATENT_BRIDGE_PUBLIC33:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/checkpoints_Seer_LIBERO_LONG/Seer/33.pth}"
LATENT_BRIDGE_VIT="${LATENT_BRIDGE_VIT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
LATENT_BRIDGE_DATASET_ROOT="${LATENT_BRIDGE_DATASET_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/seer_node2/LIBERO_DATASETS/libero_10_converted}"
LATENT_BRIDGE_LIBERO_PATH="${LATENT_BRIDGE_LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
LATENT_BRIDGE_RESULT_ROOT="${LATENT_BRIDGE_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/latent_bridge/public33_libero_long}"

export PYTHONPATH="${LATENT_BRIDGE_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export LIBERO_GL_BACKEND=osmesa
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

latent_bridge_fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

latent_bridge_require_runtime() {
    [[ "${CONDA_DEFAULT_ENV:-}" == "seer_libero" ]] || \
        latent_bridge_fail "activate conda environment seer_libero first"
    [[ "${CUDA_VISIBLE_DEVICES:-}" == "4,5,6,7" ]] || \
        latent_bridge_fail "CUDA_VISIBLE_DEVICES must be exactly 4,5,6,7 on sd1"
    [[ -s "${LATENT_BRIDGE_PUBLIC33}" ]] || latent_bridge_fail "missing public33: ${LATENT_BRIDGE_PUBLIC33}"
    [[ -s "${LATENT_BRIDGE_VIT}" ]] || latent_bridge_fail "missing ViT-MAE: ${LATENT_BRIDGE_VIT}"
    [[ -s "${LATENT_BRIDGE_DATASET_ROOT}/libero_10_converted/meta_info.h5" ]] || \
        latent_bridge_fail "invalid converted dataset root: ${LATENT_BRIDGE_DATASET_ROOT}"
    [[ -d "${LATENT_BRIDGE_LIBERO_PATH}/libero/libero" ]] || \
        latent_bridge_fail "invalid LIBERO path: ${LATENT_BRIDGE_LIBERO_PATH}"
    [[ -e "${LATENT_BRIDGE_OFFICIAL_SOURCE}/.git" ]] || \
        latent_bridge_fail "official Latent Bridge source is absent"
}

latent_bridge_prepare_stage() {
    local stage_root="$1"
    if [[ -s "${stage_root}/COMPLETE" ]]; then
        echo "[SKIP] completed stage: ${stage_root}"
        return 1
    fi
    if [[ -e "${stage_root}" ]]; then
        if [[ "${LATENT_BRIDGE_RETRY_PARTIAL:-0}" != "1" ]]; then
            latent_bridge_fail "partial stage exists (set LATENT_BRIDGE_RETRY_PARTIAL=1): ${stage_root}"
        fi
        local backup="${stage_root}.failed_attempt"
        [[ ! -e "${backup}" ]] || latent_bridge_fail "retry backup already exists: ${backup}"
        mv "${stage_root}" "${backup}"
    fi
    mkdir -p "${stage_root}"
    return 0
}

latent_bridge_common_eval_args() {
    local seed="$1"
    printf '%s\0' \
        --rgb_pad -1 --gripper_pad -1 \
        --gradient_accumulation_steps 1 \
        --bf16_module vision_encoder \
        --vit_checkpoint_path "${LATENT_BRIDGE_VIT}" \
        --libero_path "${LATENT_BRIDGE_LIBERO_PATH}" \
        --calvin_dataset "" --workers 16 --lr_scheduler cosine \
        --save_every_iter 50000 --num_epochs 20 --seed "${seed}" \
        --batch_size 64 --precision fp32 --weight_decay 1e-4 \
        --num_resampler_query 6 --num_obs_token_per_image 9 \
        --transformer_layers 24 --transformer_heads 12 --hidden_dim 384 \
        --phase evaluate --finetune_type libero_10 \
        --save_checkpoint_path "${LATENT_BRIDGE_RESULT_ROOT}/unused_checkpoints" \
        --action_pred_steps 3 --future_steps 3 --sequence_length 7 \
        --obs_pred --gripper_width --eval_libero_ensembling \
        --ensembling_temp 0.01 --multi_step_action 1 \
        --lrnode_eval_profile_full_action_head 1 \
        --libero_img_size 224 --libero_eval_max_steps 600 \
        --resume_from_checkpoint "${LATENT_BRIDGE_PUBLIC33}"
}

latent_bridge_run_eval() {
    local entry_module="$1"
    local output_root="$2"
    local run_name="$3"
    local seed="$4"
    local episodes_per_task="$5"
    local num_tasks="$6"
    local master_port="$7"
    local node_num="${NODE_NUM:-4}"
    local args=()
    while IFS= read -r -d '' item; do args+=("${item}"); done < <(latent_bridge_common_eval_args "${seed}")
    mkdir -p "${output_root}/analysis"
    export LOG_DIR="${output_root}"
    export RUN_NAME="${run_name}"
    export CKPT_TAG="public33"
    export SEER_LATENT_BRIDGE_BASE_CHECKPOINT="${LATENT_BRIDGE_PUBLIC33}"
    export EVAL_CONTROL_HZ=20
    export EVAL_NUM_EPISODES_PER_TASK="${episodes_per_task}"
    export EVAL_NUM_TASKS="${num_tasks}"
    export SAVE_VIDEO=0
    if [[ -n "${entry_module}" ]]; then
        python -m torch.distributed.run \
            --nnodes=1 --nproc_per_node="${node_num}" --master_port="${master_port}" \
            --module "${entry_module}" "${args[@]}" 2>&1 | tee "${output_root}/run.log"
    else
        python -m torch.distributed.run \
            --nnodes=1 --nproc_per_node="${node_num}" --master_port="${master_port}" \
            "${LATENT_BRIDGE_SEER_UPSTREAM}/eval_libero.py" "${args[@]}" 2>&1 | \
            tee "${output_root}/run.log"
    fi
    [[ -s "${output_root}/analysis/eval_summary.json" ]] || \
        latent_bridge_fail "evaluation summary missing: ${output_root}"
    [[ -s "${output_root}/analysis/eval_episode_metrics.csv" ]] || \
        latent_bridge_fail "episode metrics missing: ${output_root}"
}
