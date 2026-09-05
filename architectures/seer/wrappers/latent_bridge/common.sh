#!/usr/bin/env bash

set -euo pipefail

LATENT_BRIDGE_WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATENT_BRIDGE_REPO_ROOT="$(cd "${LATENT_BRIDGE_WRAPPER_DIR}/../../../.." && pwd)"
LATENT_BRIDGE_SEER_UPSTREAM="${LATENT_BRIDGE_REPO_ROOT}/architectures/seer/upstream"
LATENT_BRIDGE_OFFICIAL_SOURCE="${LATENT_BRIDGE_REPO_ROOT}/architectures/latent_bridge/upstream"

LATENT_BRIDGE_BASE_CHECKPOINT="${LATENT_BRIDGE_BASE_CHECKPOINT:-${LATENT_BRIDGE_PUBLIC33:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/checkpoints_Seer_LIBERO_LONG/Seer/33.pth}}"
LATENT_BRIDGE_PUBLIC33="${LATENT_BRIDGE_BASE_CHECKPOINT}"
LATENT_BRIDGE_BASE_CHECKPOINT_SHA256="${LATENT_BRIDGE_BASE_CHECKPOINT_SHA256:-a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646}"
LATENT_BRIDGE_VIT="${LATENT_BRIDGE_VIT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
LATENT_BRIDGE_DATASET_ROOT="${LATENT_BRIDGE_DATASET_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/seer_node2/LIBERO_DATASETS/libero_10_converted}"
LATENT_BRIDGE_DATASET_NAME="${LATENT_BRIDGE_DATASET_NAME:-libero_10_converted}"
LATENT_BRIDGE_DATASET_INFO="${LATENT_BRIDGE_DATASET_INFO:-${LATENT_BRIDGE_SEER_UPSTREAM}/data_info/libero_10_converted.json}"
LATENT_BRIDGE_LIBERO_PATH="${LATENT_BRIDGE_LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
LATENT_BRIDGE_RESULT_ROOT="${LATENT_BRIDGE_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/latent_bridge/public33_libero_long}"
LATENT_BRIDGE_RENDERER="${LATENT_BRIDGE_RENDERER:-osmesa}"
LATENT_BRIDGE_SUITE="${LATENT_BRIDGE_SUITE:-libero_10}"
LATENT_BRIDGE_RUN_LABEL="${LATENT_BRIDGE_RUN_LABEL:-seer_latent_bridge}"
LATENT_BRIDGE_EXPECTED_CONDA_ENV="${LATENT_BRIDGE_EXPECTED_CONDA_ENV:-seer_libero}"
LATENT_BRIDGE_EXPECTED_CUDA_VISIBLE_DEVICES="${LATENT_BRIDGE_EXPECTED_CUDA_VISIBLE_DEVICES:-4,5,6,7}"

case "${LATENT_BRIDGE_RENDERER}" in
    egl|osmesa) ;;
    *) echo "[ERROR] LATENT_BRIDGE_RENDERER must be egl or osmesa" >&2; exit 1 ;;
esac

export PYTHONPATH="${LATENT_BRIDGE_REPO_ROOT}:${PYTHONPATH:-}"
export LATENT_BRIDGE_REPO_ROOT LATENT_BRIDGE_SEER_UPSTREAM
export LATENT_BRIDGE_BASE_CHECKPOINT LATENT_BRIDGE_BASE_CHECKPOINT_SHA256
export LATENT_BRIDGE_DATASET_ROOT LATENT_BRIDGE_DATASET_NAME LATENT_BRIDGE_DATASET_INFO
export LATENT_BRIDGE_LIBERO_PATH LATENT_BRIDGE_RESULT_ROOT LATENT_BRIDGE_RENDERER
export LATENT_BRIDGE_SUITE LATENT_BRIDGE_RUN_LABEL
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export LATENT_BRIDGE_RENDERER
export PYOPENGL_PLATFORM="${LATENT_BRIDGE_RENDERER}"
export MUJOCO_GL="${LATENT_BRIDGE_RENDERER}"
export LIBERO_GL_BACKEND="${LATENT_BRIDGE_RENDERER}"
export SEER_LATENT_BRIDGE_BASE_CHECKPOINT_SHA256="${LATENT_BRIDGE_BASE_CHECKPOINT_SHA256}"
if [[ "${LATENT_BRIDGE_RENDERER}" == "egl" ]]; then
    export LIBERO_GL_REQUIRE_ACTUAL=1
else
    export LIBERO_GL_REQUIRE_ACTUAL=0
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

latent_bridge_fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

latent_bridge_require_base_runtime() {
    [[ "${CONDA_DEFAULT_ENV:-}" == "${LATENT_BRIDGE_EXPECTED_CONDA_ENV}" ]] || \
        latent_bridge_fail "activate conda environment ${LATENT_BRIDGE_EXPECTED_CONDA_ENV} first"
    [[ "${CUDA_VISIBLE_DEVICES:-}" == "${LATENT_BRIDGE_EXPECTED_CUDA_VISIBLE_DEVICES}" ]] || \
        latent_bridge_fail "CUDA_VISIBLE_DEVICES must be ${LATENT_BRIDGE_EXPECTED_CUDA_VISIBLE_DEVICES}"
    [[ -s "${LATENT_BRIDGE_BASE_CHECKPOINT}" ]] || \
        latent_bridge_fail "missing base checkpoint: ${LATENT_BRIDGE_BASE_CHECKPOINT}"
    [[ "$(sha256sum "${LATENT_BRIDGE_BASE_CHECKPOINT}" | awk '{print $1}')" == \
        "${LATENT_BRIDGE_BASE_CHECKPOINT_SHA256}" ]] || \
        latent_bridge_fail "base checkpoint SHA256 mismatch: ${LATENT_BRIDGE_BASE_CHECKPOINT}"
    [[ -s "${LATENT_BRIDGE_VIT}" ]] || latent_bridge_fail "missing ViT-MAE: ${LATENT_BRIDGE_VIT}"
    [[ -s "${LATENT_BRIDGE_DATASET_ROOT}/${LATENT_BRIDGE_DATASET_NAME}/meta_info.h5" ]] || \
        latent_bridge_fail "invalid converted dataset root: ${LATENT_BRIDGE_DATASET_ROOT}"
    [[ -s "${LATENT_BRIDGE_DATASET_INFO}" ]] || \
        latent_bridge_fail "missing converted dataset info: ${LATENT_BRIDGE_DATASET_INFO}"
    [[ -d "${LATENT_BRIDGE_LIBERO_PATH}/libero/libero" ]] || \
        latent_bridge_fail "invalid LIBERO path: ${LATENT_BRIDGE_LIBERO_PATH}"
    [[ -e "${LATENT_BRIDGE_OFFICIAL_SOURCE}/.git" ]] || \
        latent_bridge_fail "official Latent Bridge source is absent"
    [[ "${NODE_NUM:-4}" == "4" ]] || \
        latent_bridge_fail "the locked training/evaluation contract requires NODE_NUM=4"
}

latent_bridge_require_runtime() {
    latent_bridge_require_base_runtime
}

latent_bridge_next_backup() {
    local source_root="$1"
    local attempt=1
    local backup
    while true; do
        printf -v backup '%s.failed_attempt_%03d' "${source_root}" "${attempt}"
        if [[ ! -e "${backup}" ]]; then
            printf '%s\n' "${backup}"
            return 0
        fi
        attempt=$((attempt + 1))
    done
}

latent_bridge_archive_partial() {
    local source_root="$1"
    [[ -e "${source_root}" ]] || return 0
    local backup
    backup="$(latent_bridge_next_backup "${source_root}")"
    mv "${source_root}" "${backup}"
    echo "[RECOVERY] preserved partial output: ${backup}"
}

latent_bridge_prepare_stage() {
    local stage_root="$1"
    local recovery_mode="${2:-restart}"
    if [[ -s "${stage_root}/COMPLETE" ]]; then
        echo "[SKIP] completed stage: ${stage_root}"
        return 1
    fi
    if [[ -e "${stage_root}" ]]; then
        case "${recovery_mode}" in
            resume)
                echo "[RECOVERY] resuming partial stage: ${stage_root}"
                return 0
                ;;
            restart)
                if [[ "${LATENT_BRIDGE_RETRY_PARTIAL:-0}" != "1" ]]; then
                    latent_bridge_fail \
                        "partial stage exists (set LATENT_BRIDGE_RETRY_PARTIAL=1): ${stage_root}"
                fi
                latent_bridge_archive_partial "${stage_root}"
                ;;
            *) latent_bridge_fail "unknown recovery mode: ${recovery_mode}" ;;
        esac
    fi
    mkdir -p "${stage_root}"
    return 0
}

latent_bridge_json_has_status() {
    local path="$1"
    local expected_status="$2"
    [[ -s "${path}" ]] || return 1
    python -c \
        'import json, sys; payload=json.load(open(sys.argv[1])); raise SystemExit(0 if payload.get("status") == sys.argv[2] else 1)' \
        "${path}" "${expected_status}" >/dev/null 2>&1
}

latent_bridge_eval_row_is_complete() {
    local output_root="$1"
    local seed="$2"
    local episodes_per_task="$3"
    local num_tasks="$4"
    [[ -s "${output_root}/EVAL_COMPLETE" ]] || return 1
    python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/validate_eval_row.py" \
        --root "${output_root}" --seed "${seed}" \
        --episodes-per-task "${episodes_per_task}" --num-tasks "${num_tasks}" \
        --renderer "${LATENT_BRIDGE_RENDERER}" --suite "${LATENT_BRIDGE_SUITE}" >/dev/null
}

latent_bridge_prepare_eval_row() {
    local output_root="$1"
    local seed="$2"
    local episodes_per_task="$3"
    local num_tasks="$4"
    if latent_bridge_eval_row_is_complete \
        "${output_root}" "${seed}" "${episodes_per_task}" "${num_tasks}"; then
        echo "[SKIP] validated evaluation row: ${output_root}"
        return 1
    fi
    if [[ -e "${output_root}" ]]; then
        latent_bridge_archive_partial "${output_root}"
    fi
    mkdir -p "${output_root}/analysis"
    return 0
}

latent_bridge_common_eval_args() {
    local seed="$1"
    printf '%s\0' \
        --rgb_pad 10 --gripper_pad 4 --traj_cons \
        --gradient_accumulation_steps 1 \
        --bf16_module vision_encoder \
        --vit_checkpoint_path "${LATENT_BRIDGE_VIT}" \
        --libero_path "${LATENT_BRIDGE_LIBERO_PATH}" \
        --calvin_dataset "" --workers 16 --lr_scheduler cosine \
        --save_every_iter 50000 --num_epochs 20 --seed "${seed}" \
        --batch_size 64 --precision fp32 --weight_decay 1e-4 \
        --num_resampler_query 6 --num_obs_token_per_image 9 \
        --transformer_layers 24 --transformer_heads 12 --hidden_dim 384 \
        --phase evaluate --finetune_type "${LATENT_BRIDGE_SUITE}" \
        --save_checkpoint_path "${LATENT_BRIDGE_RESULT_ROOT}/unused_checkpoints" \
        --action_pred_steps 3 --future_steps 3 --sequence_length 7 \
        --obs_pred --gripper_width --eval_libero_ensembling \
        --ensembling_temp 0.01 --multi_step_action 1 \
        --lrnode_eval_profile_full_action_head 1 \
        --libero_img_size 128 --libero_eval_max_steps 600 \
        --resume_from_checkpoint "${LATENT_BRIDGE_BASE_CHECKPOINT}"
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
    export CKPT_TAG="${LATENT_BRIDGE_RUN_LABEL}"
    export SEER_LATENT_BRIDGE_BASE_CHECKPOINT="${LATENT_BRIDGE_BASE_CHECKPOINT}"
    export EVAL_CONTROL_HZ=20
    export EVAL_NUM_EPISODES_PER_TASK="${episodes_per_task}"
    export EVAL_NUM_TASKS="${num_tasks}"
    export SAVE_VIDEO=0
    local eval_rc=0
    set +e
    if [[ -n "${entry_module}" ]]; then
        python -m torch.distributed.run \
            --nnodes=1 --nproc_per_node="${node_num}" --master_port="${master_port}" \
            --module "${entry_module}" "${args[@]}" 2>&1 | tee "${output_root}/run.log"
        eval_rc="${PIPESTATUS[0]}"
    else
        python -m torch.distributed.run \
            --nnodes=1 --nproc_per_node="${node_num}" --master_port="${master_port}" \
            "${LATENT_BRIDGE_SEER_UPSTREAM}/eval_libero.py" "${args[@]}" 2>&1 | \
            tee "${output_root}/run.log"
        eval_rc="${PIPESTATUS[0]}"
    fi
    set -e
    if python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/validate_eval_row.py" \
        --root "${output_root}" --seed "${seed}" \
        --episodes-per-task "${episodes_per_task}" --num-tasks "${num_tasks}" \
        --renderer "${LATENT_BRIDGE_RENDERER}" --suite "${LATENT_BRIDGE_SUITE}" \
        --write-complete; then
        if [[ "${eval_rc}" -ne 0 ]]; then
            echo "[WARN] evaluator exited rc=${eval_rc}, but all expected artifacts passed validation"
        fi
        return 0
    fi
    [[ "${eval_rc}" -eq 0 ]] || \
        latent_bridge_fail "evaluation exited rc=${eval_rc} with incomplete artifacts: ${output_root}"
    latent_bridge_fail "evaluation artifacts failed validation: ${output_root}"
}
