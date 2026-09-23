#!/bin/bash

set -euo pipefail

# Frozen-baseline LR-NODE distill/adapter protocol.
# This is NOT the main from-scratch comparison. It is an isolation/adapter
# experiment:
#   - an existing Seer baseline checkpoint is loaded
#   - every non-LR-NODE module is frozen
#   - only lrnode_delta_encoder + lrnode_dynamics are trained

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
protocol_root="${LRNODE_PROTOCOL_ROOT:-${REPO_ROOT}/results/seer/lrnode/default}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-${protocol_root}/train/distill_node/}"
dataset="${DATASET:-libero_10_converted}"
root_dir="${ROOT_DIR:-${UPSTREAM_DIR}/LIBERO_DATASETS/${dataset}}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-${UPSTREAM_DIR}/checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
libero_path="${LIBERO_PATH:-}"
calvin_dataset_path="${CALVIN_DATASET_PATH:-calvin/dataset/task_ABC_D}"

baseline_env="${BASELINE_ENV:-${protocol_root}/train/_latest/scratch.env}"
if [[ -z "${BASELINE_CKPT:-}" ]]; then
    if [[ ! -f "${baseline_env}" ]]; then
        echo "[ERROR] BASELINE_CKPT is not set and baseline env does not exist: ${baseline_env}" >&2
        echo "[ERROR] Run scripts/LIBERO_LONG/Seer/scratch.sh first, evaluate it, then set BASELINE_CKPT_ID to the best checkpoint." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${baseline_env}"
    BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-${LRNODE_RUN_NAME}}"
    BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-${LRNODE_SAVE_CHECKPOINT_PATH}}"
    BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-33}"
    BASELINE_CKPT="${BASELINE_CKPT_ROOT}/${BASELINE_RUN_NAME}/${BASELINE_CKPT_ID}.pth"
else
    BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-$(basename "$(dirname "${BASELINE_CKPT}")")}"
    BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-$(dirname "$(dirname "${BASELINE_CKPT}")")}"
    BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-$(basename "${BASELINE_CKPT}" .pth)}"
fi

which_server="${WHICH_SERVER:-sd1}"
method_tag="${METHOD_TAG:-lrnode_distill_from_scratch_baseline_ckpt${BASELINE_CKPT_ID}_lronly_v1_lw05_aw01_g4}"
experiment_tag="${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}"
run_name="${RUN_NAME:-${which_server}_distill_node_${method_tag}_${experiment_tag}}"
num_epochs="${NUM_EPOCHS:-40}"
# train.py uses the strict condition `epoch > start_save_checkpoint`.
# A value of 25 therefore saves 26.pth through the final 39.pth.
start_save_checkpoint="${START_SAVE_CHECKPOINT:-25}"
if [[ ! "${num_epochs}" =~ ^[0-9]+$ || ! "${start_save_checkpoint}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] NUM_EPOCHS and START_SAVE_CHECKPOINT must be non-negative integers." >&2
    exit 1
fi
if (( start_save_checkpoint >= num_epochs - 1 )); then
    echo "[ERROR] No checkpoint would be saved: NUM_EPOCHS=${num_epochs}, START_SAVE_CHECKPOINT=${start_save_checkpoint}." >&2
    exit 1
fi
export EXPERIMENT_TAG="${experiment_tag}"
export RUN_NAME="${run_name}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
latest_dir="${protocol_root}/train/_latest"

echo "[TRAIN INFO] script=distill_node.sh"
echo "[TRAIN INFO] protocol_root=${protocol_root}"
echo "[TRAIN INFO] save_checkpoint_path=${save_checkpoint_path}"
echo "[TRAIN INFO] experiment_tag=${EXPERIMENT_TAG}"
echo "[TRAIN INFO] run_name=${RUN_NAME}"
echo "[TRAIN INFO] baseline_ckpt=${BASELINE_CKPT}"
echo "[TRAIN INFO] checkpoint_epochs=$((start_save_checkpoint + 1))-$((num_epochs - 1))"
echo "[TRAIN INFO] live_loss=console_tqdm, wandb, protocol tee log"
echo "[TRAIN INFO] latest_pointer_after_success=${latest_dir}/distill_node.env"

if [[ ! -f "${BASELINE_CKPT}" ]]; then
    echo "[ERROR] Missing baseline checkpoint: ${BASELINE_CKPT}" >&2
    exit 1
fi
if [[ ! -d "${root_dir}/${dataset}" ]]; then
    echo "[ERROR] Missing converted dataset: ${root_dir}/${dataset}" >&2
    echo "[ERROR] Set ROOT_DIR to the parent directory containing ${dataset}/." >&2
    exit 1
fi
if [[ ! -f "${vit_checkpoint_path}" ]]; then
    echo "[ERROR] Missing ViT checkpoint: ${vit_checkpoint_path}" >&2
    echo "[ERROR] Set VIT_CHECKPOINT_PATH explicitly." >&2
    exit 1
fi
if [[ -z "${libero_path}" || ! -d "${libero_path}" ]]; then
    echo "[ERROR] Set LIBERO_PATH to a valid LIBERO repository path." >&2
    exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${REPO_ROOT}:${UPSTREAM_DIR}:${libero_path}:${PYTHONPATH:-}"
master_port="${MASTER_PORT:-12423}"
node=1
node_num="${NODE_NUM:-4}"

LRNODE_EXTRA_ARGS=(
    --use_lrnode_latent_update 1
    --lrnode_train_latent_distill "${LRNODE_TRAIN_LATENT_DISTILL:-1}"
    --lrnode_teacher_target_mode "${LRNODE_TEACHER_TARGET_MODE:-shifted_context}"
    --lrnode_context_selected_step "${LRNODE_CONTEXT_SELECTED_STEP:--1}"
    --lrnode_train_protocol adapter
    --lrnode_freeze_seer_for_adapter 1
    --lrnode_assert_only_lrnode_trainable 1
    --lrnode_latent_weight "${LRNODE_LATENT_WEIGHT:-0.05}"
    --lrnode_action_distill_weight "${LRNODE_ACTION_DISTILL_WEIGHT:-0.1}"
    --lrnode_bc_weight "${LRNODE_BC_WEIGHT:-0.0}"
    --lrnode_smooth_weight "${LRNODE_SMOOTH_WEIGHT:-0.001}"
    --lrnode_hidden_dim "${LRNODE_HIDDEN_DIM:-256}"
    --lrnode_motion_dim "${LRNODE_MOTION_DIM:-128}"
    --lrnode_fast_encoder_type "${LRNODE_FAST_ENCODER_TYPE:-diffcnn}"
    --lrnode_detach_input_latent "${LRNODE_DETACH_INPUT_LATENT:-1}"
    --lrnode_detach_teacher_latent "${LRNODE_DETACH_TEACHER_LATENT:-1}"
    --lrnode_freeze_action_head_for_lrnode "${LRNODE_FREEZE_ACTION_HEAD_FOR_LRNODE:-1}"
    --lrnode_use_post_layernorm "${LRNODE_USE_POST_LAYERNORM:-0}"
    --lrnode_multistep_train "${LRNODE_MULTISTEP_TRAIN:-0}"
    --lrnode_train_max_horizon "${LRNODE_TRAIN_MAX_HORIZON:-2}"
    --lrnode_runtime_aligned_train "${LRNODE_RUNTIME_ALIGNED_TRAIN:-0}"
    --lrnode_runtime_horizon "${LRNODE_RUNTIME_HORIZON:-3}"
    --lrnode_runtime_age3_weight "${LRNODE_RUNTIME_AGE3_WEIGHT:-2.0}"
    --lrnode_frozen_teacher_eval_mode "${LRNODE_FROZEN_TEACHER_EVAL_MODE:-0}"
    --lrnode_gripper_distill_weight "${LRNODE_GRIPPER_DISTILL_WEIGHT:-0.0}"
    --lrnode_overlap_weight "${LRNODE_OVERLAP_WEIGHT:-0.0}"
    --lrnode_ensemble_weight "${LRNODE_ENSEMBLE_WEIGHT:-0.0}"
    --lrnode_gripper_switch_weight "${LRNODE_GRIPPER_SWITCH_WEIGHT:-0.0}"
    --lrnode_runtime_ensemble_temp "${LRNODE_RUNTIME_ENSEMBLE_TEMP:-0.01}"
    --lrnode_log_sanity "${LRNODE_LOG_SANITY:-1}"
    --lrnode_gate_init_bias "${LRNODE_GATE_INIT_BIAS:--4.0}"
    --lrnode_trace "${LRNODE_TRACE:-0}"
    --lrnode_debug_artifact_interval "${LRNODE_DEBUG_ARTIFACT_INTERVAL:-1000}"
    --latentloop_plan_adapter_mode "${LATENTLOOP_PLAN_ADAPTER_MODE:-off}"
    --latentloop_plan_adapter_hidden_dim "${LATENTLOOP_PLAN_ADAPTER_HIDDEN_DIM:-0}"
    --latentloop_plan_parameter_match_tolerance "${LATENTLOOP_PLAN_PARAMETER_MATCH_TOLERANCE:-0.05}"
    --latentloop_plan_arm_weight "${LATENTLOOP_PLAN_ARM_WEIGHT:-1.0}"
    --latentloop_plan_gripper_weight "${LATENTLOOP_PLAN_GRIPPER_WEIGHT:-1.0}"
    --latentloop_plan_latent_weight "${LATENTLOOP_PLAN_LATENT_WEIGHT:-1.0}"
    --latentloop_comparison_protocol "${LATENTLOOP_COMPARISON_PROTOCOL:-0}"
    --latentloop_comparison_offset_schedule "${LATENTLOOP_COMPARISON_OFFSET_SCHEDULE:-adjacent}"
    --latentloop_comparison_split_role "${LATENTLOOP_COMPARISON_SPLIT_ROLE:-full}"
    --latentloop_comparison_validation_fraction "${LATENTLOOP_COMPARISON_VALIDATION_FRACTION:-0.05}"
    --latentloop_comparison_validation_seed "${LATENTLOOP_COMPARISON_VALIDATION_SEED:-20260805}"
    --latentloop_comparison_target_microbatches "${LATENTLOOP_COMPARISON_TARGET_MICROBATCHES:-0}"
    --latentloop_comparison_warmup_microbatches "${LATENTLOOP_COMPARISON_WARMUP_MICROBATCHES:-0}"
    --latentloop_comparison_checkpoint_microbatches "${LATENTLOOP_COMPARISON_CHECKPOINT_MICROBATCHES:-0}"
    --latentloop_action_arm_weight "${LATENTLOOP_ACTION_ARM_WEIGHT:-0.0}"
    --latentloop_action_gripper_weight "${LATENTLOOP_ACTION_GRIPPER_WEIGHT:-0.0}"
    --latentloop_action_exec_weight "${LATENTLOOP_ACTION_EXEC_WEIGHT:-0.0}"
    --latentloop_action_reg_weight "${LATENTLOOP_ACTION_REG_WEIGHT:-0.0}"
    --latentloop_nonrecurrent_latent_weight "${LATENTLOOP_NONRECURRENT_LATENT_WEIGHT:-0.0}"
    --latentloop_nonrecurrent_action_weight "${LATENTLOOP_NONRECURRENT_ACTION_WEIGHT:-0.0}"
    --latentloop_nonrecurrent_smooth_weight "${LATENTLOOP_NONRECURRENT_SMOOTH_WEIGHT:-0.0}"
    --latentloop_comparison_selection_metric "${LATENTLOOP_COMPARISON_SELECTION_METRIC:-validation_total_loss}"
    --latentloop_cqpc_weight "${LATENTLOOP_CQPC_WEIGHT:-0.0}"
    --latentloop_cqpc_gamma "${LATENTLOOP_CQPC_GAMMA:-0.0}"
    --latentloop_cqpc_arm_weight "${LATENTLOOP_CQPC_ARM_WEIGHT:-1.0}"
    --latentloop_cqpc_gripper_weight "${LATENTLOOP_CQPC_GRIPPER_WEIGHT:-1.0}"
    --latentloop_cqpc_log_teacher_disagreement "${LATENTLOOP_CQPC_LOG_TEACHER_DISAGREEMENT:-0}"
)

if [[ -n "${LRNODE_INIT_ADAPTER_CKPT:-}" ]]; then
    LRNODE_EXTRA_ARGS+=(--lrnode_init_adapter_ckpt "${LRNODE_INIT_ADAPTER_CKPT}")
fi

WANDB_ARGS=()
if [[ "${REPORT_TO_WANDB:-1}" == "1" ]]; then
    WANDB_ARGS+=(--report_to_wandb)
fi

DATASET_INFO_ARGS=()
if [[ -n "${LIBERO_DATASET_INFO_PATH:-}" ]]; then
    DATASET_INFO_ARGS+=(--libero_dataset_info_path "${LIBERO_DATASET_INFO_PATH}")
fi

cd "${UPSTREAM_DIR}"
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=${master_port} train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path "${vit_checkpoint_path}" \
    --calvin_dataset "${calvin_dataset_path}" \
    --workers "${WORKERS:-4}" \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs "${num_epochs}" \
    --seed "${SEED:-42}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --precision fp32 \
    --learning_rate "${LEARNING_RATE:-1e-3}" \
    --save_checkpoint \
    --finetune_from_pretrained_ckpt "${BASELINE_CKPT}" \
    --finetune_type libero_finetune \
    --libero_dataset_name "${dataset}" \
    "${DATASET_INFO_ARGS[@]}" \
    --root_dir "${root_dir}" \
    --wandb_project "${WANDB_PROJECT:-seer}" \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name "${run_name}" \
    --save_checkpoint_path "${save_checkpoint_path}" \
    --transformer_layers 24 \
    --phase "finetune" \
    --obs_pred \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint "${start_save_checkpoint}" \
    --gripper_width \
    --warmup_epochs "${WARMUP_EPOCHS:-2}" \
    --libero_path "${libero_path}" \
    --multi_step_action 1 \
    "${WANDB_ARGS[@]}" \
    "${LRNODE_EXTRA_ARGS[@]}"

mkdir -p "${latest_dir}"
cat > "${latest_dir}/distill_node.env" <<EOF
LRNODE_PROTOCOL_SCRIPT=distill_node.sh
LRNODE_PROTOCOL_KIND=distill
LRNODE_MODULE=1
LRNODE_COUPLING=frozen_teacher_adapter
LRNODE_JOINT=0
LRNODE_BACKPROP_TO_SEER_FROM_LRNODE=0
LRNODE_TEACHER_TARGET_MODE=${LRNODE_TEACHER_TARGET_MODE:-shifted_context}
LRNODE_CONTEXT_SELECTED_STEP=${LRNODE_CONTEXT_SELECTED_STEP:--1}
LRNODE_RUNTIME_ALIGNED_TRAIN=${LRNODE_RUNTIME_ALIGNED_TRAIN:-0}
LRNODE_RUNTIME_HORIZON=${LRNODE_RUNTIME_HORIZON:-3}
LRNODE_RUNTIME_AGE3_WEIGHT=${LRNODE_RUNTIME_AGE3_WEIGHT:-2.0}
LRNODE_FROZEN_TEACHER_EVAL_MODE=${LRNODE_FROZEN_TEACHER_EVAL_MODE:-0}
LRNODE_GRIPPER_DISTILL_WEIGHT=${LRNODE_GRIPPER_DISTILL_WEIGHT:-0.0}
LRNODE_OVERLAP_WEIGHT=${LRNODE_OVERLAP_WEIGHT:-0.0}
LRNODE_ENSEMBLE_WEIGHT=${LRNODE_ENSEMBLE_WEIGHT:-0.0}
LRNODE_GRIPPER_SWITCH_WEIGHT=${LRNODE_GRIPPER_SWITCH_WEIGHT:-0.0}
LRNODE_RUNTIME_ENSEMBLE_TEMP=${LRNODE_RUNTIME_ENSEMBLE_TEMP:-0.01}
LRNODE_EXPERIMENT_TAG=${EXPERIMENT_TAG}
LRNODE_RUN_NAME=${RUN_NAME}
LRNODE_SAVE_CHECKPOINT_PATH=${save_checkpoint_path}
LRNODE_DATASET=${dataset}
LRNODE_BASELINE_CKPT=${BASELINE_CKPT}
LRNODE_BASELINE_RUN_NAME=${BASELINE_RUN_NAME}
LRNODE_BASELINE_CKPT_ID=${BASELINE_CKPT_ID}
LRNODE_NUM_EPOCHS=${num_epochs}
LRNODE_START_SAVE_CHECKPOINT=${start_save_checkpoint}
LRNODE_FIRST_SAVED_CHECKPOINT=$((start_save_checkpoint + 1))
EOF
cp "${latest_dir}/distill_node.env" "${latest_dir}/finetune_node.env"
echo "[TRAIN INFO] wrote latest pointer: ${latest_dir}/distill_node.env"
echo "[TRAIN INFO] wrote compatibility pointer: ${latest_dir}/finetune_node.env"
