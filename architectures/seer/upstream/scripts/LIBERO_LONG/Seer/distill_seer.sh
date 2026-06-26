#!/bin/bash

set -euo pipefail

# Seer-only teacher-distillation control.
#
# Purpose:
#   Test whether the teacher/distillation protocol itself improves full-Seer
#   K=1 performance, without LR-NODE latent updates or skipped full forwards.
#
# Default control:
#   - teacher: baseline Seer checkpoint, default BASELINE_CKPT_ID=33
#   - student init: same teacher checkpoint
#   - LR-NODE: disabled
#   - loss: teacher action KD only
#
# This is a control experiment for LR-NODE distill results, not a query
# reduction experiment. Evaluate it with eval_seer_distill.sh at K=1.

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-${protocol_root}/train/distill_seer/}"
dataset="${DATASET:-libero_10_converted}"
root_dir="${ROOT_DIR:-/home/mingyujung/private/seer/seer_node3/LIBERO_DATASETS/${dataset}}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
libero_path="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
calvin_dataset_path="${CALVIN_DATASET_PATH:-calvin/dataset/task_ABC_D}"

baseline_env="${BASELINE_ENV:-${protocol_root}/train/_latest/scratch.env}"
if [[ -z "${BASELINE_CKPT:-}" ]]; then
    if [[ ! -f "${baseline_env}" ]]; then
        echo "[ERROR] BASELINE_CKPT is not set and baseline env does not exist: ${baseline_env}" >&2
        echo "[ERROR] Run scripts/LIBERO_LONG/Seer/scratch.sh first, or set BASELINE_CKPT directly." >&2
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

if [[ ! -f "${BASELINE_CKPT}" ]]; then
    echo "[ERROR] Missing teacher baseline checkpoint: ${BASELINE_CKPT}" >&2
    exit 1
fi

student_init="${SEER_DISTILL_STUDENT_INIT:-teacher}"
case "${student_init}" in
    teacher)
        student_init_args=(--finetune_from_pretrained_ckpt "${BASELINE_CKPT}")
        student_tag="student_teacherinit"
        ;;
    scratch)
        student_init_args=()
        student_tag="student_scratch"
        ;;
    *)
        echo "[ERROR] Unsupported SEER_DISTILL_STUDENT_INIT=${student_init}. Use teacher or scratch." >&2
        exit 1
        ;;
esac

base_loss_args=()
if [[ "${SEER_DISTILL_USE_BASE_LOSS:-0}" == "1" ]]; then
    base_loss_args=(--loss_image --loss_action)
    base_loss_tag="withbc"
else
    base_loss_tag="kdonly"
fi

which_server="${WHICH_SERVER:-sd1}"
method_tag="${METHOD_TAG:-seer_distill_control_teacher_ckpt${BASELINE_CKPT_ID}_${student_tag}_${base_loss_tag}_v1}"
experiment_tag="${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}"
run_name="${RUN_NAME:-${which_server}_distill_seer_${method_tag}_${experiment_tag}}"
export EXPERIMENT_TAG="${experiment_tag}"
export RUN_NAME="${run_name}"
latest_dir="${protocol_root}/train/_latest"

echo "[TRAIN INFO] script=distill_seer.sh"
echo "[TRAIN INFO] protocol_root=${protocol_root}"
echo "[TRAIN INFO] save_checkpoint_path=${save_checkpoint_path}"
echo "[TRAIN INFO] experiment_tag=${EXPERIMENT_TAG}"
echo "[TRAIN INFO] run_name=${RUN_NAME}"
echo "[TRAIN INFO] teacher_ckpt=${BASELINE_CKPT}"
echo "[TRAIN INFO] student_init=${student_init}"
echo "[TRAIN INFO] use_base_loss=${SEER_DISTILL_USE_BASE_LOSS:-0}"
echo "[TRAIN INFO] latest_pointer_after_success=${latest_dir}/distill_seer.env"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
master_port="${MASTER_PORT:-12433}"
node=1
node_num="${NODE_NUM:-4}"

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=${master_port} train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 8 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path "${vit_checkpoint_path}" \
    --calvin_dataset "${calvin_dataset_path}" \
    --workers 4 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs "${NUM_EPOCHS:-40}" \
    --seed "${SEED:-42}" \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --save_checkpoint \
    --finetune_type libero_finetune \
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
    --start_save_checkpoint "${START_SAVE_CHECKPOINT:-0}" \
    --gripper_width \
    --warmup_epochs "${WARMUP_EPOCHS:-2}" \
    --libero_path "${libero_path}" \
    --report_to_wandb \
    --multi_step_action 1 \
    --use_lrnode_latent_update 0 \
    --lrnode_train_latent_distill 0 \
    --seer_distill_teacher_ckpt "${BASELINE_CKPT}" \
    --seer_distill_action_weight "${SEER_DISTILL_ACTION_WEIGHT:-1.0}" \
    --seer_distill_latent_weight "${SEER_DISTILL_LATENT_WEIGHT:-0.0}" \
    --seer_distill_teacher_eval_mode "${SEER_DISTILL_TEACHER_EVAL_MODE:-1}" \
    "${student_init_args[@]}" \
    "${base_loss_args[@]}"

mkdir -p "${latest_dir}"
cat > "${latest_dir}/distill_seer.env" <<EOF
LRNODE_PROTOCOL_SCRIPT=distill_seer.sh
LRNODE_PROTOCOL_KIND=seer_distill_control
LRNODE_MODULE=0
LRNODE_COUPLING=none
LRNODE_JOINT=0
LRNODE_BACKPROP_TO_SEER_FROM_LRNODE=0
SEER_DISTILL_CONTROL=1
SEER_DISTILL_TEACHER_CKPT=${BASELINE_CKPT}
SEER_DISTILL_STUDENT_INIT=${student_init}
SEER_DISTILL_USE_BASE_LOSS=${SEER_DISTILL_USE_BASE_LOSS:-0}
SEER_DISTILL_ACTION_WEIGHT=${SEER_DISTILL_ACTION_WEIGHT:-1.0}
SEER_DISTILL_LATENT_WEIGHT=${SEER_DISTILL_LATENT_WEIGHT:-0.0}
LRNODE_EXPERIMENT_TAG=${EXPERIMENT_TAG}
LRNODE_RUN_NAME=${RUN_NAME}
LRNODE_SAVE_CHECKPOINT_PATH=${save_checkpoint_path}
LRNODE_DATASET=${dataset}
LRNODE_BASELINE_CKPT=${BASELINE_CKPT}
LRNODE_BASELINE_RUN_NAME=${BASELINE_RUN_NAME}
LRNODE_BASELINE_CKPT_ID=${BASELINE_CKPT_ID}
EOF
echo "[TRAIN INFO] wrote latest pointer: ${latest_dir}/distill_seer.env"
