#!/bin/bash

set -euo pipefail

# Scratch LR-NODE coupled-joint protocol.
# This is the "from scratch + LR-NODE + shared-gradient coupling" setting:
#   - no pretrained Seer checkpoint is loaded
#   - normal Seer losses train the Seer backbone/action head
#   - LR-NODE distillation losses train LR-NODE modules
#   - LR-NODE losses may also update Seer/action head through z_prev/action head
#   - LR-NODE teacher target uses shifted policy contexts by default
#
# New protocol runs are intentionally saved under runs_lrnode_protocol_20260616
# so they do not mix with older scratch_checkpoints_lrnode* experiments.

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-${protocol_root}/train/scratch_node_joint/}"
dataset="${DATASET:-libero_10_converted}"
root_dir="${ROOT_DIR:-/home/mingyujung/private/seer/seer_node3/LIBERO_DATASETS/${dataset}}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
libero_path="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
calvin_dataset_path="${CALVIN_DATASET_PATH:-calvin/dataset/task_ABC_D}"

which_server="${WHICH_SERVER:-sd1}"
method_tag="${METHOD_TAG:-lrnode_scratch_coupled_v1_lw05_aw01_g4}"
experiment_tag="${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}"
run_name="${RUN_NAME:-${which_server}_scratch_node_joint_${method_tag}_${experiment_tag}}"
export EXPERIMENT_TAG="${experiment_tag}"
export RUN_NAME="${run_name}"
latest_dir="${protocol_root}/train/_latest"

echo "[TRAIN INFO] script=scratch_node_joint.sh"
echo "[TRAIN INFO] protocol_root=${protocol_root}"
echo "[TRAIN INFO] save_checkpoint_path=${save_checkpoint_path}"
echo "[TRAIN INFO] experiment_tag=${EXPERIMENT_TAG}"
echo "[TRAIN INFO] run_name=${RUN_NAME}"
echo "[TRAIN INFO] latest_pointer_after_success=${latest_dir}/scratch_node_joint.env"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
master_port="${MASTER_PORT:-10322}"
node=1
node_num="${NODE_NUM:-4}"

LRNODE_EXTRA_ARGS=(
    --use_lrnode_latent_update 1
    --lrnode_train_latent_distill 1
    --lrnode_teacher_target_mode "${LRNODE_TEACHER_TARGET_MODE:-shifted_context}"
    --lrnode_context_selected_step "${LRNODE_CONTEXT_SELECTED_STEP:--1}"
    --lrnode_train_protocol joint
    --lrnode_freeze_seer_for_adapter 0
    --lrnode_assert_only_lrnode_trainable 0
    --lrnode_latent_weight "${LRNODE_LATENT_WEIGHT:-0.05}"
    --lrnode_action_distill_weight "${LRNODE_ACTION_DISTILL_WEIGHT:-0.1}"
    --lrnode_bc_weight "${LRNODE_BC_WEIGHT:-0.0}"
    --lrnode_smooth_weight "${LRNODE_SMOOTH_WEIGHT:-0.001}"
    --lrnode_hidden_dim "${LRNODE_HIDDEN_DIM:-256}"
    --lrnode_motion_dim "${LRNODE_MOTION_DIM:-128}"
    --lrnode_fast_encoder_type "${LRNODE_FAST_ENCODER_TYPE:-diffcnn}"
    --lrnode_detach_input_latent "${LRNODE_DETACH_INPUT_LATENT:-0}"
    --lrnode_detach_teacher_latent "${LRNODE_DETACH_TEACHER_LATENT:-1}"
    --lrnode_freeze_action_head_for_lrnode "${LRNODE_FREEZE_ACTION_HEAD_FOR_LRNODE:-0}"
    --lrnode_use_post_layernorm "${LRNODE_USE_POST_LAYERNORM:-0}"
    --lrnode_multistep_train "${LRNODE_MULTISTEP_TRAIN:-0}"
    --lrnode_train_max_horizon "${LRNODE_TRAIN_MAX_HORIZON:-2}"
    --lrnode_log_sanity "${LRNODE_LOG_SANITY:-1}"
    --lrnode_gate_init_bias "${LRNODE_GATE_INIT_BIAS:--4.0}"
    --lrnode_trace "${LRNODE_TRACE:-0}"
    --lrnode_debug_artifact_interval "${LRNODE_DEBUG_ARTIFACT_INTERVAL:-1000}"
)

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
    --learning_rate "${LEARNING_RATE:-1e-3}" \
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
    --loss_image \
    --loss_action \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint "${START_SAVE_CHECKPOINT:-25}" \
    --gripper_width \
    --warmup_epochs 5 \
    --libero_path "${libero_path}" \
    --report_to_wandb \
    --multi_step_action 1 \
    "${LRNODE_EXTRA_ARGS[@]}"

mkdir -p "${latest_dir}"
cat > "${latest_dir}/scratch_node_joint.env" <<EOF
LRNODE_PROTOCOL_SCRIPT=scratch_node_joint.sh
LRNODE_PROTOCOL_KIND=scratch
LRNODE_MODULE=1
LRNODE_COUPLING=coupled_joint
LRNODE_JOINT=1
LRNODE_BACKPROP_TO_SEER_FROM_LRNODE=1
LRNODE_TEACHER_TARGET_MODE=${LRNODE_TEACHER_TARGET_MODE:-shifted_context}
LRNODE_CONTEXT_SELECTED_STEP=${LRNODE_CONTEXT_SELECTED_STEP:--1}
LRNODE_EXPERIMENT_TAG=${EXPERIMENT_TAG}
LRNODE_RUN_NAME=${RUN_NAME}
LRNODE_SAVE_CHECKPOINT_PATH=${save_checkpoint_path}
LRNODE_DATASET=${dataset}
EOF
echo "[TRAIN INFO] wrote latest pointer: ${latest_dir}/scratch_node_joint.env"
