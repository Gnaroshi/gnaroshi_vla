#!/bin/bash

set -euo pipefail

# Scratch Seer baseline protocol for LR-NODE comparisons.
# This trains the unmodified Seer policy from scratch in the current repository.
# LR-NODE is explicitly disabled, and outputs are separated from old experiments.

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-${protocol_root}/train/scratch/}"
dataset="${DATASET:-libero_10_converted}"
root_dir="${ROOT_DIR:-/home/mingyujung/private/seer/seer_node3/LIBERO_DATASETS/${dataset}}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
libero_path="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
calvin_dataset_path="${CALVIN_DATASET_PATH:-calvin/dataset/task_ABC_D}"

which_server="${WHICH_SERVER:-sd1}"
method_tag="${METHOD_TAG:-seer_scratch_baseline_v1}"
experiment_tag="${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}"
run_name="${RUN_NAME:-${which_server}_scratch_baseline_${method_tag}_${experiment_tag}}"
export EXPERIMENT_TAG="${experiment_tag}"
export RUN_NAME="${run_name}"
latest_dir="${protocol_root}/train/_latest"

echo "[TRAIN INFO] script=scratch.sh"
echo "[TRAIN INFO] protocol_root=${protocol_root}"
echo "[TRAIN INFO] save_checkpoint_path=${save_checkpoint_path}"
echo "[TRAIN INFO] experiment_tag=${EXPERIMENT_TAG}"
echo "[TRAIN INFO] run_name=${RUN_NAME}"
echo "[TRAIN INFO] latest_pointer_after_success=${latest_dir}/scratch.env"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
master_port="${MASTER_PORT:-10311}"
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
    --use_lrnode_latent_update 0 \
    --lrnode_train_latent_distill 0

mkdir -p "${latest_dir}"
cat > "${latest_dir}/scratch.env" <<EOF
LRNODE_PROTOCOL_SCRIPT=scratch.sh
LRNODE_PROTOCOL_KIND=scratch_baseline
LRNODE_MODULE=0
LRNODE_COUPLING=none
LRNODE_JOINT=0
LRNODE_BACKPROP_TO_SEER_FROM_LRNODE=0
LRNODE_EXPERIMENT_TAG=${EXPERIMENT_TAG}
LRNODE_RUN_NAME=${RUN_NAME}
LRNODE_SAVE_CHECKPOINT_PATH=${save_checkpoint_path}
LRNODE_DATASET=${dataset}
EOF
echo "[TRAIN INFO] wrote latest pointer: ${latest_dir}/scratch.env"
