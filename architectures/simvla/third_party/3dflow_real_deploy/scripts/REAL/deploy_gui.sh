#!/usr/bin/env bash
set -euo pipefail

### NEED TO CHANGE ###
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-/home/jbr/3DFlow-Seer/checkpoints/vit_mae/mae_pretrain_vit_base.pth}"

# Keep these defaults aligned with scripts/REAL/deploy.sh, but this wrapper is
# separate so the known-good CLI deploy path remains untouched.
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained_2/seer_real-world_ft_pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_filtered_40p/38.pth}"
# language_instruction="${LANGUAGE_INSTRUCTION:-Pick up the red ball and place it in the basketball hoop}"
# language_instructions="${LANGUAGE_INSTRUCTIONS:-}"
### NEED TO CHANGE ###


##### Fruit placement #####
# language_instruction="${LANGUAGE_INSTRUCTION:-Put the green apple in the pink bowl and put the orange in the green bowl}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/apple_orange/seer_baseline_real-world_ft_put_the_green_apple_in_the_pink_bowl_and_put_the_orange_in_the_green_bowl_filtered_40p/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Scratch/apple_orange/seer_baseline_real-world_scratch_40p_put_the_green_apple_in_the_pink_bowl_and_put_the_orange_in_the_green_bowl_filtered_40p/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/apple_orange/seer_real-world_ft_40p_task-green_apple_pink_bowl_orange_green_bowl_dataset-vdpm_basecoord_run20260502_211834_droidpt_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/apple_orange/seer_real-world_scratch_40p_task-green_apple_pink_bowl_orange_green_bowl_dataset-vdpm_basecoord_run20260502_211834_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"


##### Orange #####
# language_instruction="${LANGUAGE_INSTRUCTION:-Pick up the orange and place it on the plate}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/hangcup/seer_baseline_real-world_ft_pick_up_the_white_mug_and_hang_it_on_the_wooden_mug_rack_filtered_40p/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/orange/seer_real-world_ft_pick_up_the_orange_and_place_it_on_the_plate_filtered_40p_vdpm_basecoord_run20260427_075751_lang-pick_up_the_orange_and_place_it_on_the_plate_droidpt_tcp0.005_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/orange/seer_real-world_scratch_40p_lang-pick_up_the_orange_and_place_it_on_the_plate_tcp0.005_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"


##### Basketball #####
language_instruction="${LANGUAGE_INSTRUCTION:-Pick up the red ball and place it in the basketball hoop}"

# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/seer_real-world_ft_pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_filtered_40p/38.pth}"

# BASELINE + Scratch
# 100p
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Scratch/seer_baseline_real-world_scratch_100p_pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_filtered_100p/38.pth}"
# 40p
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Scratch/seer_baseline_real-world_scratch_40p_pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_filtered_40p/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/basketball/seer_real-world_ft_pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_filtered_40p_vdpm_basecoord_run20260427_075751_lang-pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_droidpt_tcp0.005_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch tcp0.005
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/basketball/seer_real-world_scratch_40p_lang-pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_tcp0.005_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch tcp0.05
resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/basketball/basketball/seer_real-world_scratch_40p_lang-pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch tcp0.05 + ref0 coord
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/basketball/seer_real-world_scratch_40p_task-pick_red_ball_basketball_hoop_dataset-vdpm_ref0_coord_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"



# Basketball scratch rebuttal ablations.
# Activate exactly one resume_from_checkpoint line in this block.
# rebuttal_checkpoint_root="/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/rebuttal_ablations"

# 1) PCD loss = 0, TCP loss = 0.05, motion-topk selection
# no variants - pos1: 5/5, pos2: -/5, pos3: -/5 
# resume_from_checkpoint="${rebuttal_checkpoint_root}/seer_rebuttal_basketball_scratch_no_pcd_task-pick_red_ball_basketball_hoop_dataset-vdpm_basecoord_run20260427_075751_pcdloss0_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only_select-motion_topk/38.pth"

# 2) PCD loss = 0.05, TCP loss = 0, motion-topk selection
# no variants - pos1: 2/5, pos2: -/5, pos3: -/5 
# resume_from_checkpoint="${rebuttal_checkpoint_root}/seer_rebuttal_basketball_scratch_no_tcp_task-pick_red_ball_basketball_hoop_dataset-vdpm_basecoord_run20260427_075751_pcdloss0.05_tcp0_pcd2048_conf1_motion0.0001_filter-motion_valid_only_select-motion_topk/38.pth"

# 3) PCD loss = 0.05, TCP loss = 0.05, uniform-random selection
# no variants - pos1: 0/5, pos2: 5/5, pos3: -/5 
# resume_from_checkpoint="${rebuttal_checkpoint_root}/seer_rebuttal_basketball_scratch_random_task-pick_red_ball_basketball_hoop_dataset-vdpm_basecoord_run20260427_075751_pcdloss0.05_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only_select-random/38.pth"

# 4) PCD loss = 0.05, TCP loss = 0.05, motion-topk, VDPM ref0 coordinates
# no variants - pos1: 10/15, pos2: -/5, pos3: -/5
# resume_from_checkpoint="${rebuttal_checkpoint_root}/seer_real-world_scratch_40p_task-pick_red_ball_basketball_hoop_dataset-vdpm_ref0_coord_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth"

##### Hang cup #####
# language_instruction="${LANGUAGE_INSTRUCTION:-Pick up the white mug and hang it on the wooden mug rack}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/hangcup/seer_baseline_real-world_ft_pick_up_the_white_mug_and_hang_it_on_the_wooden_mug_rack_filtered_40p/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Scratch/hangcup/seer_baseline_real-world_scratch_40p_pick_up_the_white_mug_and_hang_it_on_the_wooden_mug_rack_filtered_40p/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/hangcup/seer_real-world_ft_pick_up_the_white_mug_and_hang_it_on_the_wooden_mug_rack_filtered_40p_vdpm_basecoord_run20260429_001411_lang-pick_up_the_white_mug_and_hang_it_on_the_wooden_mug_rack_droidpt_tcp0.005_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/hangcup/hangcup/seer_real-world_scratch_40p_lang-pick_up_the_white_mug_and_hang_it_on_the_wooden_mug_rack_tcp0.005_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"


##### Rings #####
# language_instruction="${LANGUAGE_INSTRUCTION:-Put the blue ring on the wooden stand, then put the pink ring on the wooden stand}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/rings/seer_baseline_real-world_ft_40p_task-rings_droidpt/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/rings/seer_real-world_ft_40p_task-rings_dataset-vdpm_basecoord_run20260504_231104_droidpt_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/38.pth}"


# ##### Stacking cups #####
# language_instruction="${LANGUAGE_INSTRUCTION:-Stack the orange cup on top of the blue cup, then stack the yellow cup on top of the orange cup}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/stacking_cups/seer_baseline_real-world_ft_40p_task-stacking_cups_droidpt/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/stacking_cups/seer_real-world_ft_40p_task-stacking_cups_dataset-vdpm_basecoord_run20260504_231104_droidpt_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/stacking_cups/seer_real-world_scratch_40p_task-stacking_cups_dataset-vdpm_basecoord_run20260504_231104_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# stacking_cups_ours_DROID-FT: rollout15_success3_fail12
# view point variation


##### Wipe #####
# language_instruction="${LANGUAGE_INSTRUCTION:-Pick up the blue brush and sweep all the colorful balls on the wooden board into the white tray.}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/wipe/seer_baseline_real-world_ft_40p_task-wipe_droidpt/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Scratch/wipe/seer_baseline_real-world_scratch_40p_task-wipe/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/wipe/seer_real-world_ft_40p_task-wipe_dataset-vdpm_basecoord_run20260505_191153_droidpt_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/wipe/seer_real-world_scratch_40p_task-wipe_dataset-vdpm_basecoord_run20260505_191153_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"


##### Cabinet #####
# language_instruction="${LANGUAGE_INSTRUCTION:-Open the drawer, take the orange cup out and put it on the table, then put the blue cup in the drawer and close the drawer}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/cabinet/seer_baseline_real-world_ft_40p_task-cabinet_droidpt/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/cabinet/seer_real-world_ft_40p_task-cabinet_dataset-vdpm_basecoord_run20260507_235539_droidpt_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/cabinet/seer_real-world_ft_40p_task-cabinet_dataset-vdpm_basecoord_run20260507_235539_droidpt_tcp0.005_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/cabinet/seer_real-world_scratch_40p_task-cabinet_dataset-vdpm_basecoord_run20260507_235539_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"


##### Doll #####
# language_instruction="${LANGUAGE_INSTRUCTION:-Pick up the white cup and place it on top of the upside-down pink cup, then pick up the blue penguin plush toy and put it in the white cup}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/doll/seer_baseline_real-world_ft_40p_task-doll_droidpt/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Scratch/doll/seer_baseline_real-world_scratch_40p_task-doll/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Droid_Pre-trained/doll/seer_real-world_ft_40p_task-doll_dataset-vdpm_basecoord_run20260511_180405_droidpt_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/home/jbr/bc_data/3dflow/checkpoints_seer_ours/Real-World/Scratch/doll/seer_real-world_scratch_40p_task-doll_dataset-vdpm_basecoord_run20260511_180405_tcp0.05_pcd2048_conf1_motion0.0001_filter-motion_valid_only/38.pth}"


##### Placeholder #####
# language_instruction="${LANGUAGE_INSTRUCTION:-}"
# BASELINE + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/38.pth}"

# BASELINE + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/38.pth}"

# OURS + Droid
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/38.pth}"

# OURS + Scratch
# resume_from_checkpoint="${RESUME_FROM_CHECKPOINT:-/38.pth}"


language_instructions="${LANGUAGE_INSTRUCTIONS:-}"

camera_c="${SEER_GUI_CAMERA_C:-yes}"
extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --use-camera-c)
            camera_c="yes"
            shift
            ;;
        --no-camera-c)
            camera_c="no"
            shift
            ;;
        --camera-c)
            camera_c="$2"
            shift 2
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
run_name="${path_parts[-2]}"
log_name="${path_parts[-1]}"
log_folder="eval_logs/$run_name"
mkdir -p "$log_folder"

node=1
node_num=1

export SEER_LANGUAGE_INSTRUCTION="${language_instruction}"
if [[ -n "${language_instructions}" ]]; then
    export SEER_LANGUAGE_INSTRUCTIONS="${language_instructions}"
else
    unset SEER_LANGUAGE_INSTRUCTIONS
fi

export SEER_ENABLE_ROLLOUT_MEDIA="${SEER_ENABLE_ROLLOUT_MEDIA:-1}"
export SEER_OBSERVER_CAMERA_NAME="${SEER_OBSERVER_CAMERA_NAME:-observer}"

echo "language_instruction: ${SEER_LANGUAGE_INSTRUCTION}"
if [[ -n "${SEER_LANGUAGE_INSTRUCTIONS:-}" ]]; then
    echo "language_instructions: ${SEER_LANGUAGE_INSTRUCTIONS}"
fi
echo "camera_c: ${camera_c}"

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10113 deploy_gui.py \
    --camera-c "${camera_c}" \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path "${vit_checkpoint_path}" \
    --workers 16 \
    --calvin_dataset "" \
    --lr_scheduler cosine \
    --save_every_iter 50000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 64 \
    --precision fp32 \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --run_name test \
    --transformer_layers 24 \
    --save_checkpoint_path "checkpoint" \
    --phase "evaluate" \
    --finetune_type "real" \
    --action_pred_steps 3 \
    --future_steps 3 \
    --sequence_length 7 \
    --obs_pred \
    --resume_from_checkpoint "${resume_from_checkpoint}" \
    --real_eval_max_steps 5000 \
    --eval_libero_ensembling \
    "${extra_args[@]}"
