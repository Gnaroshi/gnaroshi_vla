#!/bin/bash

set -euo pipefail

# Current-repo LR-NODE scratch evaluation sweep.
# This script evaluates checkpoints produced by scripts/LIBERO_LONG/Seer/scratch_node.sh.

export PYTHONPATH=/home/mingyujung/private/LIBERO:$PYTHONPATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

export SAVE_LRNODE_STATS="${SAVE_LRNODE_STATS:-1}"
export SAVE_LRNODE_STATS_ALL_RANKS="${SAVE_LRNODE_STATS_ALL_RANKS:-1}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export VIDEO_FPS="${VIDEO_FPS:-20}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-1}"

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
ours_env="${OURS_ENV:-${protocol_root}/train/_latest/scratch_node.env}"
if [[ -z "${OURS_CKPT_ROOT:-}" || -z "${OURS_RUN_NAME:-}" ]]; then
    if [[ ! -f "${ours_env}" ]]; then
        echo "[ERROR] LR-NODE scratch env not found: ${ours_env}" >&2
        echo "[ERROR] Run scripts/LIBERO_LONG/Seer/scratch_node.sh first, or set OURS_CKPT_ROOT and OURS_RUN_NAME." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${ours_env}"
    OURS_RUN_NAME="${OURS_RUN_NAME:-${LRNODE_RUN_NAME}}"
    OURS_CKPT_ROOT="${OURS_CKPT_ROOT:-${LRNODE_SAVE_CHECKPOINT_PATH}}"
    DATASET="${DATASET:-${LRNODE_DATASET}}"
    echo "[EVAL INFO] loaded latest LR-NODE scratch run from ${ours_env}"
fi

run_name="${OURS_RUN_NAME}"
resume_from_checkpoint="${OURS_CKPT_ROOT}/${run_name}"
dataset="${DATASET:-libero_10_converted}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
libero_path="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-checkpoints/}"

ckpt_ids_str="${CKPT_IDS:-30 31 32 33 34 35 36 37 38 39}"
read -r -a pthlist <<< "${ckpt_ids_str}"
query_intervals_str="${LRNODE_QUERY_INTERVALS_STR:-1 2 3 4 5 6 8}"
read -r -a query_intervals <<< "${query_intervals_str}"

experiment_tag="${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}"
result_root="${EVAL_RESULT_ROOT:-${protocol_root}/eval/lrnode_scratch_sweep_${run_name}_${experiment_tag}}"

node=1
node_num="${NODE_NUM:-4}"
master_port="${MASTER_PORT:-10342}"
eval_control_hz="${EVAL_CONTROL_HZ:-20}"

LRNODE_HIDDEN_DIM="${LRNODE_HIDDEN_DIM:-256}"
LRNODE_MOTION_DIM="${LRNODE_MOTION_DIM:-128}"
LRNODE_FAST_ENCODER_TYPE="${LRNODE_FAST_ENCODER_TYPE:-diffcnn}"
LRNODE_DETACH_INPUT_LATENT="${LRNODE_DETACH_INPUT_LATENT:-1}"
LRNODE_DETACH_TEACHER_LATENT="${LRNODE_DETACH_TEACHER_LATENT:-1}"
LRNODE_FREEZE_ACTION_HEAD_FOR_LRNODE="${LRNODE_FREEZE_ACTION_HEAD_FOR_LRNODE:-1}"
LRNODE_USE_POST_LAYERNORM="${LRNODE_USE_POST_LAYERNORM:-0}"
LRNODE_MULTISTEP_TRAIN="${LRNODE_MULTISTEP_TRAIN:-0}"
LRNODE_TRAIN_MAX_HORIZON="${LRNODE_TRAIN_MAX_HORIZON:-2}"
LRNODE_LOG_SANITY="${LRNODE_LOG_SANITY:-1}"
LRNODE_GATE_INIT_BIAS="${LRNODE_GATE_INIT_BIAS:--4.0}"
LRNODE_TRACE="${LRNODE_TRACE:-0}"
LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"

echo "[EVAL INFO] script=eval_node.sh"
echo "[EVAL INFO] run_name=${run_name}"
echo "[EVAL INFO] checkpoint_dir=${resume_from_checkpoint}"
echo "[EVAL INFO] result_root=${result_root}"
echo "[EVAL INFO] ckpt_ids=${ckpt_ids_str}"
echo "[EVAL INFO] query_intervals=${query_intervals_str}"
echo "[EVAL INFO] video SAVE_VIDEO=${SAVE_VIDEO}, success=${SAVE_VIDEO_SUCC}, fail=${SAVE_VIDEO_FAIL}, all_ranks=${SAVE_VIDEO_ALL_RANKS}, stride=${VIDEO_STRIDE}"

for query_interval in "${query_intervals[@]}"; do
    if [[ "${query_interval}" -eq 1 ]]; then
        eval_skip_full_forward=0
    else
        eval_skip_full_forward=1
    fi
    effective_query_hz=$(awk -v hz="${eval_control_hz}" -v k="${query_interval}" 'BEGIN { printf "%.2f", hz / k }')
    effective_query_hz_tag="${effective_query_hz//./p}"

    for ckpt_id in "${pthlist[@]}"; do
        this_resume_from_checkpoint="${resume_from_checkpoint}/${ckpt_id}.pth"
        if [[ ! -f "${this_resume_from_checkpoint}" ]]; then
            echo "[ERROR] Missing checkpoint: ${this_resume_from_checkpoint}" >&2
            exit 1
        fi

        LOG_DIR="${result_root}/lrnode_${run_name}_ckpt_${ckpt_id}_K${query_interval}_${effective_query_hz_tag}hz"
        logfile="${LOG_DIR}/ckpt_${ckpt_id}.log"
        mkdir -p "${LOG_DIR}"

        export LOG_DIR
        export RUN_NAME="${run_name}"
        export CKPT_TAG="ckpt_${ckpt_id}"
        export EVAL_CONTROL_HZ="${eval_control_hz}"
        export EFFECTIVE_QUERY_HZ="${effective_query_hz}"

        echo "------------------------------------------------------------"
        echo "[RUN] LR-NODE ckpt=${ckpt_id}, K=${query_interval}, skip=${eval_skip_full_forward}, full_query_hz=${effective_query_hz}"
        echo "[RUN] checkpoint=${this_resume_from_checkpoint}"
        echo "[RUN] log_dir=${LOG_DIR}"
        echo "------------------------------------------------------------"

        python -m torch.distributed.run \
            --nnodes="${node}" \
            --nproc_per_node="${node_num}" \
            --master_port="${master_port}" \
            eval_libero.py \
            --traj_cons \
            --rgb_pad 10 \
            --gripper_pad 4 \
            --gradient_accumulation_steps 1 \
            --bf16_module "vision_encoder" \
            --vit_checkpoint_path "${vit_checkpoint_path}" \
            --libero_path "${libero_path}" \
            --calvin_dataset "" \
            --workers 16 \
            --lr_scheduler cosine \
            --save_every_iter 50000 \
            --num_epochs 20 \
            --seed 42 \
            --batch_size 64 \
            --precision fp32 \
            --weight_decay 1e-4 \
            --num_resampler_query 6 \
            --run_name "${run_name}" \
            --transformer_layers 24 \
            --phase "evaluate" \
            --finetune_type "libero_10" \
            --save_checkpoint_path "${save_checkpoint_path}" \
            --action_pred_steps 3 \
            --future_steps 3 \
            --sequence_length 7 \
            --obs_pred \
            --gripper_width \
            --eval_libero_ensembling \
            --multi_step_action 1 \
            --use_lrnode_latent_update 1 \
            --lrnode_eval_skip_full_forward "${eval_skip_full_forward}" \
            --lrnode_query_interval "${query_interval}" \
            --lrnode_hidden_dim "${LRNODE_HIDDEN_DIM}" \
            --lrnode_motion_dim "${LRNODE_MOTION_DIM}" \
            --lrnode_fast_encoder_type "${LRNODE_FAST_ENCODER_TYPE}" \
            --lrnode_detach_input_latent "${LRNODE_DETACH_INPUT_LATENT}" \
            --lrnode_detach_teacher_latent "${LRNODE_DETACH_TEACHER_LATENT}" \
            --lrnode_freeze_action_head_for_lrnode "${LRNODE_FREEZE_ACTION_HEAD_FOR_LRNODE}" \
            --lrnode_use_post_layernorm "${LRNODE_USE_POST_LAYERNORM}" \
            --lrnode_multistep_train "${LRNODE_MULTISTEP_TRAIN}" \
            --lrnode_train_max_horizon "${LRNODE_TRAIN_MAX_HORIZON}" \
            --lrnode_log_sanity "${LRNODE_LOG_SANITY}" \
            --lrnode_gate_init_bias "${LRNODE_GATE_INIT_BIAS}" \
            --lrnode_trace "${LRNODE_TRACE}" \
            --lrnode_eval_step_log "${LRNODE_EVAL_STEP_LOG}" \
            --lrnode_eval_shadow_full_forward "${LRNODE_EVAL_SHADOW_FULL_FORWARD}" \
            --resume_from_checkpoint "${this_resume_from_checkpoint}" | tee "${logfile}"

        for required in \
            "${LOG_DIR}/analysis/eval_summary.json" \
            "${LOG_DIR}/analysis/eval_episode_metrics.csv" \
            "${LOG_DIR}/analysis/eval_latency_profile.json" \
            "${LOG_DIR}/analysis/args_snapshot_ckpt_${ckpt_id}.json"; do
            if [[ ! -f "${required}" ]]; then
                echo "[VERIFY][FAIL] Missing expected eval artifact: ${required}" >&2
                exit 1
            fi
        done

        if [[ "${SAVE_VIDEO}" == "1" ]]; then
            if [[ -d "${LOG_DIR}/eval_videos" ]]; then
                video_count=$(find "${LOG_DIR}/eval_videos" -type f \( -name '*.mp4' -o -name '*.gif' \) | wc -l)
            else
                video_count=0
            fi
            if [[ "${video_count}" -eq 0 ]]; then
                echo "[VERIFY][FAIL] SAVE_VIDEO=1 but no videos were saved under ${LOG_DIR}/eval_videos" >&2
                exit 1
            fi
            echo "[VERIFY][OK] saved ${video_count} videos under ${LOG_DIR}/eval_videos"
        fi

        grep -q "\\[EVAL MODEL\\] use_lrnode_latent_update=True" "${logfile}" || {
            echo "[VERIFY][FAIL] Expected model log use_lrnode_latent_update=True not found in ${logfile}" >&2
            exit 1
        }
        grep -q "\\[EVAL ARGS\\] lrnode_query_interval=${query_interval}" "${logfile}" || {
            echo "[VERIFY][FAIL] Expected lrnode_query_interval=${query_interval} not found in ${logfile}" >&2
            exit 1
        }
    done
done

echo "[EVAL DONE] LR-NODE scratch sweep saved to ${result_root}"
