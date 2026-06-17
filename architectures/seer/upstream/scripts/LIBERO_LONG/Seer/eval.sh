#!/bin/bash

set -euo pipefail

# Current-repo Seer baseline evaluation sweep.
# This script evaluates checkpoints produced by scripts/LIBERO_LONG/Seer/scratch.sh.
# It intentionally does not read /home/mingyujung/private/seer/seer_main.

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
baseline_env="${BASELINE_ENV:-${protocol_root}/train/_latest/scratch.env}"
if [[ -z "${BASELINE_CKPT_ROOT:-}" || -z "${BASELINE_RUN_NAME:-}" ]]; then
    if [[ ! -f "${baseline_env}" ]]; then
        echo "[ERROR] Baseline env not found: ${baseline_env}" >&2
        echo "[ERROR] Run scripts/LIBERO_LONG/Seer/scratch.sh first, or set BASELINE_CKPT_ROOT and BASELINE_RUN_NAME." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${baseline_env}"
    BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-${LRNODE_RUN_NAME}}"
    BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-${LRNODE_SAVE_CHECKPOINT_PATH}}"
    DATASET="${DATASET:-${LRNODE_DATASET}}"
    echo "[EVAL INFO] loaded latest baseline run from ${baseline_env}"
fi

which_server="${WHICH_SERVER:-sd1}"
run_name="${BASELINE_RUN_NAME}"
resume_from_checkpoint="${BASELINE_CKPT_ROOT}/${run_name}"
dataset="${DATASET:-libero_10_converted}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
libero_path="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-checkpoints/}"

ckpt_ids_str="${CKPT_IDS:-30 31 32 33 34 35 36 37 38 39}"
read -r -a pthlist <<< "${ckpt_ids_str}"

experiment_tag="${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}"
result_root="${EVAL_RESULT_ROOT:-${protocol_root}/eval/baseline_sweep_${run_name}_${experiment_tag}}"

node=1
node_num="${NODE_NUM:-4}"
master_port="${MASTER_PORT:-10341}"
eval_control_hz="${EVAL_CONTROL_HZ:-20}"

echo "[EVAL INFO] script=eval.sh"
echo "[EVAL INFO] run_name=${run_name}"
echo "[EVAL INFO] checkpoint_dir=${resume_from_checkpoint}"
echo "[EVAL INFO] result_root=${result_root}"
echo "[EVAL INFO] ckpt_ids=${ckpt_ids_str}"
echo "[EVAL INFO] video SAVE_VIDEO=${SAVE_VIDEO}, success=${SAVE_VIDEO_SUCC}, fail=${SAVE_VIDEO_FAIL}, all_ranks=${SAVE_VIDEO_ALL_RANKS}, stride=${VIDEO_STRIDE}"

for ckpt_id in "${pthlist[@]}"; do
    this_resume_from_checkpoint="${resume_from_checkpoint}/${ckpt_id}.pth"
    if [[ ! -f "${this_resume_from_checkpoint}" ]]; then
        echo "[ERROR] Missing checkpoint: ${this_resume_from_checkpoint}" >&2
        exit 1
    fi

    effective_query_hz_tag="${eval_control_hz//./p}"
    LOG_DIR="${result_root}/baseline_${run_name}_ckpt_${ckpt_id}_K1_${effective_query_hz_tag}hz"
    logfile="${LOG_DIR}/ckpt_${ckpt_id}.log"
    mkdir -p "${LOG_DIR}"

    export LOG_DIR
    export RUN_NAME="${run_name}"
    export CKPT_TAG="ckpt_${ckpt_id}"
    export EVAL_CONTROL_HZ="${eval_control_hz}"
    export EFFECTIVE_QUERY_HZ="${eval_control_hz}"

    echo "------------------------------------------------------------"
    echo "[RUN] baseline ckpt=${ckpt_id}"
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
        --use_lrnode_latent_update 0 \
        --lrnode_eval_skip_full_forward 0 \
        --lrnode_query_interval 1 \
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

    grep -q "\\[EVAL MODEL\\] use_lrnode_latent_update=False" "${logfile}" || {
        echo "[VERIFY][FAIL] Expected baseline model log use_lrnode_latent_update=False not found in ${logfile}" >&2
        exit 1
    }
done

echo "[EVAL DONE] baseline sweep saved to ${result_root}"
