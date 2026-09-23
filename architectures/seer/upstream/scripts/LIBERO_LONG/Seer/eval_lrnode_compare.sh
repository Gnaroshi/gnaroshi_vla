#!/bin/bash

set -euo pipefail

# Single canonical LR-NODE comparison script.
# Default experiment:
#   1) baseline Seer full-forward
#   2) ours full-forward, K=1
#   3) ours LR-NODE skip-forward, K in LRNODE_QUERY_INTERVALS_STR
#
# Typical overrides:
#   METHOD_TAG="lrnode_student_v3" \
#   OURS_RUN_NAME="sd1_scratch_libero_10_converted_seer_lrnode_student_v3" \
#   OURS_CKPT_ID="42" \
#   bash scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh
#
# Or specify explicit checkpoint paths:
#   BASELINE_CKPT="/path/to/baseline.pth" \
#   OURS_CKPT="/path/to/ours.pth" \
#   OURS_NAME="lrnode_v3" \
#   bash scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export LIBERO_GL_BACKEND="${LIBERO_GL_BACKEND:-osmesa}"
if [[ "${LIBERO_GL_BACKEND}" != "osmesa" && "${LIBERO_GL_BACKEND}" != "egl" ]]; then
    echo "[ERROR] LIBERO_GL_BACKEND must be 'osmesa' or 'egl'; got ${LIBERO_GL_BACKEND}" >&2
    exit 1
fi
export MUJOCO_GL="${LIBERO_GL_BACKEND}"
export PYOPENGL_PLATFORM="${LIBERO_GL_BACKEND}"
export LIBERO_GL_REQUIRE_ACTUAL="${LIBERO_GL_REQUIRE_ACTUAL:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${UPSTREAM_DIR}/../../.." && pwd)"

# Videos are saved by default because qualitative comparison is part of this experiment.
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export VIDEO_FPS="${VIDEO_FPS:-20}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-1}"

safe_tag() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    value="${value//:/_}"
    value="${value//,/}"
    echo "${value}"
}

which_server="${WHICH_SERVER:-sd1}"
dataset="${DATASET:-libero_10_converted}"
eval_suite="${EVAL_SUITE:-libero_10}"
libero_path="${LIBERO_PATH:-}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-${UPSTREAM_DIR}/checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/seer}"
protocol_root="${LRNODE_PROTOCOL_ROOT:-${REPO_ROOT}/results/seer/lrnode/default}"
latest_baseline="${BASELINE_ENV:-${protocol_root}/train/_latest/scratch.env}"
latest_ours="${OURS_ENV:-${protocol_root}/train/_latest/scratch_node.env}"

if [[ -z "${libero_path}" || ! -d "${libero_path}" ]]; then
    echo "[ERROR] Set LIBERO_PATH to a valid LIBERO repository path." >&2
    exit 1
fi
if [[ ! -f "${vit_checkpoint_path}" ]]; then
    echo "[ERROR] Missing ViT checkpoint: ${vit_checkpoint_path}" >&2
    echo "[ERROR] Set VIT_CHECKPOINT_PATH explicitly." >&2
    exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${UPSTREAM_DIR}:${libero_path}:${PYTHONPATH:-}"

# Baseline definition. By default this uses the latest current-repo scratch.sh
# output. Set BASELINE_CKPT explicitly to evaluate any other checkpoint.
if [[ -z "${BASELINE_CKPT:-}" ]]; then
    if [[ -z "${BASELINE_RUN_NAME:-}" && -z "${BASELINE_CKPT_ROOT:-}" && -f "${latest_baseline}" ]]; then
        # shellcheck disable=SC1090
        source "${latest_baseline}"
        BASELINE_RUN_NAME="${LRNODE_RUN_NAME}"
        BASELINE_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
        BASELINE_NAME="${BASELINE_NAME:-seer_scratch_baseline}"
        echo "[EVAL INFO] loaded latest baseline run from ${latest_baseline}"
    fi
    if [[ -z "${BASELINE_RUN_NAME:-}" || -z "${BASELINE_CKPT_ROOT:-}" ]]; then
        echo "[ERROR] Baseline checkpoint is not configured." >&2
        echo "[ERROR] Run scratch.sh first or set BASELINE_CKPT=/path/to/baseline.pth." >&2
        exit 1
    fi
    BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-33}"
    BASELINE_CKPT="${BASELINE_CKPT_ROOT}/${BASELINE_RUN_NAME}/${BASELINE_CKPT_ID}.pth"
else
    BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-$(basename "$(dirname "${BASELINE_CKPT}")")}"
    BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-$(dirname "$(dirname "${BASELINE_CKPT}")")}"
    BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-$(basename "${BASELINE_CKPT}" .pth)}"
fi
BASELINE_NAME="$(safe_tag "${BASELINE_NAME:-seer_scratch_baseline}")"
BASELINE_CKPT_TAG="$(safe_tag "${BASELINE_CKPT_TAG:-${BASELINE_NAME}_ckpt_${BASELINE_CKPT_ID}}")"

# Full experiment defaults. These are resolved before ours checkpoint handling so
# baseline-only sweeps can run before scratch_node.sh exists.
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_OURS_FULL="${RUN_OURS_FULL:-1}"
LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR-2 3 4 5 6 8}"
LRNODE_QUERY_INTERVALS=()
if [[ -n "${LRNODE_QUERY_INTERVALS_STR}" ]]; then
    read -r -a LRNODE_QUERY_INTERVALS <<< "${LRNODE_QUERY_INTERVALS_STR}"
fi
NEED_OURS=0
if [[ "${RUN_OURS_FULL}" -eq 1 || "${#LRNODE_QUERY_INTERVALS[@]}" -gt 0 ]]; then
    NEED_OURS=1
fi

# Ours/method definition. By default this uses the latest current-repo
# scratch_node.sh output. Set OURS_CKPT explicitly to evaluate any other method.
if [[ -z "${OURS_CKPT:-}" ]]; then
    if [[ -z "${OURS_RUN_NAME:-}" && -z "${OURS_CKPT_ROOT:-}" && -f "${latest_ours}" ]]; then
        # shellcheck disable=SC1090
        source "${latest_ours}"
        OURS_RUN_NAME="${LRNODE_RUN_NAME}"
        OURS_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
        METHOD_TAG="${METHOD_TAG:-lrnode_scratch_ts}"
        OURS_NAME="${OURS_NAME:-${METHOD_TAG}}"
        echo "[EVAL INFO] loaded latest LR-NODE scratch run from ${latest_ours}"
    fi
    if [[ -z "${OURS_RUN_NAME:-}" || -z "${OURS_CKPT_ROOT:-}" ]]; then
        if [[ "${NEED_OURS}" -eq 1 ]]; then
            echo "[ERROR] Ours checkpoint is not configured." >&2
            echo "[ERROR] Run scratch_node.sh first or set OURS_CKPT=/path/to/ours.pth." >&2
            exit 1
        fi
        OURS_RUN_NAME="none"
        OURS_CKPT_ROOT="none"
        OURS_CKPT_ID="${OURS_CKPT_ID:-none}"
        OURS_CKPT="none"
    else
        OURS_CKPT_ID="${OURS_CKPT_ID:-${BASELINE_CKPT_ID}}"
        OURS_CKPT="${OURS_CKPT_ROOT}/${OURS_RUN_NAME}/${OURS_CKPT_ID}.pth"
    fi
else
    OURS_RUN_NAME="${OURS_RUN_NAME:-$(basename "$(dirname "${OURS_CKPT}")")}"
    OURS_CKPT_ROOT="${OURS_CKPT_ROOT:-$(dirname "$(dirname "${OURS_CKPT}")")}"
    OURS_CKPT_ID="${OURS_CKPT_ID:-$(basename "${OURS_CKPT}" .pth)}"
fi
METHOD_TAG="$(safe_tag "${METHOD_TAG:-lrnode_scratch_ts}")"
OURS_NAME="$(safe_tag "${OURS_NAME:-${METHOD_TAG}}")"
OURS_CKPT_TAG="$(safe_tag "${OURS_CKPT_TAG:-${OURS_NAME}_ckpt_${OURS_CKPT_ID}}")"

experiment_name="$(safe_tag "${EXPERIMENT_NAME:-lrnode_compare}")"
experiment_tag="$(safe_tag "${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}")"
default_result_root="${protocol_root}/eval/${experiment_name}_${OURS_NAME}_ckpt${OURS_CKPT_ID}_vs_${BASELINE_NAME}_ckpt${BASELINE_CKPT_ID}_${experiment_tag}"
result_root="${RESULT_ROOT:-${default_result_root}}"

node=1
node_num="${NODE_NUM:-4}"
master_port="${MASTER_PORT:-12452}"

EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
FASTV_ENABLED="${FASTV_ENABLED:-0}"
FASTV_PRUNE_LAYER="${FASTV_PRUNE_LAYER:-2}"
FASTV_PRUNE_RATIO="${FASTV_PRUNE_RATIO:-0.5}"
FASTV_SCORE_MODE="${FASTV_SCORE_MODE:-last_token_at_l}"
FASTV_RETENTION_DIAGNOSTICS="${FASTV_RETENTION_DIAGNOSTICS:-0}"

# LR-NODE architecture/config flags must match the trained checkpoint.
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
LRNODE_TRAIN_PROTOCOL="${LRNODE_TRAIN_PROTOCOL:-joint}"
LRNODE_FREEZE_SEER_FOR_ADAPTER="${LRNODE_FREEZE_SEER_FOR_ADAPTER:-0}"
LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE="${LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE:-0}"
LRNODE_EVAL_BASE_CKPT="${LRNODE_EVAL_BASE_CKPT:-}"

# Keep detailed metrics by default. Disable only when disk usage is a concern.
LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"
LRNODE_EVAL_REFRESH_POLICY="${LRNODE_EVAL_REFRESH_POLICY:-periodic}"
LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE="${LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE:-1}"
LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD="${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD:-1}"
LRNODE_MECHANISM_TRACE="${LRNODE_MECHANISM_TRACE:-0}"
LRNODE_TRACE_SAVE_LATENTS="${LRNODE_TRACE_SAVE_LATENTS:-0}"
LRNODE_TRACE_EPISODE_LIMIT="${LRNODE_TRACE_EPISODE_LIMIT:-0}"
LRNODE_TRACE_OUTPUT_DIR="${LRNODE_TRACE_OUTPUT_DIR:-}"
LRNODE_COUNTERFACTUAL_MODE="${LRNODE_COUNTERFACTUAL_MODE:-standard}"
LRNODE_COUNTERFACTUAL_MIX_STAGE="${LRNODE_COUNTERFACTUAL_MIX_STAGE:-pre_ensemble}"
LRNODE_LATENT_FUSION_ALPHA="${LRNODE_LATENT_FUSION_ALPHA:-0.0}"
LRNODE_LATENT_FUSION_MODE="${LRNODE_LATENT_FUSION_MODE:-every_step}"
LRNODE_MATCHED_RANDOM_SEED="${LRNODE_MATCHED_RANDOM_SEED:-20260724}"
LRNODE_MATCHED_RANDOM_NORM_MODE="${LRNODE_MATCHED_RANDOM_NORM_MODE:-per_token}"
LRNODE_EVERY_STEP_FILTER_MODE="${LRNODE_EVERY_STEP_FILTER_MODE:-off}"
LRNODE_EVERY_STEP_FILTER_ALPHA="${LRNODE_EVERY_STEP_FILTER_ALPHA:-0.5}"
LRNODE_EVERY_STEP_FILTER_BETA="${LRNODE_EVERY_STEP_FILTER_BETA:-0.5}"
LRNODE_EVERY_STEP_FILTER_DIAGNOSTICS="${LRNODE_EVERY_STEP_FILTER_DIAGNOSTICS:-0}"
LRNODE_EVAL_ABLATION_MODE="${LRNODE_EVAL_ABLATION_MODE:-stepwise}"
LRNODE_NO_DELTA_MODE="${LRNODE_NO_DELTA_MODE:-zero}"
LRNODE_CHUNK_TOKEN_POLICY="${LRNODE_CHUNK_TOKEN_POLICY:-skip_only}"
LATENTLOOP_SEGMENT_GRID_ENABLE="${LATENTLOOP_SEGMENT_GRID_ENABLE:-0}"
LATENTLOOP_FEEDBACK_SCHEDULE="${LATENTLOOP_FEEDBACK_SCHEDULE:-dense}"
LATENTLOOP_SAME_INPUT_STOCHASTICITY_REPEATS="${LATENTLOOP_SAME_INPUT_STOCHASTICITY_REPEATS:-0}"
LATENTLOOP_SAME_INPUT_STOCHASTICITY_OUTPUT="${LATENTLOOP_SAME_INPUT_STOCHASTICITY_OUTPUT:-}"
LATENTLOOP_PLAN_TRACE="${LATENTLOOP_PLAN_TRACE:-0}"
LATENTLOOP_PLAN_TRACE_SAVE_LATENTS="${LATENTLOOP_PLAN_TRACE_SAVE_LATENTS:-0}"
LATENTLOOP_PLAN_TRACE_OUTPUT_DIR="${LATENTLOOP_PLAN_TRACE_OUTPUT_DIR:-}"
LATENTLOOP_PLAN_TRACE_ROW_ID="${LATENTLOOP_PLAN_TRACE_ROW_ID:-}"
LATENTLOOP_PLAN_TRACE_PAIRED_GROUP="${LATENTLOOP_PLAN_TRACE_PAIRED_GROUP:-}"
LATENTLOOP_FEEDBACK_SOURCE="${LATENTLOOP_FEEDBACK_SOURCE:-current}"
LATENTLOOP_PLAN_ADAPTER_MODE="${LATENTLOOP_PLAN_ADAPTER_MODE:-off}"
LATENTLOOP_PLAN_ADAPTER_HIDDEN_DIM="${LATENTLOOP_PLAN_ADAPTER_HIDDEN_DIM:-0}"
LATENTLOOP_PLAN_PARAMETER_MATCH_TOLERANCE="${LATENTLOOP_PLAN_PARAMETER_MATCH_TOLERANCE:-0.05}"
LATENTLOOP_COMPARISON_PROTOCOL="${LATENTLOOP_COMPARISON_PROTOCOL:-0}"
EVAL_LIBERO_ENSEMBLING="${EVAL_LIBERO_ENSEMBLING:-1}"
EVAL_SEED="${EVAL_SEED:-42}"
LIBERO_EVAL_MAX_STEPS="${LIBERO_EVAL_MAX_STEPS:-600}"

common_eval_args=(
    --traj_cons
    --rgb_pad 10
    --gripper_pad 4
    --gradient_accumulation_steps 1
    --bf16_module "vision_encoder"
    --vit_checkpoint_path "${vit_checkpoint_path}"
    --libero_path "${libero_path}"
    --calvin_dataset ""
    --workers 16
    --lr_scheduler cosine
    --save_every_iter 50000
    --num_epochs 20
    --seed "${EVAL_SEED}"
    --batch_size 64
    --precision fp32
    --weight_decay 1e-4
    --num_resampler_query 6
    --transformer_layers 24
    --phase "evaluate"
    --finetune_type "${eval_suite}"
    --save_checkpoint_path "${save_checkpoint_path}"
    --action_pred_steps 3
    --future_steps 3
    --sequence_length 7
    --obs_pred
    --gripper_width
    --multi_step_action 1
    --libero_eval_max_steps "${LIBERO_EVAL_MAX_STEPS}"
    --fastv_enabled "${FASTV_ENABLED}"
    --fastv_prune_layer "${FASTV_PRUNE_LAYER}"
    --fastv_prune_ratio "${FASTV_PRUNE_RATIO}"
    --fastv_score_mode "${FASTV_SCORE_MODE}"
    --fastv_retention_diagnostics "${FASTV_RETENTION_DIAGNOSTICS}"
)
if [[ "${EVAL_LIBERO_ENSEMBLING}" == "1" ]]; then
    common_eval_args+=(--eval_libero_ensembling)
elif [[ "${EVAL_LIBERO_ENSEMBLING}" != "0" ]]; then
    echo "[ERROR] EVAL_LIBERO_ENSEMBLING must be 0 or 1" >&2
    exit 1
fi

run_eval() {
    local label="$1"
    local run_name="$2"
    local ckpt_path="$3"
    local ckpt_tag="$4"
    local use_lrnode="$5"
    local skip_full="$6"
    local query_interval="$7"

    if [[ ! -f "${ckpt_path}" ]]; then
        echo "[ERROR] Missing checkpoint: ${ckpt_path}" >&2
        exit 1
    fi

    local effective_query_hz
    effective_query_hz=$(awk -v hz="${EVAL_CONTROL_HZ}" -v k="${query_interval}" 'BEGIN { printf "%.2f", hz / k }')
    local effective_query_hz_tag="${effective_query_hz//./p}"

    local refresh_tag=""
    if [[ "${LRNODE_EVAL_REFRESH_POLICY}" != "periodic" ]]; then
        refresh_tag="_$(safe_tag "${LRNODE_EVAL_REFRESH_POLICY}")"
        if [[ "${LRNODE_EVAL_REFRESH_POLICY}" == "fixed_budget" ]]; then
            refresh_tag="${refresh_tag}_B${LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE}"
        fi
    fi

    local log_dir="${result_root}/${label}${refresh_tag}_K${query_interval}_${effective_query_hz_tag}hz"
    local logfile="${log_dir}/${ckpt_tag}.log"
    mkdir -p "${log_dir}"

    export LOG_DIR="${log_dir}"
    export RUN_NAME="${run_name}"
    export CKPT_TAG="${ckpt_tag}"
    export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ}"
    export EFFECTIVE_QUERY_HZ="${effective_query_hz}"
    local trace_output_dir="${LRNODE_TRACE_OUTPUT_DIR:-${log_dir}/analysis/mechanism_trace}"
    local plan_trace_output_dir="${LATENTLOOP_PLAN_TRACE_OUTPUT_DIR:-${log_dir}/analysis/plan_trace}"

    local lrnode_args=(
        --use_lrnode_latent_update "${use_lrnode}"
        --lrnode_eval_skip_full_forward "${skip_full}"
        --lrnode_train_protocol "${LRNODE_TRAIN_PROTOCOL}"
        --lrnode_freeze_seer_for_adapter "${LRNODE_FREEZE_SEER_FOR_ADAPTER}"
        --lrnode_assert_only_lrnode_trainable "${LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE}"
        --lrnode_query_interval "${query_interval}"
        --lrnode_hidden_dim "${LRNODE_HIDDEN_DIM}"
        --lrnode_motion_dim "${LRNODE_MOTION_DIM}"
        --lrnode_fast_encoder_type "${LRNODE_FAST_ENCODER_TYPE}"
        --lrnode_detach_input_latent "${LRNODE_DETACH_INPUT_LATENT}"
        --lrnode_detach_teacher_latent "${LRNODE_DETACH_TEACHER_LATENT}"
        --lrnode_freeze_action_head_for_lrnode "${LRNODE_FREEZE_ACTION_HEAD_FOR_LRNODE}"
        --lrnode_use_post_layernorm "${LRNODE_USE_POST_LAYERNORM}"
        --lrnode_multistep_train "${LRNODE_MULTISTEP_TRAIN}"
        --lrnode_train_max_horizon "${LRNODE_TRAIN_MAX_HORIZON}"
        --lrnode_log_sanity "${LRNODE_LOG_SANITY}"
        --lrnode_gate_init_bias "${LRNODE_GATE_INIT_BIAS}"
        --lrnode_trace "${LRNODE_TRACE}"
        --lrnode_eval_step_log "${LRNODE_EVAL_STEP_LOG}"
        --lrnode_eval_shadow_full_forward "${LRNODE_EVAL_SHADOW_FULL_FORWARD}"
        --lrnode_eval_profile_full_action_head "${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD}"
        --lrnode_eval_refresh_policy "${LRNODE_EVAL_REFRESH_POLICY}"
        --lrnode_eval_max_full_forwards_per_episode "${LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE}"
        --lrnode_mechanism_trace "${LRNODE_MECHANISM_TRACE}"
        --lrnode_trace_save_latents "${LRNODE_TRACE_SAVE_LATENTS}"
        --lrnode_trace_episode_limit "${LRNODE_TRACE_EPISODE_LIMIT}"
        --lrnode_trace_output_dir "${trace_output_dir}"
        --lrnode_counterfactual_mode "${LRNODE_COUNTERFACTUAL_MODE}"
        --lrnode_counterfactual_mix_stage "${LRNODE_COUNTERFACTUAL_MIX_STAGE}"
        --lrnode_latent_fusion_alpha "${LRNODE_LATENT_FUSION_ALPHA}"
        --lrnode_latent_fusion_mode "${LRNODE_LATENT_FUSION_MODE}"
        --lrnode_matched_random_seed "${LRNODE_MATCHED_RANDOM_SEED}"
        --lrnode_matched_random_norm_mode "${LRNODE_MATCHED_RANDOM_NORM_MODE}"
        --lrnode_every_step_filter_mode "${LRNODE_EVERY_STEP_FILTER_MODE}"
        --lrnode_every_step_filter_alpha "${LRNODE_EVERY_STEP_FILTER_ALPHA}"
        --lrnode_every_step_filter_beta "${LRNODE_EVERY_STEP_FILTER_BETA}"
        --lrnode_every_step_filter_diagnostics "${LRNODE_EVERY_STEP_FILTER_DIAGNOSTICS}"
        --lrnode_eval_ablation_mode "${LRNODE_EVAL_ABLATION_MODE}"
        --lrnode_no_delta_mode "${LRNODE_NO_DELTA_MODE}"
        --lrnode_chunk_token_policy "${LRNODE_CHUNK_TOKEN_POLICY}"
        --latentloop_segment_grid_enable "${LATENTLOOP_SEGMENT_GRID_ENABLE}"
        --latentloop_feedback_schedule "${LATENTLOOP_FEEDBACK_SCHEDULE}"
        --latentloop_same_input_stochasticity_repeats "${LATENTLOOP_SAME_INPUT_STOCHASTICITY_REPEATS}"
        --latentloop_same_input_stochasticity_output "${LATENTLOOP_SAME_INPUT_STOCHASTICITY_OUTPUT}"
        --latentloop_plan_trace "${LATENTLOOP_PLAN_TRACE}"
        --latentloop_plan_trace_save_latents "${LATENTLOOP_PLAN_TRACE_SAVE_LATENTS}"
        --latentloop_plan_trace_output_dir "${plan_trace_output_dir}"
        --latentloop_plan_trace_row_id "${LATENTLOOP_PLAN_TRACE_ROW_ID:-${label}}"
        --latentloop_plan_trace_paired_group "${LATENTLOOP_PLAN_TRACE_PAIRED_GROUP}"
        --latentloop_feedback_source "${LATENTLOOP_FEEDBACK_SOURCE}"
        --latentloop_plan_adapter_mode "${LATENTLOOP_PLAN_ADAPTER_MODE}"
        --latentloop_plan_adapter_hidden_dim "${LATENTLOOP_PLAN_ADAPTER_HIDDEN_DIM}"
        --latentloop_plan_parameter_match_tolerance "${LATENTLOOP_PLAN_PARAMETER_MATCH_TOLERANCE}"
        --latentloop_comparison_protocol "${LATENTLOOP_COMPARISON_PROTOCOL}"
    )
    local base_ckpt_args=()
    if [[ "${use_lrnode}" -eq 1 && -n "${LRNODE_EVAL_BASE_CKPT}" ]]; then
        if [[ ! -f "${LRNODE_EVAL_BASE_CKPT}" ]]; then
            echo "[ERROR] Missing LR-NODE eval base checkpoint: ${LRNODE_EVAL_BASE_CKPT}" >&2
            exit 1
        fi
        base_ckpt_args=(--finetune_from_pretrained_ckpt "${LRNODE_EVAL_BASE_CKPT}")
    fi

    echo "------------------------------------------------------------"
    echo "[RUN] label=${label}"
    echo "[RUN] run_name=${run_name}"
    echo "[RUN] ckpt=${ckpt_path}"
    if [[ "${#base_ckpt_args[@]}" -gt 0 ]]; then
        echo "[RUN] eval_base_ckpt=${LRNODE_EVAL_BASE_CKPT}"
    fi
    echo "[RUN] ckpt_tag=${ckpt_tag}"
    echo "[RUN] use_lrnode=${use_lrnode}, skip_full=${skip_full}, K=${query_interval}, full_query_hz=${effective_query_hz}"
    echo "[RUN] fastv_enabled=${FASTV_ENABLED}, layer=${FASTV_PRUNE_LAYER}, ratio=${FASTV_PRUNE_RATIO}, score_mode=${FASTV_SCORE_MODE}"
    echo "[RUN] refresh_policy=${LRNODE_EVAL_REFRESH_POLICY}, max_full_per_episode=${LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE}"
    echo "[RUN] mechanism_trace=${LRNODE_MECHANISM_TRACE}, counterfactual=${LRNODE_COUNTERFACTUAL_MODE}, ensemble=${EVAL_LIBERO_ENSEMBLING}"
    echo "[RUN] every_step_filter=${LRNODE_EVERY_STEP_FILTER_MODE}, alpha=${LRNODE_EVERY_STEP_FILTER_ALPHA}, beta=${LRNODE_EVERY_STEP_FILTER_BETA}, diagnostics=${LRNODE_EVERY_STEP_FILTER_DIAGNOSTICS}"
    echo "[RUN] ablation=${LRNODE_EVAL_ABLATION_MODE}, segment_grid=${LATENTLOOP_SEGMENT_GRID_ENABLE}, feedback=${LATENTLOOP_FEEDBACK_SCHEDULE}"
    echo "[RUN] renderer backend=${LIBERO_GL_BACKEND}, strict_actual=${LIBERO_GL_REQUIRE_ACTUAL}"
    echo "[RUN] video SAVE_VIDEO=${SAVE_VIDEO}, success=${SAVE_VIDEO_SUCC}, fail=${SAVE_VIDEO_FAIL}, all_ranks=${SAVE_VIDEO_ALL_RANKS}, stride=${VIDEO_STRIDE}"
    echo "[RUN] log_dir=${log_dir}"
    echo "[RUN] live_progress=${log_dir}/analysis/eval_progress.json"
    echo "------------------------------------------------------------"

    # eval_libero.py and several upstream resources are resolved relative to
    # the Seer source root. The comparison wrapper may be called from the
    # gnaroshi_vla repository root, so make that working directory explicit.
    (
        cd "${UPSTREAM_DIR}"
        python -m torch.distributed.run \
            --nnodes="${node}" \
            --nproc_per_node="${node_num}" \
            --master_port="${master_port}" \
            eval_libero.py \
            "${common_eval_args[@]}" \
            --run_name "${run_name}" \
            "${lrnode_args[@]}" \
            "${base_ckpt_args[@]}" \
            --resume_from_checkpoint "${ckpt_path}"
    ) | tee "${logfile}"

    for required in \
        "${log_dir}/analysis/eval_summary.json" \
        "${log_dir}/analysis/eval_progress.json" \
        "${log_dir}/analysis/eval_episode_metrics.csv" \
        "${log_dir}/analysis/eval_latency_profile.json" \
        "${log_dir}/analysis/args_snapshot_${ckpt_tag}.json"; do
        if [[ ! -f "${required}" ]]; then
            echo "[VERIFY][FAIL] Missing expected eval artifact: ${required}" >&2
            exit 1
        fi
    done

    if [[ "${SAVE_VIDEO}" == "1" ]]; then
        local video_count
        if [[ -d "${log_dir}/eval_videos" ]]; then
            video_count=$(find "${log_dir}/eval_videos" -type f \( -name '*.mp4' -o -name '*.gif' \) | wc -l)
        else
            video_count=0
        fi
        if [[ "${video_count}" -eq 0 ]]; then
            echo "[VERIFY][FAIL] SAVE_VIDEO=1 but no videos were saved under ${log_dir}/eval_videos" >&2
            exit 1
        fi
        echo "[VERIFY][OK] saved ${video_count} videos under ${log_dir}/eval_videos"
    fi

    echo "[VERIFY][OK] ${label} artifacts saved under ${log_dir}/analysis"
}

mkdir -p "${result_root}"
cat > "${result_root}/experiment_config.env" <<EOF
EXPERIMENT_NAME=${experiment_name}
EXPERIMENT_TAG=${experiment_tag}
LRNODE_PROTOCOL_ROOT=${protocol_root}
RESULT_ROOT=${result_root}
BASELINE_NAME=${BASELINE_NAME}
BASELINE_RUN_NAME=${BASELINE_RUN_NAME}
BASELINE_CKPT_ID=${BASELINE_CKPT_ID}
BASELINE_CKPT=${BASELINE_CKPT}
BASELINE_CKPT_TAG=${BASELINE_CKPT_TAG}
OURS_NAME=${OURS_NAME}
METHOD_TAG=${METHOD_TAG}
OURS_RUN_NAME=${OURS_RUN_NAME}
OURS_CKPT_ID=${OURS_CKPT_ID}
OURS_CKPT=${OURS_CKPT}
OURS_CKPT_TAG=${OURS_CKPT_TAG}
RUN_BASELINE=${RUN_BASELINE}
RUN_OURS_FULL=${RUN_OURS_FULL}
LRNODE_QUERY_INTERVALS_STR=${LRNODE_QUERY_INTERVALS_STR}
LRNODE_TRAIN_PROTOCOL=${LRNODE_TRAIN_PROTOCOL}
LRNODE_FREEZE_SEER_FOR_ADAPTER=${LRNODE_FREEZE_SEER_FOR_ADAPTER}
LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=${LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE}
LRNODE_EVAL_BASE_CKPT=${LRNODE_EVAL_BASE_CKPT}
LRNODE_EVAL_REFRESH_POLICY=${LRNODE_EVAL_REFRESH_POLICY}
LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE=${LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE}
LRNODE_EVAL_SHADOW_FULL_FORWARD=${LRNODE_EVAL_SHADOW_FULL_FORWARD}
LRNODE_MECHANISM_TRACE=${LRNODE_MECHANISM_TRACE}
LRNODE_TRACE_SAVE_LATENTS=${LRNODE_TRACE_SAVE_LATENTS}
LRNODE_TRACE_EPISODE_LIMIT=${LRNODE_TRACE_EPISODE_LIMIT}
LRNODE_COUNTERFACTUAL_MODE=${LRNODE_COUNTERFACTUAL_MODE}
LRNODE_COUNTERFACTUAL_MIX_STAGE=${LRNODE_COUNTERFACTUAL_MIX_STAGE}
LRNODE_LATENT_FUSION_ALPHA=${LRNODE_LATENT_FUSION_ALPHA}
LRNODE_LATENT_FUSION_MODE=${LRNODE_LATENT_FUSION_MODE}
LRNODE_MATCHED_RANDOM_SEED=${LRNODE_MATCHED_RANDOM_SEED}
LRNODE_MATCHED_RANDOM_NORM_MODE=${LRNODE_MATCHED_RANDOM_NORM_MODE}
LRNODE_EVERY_STEP_FILTER_MODE=${LRNODE_EVERY_STEP_FILTER_MODE}
LRNODE_EVERY_STEP_FILTER_ALPHA=${LRNODE_EVERY_STEP_FILTER_ALPHA}
LRNODE_EVERY_STEP_FILTER_BETA=${LRNODE_EVERY_STEP_FILTER_BETA}
LRNODE_EVERY_STEP_FILTER_DIAGNOSTICS=${LRNODE_EVERY_STEP_FILTER_DIAGNOSTICS}
LRNODE_EVAL_ABLATION_MODE=${LRNODE_EVAL_ABLATION_MODE}
LRNODE_NO_DELTA_MODE=${LRNODE_NO_DELTA_MODE}
LRNODE_CHUNK_TOKEN_POLICY=${LRNODE_CHUNK_TOKEN_POLICY}
LATENTLOOP_SEGMENT_GRID_ENABLE=${LATENTLOOP_SEGMENT_GRID_ENABLE}
LATENTLOOP_FEEDBACK_SCHEDULE=${LATENTLOOP_FEEDBACK_SCHEDULE}
LATENTLOOP_SAME_INPUT_STOCHASTICITY_REPEATS=${LATENTLOOP_SAME_INPUT_STOCHASTICITY_REPEATS}
LATENTLOOP_SAME_INPUT_STOCHASTICITY_OUTPUT=${LATENTLOOP_SAME_INPUT_STOCHASTICITY_OUTPUT}
LATENTLOOP_PLAN_TRACE=${LATENTLOOP_PLAN_TRACE}
LATENTLOOP_PLAN_TRACE_SAVE_LATENTS=${LATENTLOOP_PLAN_TRACE_SAVE_LATENTS}
LATENTLOOP_PLAN_TRACE_ROW_ID=${LATENTLOOP_PLAN_TRACE_ROW_ID}
LATENTLOOP_PLAN_TRACE_PAIRED_GROUP=${LATENTLOOP_PLAN_TRACE_PAIRED_GROUP}
LATENTLOOP_FEEDBACK_SOURCE=${LATENTLOOP_FEEDBACK_SOURCE}
LATENTLOOP_PLAN_ADAPTER_MODE=${LATENTLOOP_PLAN_ADAPTER_MODE}
LATENTLOOP_COMPARISON_PROTOCOL=${LATENTLOOP_COMPARISON_PROTOCOL}
EVAL_LIBERO_ENSEMBLING=${EVAL_LIBERO_ENSEMBLING}
EVAL_SUITE=${eval_suite}
EVAL_SEED=${EVAL_SEED}
LIBERO_EVAL_MAX_STEPS=${LIBERO_EVAL_MAX_STEPS}
EVAL_NUM_EPISODES_PER_TASK=${EVAL_NUM_EPISODES_PER_TASK:-20}
EVAL_NUM_TASKS=${EVAL_NUM_TASKS:-10}
SAVE_VIDEO=${SAVE_VIDEO}
SAVE_VIDEO_SUCC=${SAVE_VIDEO_SUCC}
SAVE_VIDEO_FAIL=${SAVE_VIDEO_FAIL}
SAVE_VIDEO_ALL_RANKS=${SAVE_VIDEO_ALL_RANKS}
VIDEO_FPS=${VIDEO_FPS}
VIDEO_STRIDE=${VIDEO_STRIDE}
EVAL_CONTROL_HZ=${EVAL_CONTROL_HZ}
FASTV_ENABLED=${FASTV_ENABLED}
FASTV_PRUNE_LAYER=${FASTV_PRUNE_LAYER}
FASTV_PRUNE_RATIO=${FASTV_PRUNE_RATIO}
FASTV_SCORE_MODE=${FASTV_SCORE_MODE}
FASTV_RETENTION_DIAGNOSTICS=${FASTV_RETENTION_DIAGNOSTICS}
NODE_NUM=${node_num}
MASTER_PORT=${master_port}
LIBERO_GL_BACKEND=${LIBERO_GL_BACKEND}
MUJOCO_GL=${MUJOCO_GL}
PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}
LIBERO_GL_REQUIRE_ACTUAL=${LIBERO_GL_REQUIRE_ACTUAL}
EOF

echo "[EXPERIMENT] result_root=${result_root}"
echo "[EXPERIMENT] baseline_name=${BASELINE_NAME}, baseline_run=${BASELINE_RUN_NAME}, baseline_ckpt=${BASELINE_CKPT}"
echo "[EXPERIMENT] ours_name=${OURS_NAME}, method=${METHOD_TAG}, ours_run=${OURS_RUN_NAME}, ours_ckpt=${OURS_CKPT}"
echo "[EXPERIMENT] run_baseline=${RUN_BASELINE}, run_ours_full=${RUN_OURS_FULL}, K intervals=${LRNODE_QUERY_INTERVALS[*]}"
echo "[EXPERIMENT] renderer backend=${LIBERO_GL_BACKEND}, strict_actual=${LIBERO_GL_REQUIRE_ACTUAL}"
echo "[EXPERIMENT] video SAVE_VIDEO=${SAVE_VIDEO}, success=${SAVE_VIDEO_SUCC}, fail=${SAVE_VIDEO_FAIL}, all_ranks=${SAVE_VIDEO_ALL_RANKS}, fps=${VIDEO_FPS}, stride=${VIDEO_STRIDE}"
echo "[EXPERIMENT] config saved: ${result_root}/experiment_config.env"

if [[ "${RUN_BASELINE}" -eq 1 ]]; then
    run_eval \
        "baseline_${BASELINE_NAME}_full" \
        "${BASELINE_RUN_NAME}" \
        "${BASELINE_CKPT}" \
        "${BASELINE_CKPT_TAG}" \
        0 \
        0 \
        1
fi

if [[ "${RUN_OURS_FULL}" -eq 1 ]]; then
    run_eval \
        "ours_${OURS_NAME}_full" \
        "${OURS_RUN_NAME}" \
        "${OURS_CKPT}" \
        "${OURS_CKPT_TAG}" \
        1 \
        0 \
        1
fi

for k in "${LRNODE_QUERY_INTERVALS[@]}"; do
    run_eval \
        "ours_${OURS_NAME}_skip" \
        "${OURS_RUN_NAME}" \
        "${OURS_CKPT}" \
        "${OURS_CKPT_TAG}" \
        1 \
        1 \
        "${k}"
done

RESULT_ROOT="${result_root}" \
BASELINE_NAME="${BASELINE_NAME}" \
BASELINE_RUN_NAME="${BASELINE_RUN_NAME}" \
BASELINE_CKPT="${BASELINE_CKPT}" \
BASELINE_CKPT_ID="${BASELINE_CKPT_ID}" \
OURS_NAME="${OURS_NAME}" \
METHOD_TAG="${METHOD_TAG}" \
OURS_RUN_NAME="${OURS_RUN_NAME}" \
OURS_CKPT="${OURS_CKPT}" \
OURS_CKPT_ID="${OURS_CKPT_ID}" \
python - <<'PY'
import csv
import json
import os
from pathlib import Path


def count_videos(run_dir: Path):
    video_root = run_dir / "eval_videos"
    success = 0
    fail = 0
    if video_root.exists():
        for path in video_root.glob("**/*"):
            if path.suffix.lower() not in {".mp4", ".gif"}:
                continue
            if "success" in path.parts:
                success += 1
            elif "fail" in path.parts:
                fail += 1
    return success, fail


root = Path(os.environ["RESULT_ROOT"])
rows = []
for path in sorted(root.glob("*/analysis/eval_summary.json")):
    run_dir = path.parents[1]
    data = json.loads(path.read_text())
    lr = data.get("lrnode", {})
    qr = data.get("query_reduction", {})
    smooth = data.get("action_smoothness", {})
    renderer = data.get("renderer_backend", {})
    success_videos, fail_videos = count_videos(run_dir)
    is_baseline = run_dir.name.startswith("baseline_")
    rows.append({
        "baseline_name": os.environ["BASELINE_NAME"],
        "baseline_run_name": os.environ["BASELINE_RUN_NAME"],
        "baseline_ckpt_id": os.environ["BASELINE_CKPT_ID"],
        "baseline_ckpt": os.environ["BASELINE_CKPT"],
        "ours_name": os.environ["OURS_NAME"],
        "method_tag": os.environ["METHOD_TAG"],
        "ours_run_name": os.environ["OURS_RUN_NAME"],
        "ours_ckpt_id": os.environ["OURS_CKPT_ID"],
        "ours_ckpt": os.environ["OURS_CKPT"],
        "run_dir": run_dir.name,
        "run_type": "baseline" if is_baseline else "ours",
        "renderer_backend": renderer.get("effective_backend"),
        "renderer_classification": renderer.get("backend_classification"),
        "gl_vendor": renderer.get("actual_gl_vendor"),
        "gl_renderer": renderer.get("actual_gl_renderer"),
        "render_gpu_device_id": renderer.get("render_gpu_device_id"),
        "success_rate_pct": round(float(data.get("success_rate", 0.0)) * 100.0, 3),
        "lrnode_enabled": lr.get("enabled"),
        "skip_full_forward": lr.get("eval_skip_full_forward"),
        "query_interval": lr.get("query_interval"),
        "eval_refresh_policy": lr.get("eval_refresh_policy", "periodic"),
        "max_full_forwards_per_episode": lr.get("max_full_forwards_per_episode", 1),
        "nominal_full_query_hz": lr.get("nominal_full_query_hz"),
        "effective_full_query_hz": lr.get("effective_full_query_hz"),
        "effective_lrnode_update_hz": lr.get("effective_lrnode_update_hz"),
        "effective_action_head_hz": lr.get("effective_action_head_hz"),
        "full_forward_calls": qr.get("num_full_forward_calls"),
        "lrnode_update_calls": qr.get("num_lrnode_update_calls"),
        "skip_action_head_calls": qr.get("num_skip_action_head_calls", qr.get("num_action_head_calls")),
        "total_action_head_calls": qr.get("num_total_action_head_calls"),
        "full_query_reduction_pct": round(float(qr.get("full_query_reduction_ratio", 0.0)) * 100.0, 3),
        "effective_query_interval": round(float(qr.get("effective_query_interval", 0.0)), 3),
        "avg_full_forward_ms": round(float(lr.get("avg_full_forward_latency_sec", 0.0)) * 1000.0, 3),
        "avg_full_action_head_ms": round(float(lr.get("avg_full_action_head_latency_sec", 0.0)) * 1000.0, 3),
        "avg_full_non_action_head_ms": round(float(lr.get("avg_full_non_action_head_latency_sec", 0.0)) * 1000.0, 3),
        "avg_lrnode_ms": round(float(lr.get("avg_lrnode_latency_sec", 0.0)) * 1000.0, 3),
        "avg_skip_action_head_ms": round(float(lr.get("avg_action_head_latency_sec", 0.0)) * 1000.0, 3),
        "avg_policy_step_ms": round(float(lr.get("avg_policy_step_latency_sec", 0.0)) * 1000.0, 3),
        "action_jerk_l2_mean": round(float(smooth.get("action_jerk_l2_mean", 0.0)), 6),
        "action_jerk_l2_p95": round(float(smooth.get("action_jerk_l2_p95", 0.0)), 6),
        "gripper_switch_rate": round(float(smooth.get("gripper_switch_rate", 0.0)), 6),
        "success_videos": success_videos,
        "fail_videos": fail_videos,
    })

out_path = root / "experiment_summary.csv"
if rows:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

print("[EXPERIMENT SUMMARY]")
for row in rows:
    print(row)
print(f"[EXPERIMENT SUMMARY] saved: {out_path}")
PY

echo "[DONE] LR-NODE comparison completed: ${result_root}"
