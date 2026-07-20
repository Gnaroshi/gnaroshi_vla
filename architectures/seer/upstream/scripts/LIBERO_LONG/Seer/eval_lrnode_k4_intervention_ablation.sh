#!/bin/bash

set -euo pipefail

# LR-NODE K=4 intervention ablations for the frozen-baseline distill protocol.
#
# This script does not train or modify weights. It evaluates:
#   1) baseline_k1_full
#   2) ours_k1_full
#   3) ours_k4_stepwise
#   4) ours_k4_hold_action
#   5) ours_k4_hold_latent
#   6) ours_k4_seer_token_chunk
#   7) ours_k4_no_delta
#
# Interpretation rules:
# - stepwise strong, hold/chunk weak:
#     supports learned delta-conditioned latent update.
# - seer_token_chunk similarly strong:
#     Seer native action chunking may explain part of the K=4 gain.
# - hold_action or hold_latent strong:
#     sparse refresh / temporal persistence may explain part of the gain.
# - no_delta similar to stepwise:
#     visual/proprio delta is not the main contributor.
# - no_delta weak while stepwise strong:
#     delta-conditioned update is important.
#
# Terminology:
# This is delta-conditioned fixed-Euler latent dynamics. It is not an NCDE,
# not an adaptive ODE solver, and not evidence by itself that the learned
# dynamics caused the K=4 improvement. These ablations are meant to test that.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${UPSTREAM_DIR}/../../.." && pwd)"

export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
export EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ:-20}"
export EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ:-1}"
export EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ:-1}"

if [[ "${DEBUG_ABLATION:-0}" == "1" ]]; then
    export EVAL_NUM_TASKS="${EVAL_NUM_TASKS:-1}"
    export EVAL_NUM_EPISODES_PER_TASK="${EVAL_NUM_EPISODES_PER_TASK:-2}"
    export LIBERO_EVAL_MAX_STEPS="${LIBERO_EVAL_MAX_STEPS:-120}"
    export VIDEO_STRIDE="${VIDEO_STRIDE:-4}"
    echo "[DEBUG] DEBUG_ABLATION=1: tasks=${EVAL_NUM_TASKS}, episodes_per_task=${EVAL_NUM_EPISODES_PER_TASK}, max_steps=${LIBERO_EVAL_MAX_STEPS}"
else
    export LIBERO_EVAL_MAX_STEPS="${LIBERO_EVAL_MAX_STEPS:-600}"
fi

safe_tag() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    value="${value//:/_}"
    value="${value//,/}"
    echo "${value}"
}

protocol_root="${LRNODE_PROTOCOL_ROOT:-${REPO_ROOT}/results/seer/lrnode/default}"
latest_baseline="${protocol_root}/train/_latest/scratch.env"
latest_distill="${protocol_root}/train/_latest/distill_node.env"

if [[ -z "${BASELINE_CKPT:-}" && -z "${BASELINE_RUN_NAME:-}" && -z "${BASELINE_CKPT_ROOT:-}" ]]; then
    if [[ ! -f "${latest_baseline}" ]]; then
        echo "[ERROR] Missing baseline env: ${latest_baseline}" >&2
        echo "[ERROR] Set BASELINE_CKPT explicitly or run scratch.sh first." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${latest_baseline}"
    BASELINE_RUN_NAME="${LRNODE_RUN_NAME}"
    BASELINE_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
    echo "[EVAL INFO] loaded baseline run from ${latest_baseline}"
fi

BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-33}"
BASELINE_CKPT="${BASELINE_CKPT:-${BASELINE_CKPT_ROOT}/${BASELINE_RUN_NAME}/${BASELINE_CKPT_ID}.pth}"

if [[ -z "${OURS_CKPT:-}" && -z "${OURS_RUN_NAME:-}" && -z "${OURS_CKPT_ROOT:-}" ]]; then
    if [[ ! -f "${latest_distill}" ]]; then
        echo "[ERROR] Missing distill env: ${latest_distill}" >&2
        echo "[ERROR] Set OURS_CKPT explicitly or run distill_node.sh first." >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${latest_distill}"
    OURS_RUN_NAME="${LRNODE_RUN_NAME}"
    OURS_CKPT_ROOT="${LRNODE_SAVE_CHECKPOINT_PATH}"
    echo "[EVAL INFO] loaded distill run from ${latest_distill}"
fi

OURS_CKPT_ID="${OURS_CKPT_ID:-39}"
OURS_CKPT="${OURS_CKPT:-${OURS_CKPT_ROOT}/${OURS_RUN_NAME}/${OURS_CKPT_ID}.pth}"

if [[ ! -f "${BASELINE_CKPT}" ]]; then
    echo "[ERROR] Missing baseline checkpoint: ${BASELINE_CKPT}" >&2
    exit 1
fi
if [[ ! -f "${OURS_CKPT}" ]]; then
    echo "[ERROR] Missing LR-NODE checkpoint: ${OURS_CKPT}" >&2
    exit 1
fi

BASELINE_NAME="$(safe_tag "${BASELINE_NAME:-seer_scratch_baseline}")"
OURS_NAME="$(safe_tag "${OURS_NAME:-lrnode_distill_ckpt${OURS_CKPT_ID}}")"
EXPERIMENT_TAG="$(safe_tag "${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}")"
RESULT_ROOT="${RESULT_ROOT:-${protocol_root}/eval/lrnode_k4_intervention_ablation_${EXPERIMENT_TAG}}"

which_server="${WHICH_SERVER:-sd1}"
dataset="${DATASET:-libero_10_converted}"
libero_path="${LIBERO_PATH:-}"
vit_checkpoint_path="${VIT_CHECKPOINT_PATH:-${UPSTREAM_DIR}/checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
save_checkpoint_path="${SAVE_CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/seer}"
node=1
node_num="${NODE_NUM:-4}"
master_port_base="${MASTER_PORT:-12740}"

if [[ -z "${libero_path}" || ! -d "${libero_path}" ]]; then
    echo "[ERROR] Set LIBERO_PATH to a valid LIBERO repository path." >&2
    exit 1
fi
if [[ ! -f "${vit_checkpoint_path}" ]]; then
    echo "[ERROR] Missing ViT checkpoint: ${vit_checkpoint_path}" >&2
    exit 1
fi
export PYTHONPATH="${libero_path}:${PYTHONPATH:-}"

# LR-NODE architecture/config flags must match the distill checkpoint.
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
LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD="${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD:-1}"

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
    --seed 42
    --batch_size 64
    --precision fp32
    --weight_decay 1e-4
    --num_resampler_query 6
    --transformer_layers 24
    --phase "evaluate"
    --finetune_type "libero_10"
    --save_checkpoint_path "${save_checkpoint_path}"
    --action_pred_steps 3
    --future_steps 3
    --sequence_length 7
    --obs_pred
    --gripper_width
    --eval_libero_ensembling
    --multi_step_action 1
    --libero_eval_max_steps "${LIBERO_EVAL_MAX_STEPS}"
)

mkdir -p "${RESULT_ROOT}"
manifest="${RESULT_ROOT}/run_manifest.csv"
printf "variant_name,ablation_mode,k,control_hz,checkpoint,log_dir\n" > "${manifest}"

cat > "${RESULT_ROOT}/experiment_config.env" <<EOF
EXPERIMENT=lrnode_k4_intervention_ablation
EXPERIMENT_TAG=${EXPERIMENT_TAG}
RESULT_ROOT=${RESULT_ROOT}
BASELINE_NAME=${BASELINE_NAME}
BASELINE_RUN_NAME=${BASELINE_RUN_NAME}
BASELINE_CKPT_ID=${BASELINE_CKPT_ID}
BASELINE_CKPT=${BASELINE_CKPT}
OURS_NAME=${OURS_NAME}
OURS_RUN_NAME=${OURS_RUN_NAME}
OURS_CKPT_ID=${OURS_CKPT_ID}
OURS_CKPT=${OURS_CKPT}
EVAL_CONTROL_HZ=${EVAL_CONTROL_HZ}
NODE_NUM=${node_num}
MASTER_PORT_BASE=${master_port_base}
SAVE_VIDEO=${SAVE_VIDEO}
SAVE_VIDEO_SUCC=${SAVE_VIDEO_SUCC}
SAVE_VIDEO_FAIL=${SAVE_VIDEO_FAIL}
SAVE_VIDEO_ALL_RANKS=${SAVE_VIDEO_ALL_RANKS}
VIDEO_STRIDE=${VIDEO_STRIDE}
LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD}
DEBUG_ABLATION=${DEBUG_ABLATION:-0}
EVAL_NUM_TASKS=${EVAL_NUM_TASKS:-10}
EVAL_NUM_EPISODES_PER_TASK=${EVAL_NUM_EPISODES_PER_TASK:-20}
LIBERO_EVAL_MAX_STEPS=${LIBERO_EVAL_MAX_STEPS}
EOF

echo "[EXPERIMENT] result_root=${RESULT_ROOT}"
echo "[EXPERIMENT] baseline=${BASELINE_CKPT}"
echo "[EXPERIMENT] ours=${OURS_CKPT}"
echo "[EXPERIMENT] control_hz=${EVAL_CONTROL_HZ}, node_num=${node_num}, cuda=${CUDA_VISIBLE_DEVICES}"
echo "[EXPERIMENT] config=${RESULT_ROOT}/experiment_config.env"

run_idx=0
run_eval() {
    local variant="$1"
    local ckpt="$2"
    local use_lrnode="$3"
    local skip_full="$4"
    local k="$5"
    local ablation_mode="$6"
    local no_delta_mode="$7"
    local chunk_policy="$8"
    local base_ckpt="${9:-}"

    run_idx=$((run_idx + 1))
    local log_dir="${RESULT_ROOT}/${variant}"
    local logfile="${log_dir}/${variant}.log"
    local master_port=$((master_port_base + run_idx))
    mkdir -p "${log_dir}"

    local base_args=()
    if [[ -n "${base_ckpt}" ]]; then
        base_args=(--finetune_from_pretrained_ckpt "${base_ckpt}")
    fi

    export LOG_DIR="${log_dir}"
    export RUN_NAME="${variant}"
    export CKPT_TAG="${variant}"
    export EFFECTIVE_QUERY_HZ
    EFFECTIVE_QUERY_HZ="$(awk -v hz="${EVAL_CONTROL_HZ}" -v kk="${k}" 'BEGIN { printf "%.6f", hz / kk }')"

    echo "------------------------------------------------------------"
    echo "[RUN] ${variant}"
    echo "[RUN] ckpt=${ckpt}"
    if [[ -n "${base_ckpt}" ]]; then
        echo "[RUN] base_ckpt=${base_ckpt}"
    fi
    echo "[RUN] use_lrnode=${use_lrnode}, skip_full=${skip_full}, K=${k}, mode=${ablation_mode}"
    echo "[RUN] log_dir=${log_dir}"
    echo "------------------------------------------------------------"

    python -m torch.distributed.run \
        --nnodes="${node}" \
        --nproc_per_node="${node_num}" \
        --master_port="${master_port}" \
        eval_libero.py \
        "${common_eval_args[@]}" \
        --run_name "${variant}" \
        --use_lrnode_latent_update "${use_lrnode}" \
        --lrnode_eval_skip_full_forward "${skip_full}" \
        --lrnode_train_protocol "adapter" \
        --lrnode_freeze_seer_for_adapter "1" \
        --lrnode_assert_only_lrnode_trainable "1" \
        --lrnode_query_interval "${k}" \
        --lrnode_eval_ablation_mode "${ablation_mode}" \
        --lrnode_no_delta_mode "${no_delta_mode}" \
        --lrnode_chunk_token_policy "${chunk_policy}" \
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
        --lrnode_eval_profile_full_action_head "${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD}" \
        "${base_args[@]}" \
        --resume_from_checkpoint "${ckpt}" | tee "${logfile}"

    for required in \
        "${log_dir}/analysis/eval_summary.json" \
        "${log_dir}/analysis/eval_episode_metrics.csv" \
        "${log_dir}/analysis/eval_latency_profile.json" \
        "${log_dir}/analysis/args_snapshot_${variant}.json"; do
        if [[ ! -f "${required}" ]]; then
            echo "[VERIFY][FAIL] Missing expected artifact: ${required}" >&2
            exit 1
        fi
    done

    printf "%s,%s,%s,%s,%s,%s\n" \
        "${variant}" "${ablation_mode}" "${k}" "${EVAL_CONTROL_HZ}" "${ckpt}" "${log_dir}" >> "${manifest}"
}

run_eval "baseline_k1_full" "${BASELINE_CKPT}" 0 0 1 "stepwise" "zero" "skip_only" ""
run_eval "ours_k1_full" "${OURS_CKPT}" 1 0 1 "stepwise" "zero" "skip_only" "${BASELINE_CKPT}"
run_eval "ours_k4_stepwise" "${OURS_CKPT}" 1 1 4 "stepwise" "zero" "skip_only" "${BASELINE_CKPT}"
run_eval "ours_k4_hold_action" "${OURS_CKPT}" 1 1 4 "hold_action" "zero" "skip_only" "${BASELINE_CKPT}"
run_eval "ours_k4_hold_latent" "${OURS_CKPT}" 1 1 4 "hold_latent" "zero" "skip_only" "${BASELINE_CKPT}"
run_eval "ours_k4_seer_token_chunk" "${OURS_CKPT}" 1 1 4 "seer_token_chunk" "zero" "skip_only" "${BASELINE_CKPT}"
run_eval "ours_k4_no_delta" "${OURS_CKPT}" 1 1 4 "no_delta" "zero" "skip_only" "${BASELINE_CKPT}"

RESULT_ROOT="${RESULT_ROOT}" python - <<'PY'
import csv
import json
import os
from pathlib import Path


root = Path(os.environ["RESULT_ROOT"])
manifest_path = root / "run_manifest.csv"
manifest_rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))


def read_summary(row):
    path = Path(row["log_dir"]) / "analysis" / "eval_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


summary_rows = []
for row in manifest_rows:
    data = read_summary(row)
    lr = data.get("lrnode", {})
    qr = data.get("query_reduction", {})
    smooth = data.get("action_smoothness", {})
    summary_rows.append({
        "run_name": row["variant_name"],
        "ablation_mode": row["ablation_mode"],
        "K": row["k"],
        "control_freq": data.get("control_freq", data.get("environment", {}).get("control_freq", "")),
        "success_rate": data.get("success_rate", 0.0),
        "success_rate_pct": round(float(data.get("success_rate", 0.0)) * 100.0, 3),
        "full_query_reduction": data.get("full_query_reduction_ratio", qr.get("full_query_reduction_ratio", 0.0)),
        "effective_full_query_hz": data.get("effective_full_query_hz", lr.get("effective_full_query_hz", 0.0)),
        "effective_lrnode_update_hz": data.get("effective_lrnode_update_hz", lr.get("effective_lrnode_update_hz", 0.0)),
        "effective_action_head_hz": data.get("effective_action_head_hz", lr.get("effective_action_head_hz", 0.0)),
        "avg_policy_step_latency_ms": data.get("avg_policy_step_latency_ms", 0.0),
        "avg_full_forward_latency_ms": data.get("avg_full_forward_latency_ms", 0.0),
        "avg_full_action_head_latency_ms": data.get("avg_full_action_head_latency_ms", 0.0),
        "avg_full_non_action_head_latency_ms": data.get("avg_full_non_action_head_latency_ms", 0.0),
        "avg_lrnode_latency_ms": data.get("avg_lrnode_latency_ms", 0.0),
        "avg_fast_encoder_latency_ms": data.get("avg_fast_encoder_latency_ms", 0.0),
        "avg_action_head_latency_ms": data.get("avg_action_head_latency_ms", 0.0),
        "avg_skip_action_head_latency_ms": data.get(
            "avg_skip_action_head_latency_ms",
            data.get("avg_action_head_latency_ms", 0.0),
        ),
        "action_delta_l2_mean": data.get("action_delta_l2_mean", smooth.get("action_delta_l2_mean", 0.0)),
        "action_delta_l2_p95": data.get("action_delta_l2_p95", smooth.get("action_delta_l2_p95", 0.0)),
        "action_jerk_l2_mean": data.get("action_jerk_l2_mean", smooth.get("action_jerk_l2_mean", 0.0)),
        "action_jerk_l2_p95": data.get("action_jerk_l2_p95", smooth.get("action_jerk_l2_p95", 0.0)),
        "gripper_switch_rate": data.get("gripper_switch_rate", smooth.get("gripper_switch_rate", 0.0)),
        "num_env_steps": data.get("num_env_steps", qr.get("num_env_steps", 0)),
        "num_full_forward_calls": data.get("num_full_forward_calls", qr.get("num_full_forward_calls", 0)),
        "num_lrnode_update_calls": data.get("num_lrnode_update_calls", qr.get("num_lrnode_update_calls", 0)),
        "num_fast_encoder_calls": data.get("num_fast_encoder_calls", qr.get("num_fast_encoder_calls", 0)),
        "num_action_head_calls": data.get("num_action_head_calls", qr.get("num_action_head_calls", 0)),
        "num_skip_action_head_calls": data.get(
            "num_skip_action_head_calls",
            qr.get("num_skip_action_head_calls", qr.get("num_action_head_calls", 0)),
        ),
        "num_total_action_head_calls": data.get(
            "num_total_action_head_calls",
            qr.get("num_total_action_head_calls", 0),
        ),
        "num_hold_action_steps": data.get("num_hold_action_steps", qr.get("num_hold_action_steps", 0)),
        "num_hold_latent_steps": data.get("num_hold_latent_steps", qr.get("num_hold_latent_steps", 0)),
        "num_chunk_token_steps": data.get("num_chunk_token_steps", qr.get("num_chunk_token_steps", 0)),
        "num_no_delta_steps": data.get("num_no_delta_steps", qr.get("num_no_delta_steps", 0)),
    })

summary_path = root / "experiment_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)


def read_episode_success(row):
    path = Path(row["log_dir"]) / "analysis" / "eval_episode_metrics.csv"
    out = {}
    for item in csv.DictReader(path.open("r", encoding="utf-8")):
        key = (int(item["task_id"]), int(item["episode_id"]))
        out[key] = int(float(item["success"]))
    return out


baseline_row = next(row for row in manifest_rows if row["variant_name"] == "baseline_k1_full")
baseline = read_episode_success(baseline_row)
flip_rows = []
for row in manifest_rows:
    if row["variant_name"] == "baseline_k1_full":
        continue
    variant = read_episode_success(row)
    common_keys = sorted(set(baseline) & set(variant))
    fail_to_success = sum(1 for key in common_keys if baseline[key] == 0 and variant[key] == 1)
    success_to_fail = sum(1 for key in common_keys if baseline[key] == 1 and variant[key] == 0)
    same_success = sum(1 for key in common_keys if baseline[key] == 1 and variant[key] == 1)
    same_fail = sum(1 for key in common_keys if baseline[key] == 0 and variant[key] == 0)
    flip_rows.append({
        "variant_name": row["variant_name"],
        "baseline_success": sum(baseline[key] for key in common_keys),
        "variant_success": sum(variant[key] for key in common_keys),
        "fail_to_success_count": fail_to_success,
        "success_to_fail_count": success_to_fail,
        "same_success_count": same_success,
        "same_fail_count": same_fail,
        "net_gain": fail_to_success - success_to_fail,
        "total_episodes": len(common_keys),
    })

flip_path = root / "paired_flip_summary.csv"
with flip_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(flip_rows[0].keys()))
    writer.writeheader()
    writer.writerows(flip_rows)

print("[SUMMARY] experiment_summary.csv")
for row in summary_rows:
    print(row)
print(f"[SUMMARY] saved: {summary_path}")
print("[SUMMARY] paired_flip_summary.csv")
for row in flip_rows:
    print(row)
print(f"[SUMMARY] saved: {flip_path}")
PY

echo "[DONE] LR-NODE K=4 intervention ablation completed: ${RESULT_ROOT}"
