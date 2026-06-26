#!/bin/bash

set -euo pipefail

# DISTILL-EXTREME-HZUP-FIRSTONLY
#
# Purpose:
#   Raise actual LIBERO control_freq while allowing only the first policy step
#   of each episode to query full Seer. This tests the extreme claim:
#     high-rate control with almost no repeated VLA querying.
#
# Rows:
#   for each HZ in HZS_STR:
#     adapter-composed first_only rollout
#   for the first HZ only:
#     baseline full and adapter-composed full references

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
experiment_name="${EXPERIMENT_NAME:-lrnode_distill_extreme_hzup_firstonly}"
experiment_tag="${EXPERIMENT_TAG:-distill_extreme_hzup_firstonly_$(date +%Y%m%d_%H%M%S)}"
result_root="${EVAL_RESULT_ROOT:-${protocol_root}/eval/${experiment_name}_${experiment_tag}}"
launch_root="${result_root}/_launch_logs"
mkdir -p "${launch_root}"

export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
export EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ:-20}"
export EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ:-1}"
export EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ:-1}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
export LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"

hzs_str="${HZS_STR:-20 40 60 80}"
read -r -a hzs <<< "${hzs_str}"

master_port_base="${MASTER_PORT:-12680}"
node_num="${NODE_NUM:-4}"

cat > "${result_root}/experiment_config.env" <<EOF
SCRIPT=scripts/LIBERO_LONG/Seer/eval_lrnode_distill_extreme_hzup_firstonly.sh
EXPERIMENT_NAME=${experiment_name}
EXPERIMENT_TAG=${experiment_tag}
RESULT_ROOT=${result_root}
HZS_STR=${hzs_str}
MASTER_PORT_BASE=${master_port_base}
NODE_NUM=${node_num}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
SAVE_VIDEO=${SAVE_VIDEO}
SAVE_VIDEO_SUCC=${SAVE_VIDEO_SUCC}
SAVE_VIDEO_FAIL=${SAVE_VIDEO_FAIL}
SAVE_VIDEO_ALL_RANKS=${SAVE_VIDEO_ALL_RANKS}
VIDEO_STRIDE=${VIDEO_STRIDE}
EVAL_BASE_CONTROL_HZ=${EVAL_BASE_CONTROL_HZ}
EVAL_SCALE_MAX_STEPS_WITH_HZ=${EVAL_SCALE_MAX_STEPS_WITH_HZ}
EVAL_SCALE_SETTLE_STEPS_WITH_HZ=${EVAL_SCALE_SETTLE_STEPS_WITH_HZ}
LRNODE_EVAL_STEP_LOG=${LRNODE_EVAL_STEP_LOG}
LRNODE_EVAL_SHADOW_FULL_FORWARD=${LRNODE_EVAL_SHADOW_FULL_FORWARD}
EOF

echo "[DISTILL EXTREME HZUP FIRSTONLY] result_root=${result_root}"
echo "[DISTILL EXTREME HZUP FIRSTONLY] hzs=${hzs_str}"

idx=0
for hz in "${hzs[@]}"; do
    hz_tag="${hz//./p}"
    hz_root="${result_root}/hz_${hz_tag}_firstonly"
    hz_log="${launch_root}/hz_${hz_tag}_firstonly.log"
    port=$((master_port_base + idx))
    idx=$((idx + 1))

    if [[ "${hz}" == "${hzs[0]}" ]]; then
        run_baseline=1
        run_ours_full=1
    else
        run_baseline=0
        run_ours_full=0
    fi

    video_fps="$(awk -v hz="${hz}" 'BEGIN { printf "%d", hz }')"

    echo "------------------------------------------------------------"
    echo "[DISTILL EXTREME HZUP FIRSTONLY RUN] control_hz=${hz}, port=${port}"
    echo "[DISTILL EXTREME HZUP FIRSTONLY RUN] result=${hz_root}"
    echo "------------------------------------------------------------"

    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_TAG="${experiment_tag}_hz_${hz_tag}_firstonly" \
    RESULT_ROOT="${hz_root}" \
    RUN_BASELINE="${run_baseline}" \
    RUN_OURS_FULL="${run_ours_full}" \
    LRNODE_QUERY_INTERVALS_STR="1" \
    LRNODE_EVAL_REFRESH_POLICY="first_only" \
    LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE="1" \
    EVAL_CONTROL_HZ="${hz}" \
    EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ}" \
    EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ}" \
    EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ}" \
    VIDEO_FPS="${video_fps}" \
    MASTER_PORT="${port}" \
    NODE_NUM="${node_num}" \
    bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh 2>&1 | tee "${hz_log}"
done

RESULT_ROOT="${result_root}" python - <<'PY'
import csv
import json
from pathlib import Path
import os

root = Path(os.environ["RESULT_ROOT"])
rows = []
for path in sorted(root.glob("hz_*_firstonly/*/analysis/eval_summary.json")):
    data = json.loads(path.read_text())
    lr = data.get("lrnode", {})
    qr = data.get("query_reduction", {})
    smooth = data.get("action_smoothness", {})
    env = data.get("environment", {})
    control_hz = float(lr.get("control_hz", env.get("control_hz", 20.0)))
    policy_ms = float(lr.get("avg_policy_step_latency_sec", 0.0)) * 1000.0
    budget_ms = 1000.0 / control_hz if control_hz > 0 else 0.0
    rows.append({
        "hz_dir": path.parents[2].name,
        "run_dir": path.parents[1].name,
        "success_rate_pct": round(float(data.get("success_rate", 0.0)) * 100.0, 3),
        "refresh_policy": lr.get("eval_refresh_policy", "periodic"),
        "control_hz": control_hz,
        "env_control_freq": env.get("control_freq"),
        "eval_max_steps": env.get("eval_max_steps"),
        "effective_full_query_hz": round(float(lr.get("effective_full_query_hz", 0.0)), 3),
        "effective_lrnode_update_hz": round(float(lr.get("effective_lrnode_update_hz", 0.0)), 3),
        "full_forward_calls": qr.get("num_full_forward_calls"),
        "lrnode_update_calls": qr.get("num_lrnode_update_calls"),
        "full_query_reduction_pct": round(float(qr.get("full_query_reduction_ratio", 0.0)) * 100.0, 3),
        "effective_query_interval": round(float(qr.get("effective_query_interval", 0.0)), 3),
        "avg_policy_step_ms": round(policy_ms, 3),
        "policy_budget_ms": round(budget_ms, 3),
        "policy_latency_over_budget": round(policy_ms / budget_ms, 3) if budget_ms > 0 else 0.0,
        "avg_full_forward_ms": round(float(lr.get("avg_full_forward_latency_sec", 0.0)) * 1000.0, 3),
        "avg_lrnode_ms": round(float(lr.get("avg_lrnode_latency_sec", 0.0)) * 1000.0, 3),
        "action_jerk_l2_p95": round(float(smooth.get("action_jerk_l2_p95", 0.0)), 6),
        "summary_path": str(path),
    })

csv_path = root / "extreme_hzup_firstonly_summary.csv"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[DISTILL EXTREME HZUP FIRSTONLY SUMMARY] saved: {csv_path}")
for row in rows:
    print(row)
PY

echo "[DISTILL EXTREME HZUP FIRSTONLY DONE] ${result_root}"
