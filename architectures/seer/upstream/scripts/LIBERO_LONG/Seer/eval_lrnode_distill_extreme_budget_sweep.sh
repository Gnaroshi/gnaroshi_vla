#!/bin/bash

set -euo pipefail

# DISTILL-EXTREME-BUDGET-SWEEP
#
# Purpose:
#   At normal LIBERO control_freq=20, measure how many full-Seer refreshes per
#   episode are needed before LR-NODE recovers performance.
#
# Rows:
#   one reference block with baseline full and adapter-composed full
#   adapter-composed fixed_budget rollouts with B in FULL_BUDGETS_STR
#
# Definition:
#   fixed_budget B spreads B full-Seer calls across the episode horizon. Between
#   those calls, actions come from LR-NODE updates.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
experiment_name="${EXPERIMENT_NAME:-lrnode_distill_extreme_budget_sweep}"
experiment_tag="${EXPERIMENT_TAG:-distill_extreme_budget_sweep_$(date +%Y%m%d_%H%M%S)}"
result_root="${EVAL_RESULT_ROOT:-${protocol_root}/eval/${experiment_name}_${experiment_tag}}"
launch_root="${result_root}/_launch_logs"
mkdir -p "${launch_root}"

export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
export LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"

budgets_str="${FULL_BUDGETS_STR:-1 2 4 8}"
read -r -a budgets <<< "${budgets_str}"

master_port_base="${MASTER_PORT:-12670}"
node_num="${NODE_NUM:-4}"

cat > "${result_root}/experiment_config.env" <<EOF
SCRIPT=scripts/LIBERO_LONG/Seer/eval_lrnode_distill_extreme_budget_sweep.sh
EXPERIMENT_NAME=${experiment_name}
EXPERIMENT_TAG=${experiment_tag}
RESULT_ROOT=${result_root}
FULL_BUDGETS_STR=${budgets_str}
MASTER_PORT_BASE=${master_port_base}
NODE_NUM=${node_num}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
EVAL_CONTROL_HZ=${EVAL_CONTROL_HZ}
SAVE_VIDEO=${SAVE_VIDEO}
SAVE_VIDEO_SUCC=${SAVE_VIDEO_SUCC}
SAVE_VIDEO_FAIL=${SAVE_VIDEO_FAIL}
SAVE_VIDEO_ALL_RANKS=${SAVE_VIDEO_ALL_RANKS}
LRNODE_EVAL_STEP_LOG=${LRNODE_EVAL_STEP_LOG}
LRNODE_EVAL_SHADOW_FULL_FORWARD=${LRNODE_EVAL_SHADOW_FULL_FORWARD}
EOF

echo "[DISTILL EXTREME BUDGET] result_root=${result_root}"
echo "[DISTILL EXTREME BUDGET] budgets=${budgets_str}"

idx=0
for budget in "${budgets[@]}"; do
    if [[ "${budget}" -lt 1 ]]; then
        echo "[ERROR] FULL_BUDGETS_STR contains invalid budget '${budget}'." >&2
        exit 1
    fi

    budget_root="${result_root}/budget_B${budget}"
    budget_log="${launch_root}/budget_B${budget}.log"
    port=$((master_port_base + idx))
    idx=$((idx + 1))

    if [[ "${budget}" == "${budgets[0]}" ]]; then
        run_baseline=1
        run_ours_full=1
    else
        run_baseline=0
        run_ours_full=0
    fi

    echo "------------------------------------------------------------"
    echo "[DISTILL EXTREME BUDGET RUN] B=${budget}, port=${port}"
    echo "[DISTILL EXTREME BUDGET RUN] result=${budget_root}"
    echo "------------------------------------------------------------"

    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_TAG="${experiment_tag}_B${budget}" \
    RESULT_ROOT="${budget_root}" \
    RUN_BASELINE="${run_baseline}" \
    RUN_OURS_FULL="${run_ours_full}" \
    LRNODE_QUERY_INTERVALS_STR="1" \
    LRNODE_EVAL_REFRESH_POLICY="fixed_budget" \
    LRNODE_EVAL_MAX_FULL_FORWARDS_PER_EPISODE="${budget}" \
    EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ}" \
    MASTER_PORT="${port}" \
    NODE_NUM="${node_num}" \
    bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh 2>&1 | tee "${budget_log}"
done

RESULT_ROOT="${result_root}" python - <<'PY'
import csv
import json
from pathlib import Path
import os

root = Path(os.environ["RESULT_ROOT"])
rows = []
for path in sorted(root.glob("budget_B*/*/analysis/eval_summary.json")):
    data = json.loads(path.read_text())
    lr = data.get("lrnode", {})
    qr = data.get("query_reduction", {})
    smooth = data.get("action_smoothness", {})
    rows.append({
        "budget_dir": path.parents[2].name,
        "run_dir": path.parents[1].name,
        "success_rate_pct": round(float(data.get("success_rate", 0.0)) * 100.0, 3),
        "refresh_policy": lr.get("eval_refresh_policy", "periodic"),
        "max_full_forwards_per_episode": lr.get("max_full_forwards_per_episode", 1),
        "control_hz": lr.get("control_hz", 20.0),
        "effective_full_query_hz": round(float(lr.get("effective_full_query_hz", 0.0)), 3),
        "effective_lrnode_update_hz": round(float(lr.get("effective_lrnode_update_hz", 0.0)), 3),
        "num_env_steps": qr.get("num_env_steps"),
        "full_forward_calls": qr.get("num_full_forward_calls"),
        "lrnode_update_calls": qr.get("num_lrnode_update_calls"),
        "full_query_reduction_pct": round(float(qr.get("full_query_reduction_ratio", 0.0)) * 100.0, 3),
        "effective_query_interval": round(float(qr.get("effective_query_interval", 0.0)), 3),
        "avg_policy_step_ms": round(float(lr.get("avg_policy_step_latency_sec", 0.0)) * 1000.0, 3),
        "avg_full_forward_ms": round(float(lr.get("avg_full_forward_latency_sec", 0.0)) * 1000.0, 3),
        "avg_lrnode_ms": round(float(lr.get("avg_lrnode_latency_sec", 0.0)) * 1000.0, 3),
        "action_jerk_l2_p95": round(float(smooth.get("action_jerk_l2_p95", 0.0)), 6),
        "summary_path": str(path),
    })

csv_path = root / "extreme_budget_sweep_summary.csv"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[DISTILL EXTREME BUDGET SUMMARY] saved: {csv_path}")
for row in rows:
    print(row)
PY

echo "[DISTILL EXTREME BUDGET DONE] ${result_root}"
