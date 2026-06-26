#!/bin/bash

set -euo pipefail

# DISTILL-HZUP20Q
#
# Purpose:
#   Raise actual LIBERO control_freq while keeping expensive full-Seer query
#   near 20 Hz, using a frozen-baseline LR-NODE adapter checkpoint.
#
# Default rows:
#   20:1, 40:1, 40:2, 60:1, 60:3, 80:1, 80:4
#
# K=1 rows run baseline full and adapter-composed full references.
# K>1 rows run adapter-composed LR-NODE skip only.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
experiment_name="${EXPERIMENT_NAME:-lrnode_distill_hzup20q}"
experiment_tag="${EXPERIMENT_TAG:-distill_hzup20q_$(date +%Y%m%d_%H%M%S)}"
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

hz_k_pairs_str="${HZ_K_PAIRS_STR:-20:1 40:1 40:2 60:1 60:3 80:1 80:4}"
read -r -a hz_k_pairs <<< "${hz_k_pairs_str}"

master_port_base="${MASTER_PORT:-12650}"
node_num="${NODE_NUM:-4}"

cat > "${result_root}/experiment_config.env" <<EOF
SCRIPT=scripts/LIBERO_LONG/Seer/eval_lrnode_distill_hzup20q.sh
EXPERIMENT_NAME=${experiment_name}
EXPERIMENT_TAG=${experiment_tag}
RESULT_ROOT=${result_root}
HZ_K_PAIRS_STR=${hz_k_pairs_str}
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
EOF

echo "[DISTILL HZUP20Q] result_root=${result_root}"
echo "[DISTILL HZUP20Q] hz_k_pairs=${hz_k_pairs_str}"

pair_idx=0
for pair in "${hz_k_pairs[@]}"; do
    if [[ "${pair}" != *:* ]]; then
        echo "[ERROR] Invalid HZ_K pair '${pair}'. Expected format like 40:2." >&2
        exit 1
    fi
    hz="${pair%%:*}"
    query_interval="${pair##*:}"
    hz_tag="${hz//./p}"
    pair_result_root="${result_root}/hz_${hz_tag}_K${query_interval}"
    pair_log="${launch_root}/hz_${hz_tag}_K${query_interval}.log"
    pair_port=$((master_port_base + pair_idx))
    pair_idx=$((pair_idx + 1))

    if [[ "${query_interval}" == "1" ]]; then
        run_baseline=1
        run_ours_full=1
        query_intervals=""
    else
        run_baseline=0
        run_ours_full=0
        query_intervals="${query_interval}"
    fi

    video_fps="$(awk -v hz="${hz}" 'BEGIN { printf "%d", hz }')"

    echo "------------------------------------------------------------"
    echo "[DISTILL HZUP20Q RUN] control_hz=${hz}, K=${query_interval}, port=${pair_port}"
    echo "[DISTILL HZUP20Q RUN] result=${pair_result_root}"
    echo "------------------------------------------------------------"

    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_TAG="${experiment_tag}_hz_${hz_tag}_K${query_interval}" \
    RESULT_ROOT="${pair_result_root}" \
    RUN_BASELINE="${run_baseline}" \
    RUN_OURS_FULL="${run_ours_full}" \
    LRNODE_QUERY_INTERVALS_STR="${query_intervals}" \
    EVAL_CONTROL_HZ="${hz}" \
    EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ}" \
    EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ}" \
    EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ}" \
    VIDEO_FPS="${video_fps}" \
    MASTER_PORT="${pair_port}" \
    NODE_NUM="${node_num}" \
    bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh 2>&1 | tee "${pair_log}"
done

RESULT_ROOT="${result_root}" python - <<'PY'
import csv
import json
import re
from pathlib import Path
import os

root = Path(os.environ["RESULT_ROOT"])
rows = []
for path in sorted(root.glob("hz_*_K*/*/analysis/eval_summary.json")):
    data = json.loads(path.read_text())
    lr = data.get("lrnode", {})
    qr = data.get("query_reduction", {})
    smooth = data.get("action_smoothness", {})
    env = data.get("environment", {})
    match = re.search(r"_K(\d+)_", str(path))
    k = int(lr.get("query_interval", match.group(1) if match else 1))
    hz = float(lr.get("control_hz", 20.0))
    policy_ms = float(lr.get("avg_policy_step_latency_sec", 0.0)) * 1000.0
    budget_ms = 1000.0 / hz if hz > 0 else 0.0
    rows.append({
        "control_hz": hz,
        "query_interval": k,
        "run_dir": path.parents[1].name,
        "success_rate_pct": round(float(data.get("success_rate", 0.0)) * 100.0, 3),
        "skip_full_forward": bool(lr.get("eval_skip_full_forward", False)),
        "effective_full_query_hz": round(float(lr.get("effective_full_query_hz", hz / max(k, 1))), 3),
        "effective_lrnode_update_hz": round(float(lr.get("effective_lrnode_update_hz", 0.0)), 3),
        "env_control_freq": int(env.get("control_freq", round(hz))),
        "eval_max_steps": int(env.get("eval_max_steps", 0)),
        "full_forward_calls": int(qr.get("num_full_forward_calls", lr.get("full_forward_calls", 0))),
        "lrnode_update_calls": int(qr.get("num_lrnode_update_calls", lr.get("lrnode_update_calls", 0))),
        "full_query_reduction_pct": round(float(qr.get("full_query_reduction_ratio", 0.0)) * 100.0, 3),
        "avg_full_forward_ms": round(float(lr.get("avg_full_forward_latency_sec", 0.0)) * 1000.0, 3),
        "avg_lrnode_ms": round(float(lr.get("avg_lrnode_latency_sec", 0.0)) * 1000.0, 3),
        "avg_policy_step_ms": round(policy_ms, 3),
        "policy_budget_ms": round(budget_ms, 3),
        "policy_latency_over_budget": round(policy_ms / budget_ms, 3) if budget_ms > 0 else 0.0,
        "action_jerk_l2_p95": round(float(smooth.get("action_jerk_l2_p95", 0.0)), 6),
        "summary_path": str(path),
    })

rows.sort(key=lambda r: (r["control_hz"], r["query_interval"], r["run_dir"]))
csv_path = root / "hzup20q_summary.csv"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"[DISTILL HZUP20Q SUMMARY] saved: {csv_path}")
for row in rows:
    print(row)
PY

echo "[DISTILL HZUP20Q DONE] ${result_root}"
