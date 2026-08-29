#!/bin/bash

set -euo pipefail

# DISTILL-QRED20-CKPT-SWEEP
#
# Purpose:
#   Evaluate multiple frozen-baseline LR-NODE adapter checkpoints under the
#   same QRED20 protocol. This is for checking whether ckpt39 is special or
#   whether earlier adapter checkpoints show the same K-dependent behavior.
#
# Default:
#   ckpts: 31 32 33 34 35 36 37 38
#   rows per ckpt:
#     adapter-composed full K=1
#     adapter-composed skip K=2,3,4
#
# Baseline full K=1 is not repeated by default because it is shared across all
# adapter checkpoints and was already measured in the main QRED20 run.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${UPSTREAM_DIR}/../../.." && pwd)"
protocol_root="${LRNODE_PROTOCOL_ROOT:-${REPO_ROOT}/results/seer/lrnode/default}"
experiment_name="${EXPERIMENT_NAME:-lrnode_distill_qred20_ckpt_sweep}"
experiment_tag="${EXPERIMENT_TAG:-distill_qred20_ckpt_sweep_$(date +%Y%m%d_%H%M%S)}"
result_root="${EVAL_RESULT_ROOT:-${protocol_root}/eval/${experiment_name}_${experiment_tag}}"
launch_root="${result_root}/_launch_logs"
mkdir -p "${launch_root}"

ckpts_str="${CKPT_IDS_STR:-31 32 33 34 35 36 37 38}"
read -r -a ckpts <<< "${ckpts_str}"

export BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-33}"
export RUN_BASELINE="${RUN_BASELINE:-0}"
export RUN_OURS_FULL="${RUN_OURS_FULL:-1}"
export LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR-2 3 4}"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
export NODE_NUM="${NODE_NUM:-4}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
export LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"

master_port_base="${MASTER_PORT:-12710}"

if [[ "${ALLOW_CONCURRENT_EVAL:-0}" != "1" ]]; then
    active_eval="$(
        ps -eo pid,cmd \
            | awk '$0 ~ /eval_libero.py|torch.distributed.run|eval_lrnode_distill_hzup20q.sh/ && $0 !~ /awk/ {print}'
    )"
    if [[ -n "${active_eval}" ]]; then
        echo "[ERROR] Another eval appears to be running. Refusing to start ckpt sweep." >&2
        echo "${active_eval}" >&2
        echo "[ERROR] Wait for the active eval to finish, or set ALLOW_CONCURRENT_EVAL=1 if you intentionally want overlap." >&2
        exit 1
    fi
fi

cat > "${result_root}/experiment_config.env" <<EOF
SCRIPT=scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20_ckpt_sweep.sh
EXPERIMENT_NAME=${experiment_name}
EXPERIMENT_TAG=${experiment_tag}
RESULT_ROOT=${result_root}
CKPT_IDS_STR=${ckpts_str}
BASELINE_CKPT_ID=${BASELINE_CKPT_ID}
RUN_BASELINE=${RUN_BASELINE}
RUN_OURS_FULL=${RUN_OURS_FULL}
LRNODE_QUERY_INTERVALS_STR=${LRNODE_QUERY_INTERVALS_STR}
EVAL_CONTROL_HZ=${EVAL_CONTROL_HZ}
MASTER_PORT_BASE=${master_port_base}
NODE_NUM=${NODE_NUM}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
SAVE_VIDEO=${SAVE_VIDEO}
SAVE_VIDEO_SUCC=${SAVE_VIDEO_SUCC}
SAVE_VIDEO_FAIL=${SAVE_VIDEO_FAIL}
SAVE_VIDEO_ALL_RANKS=${SAVE_VIDEO_ALL_RANKS}
LRNODE_EVAL_STEP_LOG=${LRNODE_EVAL_STEP_LOG}
LRNODE_EVAL_SHADOW_FULL_FORWARD=${LRNODE_EVAL_SHADOW_FULL_FORWARD}
EOF

echo "[DISTILL QRED20 CKPT SWEEP] result_root=${result_root}"
echo "[DISTILL QRED20 CKPT SWEEP] ckpts=${ckpts_str}"
echo "[DISTILL QRED20 CKPT SWEEP] K=${LRNODE_QUERY_INTERVALS_STR}"

idx=0
for ckpt_id in "${ckpts[@]}"; do
    ckpt_root="${result_root}/ckpt_${ckpt_id}"
    ckpt_log="${launch_root}/ckpt_${ckpt_id}.log"
    port=$((master_port_base + idx))
    idx=$((idx + 1))

    echo "------------------------------------------------------------"
    echo "[DISTILL QRED20 CKPT RUN] ckpt=${ckpt_id}, port=${port}"
    echo "[DISTILL QRED20 CKPT RUN] result=${ckpt_root}"
    echo "------------------------------------------------------------"

    EXPERIMENT_NAME="${experiment_name}" \
    EXPERIMENT_TAG="${experiment_tag}_ckpt${ckpt_id}" \
    RESULT_ROOT="${ckpt_root}" \
    BASELINE_CKPT_ID="${BASELINE_CKPT_ID}" \
    OURS_CKPT_ID="${ckpt_id}" \
    RUN_BASELINE="${RUN_BASELINE}" \
    RUN_OURS_FULL="${RUN_OURS_FULL}" \
    LRNODE_QUERY_INTERVALS_STR="${LRNODE_QUERY_INTERVALS_STR}" \
    EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ}" \
    MASTER_PORT="${port}" \
    NODE_NUM="${NODE_NUM}" \
    bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh 2>&1 | tee "${ckpt_log}"
done

RESULT_ROOT="${result_root}" python - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

root = Path(os.environ["RESULT_ROOT"])
rows = []
for path in sorted(root.glob("ckpt_*/*/analysis/eval_summary.json")):
    data = json.loads(path.read_text())
    lr = data.get("lrnode", {})
    qr = data.get("query_reduction", {})
    smooth = data.get("action_smoothness", {})
    ckpt_match = re.search(r"ckpt_(\d+)", str(path))
    rows.append({
        "ckpt_id": int(ckpt_match.group(1)) if ckpt_match else -1,
        "run_dir": path.parents[1].name,
        "success_rate_pct": round(float(data.get("success_rate", 0.0)) * 100.0, 3),
        "skip_full_forward": bool(lr.get("eval_skip_full_forward", False)),
        "query_interval": int(lr.get("query_interval", 1)),
        "control_hz": round(float(lr.get("control_hz", 20.0)), 3),
        "effective_full_query_hz": round(float(lr.get("effective_full_query_hz", 0.0)), 3),
        "effective_lrnode_update_hz": round(float(lr.get("effective_lrnode_update_hz", 0.0)), 3),
        "full_forward_calls": int(qr.get("num_full_forward_calls", 0)),
        "lrnode_update_calls": int(qr.get("num_lrnode_update_calls", 0)),
        "full_query_reduction_pct": round(float(qr.get("full_query_reduction_ratio", 0.0)) * 100.0, 3),
        "effective_query_interval": round(float(qr.get("effective_query_interval", 0.0)), 3),
        "avg_policy_step_ms": round(float(lr.get("avg_policy_step_latency_sec", 0.0)) * 1000.0, 3),
        "avg_full_forward_ms": round(float(lr.get("avg_full_forward_latency_sec", 0.0)) * 1000.0, 3),
        "avg_lrnode_ms": round(float(lr.get("avg_lrnode_latency_sec", 0.0)) * 1000.0, 3),
        "action_jerk_l2_p95": round(float(smooth.get("action_jerk_l2_p95", 0.0)), 6),
        "gripper_switch_rate": round(float(smooth.get("gripper_switch_rate", 0.0)), 6),
        "summary_path": str(path),
    })

rows.sort(key=lambda r: (r["ckpt_id"], r["query_interval"], r["run_dir"]))
csv_path = root / "qred20_ckpt_sweep_summary.csv"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

print(f"[DISTILL QRED20 CKPT SWEEP SUMMARY] saved: {csv_path}")
for row in rows:
    print(row)
PY

echo "[DISTILL QRED20 CKPT SWEEP DONE] ${result_root}"
