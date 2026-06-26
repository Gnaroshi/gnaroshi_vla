#!/bin/bash

set -euo pipefail

# Scratch-node internal Hz/query sweep.
#
# Purpose:
#   Use the same scratch_node checkpoint for all rows.
#   Treat K=1 as the full-query Seer reference for that checkpoint.
#   Then increase the action/control Hz while reducing full Seer query Hz via K>1.
#
# Important:
#   EVAL_CONTROL_HZ is passed into LIBERO OffScreenRenderEnv(control_freq=...).
#   To keep the same nominal episode duration as the 20 Hz baseline, eval max
#   steps and env horizon are scaled by control_hz / EVAL_BASE_CONTROL_HZ unless
#   EVAL_SCALE_MAX_STEPS_WITH_HZ=0 is explicitly set.
#
# Default grid:
#   20:1  -> full Seer at 20 Hz reference
#   20:2  -> action 20 Hz, full Seer 10 Hz
#   20:3  -> action 20 Hz, full Seer 6.67 Hz
#   20:4  -> action 20 Hz, full Seer 5 Hz
#   40:1  -> full Seer at 40 Hz upper-bound reference
#   40:2  -> action 40 Hz, full Seer 20 Hz
#   60:1  -> full Seer at 60 Hz upper-bound reference
#   60:3  -> action 60 Hz, full Seer 20 Hz
#   80:1  -> full Seer at 80 Hz upper-bound reference
#   80:4  -> action 80 Hz, full Seer 20 Hz

export PYTHONPATH=/home/mingyujung/private/LIBERO:$PYTHONPATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

export SAVE_LRNODE_STATS="${SAVE_LRNODE_STATS:-1}"
export SAVE_LRNODE_STATS_ALL_RANKS="${SAVE_LRNODE_STATS_ALL_RANKS:-1}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
export EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ:-20}"
export EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ:-1}"
export EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ:-1}"

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
ours_env="${OURS_ENV:-${protocol_root}/train/_latest/scratch_node.env}"

if [[ ! -f "${ours_env}" ]]; then
    echo "[ERROR] scratch_node env not found: ${ours_env}" >&2
    echo "[ERROR] Run scripts/LIBERO_LONG/Seer/scratch_node.sh first, or set OURS_ENV." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ours_env}"

run_name="${OURS_RUN_NAME:-${LRNODE_RUN_NAME}}"
ckpt_root="${OURS_CKPT_ROOT:-${LRNODE_SAVE_CHECKPOINT_PATH}}"
ckpt_dir="${ckpt_root}/${run_name}"

if [[ ! -d "${ckpt_dir}" ]]; then
    echo "[ERROR] scratch_node checkpoint dir not found: ${ckpt_dir}" >&2
    exit 1
fi

safe_tag() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    value="${value//:/_}"
    value="${value//,/}"
    echo "${value}"
}

find_latest_k1_root() {
    python - "${protocol_root}" "${run_name}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "eval"
run_name = sys.argv[2]
candidates = []
for path in root.glob(f"lrnode_scratch_sweep_{run_name}_*"):
    if any(path.glob("*/analysis/eval_summary.json")):
        candidates.append(path)
if not candidates:
    raise SystemExit(1)
candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
print(candidates[0])
PY
}

select_best_ckpt_from_k1_root() {
    local root="$1"
    python - "${root}" "${run_name}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_name = sys.argv[2]
rows = []
for path in root.glob("*/analysis/eval_summary.json"):
    data = json.loads(path.read_text())
    lr = data.get("lrnode", {})
    if data.get("run_name") != run_name:
        continue
    if int(lr.get("query_interval", -1)) != 1:
        continue
    if bool(lr.get("eval_skip_full_forward", True)):
        continue
    match = re.search(r"_ckpt_(\d+)_K1_", str(path))
    if not match:
        continue
    ckpt = int(match.group(1))
    sr = float(data.get("success_rate", 0.0))
    env_steps = int(lr.get("num_env_steps", 10**18))
    rows.append((sr, -env_steps, ckpt))

if not rows:
    raise SystemExit(f"No K=1 full-forward eval summaries found under {root}")
rows.sort(reverse=True)
print(rows[0][2])
PY
}

if [[ -z "${CKPT_IDS:-}" ]]; then
    auto_select_best="${AUTO_SELECT_BEST:-1}"
    if [[ "${auto_select_best}" == "1" ]]; then
        k1_root="${BEST_FROM_K1_ROOT:-}"
        if [[ -z "${k1_root}" ]]; then
            if k1_root="$(find_latest_k1_root)"; then
                true
            else
                echo "[ERROR] Could not auto-detect a previous K=1 eval root." >&2
                echo "[ERROR] Set CKPT_IDS manually, e.g. CKPT_IDS=30, or set BEST_FROM_K1_ROOT." >&2
                exit 1
            fi
        fi
        CKPT_IDS="$(select_best_ckpt_from_k1_root "${k1_root}")"
        echo "[HZ SWEEP] auto-selected best K=1 ckpt=${CKPT_IDS} from ${k1_root}"
    else
        CKPT_IDS="30"
        echo "[HZ SWEEP] AUTO_SELECT_BEST=0 and CKPT_IDS unset; using CKPT_IDS=${CKPT_IDS}"
    fi
fi

ckpt_ids_str="${CKPT_IDS}"
read -r -a ckpt_ids <<< "${ckpt_ids_str}"
for ckpt_id in "${ckpt_ids[@]}"; do
    if [[ ! -f "${ckpt_dir}/${ckpt_id}.pth" ]]; then
        echo "[ERROR] Missing checkpoint: ${ckpt_dir}/${ckpt_id}.pth" >&2
        exit 1
    fi
done

experiment_name="$(safe_tag "${EXPERIMENT_NAME:-scratch_hz_sweep}")"
hz_k_pairs_str="${HZ_K_PAIRS_STR:-20:1 20:2 20:3 20:4 40:1 40:2 60:1 60:3 80:1 80:4}"
read -r -a hz_k_pairs <<< "${hz_k_pairs_str}"

experiment_tag="$(safe_tag "${EXPERIMENT_TAG:-${experiment_name}_$(date +%Y%m%d_%H%M%S)}")"
result_root="${EVAL_RESULT_ROOT:-${protocol_root}/eval/lrnode_${experiment_name}_${run_name}_${experiment_tag}}"
launch_root="${result_root}/_launch_logs"
mkdir -p "${launch_root}"

master_port_base="${MASTER_PORT:-12472}"
node_num="${NODE_NUM:-4}"
video_fps_override="${VIDEO_FPS_OVERRIDE:-}"

cat > "${result_root}/experiment_config.env" <<EOF
SCRIPT=scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
EXPERIMENT_NAME=${experiment_name}
EXPERIMENT_TAG=${experiment_tag}
LRNODE_PROTOCOL_ROOT=${protocol_root}
OURS_ENV=${ours_env}
OURS_RUN_NAME=${run_name}
OURS_CKPT_DIR=${ckpt_dir}
CKPT_IDS=${ckpt_ids_str}
HZ_K_PAIRS_STR=${hz_k_pairs_str}
MASTER_PORT_BASE=${master_port_base}
NODE_NUM=${node_num}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
SAVE_VIDEO=${SAVE_VIDEO}
SAVE_VIDEO_SUCC=${SAVE_VIDEO_SUCC}
SAVE_VIDEO_FAIL=${SAVE_VIDEO_FAIL}
SAVE_VIDEO_ALL_RANKS=${SAVE_VIDEO_ALL_RANKS}
VIDEO_STRIDE=${VIDEO_STRIDE}
VIDEO_FPS_OVERRIDE=${video_fps_override}
EVAL_BASE_CONTROL_HZ=${EVAL_BASE_CONTROL_HZ}
EVAL_SCALE_MAX_STEPS_WITH_HZ=${EVAL_SCALE_MAX_STEPS_WITH_HZ}
EVAL_SCALE_SETTLE_STEPS_WITH_HZ=${EVAL_SCALE_SETTLE_STEPS_WITH_HZ}
EOF

echo "[HZ SWEEP] run_name=${run_name}"
echo "[HZ SWEEP] experiment_name=${experiment_name}"
echo "[HZ SWEEP] ckpt_dir=${ckpt_dir}"
echo "[HZ SWEEP] ckpt_ids=${ckpt_ids_str}"
echo "[HZ SWEEP] hz_k_pairs=${hz_k_pairs_str}"
echo "[HZ SWEEP] result_root=${result_root}"
echo "[HZ SWEEP] config=${result_root}/experiment_config.env"

pair_idx=0
for pair in "${hz_k_pairs[@]}"; do
    if [[ "${pair}" != *:* ]]; then
        echo "[ERROR] Invalid HZ_K pair '${pair}'. Expected format like 40:2." >&2
        exit 1
    fi
    hz="${pair%%:*}"
    query_interval="${pair##*:}"
    if [[ -z "${hz}" || -z "${query_interval}" ]]; then
        echo "[ERROR] Invalid HZ_K pair '${pair}'." >&2
        exit 1
    fi
    hz_tag="${hz//./p}"
    pair_result_root="${result_root}/hz_${hz_tag}_K${query_interval}"
    pair_log="${launch_root}/hz_${hz_tag}_K${query_interval}.log"
    pair_port=$((master_port_base + pair_idx))
    pair_idx=$((pair_idx + 1))

    if [[ -n "${video_fps_override}" ]]; then
        video_fps="${video_fps_override}"
    else
        video_fps="$(awk -v hz="${hz}" 'BEGIN { printf "%d", hz }')"
    fi

    echo "------------------------------------------------------------"
    echo "[HZ SWEEP RUN] control_hz=${hz}, K=${query_interval}, port=${pair_port}"
    echo "[HZ SWEEP RUN] result=${pair_result_root}"
    echo "------------------------------------------------------------"

    CKPT_IDS="${ckpt_ids_str}" \
    LRNODE_QUERY_INTERVALS_STR="${query_interval}" \
    EVAL_CONTROL_HZ="${hz}" \
    EVAL_BASE_CONTROL_HZ="${EVAL_BASE_CONTROL_HZ}" \
    EVAL_SCALE_MAX_STEPS_WITH_HZ="${EVAL_SCALE_MAX_STEPS_WITH_HZ}" \
    EVAL_SCALE_SETTLE_STEPS_WITH_HZ="${EVAL_SCALE_SETTLE_STEPS_WITH_HZ}" \
    VIDEO_FPS="${video_fps}" \
    EVAL_RESULT_ROOT="${pair_result_root}" \
    MASTER_PORT="${pair_port}" \
    NODE_NUM="${node_num}" \
    bash scripts/LIBERO_LONG/Seer/eval_node.sh 2>&1 | tee "${pair_log}"
done

RESULT_ROOT="${result_root}" python - <<'PY'
import csv
import json
import math
import re
from pathlib import Path

root = Path(__import__("os").environ["RESULT_ROOT"])
rows = []
for path in sorted(root.glob("hz_*_K*/*/analysis/eval_summary.json")):
    data = json.loads(path.read_text())
    lr = data.get("lrnode", {})
    qr = data.get("query_reduction", {})
    smooth = data.get("action_smoothness", {})
    env = data.get("environment", {})
    match = re.search(r"_ckpt_(\d+)_K(\d+)_", str(path))
    ckpt = int(match.group(1)) if match else -1
    k = int(lr.get("query_interval", match.group(2) if match else 1))
    hz = float(lr.get("control_hz", 20.0))
    policy_ms = float(lr.get("avg_policy_step_latency_sec", 0.0)) * 1000.0
    full_ms = float(lr.get("avg_full_forward_latency_sec", 0.0)) * 1000.0
    lrnode_ms = float(lr.get("avg_lrnode_latency_sec", 0.0)) * 1000.0
    budget_ms = 1000.0 / hz if hz > 0 else 0.0
    rows.append({
        "ckpt": ckpt,
        "control_hz": hz,
        "query_interval": k,
        "run_dir": path.parents[1].name,
        "success_rate": float(data.get("success_rate", 0.0)),
        "success_rate_pct": round(float(data.get("success_rate", 0.0)) * 100.0, 3),
        "skip_full_forward": bool(lr.get("eval_skip_full_forward", False)),
        "effective_action_hz": float(lr.get("effective_action_hz", hz)),
        "effective_full_query_hz": float(lr.get("effective_full_query_hz", hz / max(k, 1))),
        "effective_lrnode_update_hz": float(lr.get("effective_lrnode_update_hz", 0.0)),
        "env_control_freq": int(env.get("control_freq", round(hz))),
        "eval_max_steps": int(env.get("eval_max_steps", 0)),
        "env_horizon": int(env.get("env_horizon", 0)),
        "settle_steps": int(env.get("settle_steps", 0)),
        "env_steps": int(qr.get("num_env_steps", lr.get("num_env_steps", 0))),
        "full_forward_calls": int(qr.get("num_full_forward_calls", lr.get("full_forward_calls", 0))),
        "lrnode_update_calls": int(qr.get("num_lrnode_update_calls", lr.get("lrnode_update_calls", 0))),
        "full_query_reduction_pct": round(float(qr.get("full_query_reduction_ratio", 0.0)) * 100.0, 3),
        "avg_full_forward_ms": round(full_ms, 3),
        "avg_lrnode_ms": round(lrnode_ms, 3),
        "avg_fast_encoder_ms": round(float(lr.get("avg_fast_encoder_latency_sec", 0.0)) * 1000.0, 3),
        "avg_node_update_ms": round(float(lr.get("avg_node_update_latency_sec", 0.0)) * 1000.0, 3),
        "avg_action_head_ms": round(float(lr.get("avg_action_head_latency_sec", 0.0)) * 1000.0, 3),
        "avg_policy_step_ms": round(policy_ms, 3),
        "avg_env_step_ms": round(float(lr.get("avg_env_step_latency_sec", 0.0)) * 1000.0, 3),
        "policy_budget_ms": round(budget_ms, 3),
        "policy_latency_over_budget": round(policy_ms / budget_ms, 3) if budget_ms > 0 else 0.0,
        "full_forward_over_budget": round(full_ms / budget_ms, 3) if budget_ms > 0 else 0.0,
        "lrnode_over_budget": round(lrnode_ms / budget_ms, 3) if budget_ms > 0 else 0.0,
        "policy_meets_budget": bool(policy_ms <= budget_ms) if budget_ms > 0 else False,
        "lrnode_meets_budget": bool((lrnode_ms > 0.0) and (lrnode_ms <= budget_ms)) if budget_ms > 0 else False,
        "action_delta_l2_mean": round(float(smooth.get("action_delta_l2_mean", 0.0)), 6),
        "action_delta_l2_p95": round(float(smooth.get("action_delta_l2_p95", 0.0)), 6),
        "action_jerk_l2_mean": round(float(smooth.get("action_jerk_l2_mean", 0.0)), 6),
        "action_jerk_l2_p95": round(float(smooth.get("action_jerk_l2_p95", 0.0)), 6),
        "gripper_switch_rate": round(float(smooth.get("gripper_switch_rate", 0.0)), 6),
        "summary_path": str(path),
    })

baselines = {}
for row in rows:
    if row["query_interval"] == 1 and not row["skip_full_forward"]:
        baselines[(row["ckpt"], row["control_hz"])] = row["success_rate"]

for row in rows:
    ref = baselines.get((row["ckpt"], row["control_hz"]))
    row["same_ckpt_hz_k1_sr_pct"] = round(ref * 100.0, 3) if ref is not None else ""
    row["same_ckpt_hz_preservation_pct"] = (
        round(row["success_rate"] / ref * 100.0, 3)
        if ref and ref > 0
        else ""
    )

rows.sort(key=lambda r: (r["ckpt"], r["control_hz"], r["query_interval"]))
csv_path = root / "hz_sweep_summary.csv"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

md_path = root / "hz_sweep_summary.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# LR-NODE Scratch Hz Sweep Summary\\n\\n")
    f.write("K=1 is the full-query reference for the same scratch_node checkpoint and the same LIBERO control_freq.\\n\\n")
    f.write("| ckpt | Hz | K | max steps | SR % | K=1 ref % | preservation % | full query Hz | LR-NODE Hz | query reduction % | policy ms | budget ms | policy/budget | LR-NODE ms | LR-NODE/budget | jerk p95 |\\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\\n")
    for r in rows:
        f.write(
            f"| {r['ckpt']} | {r['control_hz']:.1f} | {r['query_interval']} | "
            f"{r['eval_max_steps']} | {r['success_rate_pct']:.3f} | {r['same_ckpt_hz_k1_sr_pct']} | "
            f"{r['same_ckpt_hz_preservation_pct']} | {r['effective_full_query_hz']:.3f} | "
            f"{r['effective_lrnode_update_hz']:.3f} | {r['full_query_reduction_pct']:.3f} | "
            f"{r['avg_policy_step_ms']:.3f} | {r['policy_budget_ms']:.3f} | "
            f"{r['policy_latency_over_budget']:.3f} | {r['avg_lrnode_ms']:.3f} | "
            f"{r['lrnode_over_budget']:.3f} | {r['action_jerk_l2_p95']:.6f} |\\n"
        )
    f.write("\\nNotes:\\n\\n")
    f.write("- `control_hz` is passed to LIBERO as `OffScreenRenderEnv(control_freq=control_hz)`.\\n")
    f.write("- By default, `eval_max_steps` and env `horizon` are scaled by `control_hz / 20` to preserve the nominal episode duration.\\n")
    f.write("- `preservation % = SR(H, K) / SR(H, K=1 same checkpoint) * 100`.\\n")
    f.write("- `policy/budget <= 1` means the measured policy step latency fits the nominal real-time action budget `1000 / Hz` ms.\\n")

print(f"[HZ SWEEP SUMMARY] saved CSV: {csv_path}")
print(f"[HZ SWEEP SUMMARY] saved MD: {md_path}")
for row in rows:
    print(row)
PY

echo "[HZ SWEEP DONE] ${result_root}"
