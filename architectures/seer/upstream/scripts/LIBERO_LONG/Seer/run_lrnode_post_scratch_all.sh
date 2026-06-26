#!/bin/bash

set -euo pipefail

# One-command launcher after scratch_node.sh finishes.
#
# Runs:
#   1) scratch_node K=1 full-forward checkpoint selection over ckpt 30-39
#   2) distill_node from the best current baseline checkpoint, default ckpt 33
#   3) optional scratch_node K sweep for the best K=1 checkpoint
#
# eval_node.sh and distill_node.sh intentionally use different master ports.
# This script keeps the same CUDA_VISIBLE_DEVICES by default because the target
# machine has enough VRAM for the intended concurrent run.

protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
scratch_node_env="${OURS_ENV:-${protocol_root}/train/_latest/scratch_node.env}"
baseline_env="${BASELINE_ENV:-${protocol_root}/train/_latest/scratch.env}"

ckpt_ids="${CKPT_IDS:-30 31 32 33 34 35 36 37 38 39}"
baseline_ckpt_id="${BASELINE_CKPT_ID:-33}"
experiment_tag="${EXPERIMENT_TAG:-post_scratch_node_$(date +%Y%m%d_%H%M%S)}"

eval_k1_master_port="${EVAL_K1_MASTER_PORT:-12442}"
eval_ksweep_master_port="${EVAL_KSWEEP_MASTER_PORT:-12443}"
distill_master_port="${DISTILL_MASTER_PORT:-12423}"

run_distill="${RUN_DISTILL:-1}"
run_scratch_k1_eval="${RUN_SCRATCH_K1_EVAL:-1}"
run_scratch_best_k_sweep="${RUN_SCRATCH_BEST_K_SWEEP:-1}"
best_sweep_query_intervals="${BEST_SWEEP_QUERY_INTERVALS:-2 3 4 5 6 8}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export SAVE_LRNODE_STATS="${SAVE_LRNODE_STATS:-1}"
export SAVE_LRNODE_STATS_ALL_RANKS="${SAVE_LRNODE_STATS_ALL_RANKS:-1}"

launch_root="${protocol_root}/launch_logs/${experiment_tag}"
mkdir -p "${launch_root}"

if [[ ! -f "${scratch_node_env}" ]]; then
    echo "[ERROR] scratch_node env not found: ${scratch_node_env}" >&2
    exit 1
fi
if [[ ! -f "${baseline_env}" ]]; then
    echo "[ERROR] baseline env not found: ${baseline_env}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${scratch_node_env}"
ours_run_name="${LRNODE_RUN_NAME}"
ours_ckpt_root="${LRNODE_SAVE_CHECKPOINT_PATH}"
ours_dataset="${LRNODE_DATASET}"
ours_ckpt_dir="${ours_ckpt_root}/${ours_run_name}"

for ckpt_id in ${ckpt_ids}; do
    if [[ ! -f "${ours_ckpt_dir}/${ckpt_id}.pth" ]]; then
        echo "[ERROR] Missing scratch_node checkpoint: ${ours_ckpt_dir}/${ckpt_id}.pth" >&2
        exit 1
    fi
done

# shellcheck disable=SC1090
source "${baseline_env}"
baseline_run_name="${LRNODE_RUN_NAME}"
baseline_ckpt_root="${LRNODE_SAVE_CHECKPOINT_PATH}"
baseline_ckpt="${baseline_ckpt_root}/${baseline_run_name}/${baseline_ckpt_id}.pth"
if [[ ! -f "${baseline_ckpt}" ]]; then
    echo "[ERROR] Missing baseline checkpoint: ${baseline_ckpt}" >&2
    exit 1
fi

cat > "${launch_root}/run_plan.env" <<EOF
EXPERIMENT_TAG=${experiment_tag}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
SCRATCH_NODE_ENV=${scratch_node_env}
SCRATCH_NODE_RUN_NAME=${ours_run_name}
SCRATCH_NODE_CKPT_DIR=${ours_ckpt_dir}
SCRATCH_NODE_CKPT_IDS=${ckpt_ids}
BASELINE_ENV=${baseline_env}
BASELINE_RUN_NAME=${baseline_run_name}
BASELINE_CKPT_ID=${baseline_ckpt_id}
BASELINE_CKPT=${baseline_ckpt}
EVAL_K1_MASTER_PORT=${eval_k1_master_port}
EVAL_KSWEEP_MASTER_PORT=${eval_ksweep_master_port}
DISTILL_MASTER_PORT=${distill_master_port}
RUN_DISTILL=${run_distill}
RUN_SCRATCH_K1_EVAL=${run_scratch_k1_eval}
RUN_SCRATCH_BEST_K_SWEEP=${run_scratch_best_k_sweep}
BEST_SWEEP_QUERY_INTERVALS=${best_sweep_query_intervals}
EOF

echo "[POST SCRATCH] launch_root=${launch_root}"
echo "[POST SCRATCH] cuda=${CUDA_VISIBLE_DEVICES}"
echo "[POST SCRATCH] scratch_node=${ours_run_name}"
echo "[POST SCRATCH] baseline_ckpt=${baseline_ckpt}"
echo "[POST SCRATCH] ports eval_k1=${eval_k1_master_port}, eval_ksweep=${eval_ksweep_master_port}, distill=${distill_master_port}"
echo "[POST SCRATCH] save_video=${SAVE_VIDEO}, success=${SAVE_VIDEO_SUCC}, fail=${SAVE_VIDEO_FAIL}, all_ranks=${SAVE_VIDEO_ALL_RANKS}"

distill_pid=""
cleanup() {
    if [[ -n "${distill_pid}" ]] && kill -0 "${distill_pid}" 2>/dev/null; then
        echo "[POST SCRATCH] stopping background distill pid=${distill_pid}"
        kill "${distill_pid}" 2>/dev/null || true
    fi
}
trap cleanup ERR INT TERM

if [[ "${run_distill}" == "1" ]]; then
    distill_log="${launch_root}/distill_node_baseline_ckpt${baseline_ckpt_id}.log"
    echo "[POST SCRATCH] starting distill in background: ${distill_log}"
    BASELINE_CKPT_ID="${baseline_ckpt_id}" \
    MASTER_PORT="${distill_master_port}" \
    EXPERIMENT_TAG="distill_from_baseline_ckpt${baseline_ckpt_id}_${experiment_tag}" \
    bash scripts/LIBERO_LONG/Seer/distill_node.sh > "${distill_log}" 2>&1 &
    distill_pid=$!
    echo "[POST SCRATCH] distill_pid=${distill_pid}"
fi

k1_result_root="${protocol_root}/eval/lrnode_scratch_k1_${ours_run_name}_${experiment_tag}"
if [[ "${run_scratch_k1_eval}" == "1" ]]; then
    k1_log="${launch_root}/eval_node_k1_ckpt_selection.log"
    echo "[POST SCRATCH] running scratch_node K=1 ckpt selection: ${k1_log}"
    CKPT_IDS="${ckpt_ids}" \
    LRNODE_QUERY_INTERVALS_STR="1" \
    EVAL_RESULT_ROOT="${k1_result_root}" \
    MASTER_PORT="${eval_k1_master_port}" \
    EXPERIMENT_TAG="scratch_node_k1_${experiment_tag}" \
    bash scripts/LIBERO_LONG/Seer/eval_node.sh 2>&1 | tee "${k1_log}"
fi

best_ckpt=""
if [[ "${run_scratch_best_k_sweep}" == "1" ]]; then
    best_ckpt=$(python - "${k1_result_root}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in root.glob("*/analysis/eval_summary.json"):
    name = path.parent.parent.name
    match = re.search(r"_ckpt_(\d+)_", name)
    if not match:
        continue
    data = json.loads(path.read_text())
    ckpt = int(match.group(1))
    sr = float(data.get("success_rate", 0.0))
    steps = int(data.get("lrnode", {}).get("num_env_steps", 10**18))
    rows.append((sr, -steps, ckpt))

if not rows:
    raise SystemExit(f"No eval_summary.json files found under {root}")

rows.sort(reverse=True)
print(rows[0][2])
PY
)
    echo "[POST SCRATCH] selected best scratch_node K=1 ckpt=${best_ckpt}"
    echo "SCRATCH_NODE_BEST_CKPT=${best_ckpt}" >> "${launch_root}/run_plan.env"

    ksweep_result_root="${protocol_root}/eval/lrnode_scratch_best_ckpt${best_ckpt}_ksweep_${ours_run_name}_${experiment_tag}"
    ksweep_log="${launch_root}/eval_node_best_ckpt${best_ckpt}_ksweep.log"
    echo "[POST SCRATCH] running scratch_node best ckpt K sweep: ${ksweep_log}"
    CKPT_IDS="${best_ckpt}" \
    LRNODE_QUERY_INTERVALS_STR="${best_sweep_query_intervals}" \
    EVAL_RESULT_ROOT="${ksweep_result_root}" \
    MASTER_PORT="${eval_ksweep_master_port}" \
    EXPERIMENT_TAG="scratch_node_best_ckpt${best_ckpt}_ksweep_${experiment_tag}" \
    bash scripts/LIBERO_LONG/Seer/eval_node.sh 2>&1 | tee "${ksweep_log}"
fi

if [[ -n "${distill_pid}" ]]; then
    echo "[POST SCRATCH] waiting for distill pid=${distill_pid}"
    wait "${distill_pid}"
    echo "[POST SCRATCH] distill finished"
fi

echo "[POST SCRATCH] done"
echo "[POST SCRATCH] launch_root=${launch_root}"
