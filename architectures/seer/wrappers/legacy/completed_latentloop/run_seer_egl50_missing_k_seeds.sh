#!/usr/bin/env bash

set -Eeuo pipefail

# Complete the missing evaluation seeds for the useful K-curve region of the
# locked public33 EGL-50 Seer campaign. Each row uses four GPUs sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
EVAL_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
EXPECTED_REPO="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PAPER_ROOT="${PAPER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/paper_egl50/seer_public33_egl50_main_v1}"
LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
PUBLIC33_CKPT="${PUBLIC33_CKPT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/checkpoints_Seer_LIBERO_LONG/Seer/33.pth}"
PUBLIC33_ADAPTER="${PUBLIC33_ADAPTER:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/seer.incoming_20260817/lrnode/official_seer_libero_k4_v1/train/adapters/official_seer_ckpt33_lrnode_adapter_v1/39.pth}"

PUBLIC33_SHA256=a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646
PUBLIC33_ADAPTER_SHA256=3f70179ab9b1bae64fc772d71c57a93592b9f82e53b5fcaf1a6beb319c280462
VIT_SHA256=aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d

EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-43 44}"
K_LIST_STR="${K_LIST_STR:-3 5 6 7}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS_STR}"
read -r -a K_LIST <<< "${K_LIST_STR}"
EPISODES_PER_TASK=50
NUM_TASKS=10
MASTER_PORT_BASE="${MASTER_PORT_BASE:-17600}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SOURCE_LOCK="${PAPER_ROOT}/source_sha256.lock"
QUEUE_STATUS="${PAPER_ROOT}/missing_k_seed_queue.env"
ROW_INDEX=0

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

verify_hash() {
    local path="$1" expected="$2" label="$3" actual
    require_file "${path}"
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || fail "${label} SHA256 mismatch: ${actual}"
    echo "[VERIFY][OK] ${label} sha256=${actual}"
}

on_exit() {
    local rc=$?
    if (( rc == 0 )); then
        printf 'status=COMPLETE\ntime=%s\n' "$(date --iso-8601=seconds)" > "${QUEUE_STATUS}"
    else
        printf 'status=FAILED\nexit_code=%s\ntime=%s\n' "${rc}" "$(date --iso-8601=seconds)" > "${QUEUE_STATUS}"
    fi
}

validate_row() {
    local root="$1" seed="$2" k="$3"
    python - "${root}" "${seed}" "${k}" "${EPISODES_PER_TASK}" "${NUM_TASKS}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seed, k = int(sys.argv[2]), int(sys.argv[3])
episodes_per_task, num_tasks = int(sys.argv[4]), int(sys.argv[5])
summaries = list(root.glob("*/analysis/eval_summary.json"))
if len(summaries) != 1:
    raise RuntimeError(f"expected one eval summary, found {len(summaries)} under {root}")
summary_path = summaries[0]
analysis = summary_path.parent
summary = json.loads(summary_path.read_text(encoding="utf-8"))
with (analysis / "eval_episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
expected_count = episodes_per_task * num_tasks
if len(rows) != expected_count:
    raise RuntimeError(f"expected {expected_count} episodes, found {len(rows)}")
coverage = {(int(row["task_id"]), int(row["episode_id"])) for row in rows}
expected = {(task, episode) for task in range(num_tasks) for episode in range(episodes_per_task)}
if coverage != expected:
    raise RuntimeError("task/init-state coverage mismatch")
if {int(float(row["seed"])) for row in rows} != {seed}:
    raise RuntimeError("evaluation seed mismatch")
if summary.get("suite") != "libero_10":
    raise RuntimeError(f"suite mismatch: {summary.get('suite')}")
renderer = summary.get("renderer_backend", {})
if renderer.get("requested_backend") != "egl" or renderer.get("effective_backend") != "egl":
    raise RuntimeError(f"renderer mismatch: {renderer}")
rank_contexts = renderer.get("rank_contexts", [])
if not renderer.get("all_ranks_actual_context_verified", False):
    raise RuntimeError("EGL context was not verified on all ranks")
if len(rank_contexts) != 4 or not all(row.get("actual_context_verified") for row in rank_contexts):
    raise RuntimeError("expected four verified EGL rank contexts")
lrnode = summary.get("lrnode", {})
query = summary.get("query_reduction", {})
if not lrnode.get("enabled") or not lrnode.get("eval_skip_full_forward"):
    raise RuntimeError("LatentLoop skip path was not enabled")
if int(lrnode.get("query_interval", -1)) != k:
    raise RuntimeError(f"K mismatch: {lrnode.get('query_interval')} != {k}")
if summary.get("lrnode_eval_ablation_mode") != "stepwise":
    raise RuntimeError("expected stepwise ablation mode")
if int(query.get("num_fallback_full_calls", -1)) != 0:
    raise RuntimeError("fallback full forward was used")
env_steps = int(query.get("num_env_steps", -1))
full_calls = int(query.get("num_full_forward_calls", -1))
update_calls = int(query.get("num_lrnode_update_calls", -1))
if full_calls + update_calls != env_steps:
    raise RuntimeError("full and LatentLoop calls do not partition environment steps")
for name in ("eval_progress.json", "eval_latency_profile.json"):
    if not (analysis / name).is_file():
        raise FileNotFoundError(analysis / name)
print(f"[ROW PASS] seed={seed} K={k} episodes={len(rows)} SR={100.0 * float(summary['success_rate']):.2f}%")
PY
}

write_contract() {
    local root="$1" row_id="$2" seed="$3" k="$4"
    printf '%s\n' \
        "ROW_ID=${row_id}" \
        "SUITE=libero_10" \
        "EVAL_SEED=${seed}" \
        "METHOD=latentloop" \
        "QUERY_INTERVAL=${k}" \
        "ABLATION_MODE=stepwise" \
        "EPISODES_PER_TASK=${EPISODES_PER_TASK}" \
        "NUM_TASKS=${NUM_TASKS}" \
        "RENDERER=egl" \
        "BASELINE_CKPT=${PUBLIC33_CKPT}" \
        "ADAPTER_CKPT=${PUBLIC33_ADAPTER}" \
        > "${root}/row_contract.env"
}

write_campaign_table() {
    python - "${PAPER_ROOT}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = []
for contract_path in sorted((root / "eval").glob("*/row_contract.env")):
    contract = {}
    for line in contract_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            contract[key] = value
    summaries = list(contract_path.parent.glob("*/analysis/eval_summary.json"))
    if len(summaries) != 1:
        continue
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    query = summary.get("query_reduction", {})
    lrnode = summary.get("lrnode", {})
    records.append({
        "row_id": contract["ROW_ID"],
        "suite": contract["SUITE"],
        "seed": int(contract["EVAL_SEED"]),
        "method": contract["METHOD"],
        "k": int(contract["QUERY_INTERVAL"]),
        "ablation": contract["ABLATION_MODE"],
        "episodes": sum(int(row.get("num_episodes", 0)) for row in summary.get("task_results", [])),
        "success_rate_pct": 100.0 * float(summary.get("success_rate", 0.0)),
        "env_steps": int(query.get("num_env_steps", 0)),
        "full_forward_calls": int(query.get("num_full_forward_calls", 0)),
        "latentloop_update_calls": int(query.get("num_lrnode_update_calls", 0)),
        "query_reduction_pct": 100.0 * float(query.get("full_query_reduction_ratio", 0.0)),
        "avg_policy_step_ms": 1000.0 * float(lrnode.get("avg_policy_step_latency_sec", 0.0)),
        "renderer": summary.get("renderer_backend", {}).get("effective_backend"),
    })
output = root / "campaign_rows.csv"
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print(f"[CAMPAIGN TABLE] complete_rows={len(records)} path={output}")
PY
}

run_row() {
    local seed="$1" k="$2" row_id root logfile port rc
    row_id="long_seed${seed}_latentloop_k${k}"
    root="${PAPER_ROOT}/eval/${row_id}"
    logfile="${PAPER_ROOT}/logs/${row_id}.log"
    ROW_INDEX=$((ROW_INDEX + 1))
    port=$((MASTER_PORT_BASE + ROW_INDEX))

    if [[ -d "${root}" ]] && validate_row "${root}" "${seed}" "${k}" >/dev/null 2>&1; then
        write_contract "${root}" "${row_id}" "${seed}" "${k}"
        echo "[SKIP] verified complete row: ${row_id}"
        return 0
    fi
    if [[ -e "${root}" ]]; then
        local quarantine="${root}.incomplete.$(date +%Y%m%d_%H%M%S)"
        echo "[RESUME] preserving incomplete row at ${quarantine}"
        mv "${root}" "${quarantine}"
    fi

    echo "[EVAL START] row=${row_id} seed=${seed} K=${k} port=${port}"
    set +e
    env \
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        LIBERO_GL_BACKEND=egl MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_GL_REQUIRE_ACTUAL=1 \
        LIBERO_PATH="${LIBERO_PATH}" VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
        LRNODE_PROTOCOL_ROOT="${PAPER_ROOT}" SAVE_CHECKPOINT_PATH="${PAPER_ROOT}/eval_checkpoints" \
        EVAL_SUITE=libero_10 EVAL_SEED="${seed}" \
        EVAL_NUM_EPISODES_PER_TASK="${EPISODES_PER_TASK}" EVAL_NUM_TASKS="${NUM_TASKS}" \
        EVAL_CONTROL_HZ=20 LIBERO_EVAL_MAX_STEPS=600 EVAL_LIBERO_ENSEMBLING=1 \
        BASELINE_CKPT="${PUBLIC33_CKPT}" BASELINE_CKPT_ID=33 BASELINE_NAME=seer_libero_10_teacher \
        OURS_CKPT="${PUBLIC33_ADAPTER}" OURS_CKPT_ID=39 OURS_NAME=latentloop_libero_10 \
        METHOD_TAG=latentloop_v0 LRNODE_EVAL_BASE_CKPT="${PUBLIC33_CKPT}" \
        LRNODE_TRAIN_PROTOCOL=adapter LRNODE_FREEZE_SEER_FOR_ADAPTER=1 \
        LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1 LRNODE_EVAL_STEP_LOG=1 \
        LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=1 LRNODE_EVAL_SHADOW_FULL_FORWARD=0 \
        LRNODE_GATE_INIT_BIAS=-4.0 LRNODE_EVAL_ABLATION_MODE=stepwise \
        LRNODE_NO_DELTA_MODE=zero LRNODE_CHUNK_TOKEN_POLICY=skip_only \
        RUN_BASELINE=0 RUN_OURS_FULL=0 LRNODE_QUERY_INTERVALS_STR="${k}" \
        NODE_NUM=4 MASTER_PORT="${port}" SAVE_VIDEO=0 SAVE_VIDEO_SUCC=0 \
        SAVE_VIDEO_FAIL=0 SAVE_VIDEO_ALL_RANKS=0 \
        EXPERIMENT_NAME=seer_egl50 EXPERIMENT_TAG="${row_id}" RESULT_ROOT="${root}" \
        bash "${EVAL_SCRIPT}" 2>&1 | tee -a "${logfile}"
    rc=${PIPESTATUS[0]}
    set -e
    validate_row "${root}" "${seed}" "${k}"
    write_contract "${root}" "${row_id}" "${seed}" "${k}"
    if (( rc != 0 )); then
        echo "[WARN] evaluator rc=${rc}, but all 500 episodes and artifacts passed validation"
    fi
}

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || fail "expected ${EXPECTED_HOST}, got $(hostname)"
[[ "$(readlink -f "${REPO_ROOT}")" == "$(readlink -f "${EXPECTED_REPO}")" ]] || fail "unexpected repo: ${REPO_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "seer_libero" ]] || fail "activate conda environment seer_libero first"
[[ "${GPU_LIST}" == "0,1,2,3" ]] || fail "this completion queue is fixed to physical GPUs 0,1,2,3"
[[ "${EVAL_SEEDS_STR}" == "43 44" ]] || fail "expected missing seeds: 43 44"
[[ "${K_LIST_STR}" == "3 5 6 7" ]] || fail "expected missing K list: 3 5 6 7"
[[ "${PREFLIGHT_ONLY}" == "0" || "${PREFLIGHT_ONLY}" == "1" ]] || fail "PREFLIGHT_ONLY must be 0 or 1"
require_file "${EVAL_SCRIPT}"
require_file "${SOURCE_LOCK}"
verify_hash "${PUBLIC33_CKPT}" "${PUBLIC33_SHA256}" "public Seer 33"
verify_hash "${PUBLIC33_ADAPTER}" "${PUBLIC33_ADAPTER_SHA256}" "public33 adapter39"
verify_hash "${VIT_CHECKPOINT_PATH}" "${VIT_SHA256}" "MAE ViT"
(cd "${REPO_ROOT}" && sha256sum --check --quiet "${SOURCE_LOCK}") \
    || fail "source differs from the locked campaign"

SELECTED_GPU_LIST="${GPU_LIST}" python - <<'PY'
import os
import subprocess
import torch

if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected four visible CUDA devices, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(4)]
if not all("RTX 3090" in name for name in names):
    raise RuntimeError(f"expected four RTX 3090 GPUs: {names}")
selected_indices = {int(value) for value in os.environ["SELECTED_GPU_LIST"].split(",")}
gpu_rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
    text=True,
)
selected_uuids = set()
for row in gpu_rows.splitlines():
    index, uuid = [value.strip() for value in row.split(",", 1)]
    if int(index) in selected_indices:
        selected_uuids.add(uuid)
app_rows = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
    text=True,
).strip()
busy = []
for row in app_rows.splitlines():
    if not row.strip():
        continue
    uuid, pid, process_name = [value.strip() for value in row.split(",", 2)]
    if uuid in selected_uuids:
        busy.append(f"{uuid} pid={pid} process={process_name}")
if busy:
    raise RuntimeError("selected GPUs already have compute processes:\n" + "\n".join(busy))
print(f"[VERIFY][OK] four idle RTX 3090 GPUs: {names}")
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PREFLIGHT][PASS] source, artifacts, renderer inputs, and GPUs are valid; no evaluation launched"
    exit 0
fi

mkdir -p "${PAPER_ROOT}/logs" "${PAPER_ROOT}/eval"
exec 9>"${PAPER_ROOT}/queue.lock"
flock -n 9 || fail "another campaign process owns ${PAPER_ROOT}/queue.lock"
trap on_exit EXIT

cat > "${PAPER_ROOT}/missing_k_seed_contract.env" <<EOF
PROTOCOL=seer_public33_egl50_missing_k_seeds
HOST=${EXPECTED_HOST}
SOURCE_REPO=${REPO_ROOT}
SOURCE_LOCK=${SOURCE_LOCK}
LAUNCHER=${BASH_SOURCE[0]}
LAUNCHER_SHA256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')
GPU_LIST=${GPU_LIST}
RENDERER=egl
EPISODES_PER_TASK=${EPISODES_PER_TASK}
NUM_TASKS=${NUM_TASKS}
EVAL_SEEDS=${EVAL_SEEDS_STR}
K_LIST=${K_LIST_STR}
PUBLIC33_CKPT=${PUBLIC33_CKPT}
PUBLIC33_SHA256=${PUBLIC33_SHA256}
PUBLIC33_ADAPTER=${PUBLIC33_ADAPTER}
PUBLIC33_ADAPTER_SHA256=${PUBLIC33_ADAPTER_SHA256}
EOF

echo "[QUEUE] public33 EGL-50 missing K seeds: seeds=${EVAL_SEEDS_STR}; K=${K_LIST_STR}"
for seed in "${EVAL_SEEDS[@]}"; do
    for k in "${K_LIST[@]}"; do
        run_row "${seed}" "${k}"
        write_campaign_table
    done
done
echo "[DONE] all eight missing K-curve rows are complete"
