#!/usr/bin/env bash

set -Eeuo pipefail

# Isolate the effect of the Spatial teacher checkpoint:
#   1) evaluate suite-specific Spatial teacher33 at K=1 for three seeds,
#   2) train a frozen-teacher LatentLoop adapter from teacher33,
#   3) evaluate adapter39 at K=4 for the same three seeds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
DISTILL_SCRIPT="${SCRIPT_DIR}/distill_node.sh"
EVAL_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
EXPECTED_REPO="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer}"
SHARED_GNAROSHI_ROOT="${SHARED_GNAROSHI_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla}"
PAPER_CHECKPOINT_ROOT="${PAPER_CHECKPOINT_ROOT:-${SHARED_GNAROSHI_ROOT}/artifacts/checkpoints/seer/paper}"
PAPER_ROOT="${PAPER_ROOT:-${SHARED_GNAROSHI_ROOT}/results/seer/latentloop/reproductions/spatial_teacher33_adapter39_egl50}"
SUITE_STUDY_ROOT="${SUITE_STUDY_ROOT:-${SHARED_SEER_ROOT}/libero_suite_study}"
DATASET_ROOT="${DATASET_ROOT:-${SUITE_STUDY_ROOT}/datasets}"
DATASET="libero_spatial_converted"
DATASET_INFO="${DATASET_ROOT}/${DATASET}/data_info.json"
DATASET_META="${DATASET_ROOT}/${DATASET}/meta_info.h5"
TEACHER_CKPT="${TEACHER_CKPT:-${PAPER_CHECKPOINT_ROOT}/libero_spatial/teacher_scratch33.pth}"
ADAPTER_SAVE_ROOT="${PAPER_ROOT}/train/adapter"
ADAPTER_RUN_NAME="latentloop_libero_spatial_teacher33_seed42"
ADAPTER_CKPT="${ADAPTER_SAVE_ROOT}/${ADAPTER_RUN_NAME}/39.pth"
LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
VIT_CHECKPOINT_PATH="${SHARED_SEER_ROOT}/vit_mae/mae_pretrain_vit_base.pth"

TEACHER_SHA256="05965065657b3a8797df034292baed648d73e4c3c1f2466617781242a945925c"
VIT_SHA256="aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
DATASET_INFO_SHA256="9b935ac9a281bb708d7e420e9a2aa4f4fa54c35aa27b41a575ca3820debbfba0"
DATASET_META_SHA256="08f765dbc4695e9618517762a4e1b297d3062a485a06f15ff52f6e52000e3352"

EVAL_SEEDS=(42 43 44)
EPISODES_PER_TASK=50
NUM_TASKS=10
TRAIN_SEED=42
ADAPTER_EPOCHS=40
ADAPTER_CKPT_ID=39
START_SAVE_CHECKPOINT=29
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-17600}"
EVAL_MASTER_PORT_BASE="${EVAL_MASTER_PORT_BASE:-17700}"
REPORT_TO_WANDB="${REPORT_TO_WANDB:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

CURRENT_STAGE="preflight"
ROW_INDEX=0

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "missing directory: $1"
}

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

require_hash() {
    local label="$1" path="$2" expected="$3" actual
    require_file "${path}"
    actual="$(sha256_of "${path}")"
    [[ "${actual}" == "${expected}" ]] \
        || fail "${label} SHA256 mismatch: expected=${expected}, actual=${actual}, path=${path}"
    echo "[VERIFY][OK] ${label} sha256=${actual}"
}

source_lock_lines() {
    (
        cd "${REPO_ROOT}"
        sha256sum \
            "${UPSTREAM_DIR}/eval_libero.py" \
            "${UPSTREAM_DIR}/models/seer_model.py" \
            "${UPSTREAM_DIR}/models/lrnode_modules.py" \
            "${UPSTREAM_DIR}/utils/arguments_utils.py" \
            "${UPSTREAM_DIR}/utils/eval_utils_libero.py" \
            "${UPSTREAM_DIR}/utils/train_utils.py" \
            "${DISTILL_SCRIPT}" \
            "${EVAL_SCRIPT}" \
            "architectures/seer/wrappers/lrnode/$(basename "${BASH_SOURCE[0]}")"
    )
}

initialize_source_lock() {
    local lock="${PAPER_ROOT}/source_sha256.lock" current
    current="$(mktemp /tmp/spatial_teacher33_source.XXXXXX)"
    source_lock_lines > "${current}"
    if [[ -s "${lock}" ]]; then
        if ! cmp -s "${lock}" "${current}"; then
            diff -u "${lock}" "${current}" || true
            rm -f "${current}"
            fail "source changed relative to campaign lock: ${lock}"
        fi
        rm -f "${current}"
    else
        mv "${current}" "${lock}"
    fi
    echo "[VERIFY][OK] source lock: ${lock}"
}

verify_source_lock() {
    local lock="${PAPER_ROOT}/source_sha256.lock" current
    current="$(mktemp /tmp/spatial_teacher33_source.XXXXXX)"
    source_lock_lines > "${current}"
    if ! cmp -s "${lock}" "${current}"; then
        diff -u "${lock}" "${current}" || true
        rm -f "${current}"
        fail "source changed during campaign"
    fi
    rm -f "${current}"
}

verify_dataset() {
    python - "${DATASET_ROOT}/${DATASET}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "conversion_manifest.json").read_text())
expected = {
    "status": "complete",
    "suite": "libero_spatial",
    "num_tasks": 10,
    "num_episodes": 500,
}
actual = {key: manifest.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"Spatial conversion mismatch: expected={expected}, actual={actual}")
print(
    f"[VERIFY][OK] Spatial converted dataset tasks={manifest['num_tasks']} "
    f"episodes={manifest['num_episodes']} steps={manifest['num_steps']}"
)
PY
}

selected_gpu_processes() {
    python - "${GPU_LIST}" <<'PY'
import subprocess
import sys

selected = {int(item) for item in sys.argv[1].split(",")}
gpu_rows = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
    text=True,
)
selected_uuids = set()
for line in gpu_rows.splitlines():
    index, uuid = [part.strip() for part in line.split(",", 1)]
    if int(index) in selected:
        selected_uuids.add(uuid)
app_rows = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
    text=True,
).strip()
for line in app_rows.splitlines():
    if not line.strip():
        continue
    uuid, pid, name = [part.strip() for part in line.split(",", 2)]
    if uuid in selected_uuids:
        print(f"{pid}:{name}")
PY
}

wait_for_selected_gpus() {
    CURRENT_STAGE="wait_for_gpus"
    while true; do
        local busy
        busy="$(selected_gpu_processes)" || fail "nvidia-smi GPU occupancy query failed"
        if [[ -z "${busy}" ]]; then
            echo "[WAIT][DONE] physical GPUs ${GPU_LIST} are free"
            return
        fi
        echo "[WAIT] physical GPUs ${GPU_LIST} busy: ${busy//$'\n'/, }"
        sleep "${WAIT_POLL_SECONDS}"
    done
}

verify_checkpoint_pair() {
    local out="${PAPER_ROOT}/parity/spatial_teacher33_adapter39.json"
    mkdir -p "$(dirname "${out}")"
    BASE_CKPT="${TEACHER_CKPT}" ADAPTER_CKPT="${ADAPTER_CKPT}" OUTPUT_JSON="${out}" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

base_path = Path(os.environ["BASE_CKPT"])
adapter_path = Path(os.environ["ADAPTER_CKPT"])
base = torch.load(base_path, map_location="cpu")["model_state_dict"]
adapter = torch.load(adapter_path, map_location="cpu")["model_state_dict"]
allowed = ("module.lrnode_delta_encoder.", "module.lrnode_dynamics.")
bad = sorted(key for key in adapter if not key.startswith(allowed))
overlap = sorted(set(base) & set(adapter))
if bad:
    raise RuntimeError(f"adapter contains non-LatentLoop tensors: {bad[:8]}")
if overlap:
    raise RuntimeError(f"adapter overwrites teacher tensors: {overlap[:8]}")
if any("lrnode" in key.lower() for key in base):
    raise RuntimeError("teacher checkpoint unexpectedly contains LatentLoop tensors")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "status": "PASS",
    "teacher_checkpoint": str(base_path),
    "teacher_sha256": digest(base_path),
    "teacher_tensor_count": len(base),
    "adapter_checkpoint": str(adapter_path),
    "adapter_sha256": digest(adapter_path),
    "adapter_tensor_count": len(adapter),
    "adapter_numel": sum(value.numel() for value in adapter.values()),
    "shared_tensor_overwrite_count": len(overlap),
    "non_latentloop_adapter_tensor_count": len(bad),
}
Path(os.environ["OUTPUT_JSON"]).write_text(json.dumps(payload, indent=2) + "\n")
print(f"[PARITY][PASS] {os.environ['OUTPUT_JSON']}")
PY
}

validate_eval_row() {
    local root="$1" seed="$2" kind="$3" k="$4"
    python - "${root}" "${seed}" "${kind}" "${k}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seed, kind, k = int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
summaries = sorted(root.glob("*/analysis/eval_summary.json"))
if len(summaries) != 1:
    raise RuntimeError(f"expected one summary under {root}, found {len(summaries)}")
summary_path = summaries[0]
analysis = summary_path.parent
summary = json.loads(summary_path.read_text())
with (analysis / "eval_episode_metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 500:
    raise RuntimeError(f"expected 500 episode rows, found {len(rows)}")
coverage = {(int(row["task_id"]), int(row["episode_id"])) for row in rows}
expected = {(task, episode) for task in range(10) for episode in range(50)}
if coverage != expected:
    raise RuntimeError("task/init-state coverage mismatch")
if {int(float(row["seed"])) for row in rows} != {seed}:
    raise RuntimeError("execution-seed mismatch")
if summary.get("suite") != "libero_spatial":
    raise RuntimeError(f"suite mismatch: {summary.get('suite')}")
renderer = summary.get("renderer_backend", {})
if renderer.get("effective_backend") != "egl":
    raise RuntimeError(f"renderer is not EGL: {renderer}")
if not renderer.get("all_ranks_actual_context_verified"):
    raise RuntimeError("not all ranks verified hardware EGL")
contexts = renderer.get("rank_contexts", [])
if len(contexts) != 4 or not all(item.get("actual_context_verified") for item in contexts):
    raise RuntimeError("expected four verified EGL rank contexts")
lrnode = summary.get("lrnode", {})
query = summary.get("query_reduction", {})
if int(lrnode.get("query_interval", -1)) != k:
    raise RuntimeError(f"K mismatch: {lrnode.get('query_interval')} != {k}")
if kind == "baseline":
    if lrnode.get("enabled") or int(query.get("num_lrnode_update_calls", -1)) != 0:
        raise RuntimeError("baseline unexpectedly used LatentLoop")
else:
    if not lrnode.get("enabled") or not lrnode.get("eval_skip_full_forward"):
        raise RuntimeError("LatentLoop skip path was not enabled")
    if int(query.get("num_fallback_full_calls", -1)) != 0:
        raise RuntimeError("LatentLoop used fallback full calls")
    env_steps = int(query.get("num_env_steps", -1))
    full = int(query.get("num_full_forward_calls", -1))
    updates = int(query.get("num_lrnode_update_calls", -1))
    if full + updates != env_steps:
        raise RuntimeError("full/update calls do not partition environment steps")
for name in ("eval_progress.json", "eval_latency_profile.json"):
    if not (analysis / name).is_file():
        raise FileNotFoundError(analysis / name)
print(
    f"[EVAL VERIFY] Spatial seed={seed} method={kind} K={k} "
    f"episodes={len(rows)} SR={100.0 * float(summary['success_rate']):.2f}%"
)
PY
}

write_row_contract() {
    local root="$1" row_id="$2" seed="$3" kind="$4" k="$5" adapter="$6"
    printf '%s\n' \
        "ROW_ID=${row_id}" \
        "SUITE=libero_spatial" \
        "EVAL_SEED=${seed}" \
        "METHOD=${kind}" \
        "QUERY_INTERVAL=${k}" \
        "EPISODES_PER_TASK=${EPISODES_PER_TASK}" \
        "NUM_TASKS=${NUM_TASKS}" \
        "RENDERER=egl" \
        "TEACHER_CKPT=${TEACHER_CKPT}" \
        "TEACHER_SHA256=${TEACHER_SHA256}" \
        "ADAPTER_CKPT=${adapter}" \
        > "${root}/row_contract.env"
}

run_eval_row() {
    local seed="$1" kind="$2" k="$3" adapter="$4"
    local row_id="spatial_teacher33_seed${seed}_${kind}_k${k}"
    local root="${PAPER_ROOT}/eval/${row_id}"
    local logfile="${PAPER_ROOT}/logs/${row_id}.log"
    local run_baseline=0 intervals="${k}" ours_ckpt="${adapter}"
    ROW_INDEX=$((ROW_INDEX + 1))
    CURRENT_STAGE="eval_${row_id}"
    if [[ -d "${root}" ]] && validate_eval_row "${root}" "${seed}" "${kind}" "${k}" >/dev/null 2>&1; then
        echo "[SKIP] verified complete row: ${row_id}"
        return
    fi
    if [[ -e "${root}" ]]; then
        local quarantine="${root}.incomplete.$(date +%Y%m%d_%H%M%S)"
        echo "[RESUME] preserving incomplete row at ${quarantine}"
        mv "${root}" "${quarantine}"
    fi
    verify_source_lock
    if [[ "${kind}" == "baseline" ]]; then
        run_baseline=1
        intervals=""
        ours_ckpt=""
    else
        require_file "${adapter}"
    fi
    local port=$((EVAL_MASTER_PORT_BASE + ROW_INDEX))
    echo "[EVAL START] row=${row_id} seed=${seed} method=${kind} K=${k} port=${port}"
    local rc
    set +e
    env \
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        LIBERO_GL_BACKEND=egl \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        LIBERO_GL_REQUIRE_ACTUAL=1 \
        LIBERO_PATH="${LIBERO_PATH}" \
        VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
        LRNODE_PROTOCOL_ROOT="${PAPER_ROOT}" \
        SAVE_CHECKPOINT_PATH="${PAPER_ROOT}/eval_checkpoints" \
        EVAL_SUITE=libero_spatial \
        EVAL_SEED="${seed}" \
        EVAL_NUM_EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
        EVAL_NUM_TASKS="${NUM_TASKS}" \
        EVAL_CONTROL_HZ=20 \
        LIBERO_EVAL_MAX_STEPS=600 \
        EVAL_LIBERO_ENSEMBLING=1 \
        BASELINE_CKPT="${TEACHER_CKPT}" \
        BASELINE_CKPT_ID=33 \
        BASELINE_NAME=seer_libero_spatial_teacher33 \
        OURS_CKPT="${ours_ckpt}" \
        OURS_CKPT_ID="${ADAPTER_CKPT_ID}" \
        OURS_NAME=latentloop_libero_spatial_teacher33_adapter39 \
        METHOD_TAG=latentloop_v0 \
        LRNODE_EVAL_BASE_CKPT="${TEACHER_CKPT}" \
        LRNODE_TRAIN_PROTOCOL=adapter \
        LRNODE_FREEZE_SEER_FOR_ADAPTER=1 \
        LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1 \
        LRNODE_EVAL_STEP_LOG=1 \
        LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=1 \
        LRNODE_EVAL_SHADOW_FULL_FORWARD=0 \
        LRNODE_GATE_INIT_BIAS=-4.0 \
        LRNODE_EVAL_ABLATION_MODE=stepwise \
        RUN_BASELINE="${run_baseline}" \
        RUN_OURS_FULL=0 \
        LRNODE_QUERY_INTERVALS_STR="${intervals}" \
        NODE_NUM=4 \
        MASTER_PORT="${port}" \
        SAVE_VIDEO=0 \
        SAVE_VIDEO_SUCC=0 \
        SAVE_VIDEO_FAIL=0 \
        SAVE_VIDEO_ALL_RANKS=0 \
        EXPERIMENT_NAME=spatial_teacher33_egl50 \
        EXPERIMENT_TAG="${row_id}" \
        RESULT_ROOT="${root}" \
        bash "${EVAL_SCRIPT}" 2>&1 | tee -a "${logfile}"
    rc=${PIPESTATUS[0]}
    set -e
    if ! validate_eval_row "${root}" "${seed}" "${kind}" "${k}"; then
        fail "incomplete or invalid eval row: ${row_id}; rc=${rc}; log=${logfile}"
    fi
    if (( rc != 0 )); then
        echo "[WARN] eval wrapper rc=${rc}, but all 500 episodes passed validation"
    fi
    write_row_contract "${root}" "${row_id}" "${seed}" "${kind}" "${k}" "${adapter}"
}

train_adapter() {
    local run_dir="${ADAPTER_SAVE_ROOT}/${ADAPTER_RUN_NAME}"
    CURRENT_STAGE="train_spatial_teacher33_adapter"
    if [[ -s "${ADAPTER_CKPT}" ]]; then
        echo "[SKIP] adapter checkpoint exists: ${ADAPTER_CKPT}"
        verify_checkpoint_pair
        return
    fi
    [[ ! -e "${run_dir}" ]] || fail "partial adapter directory requires review: ${run_dir}"
    verify_source_lock
    echo "[TRAIN START] Spatial teacher33 -> LatentLoop adapter $(date --iso-8601=seconds)"
    local rc
    set +e
    env \
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        LRNODE_PROTOCOL_ROOT="${PAPER_ROOT}" \
        DATASET="${DATASET}" \
        ROOT_DIR="${DATASET_ROOT}" \
        LIBERO_DATASET_INFO_PATH="${DATASET_INFO}" \
        VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
        LIBERO_PATH="${LIBERO_PATH}" \
        BASELINE_CKPT="${TEACHER_CKPT}" \
        BASELINE_CKPT_ID=33 \
        SAVE_CHECKPOINT_PATH="${ADAPTER_SAVE_ROOT}" \
        RUN_NAME="${ADAPTER_RUN_NAME}" \
        METHOD_TAG=latentloop_libero_spatial_teacher33_adapter \
        EXPERIMENT_TAG=spatial_teacher33_adapter39_egl50_v1 \
        NUM_EPOCHS="${ADAPTER_EPOCHS}" \
        START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT}" \
        SEED="${TRAIN_SEED}" \
        LEARNING_RATE=1e-3 \
        WARMUP_EPOCHS=2 \
        LRNODE_TEACHER_TARGET_MODE=shifted_context \
        LRNODE_CONTEXT_SELECTED_STEP=-1 \
        LRNODE_LATENT_WEIGHT=0.05 \
        LRNODE_ACTION_DISTILL_WEIGHT=0.1 \
        LRNODE_BC_WEIGHT=0.0 \
        LRNODE_SMOOTH_WEIGHT=0.001 \
        LRNODE_GATE_INIT_BIAS=-4.0 \
        LRNODE_DETACH_INPUT_LATENT=1 \
        LRNODE_DETACH_TEACHER_LATENT=1 \
        LRNODE_FREEZE_ACTION_HEAD_FOR_LRNODE=1 \
        LRNODE_MULTISTEP_TRAIN=0 \
        LRNODE_RUNTIME_ALIGNED_TRAIN=0 \
        REPORT_TO_WANDB="${REPORT_TO_WANDB}" \
        WANDB_PROJECT=seer_libero_suite \
        NODE_NUM=4 \
        MASTER_PORT="${TRAIN_MASTER_PORT}" \
        bash "${DISTILL_SCRIPT}" 2>&1 | tee -a "${PAPER_ROOT}/logs/spatial_teacher33_adapter_train.log"
    rc=${PIPESTATUS[0]}
    set -e
    require_file "${ADAPTER_CKPT}"
    if (( rc != 0 )); then
        echo "[WARN] training wrapper rc=${rc}, but adapter39 exists; validating checkpoint"
    fi
    verify_checkpoint_pair
    echo "[TRAIN DONE] adapter=${ADAPTER_CKPT}"
}

write_campaign_summary() {
    TEACHER_CKPT="${TEACHER_CKPT}" ADAPTER_CKPT="${ADAPTER_CKPT}" PAPER_ROOT="${PAPER_ROOT}" python - <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import statistics
from datetime import datetime, timezone

root = pathlib.Path(os.environ["PAPER_ROOT"])

def digest(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
for contract_path in sorted((root / "eval").glob("*/row_contract.env")):
    contract = {}
    for line in contract_path.read_text().splitlines():
        key, value = line.split("=", 1)
        contract[key] = value
    summaries = sorted(contract_path.parent.glob("*/analysis/eval_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"missing summary for {contract_path.parent}")
    summary = json.loads(summaries[0].read_text())
    rows.append(
        {
            "row_id": contract["ROW_ID"],
            "seed": int(contract["EVAL_SEED"]),
            "method": contract["METHOD"],
            "query_interval": int(contract["QUERY_INTERVAL"]),
            "success_rate": float(summary["success_rate"]),
            "successes": int(round(float(summary["success_rate"]) * 500)),
            "episodes": 500,
            "avg_policy_step_latency_ms": float(summary["avg_policy_step_latency_ms"]),
            "avg_full_forward_latency_ms": float(summary["avg_full_forward_latency_ms"]),
            "avg_lrnode_latency_ms": float(summary["avg_lrnode_latency_ms"]),
            "full_query_reduction_ratio": float(summary["full_query_reduction_ratio"]),
        }
    )
aggregates = {}
for method in ("baseline", "latentloop"):
    values = [row["success_rate"] for row in rows if row["method"] == method]
    aggregates[method] = {
        "num_execution_seeds": len(values),
        "mean_success_rate": statistics.mean(values),
        "sample_std_success_rate": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
payload = {
    "status": "COMPLETE",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "suite": "libero_spatial",
    "teacher_checkpoint": os.environ["TEACHER_CKPT"],
    "teacher_sha256": digest(os.environ["TEACHER_CKPT"]),
    "adapter_checkpoint": os.environ["ADAPTER_CKPT"],
    "adapter_sha256": digest(os.environ["ADAPTER_CKPT"]),
    "training": {
        "seed": 42,
        "epochs": 40,
        "selected_adapter_epoch": 39,
        "learning_rate": 1e-3,
        "warmup_epochs": 2,
        "teacher_frozen": True,
        "action_head_frozen": True,
        "target_mode": "shifted_context",
        "runtime_aligned_training": False,
    },
    "evaluation": {
        "renderer": "egl",
        "episodes_per_task": 50,
        "num_tasks": 10,
        "execution_seeds": [42, 43, 44],
        "baseline_query_interval": 1,
        "latentloop_query_interval": 4,
    },
    "rows": rows,
    "aggregates": aggregates,
}
(root / "campaign_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
with (root / "campaign_rows.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"[SUMMARY] {root / 'campaign_summary.json'}")
PY
}

on_exit() {
    local rc=$?
    if (( rc == 0 )); then
        printf 'status=COMPLETE\nstage=%s\ntime=%s\n' "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" \
            > "${PAPER_ROOT}/queue_complete.env"
    else
        printf 'status=FAILED\nexit_code=%s\nstage=%s\ntime=%s\n' \
            "${rc}" "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" \
            > "${PAPER_ROOT}/queue_failed.env"
    fi
}

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || fail "expected ${EXPECTED_HOST}, got $(hostname)"
[[ "$(readlink -f "${REPO_ROOT}")" == "$(readlink -f "${EXPECTED_REPO}")" ]] \
    || fail "unexpected source tree: ${REPO_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == seer_libero ]] || fail "activate conda environment seer_libero"
[[ "${GPU_LIST}" == 0,1,2,3 ]] || fail "set CUDA_VISIBLE_DEVICES=0,1,2,3"
[[ "${PREFLIGHT_ONLY}" == 0 || "${PREFLIGHT_ONLY}" == 1 ]] || fail "PREFLIGHT_ONLY must be 0 or 1"
[[ "${REPORT_TO_WANDB}" == 0 || "${REPORT_TO_WANDB}" == 1 ]] || fail "REPORT_TO_WANDB must be 0 or 1"
require_dir "${LIBERO_PATH}"
require_dir "${DATASET_ROOT}/${DATASET}"
require_file "${DISTILL_SCRIPT}"
require_file "${EVAL_SCRIPT}"
require_hash "Spatial teacher33" "${TEACHER_CKPT}" "${TEACHER_SHA256}"
require_hash "MAE ViT" "${VIT_CHECKPOINT_PATH}" "${VIT_SHA256}"
require_hash "Spatial data_info" "${DATASET_INFO}" "${DATASET_INFO_SHA256}"
require_hash "Spatial meta_info" "${DATASET_META}" "${DATASET_META_SHA256}"
verify_dataset

python - "${GPU_LIST}" <<'PY'
import sys
import torch

expected = sys.argv[1].split(",")
if torch.cuda.device_count() != len(expected):
    raise RuntimeError(
        f"expected {len(expected)} visible CUDA devices, got {torch.cuda.device_count()}"
    )
names = [torch.cuda.get_device_name(index) for index in range(len(expected))]
if not all("RTX 3090" in name for name in names):
    raise RuntimeError(f"paper evaluation requires four RTX 3090 GPUs: {names}")
print(f"[VERIFY][OK] visible CUDA devices: {names}")
PY

if [[ "${PREFLIGHT_ONLY}" == 1 ]]; then
    tmp_lock="$(mktemp /tmp/spatial_teacher33_preflight_lock.XXXXXX)"
    source_lock_lines > "${tmp_lock}"
    echo "[PREFLIGHT][PASS] source, checkpoints, dataset, and four-GPU contract verified"
    echo "[PREFLIGHT] source lock candidate: ${tmp_lock}"
    exit 0
fi

mkdir -p "${PAPER_ROOT}/logs" "${PAPER_ROOT}/eval" "${PAPER_ROOT}/parity"
exec 9>"${PAPER_ROOT}/queue.lock"
flock -n 9 || fail "another process owns ${PAPER_ROOT}/queue.lock"
rm -f "${PAPER_ROOT}/queue_failed.env" "${PAPER_ROOT}/queue_complete.env"
exec > >(tee -a "${PAPER_ROOT}/sequential_run.log") 2>&1
trap on_exit EXIT
initialize_source_lock
cat > "${PAPER_ROOT}/campaign_contract.env" <<EOF
PROTOCOL=spatial_teacher33_adapter39_egl50_v1
HOST=${EXPECTED_HOST}
SOURCE_REPO=${REPO_ROOT}
GPU_LIST=${GPU_LIST}
SUITE=libero_spatial
RENDERER=egl
TRAIN_SEED=${TRAIN_SEED}
EVAL_SEEDS=42,43,44
EPISODES_PER_TASK=${EPISODES_PER_TASK}
NUM_TASKS=${NUM_TASKS}
TEACHER_CKPT=${TEACHER_CKPT}
TEACHER_SHA256=${TEACHER_SHA256}
ADAPTER_CKPT=${ADAPTER_CKPT}
ADAPTER_EPOCHS=${ADAPTER_EPOCHS}
ADAPTER_CKPT_ID=${ADAPTER_CKPT_ID}
BASELINE_K=1
LATENTLOOP_K=4
EOF

echo "============================================================"
echo "[CAMPAIGN] Spatial teacher33 -> adapter39, EGL-50, seeds 42/43/44"
echo "[ORDER] teacher33 K1 evaluation -> adapter training -> adapter39 K4 evaluation"
echo "[TRAIN] 40 epochs, LR=1e-3, warmup=2, frozen teacher/action head"
echo "[EVAL] 50 episodes/task x 10 tasks x 3 execution seeds per method"
echo "[ROOT] ${PAPER_ROOT}"
echo "============================================================"

wait_for_selected_gpus
for seed in "${EVAL_SEEDS[@]}"; do
    run_eval_row "${seed}" baseline 1 ""
done

train_adapter

for seed in "${EVAL_SEEDS[@]}"; do
    run_eval_row "${seed}" latentloop 4 "${ADAPTER_CKPT}"
done

CURRENT_STAGE="summarize"
write_campaign_summary
CURRENT_STAGE="complete"
printf '%s\n' "$(date --iso-8601=seconds)" > "${PAPER_ROOT}/sequential_run_complete.txt"
echo "[DONE] ${PAPER_ROOT}/campaign_summary.json"
