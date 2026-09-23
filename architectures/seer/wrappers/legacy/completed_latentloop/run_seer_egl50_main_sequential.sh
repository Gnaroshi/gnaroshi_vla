#!/usr/bin/env bash

set -Eeuo pipefail

# Resumable sd1 paper-evaluation queue.
#
# The queue waits for the in-flight Object scratch run, retires the stopped
# legacy OSMesa wrapper, and then runs the EGL-50 efficacy protocol. Large
# artifacts stay under shared storage; this source file has a stable name so
# process listings expose an auditable experiment entry point.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
SCRATCH_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/scratch.sh"
DISTILL_SCRIPT="${SCRIPT_DIR}/distill_node.sh"
EVAL_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
EXPECTED_REPO="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-seer_public33_egl50_main_v1}"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer}"
PAPER_ROOT="${PAPER_ROOT:-${SHARED_SEER_ROOT}/paper_egl50/${CAMPAIGN_TAG}}"
SUITE_TRAIN_ROOT="${SUITE_TRAIN_ROOT:-${SHARED_SEER_ROOT}/libero_suite_study/campaigns/spatial_object_goal_v1}"
CONVERTED_ROOT="${CONVERTED_ROOT:-${SHARED_SEER_ROOT}/libero_suite_study/datasets}"
LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH:-${SHARED_SEER_ROOT}/vit_mae/mae_pretrain_vit_base.pth}"
PUBLIC33_CKPT="${PUBLIC33_CKPT:-${SHARED_SEER_ROOT}/checkpoints_Seer_LIBERO_LONG/Seer/33.pth}"
PUBLIC33_ADAPTER="${PUBLIC33_ADAPTER:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/seer.incoming_20260817/lrnode/official_seer_libero_k4_v1/train/adapters/official_seer_ckpt33_lrnode_adapter_v1/39.pth}"

PUBLIC33_SHA256="${PUBLIC33_SHA256:-a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646}"
PUBLIC33_ADAPTER_SHA256="${PUBLIC33_ADAPTER_SHA256:-3f70179ab9b1bae64fc772d71c57a93592b9f82e53b5fcaf1a6beb319c280462}"
VIT_SHA256="${VIT_SHA256:-aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d}"

TRAIN_SEED="${TRAIN_SEED:-42}"
EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-42 43 44}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS_STR}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
NUM_TASKS="${NUM_TASKS:-10}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-40}"
ADAPTER_EPOCHS="${ADAPTER_EPOCHS:-40}"
BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-39}"
ADAPTER_CKPT_ID="${ADAPTER_CKPT_ID:-39}"
START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT:-29}"
REPORT_TO_WANDB="${REPORT_TO_WANDB:-1}"
TRAIN_MASTER_PORT_BASE="${TRAIN_MASTER_PORT_BASE:-16800}"
EVAL_MASTER_PORT_BASE="${EVAL_MASTER_PORT_BASE:-17100}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
RUN_DIRECT_MECHANISMS="${RUN_DIRECT_MECHANISMS:-1}"
RUN_K_CURVE="${RUN_K_CURVE:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

OBJECT_RUN_NAME="seer_libero_object_scratch_seed42"
OBJECT_CKPT="${SUITE_TRAIN_ROOT}/train/libero_object/baseline/${OBJECT_RUN_NAME}/39.pth"
LEGACY_WRAPPER_PATTERN="architectures/seer/wrappers/lrnode/run_libero_suite_baseline_latentloop.sh"

CURRENT_STAGE="startup"
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

is_bool() {
    [[ "$1" == "0" || "$1" == "1" ]]
}

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

require_sha256() {
    local label="$1" path="$2" expected="$3" actual
    require_file "${path}"
    actual="$(sha256_of "${path}")"
    [[ "${actual}" == "${expected}" ]] \
        || fail "${label} SHA256 mismatch: expected=${expected}, actual=${actual}, path=${path}"
    echo "[VERIFY][OK] ${label} sha256=${actual}"
}

on_exit() {
    local rc=$?
    if (( rc == 0 )); then
        printf 'status=COMPLETE\ntime=%s\n' "$(date --iso-8601=seconds)" \
            > "${PAPER_ROOT}/queue_complete.env"
    else
        printf 'status=FAILED\nexit_code=%s\nstage=%s\ntime=%s\n' \
            "${rc}" "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" \
            > "${PAPER_ROOT}/queue_failed.env"
    fi
}

selected_gpu_processes() {
    python - "${GPU_LIST}" <<'PY'
import subprocess
import sys

selected = {int(x) for x in sys.argv[1].split(",")}
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
        echo "[WAIT] physical GPUs ${GPU_LIST} still busy: ${busy//$'\n'/, }"
        sleep "${WAIT_POLL_SECONDS}"
    done
}

wait_for_object_training() {
    CURRENT_STAGE="wait_for_object_teacher39"
    local pattern="torchrun.*--run_name ${OBJECT_RUN_NAME}"
    while pgrep -f -- "${pattern}" >/dev/null 2>&1; do
        local latest="none"
        if [[ -d "$(dirname "${OBJECT_CKPT}")" ]]; then
            latest="$(find "$(dirname "${OBJECT_CKPT}")" -maxdepth 1 -name '*.pth' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1)"
            latest="${latest:-none}"
        fi
        echo "[WAIT] Object teacher is training; latest_checkpoint=${latest}; next_check=${WAIT_POLL_SECONDS}s"
        sleep "${WAIT_POLL_SECONDS}"
    done
    require_file "${OBJECT_CKPT}"
    echo "[WAIT][DONE] Object teacher39 is complete: ${OBJECT_CKPT}"
}

retire_legacy_wrapper() {
    CURRENT_STAGE="retire_legacy_osmesa_wrapper"
    local pid stat cmd
    mapfile -t legacy_pids < <(pgrep -f -- "bash ${LEGACY_WRAPPER_PATTERN}" || true)
    for pid in "${legacy_pids[@]:-}"; do
        [[ -n "${pid}" && "${pid}" != "$$" ]] || continue
        stat="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
        cmd="$(ps -o cmd= -p "${pid}" 2>/dev/null || true)"
        [[ "${cmd}" == *"${LEGACY_WRAPPER_PATTERN}"* ]] || continue
        echo "[CLEANUP] retiring legacy wrapper pid=${pid} stat=${stat}"
        kill -TERM "${pid}" 2>/dev/null || true
        if [[ "${stat}" == T* ]]; then
            kill -CONT "${pid}" 2>/dev/null || true
        fi
    done
    sleep 2
    if pgrep -f -- "eval_libero.py.*${OBJECT_RUN_NAME}" >/dev/null 2>&1; then
        fail "legacy Object OSMesa evaluation started unexpectedly"
    fi
}

source_lock_lines() {
    sha256sum \
        "${UPSTREAM_DIR}/eval_libero.py" \
        "${UPSTREAM_DIR}/models/seer_model.py" \
        "${UPSTREAM_DIR}/models/lrnode_modules.py" \
        "${UPSTREAM_DIR}/utils/arguments_utils.py" \
        "${UPSTREAM_DIR}/utils/eval_utils_libero.py" \
        "${UPSTREAM_DIR}/utils/train_utils.py" \
        "${SCRATCH_SCRIPT}" \
        "${DISTILL_SCRIPT}" \
        "${EVAL_SCRIPT}" \
        "${BASH_SOURCE[0]}"
}

initialize_source_lock() {
    local lock="${PAPER_ROOT}/source_sha256.lock" current
    current="$(mktemp /tmp/seer_egl50_source.XXXXXX)"
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
    current="$(mktemp /tmp/seer_egl50_source.XXXXXX)"
    source_lock_lines > "${current}"
    if ! cmp -s "${lock}" "${current}"; then
        diff -u "${lock}" "${current}" || true
        rm -f "${current}"
        fail "source changed while campaign was running"
    fi
    rm -f "${current}"
}

verify_converted_suite() {
    local suite="$1"
    local target="${CONVERTED_ROOT}/${suite}_converted"
    python - "${target}" "${suite}" <<'PY'
import json
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
suite = sys.argv[2]
manifest = json.loads((target / "conversion_manifest.json").read_text(encoding="utf-8"))
expected = {"status": "complete", "suite": suite, "num_tasks": 10, "num_episodes": 500}
actual = {key: manifest.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(f"invalid conversion manifest: expected={expected}, actual={actual}")
for name in ("meta_info.h5", "data_info.json"):
    if not (target / name).is_file():
        raise FileNotFoundError(target / name)
print(f"[VERIFY][OK] converted dataset {suite}: episodes=500")
PY
}

verify_checkpoint_pair() {
    local label="$1" base="$2" adapter="$3"
    local out="${PAPER_ROOT}/parity/${label}.json"
    mkdir -p "$(dirname "${out}")"
    BASE_CKPT="${base}" ADAPTER_CKPT="${adapter}" OUTPUT_JSON="${out}" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch

base_path = Path(os.environ["BASE_CKPT"])
adapter_path = Path(os.environ["ADAPTER_CKPT"])
base_payload = torch.load(base_path, map_location="cpu")
adapter_payload = torch.load(adapter_path, map_location="cpu")
base = base_payload["model_state_dict"]
adapter = adapter_payload["model_state_dict"]
allowed = ("module.lrnode_delta_encoder.", "module.lrnode_dynamics.")
bad = sorted(key for key in adapter if not key.startswith(allowed))
overlap = sorted(set(base) & set(adapter))
if bad:
    raise RuntimeError(f"adapter contains non-LatentLoop tensors: {bad[:8]}")
if overlap:
    raise RuntimeError(f"adapter overwrites shared Seer tensors: {overlap[:8]}")
if any("lrnode" in key.lower() for key in base):
    raise RuntimeError("teacher checkpoint unexpectedly contains LatentLoop tensors")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

result = {
    "status": "PASS",
    "base_checkpoint": str(base_path),
    "base_sha256": digest(base_path),
    "base_tensor_count": len(base),
    "base_numel": sum(value.numel() for value in base.values()),
    "adapter_checkpoint": str(adapter_path),
    "adapter_sha256": digest(adapter_path),
    "adapter_tensor_count": len(adapter),
    "adapter_numel": sum(value.numel() for value in adapter.values()),
    "shared_tensor_overwrite_count": len(overlap),
    "non_latentloop_adapter_tensor_count": len(bad),
    "k1_contract": "full Seer path; updater call count must remain zero",
}
Path(os.environ["OUTPUT_JSON"]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"[PARITY][PASS] {os.environ['OUTPUT_JSON']}")
PY
}

run_training_command() {
    local label="$1" logfile="$2" artifact="$3"
    shift 3
    verify_source_lock
    mkdir -p "$(dirname "${logfile}")"
    echo "[TRAIN START] ${label} $(date --iso-8601=seconds)"
    local started rc elapsed
    started="$(date +%s)"
    set +e
    "$@" 2>&1 | tee -a "${logfile}"
    rc=${PIPESTATUS[0]}
    set -e
    elapsed=$(( $(date +%s) - started ))
    if [[ ! -s "${artifact}" ]]; then
        fail "training did not produce ${artifact}; rc=${rc}; log=${logfile}"
    fi
    if (( rc != 0 )); then
        echo "[WARN] training wrapper rc=${rc}, but terminal checkpoint is complete; continuing"
    fi
    echo "[TRAIN DONE] ${label} elapsed=$((elapsed / 3600))h$(((elapsed % 3600) / 60))m artifact=${artifact}"
}

suite_baseline_ckpt() {
    local suite="$1"
    echo "${SUITE_TRAIN_ROOT}/train/${suite}/baseline/seer_${suite}_scratch_seed${TRAIN_SEED}/${BASELINE_CKPT_ID}.pth"
}

suite_adapter_ckpt() {
    local suite="$1"
    echo "${SUITE_TRAIN_ROOT}/train/${suite}/adapter/latentloop_${suite}_teacher${BASELINE_CKPT_ID}_seed${TRAIN_SEED}/${ADAPTER_CKPT_ID}.pth"
}

train_suite_baseline() {
    local suite="$1" index="$2"
    local dataset="${suite}_converted"
    local root_dir="${CONVERTED_ROOT}"
    local info="${CONVERTED_ROOT}/${dataset}/data_info.json"
    local save_root="${SUITE_TRAIN_ROOT}/train/${suite}/baseline"
    local run_name="seer_${suite}_scratch_seed${TRAIN_SEED}"
    local run_dir="${save_root}/${run_name}"
    local ckpt="${run_dir}/${BASELINE_CKPT_ID}.pth"
    if [[ -s "${ckpt}" ]]; then
        echo "[SKIP] suite baseline exists: ${ckpt}"
        return
    fi
    [[ ! -e "${run_dir}" ]] || fail "partial suite baseline directory requires review: ${run_dir}"
    CURRENT_STAGE="train_${suite}_baseline"
    run_training_command "${suite} baseline" "${SUITE_TRAIN_ROOT}/logs/${suite}_baseline_train.log" "${ckpt}" \
        env \
            CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            LRNODE_PROTOCOL_ROOT="${SUITE_TRAIN_ROOT}" \
            DATASET="${dataset}" \
            ROOT_DIR="${root_dir}" \
            LIBERO_DATASET_INFO_PATH="${info}" \
            VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
            LIBERO_PATH="${LIBERO_PATH}" \
            SAVE_CHECKPOINT_PATH="${save_root}" \
            RUN_NAME="${run_name}" \
            METHOD_TAG="seer_${suite}_scratch" \
            EXPERIMENT_TAG="${CAMPAIGN_TAG}_${suite}" \
            NUM_EPOCHS="${BASELINE_EPOCHS}" \
            START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT}" \
            SEED="${TRAIN_SEED}" \
            LEARNING_RATE=1e-3 \
            REPORT_TO_WANDB="${REPORT_TO_WANDB}" \
            WANDB_PROJECT=seer_libero_suite \
            NODE_NUM=4 \
            MASTER_PORT="$((TRAIN_MASTER_PORT_BASE + index * 20))" \
            bash "${SCRATCH_SCRIPT}"
}

train_suite_adapter() {
    local suite="$1" index="$2"
    local dataset="${suite}_converted"
    local root_dir="${CONVERTED_ROOT}"
    local info="${CONVERTED_ROOT}/${dataset}/data_info.json"
    local baseline="$(suite_baseline_ckpt "${suite}")"
    local save_root="${SUITE_TRAIN_ROOT}/train/${suite}/adapter"
    local run_name="latentloop_${suite}_teacher${BASELINE_CKPT_ID}_seed${TRAIN_SEED}"
    local run_dir="${save_root}/${run_name}"
    local ckpt="${run_dir}/${ADAPTER_CKPT_ID}.pth"
    require_file "${baseline}"
    if [[ -s "${ckpt}" ]]; then
        echo "[SKIP] suite adapter exists: ${ckpt}"
        verify_checkpoint_pair "${suite}_teacher_adapter" "${baseline}" "${ckpt}"
        return
    fi
    [[ ! -e "${run_dir}" ]] || fail "partial suite adapter directory requires review: ${run_dir}"
    CURRENT_STAGE="train_${suite}_adapter"
    run_training_command "${suite} LatentLoop adapter" "${SUITE_TRAIN_ROOT}/logs/${suite}_adapter_train.log" "${ckpt}" \
        env \
            CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
            LRNODE_PROTOCOL_ROOT="${SUITE_TRAIN_ROOT}" \
            DATASET="${dataset}" \
            ROOT_DIR="${root_dir}" \
            LIBERO_DATASET_INFO_PATH="${info}" \
            VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
            LIBERO_PATH="${LIBERO_PATH}" \
            BASELINE_CKPT="${baseline}" \
            BASELINE_CKPT_ID="${BASELINE_CKPT_ID}" \
            SAVE_CHECKPOINT_PATH="${save_root}" \
            RUN_NAME="${run_name}" \
            METHOD_TAG="latentloop_${suite}_adapter" \
            EXPERIMENT_TAG="${CAMPAIGN_TAG}_${suite}" \
            NUM_EPOCHS="${ADAPTER_EPOCHS}" \
            START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT}" \
            SEED="${TRAIN_SEED}" \
            LEARNING_RATE=1e-3 \
            WARMUP_EPOCHS=2 \
            REPORT_TO_WANDB="${REPORT_TO_WANDB}" \
            WANDB_PROJECT=seer_libero_suite \
            NODE_NUM=4 \
            MASTER_PORT="$((TRAIN_MASTER_PORT_BASE + index * 20 + 10))" \
            bash "${DISTILL_SCRIPT}"
    verify_checkpoint_pair "${suite}_teacher_adapter" "${baseline}" "${ckpt}"
}

validate_eval_row() {
    local root="$1" suite="$2" seed="$3" kind="$4" k="$5" ablation="$6"
    python - "${root}" "${suite}" "${seed}" "${kind}" "${k}" "${ablation}" \
        "${EPISODES_PER_TASK}" "${NUM_TASKS}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
suite, seed, kind = sys.argv[2], int(sys.argv[3]), sys.argv[4]
k, ablation = int(sys.argv[5]), sys.argv[6]
episodes_per_task, num_tasks = int(sys.argv[7]), int(sys.argv[8])
summaries = sorted(root.glob("*/analysis/eval_summary.json"))
if len(summaries) != 1:
    raise RuntimeError(f"expected one summary under {root}, found {len(summaries)}")
summary_path = summaries[0]
analysis = summary_path.parent
summary = json.loads(summary_path.read_text(encoding="utf-8"))
with (analysis / "eval_episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
expected_count = episodes_per_task * num_tasks
if len(rows) != expected_count:
    raise RuntimeError(f"expected {expected_count} episode rows, found {len(rows)}")
coverage = {(int(row["task_id"]), int(row["episode_id"])) for row in rows}
expected_coverage = {(task, episode) for task in range(num_tasks) for episode in range(episodes_per_task)}
if coverage != expected_coverage:
    raise RuntimeError("task/init-state coverage mismatch")
if {int(float(row["seed"])) for row in rows} != {seed}:
    raise RuntimeError("episode seed mismatch")
if summary.get("suite") != suite:
    raise RuntimeError(f"suite mismatch: {summary.get('suite')} != {suite}")
renderer = summary.get("renderer_backend", {})
if renderer.get("requested_backend") != "egl" or renderer.get("effective_backend") != "egl":
    raise RuntimeError(f"renderer is not EGL: {renderer}")
if not renderer.get("all_ranks_actual_context_verified", False):
    raise RuntimeError("not all ranks verified an actual EGL context")
rank_contexts = renderer.get("rank_contexts", [])
if len(rank_contexts) != 4 or not all(item.get("actual_context_verified") for item in rank_contexts):
    raise RuntimeError("expected four verified EGL rank contexts")
lrnode = summary.get("lrnode", {})
query = summary.get("query_reduction", {})
if int(lrnode.get("query_interval", -1)) != k:
    raise RuntimeError(f"K mismatch: {lrnode.get('query_interval')} != {k}")
if summary.get("lrnode_eval_ablation_mode") != ablation:
    raise RuntimeError("ablation mode mismatch")
if kind == "baseline":
    if lrnode.get("enabled") or lrnode.get("eval_skip_full_forward"):
        raise RuntimeError("baseline unexpectedly enabled LatentLoop")
    if int(query.get("num_lrnode_update_calls", -1)) != 0:
        raise RuntimeError("baseline made LatentLoop update calls")
else:
    if not lrnode.get("enabled") or not lrnode.get("eval_skip_full_forward"):
        raise RuntimeError("LatentLoop skip path was not enabled")
    if int(query.get("num_fallback_full_calls", -1)) != 0:
        raise RuntimeError("LatentLoop used fallback full calls")
    env_steps = int(query.get("num_env_steps", -1))
    full = int(query.get("num_full_forward_calls", -1))
    if ablation in {"stepwise", "no_delta"}:
        updates = int(query.get("num_lrnode_update_calls", -1))
        if full + updates != env_steps:
            raise RuntimeError("full/update calls do not partition environment steps")
    elif ablation == "seer_token_chunk":
        if full + int(query.get("num_chunk_token_steps", -1)) != env_steps:
            raise RuntimeError("full/chunk calls do not partition environment steps")
    elif ablation == "hold_latent":
        if full + int(query.get("num_hold_latent_steps", -1)) != env_steps:
            raise RuntimeError("full/hold-latent calls do not partition environment steps")
    elif ablation == "hold_action":
        if full + int(query.get("num_hold_action_steps", -1)) != env_steps:
            raise RuntimeError("full/hold-action calls do not partition environment steps")
for name in ("eval_progress.json", "eval_latency_profile.json"):
    if not (analysis / name).is_file():
        raise FileNotFoundError(analysis / name)
print(
    f"[EVAL VERIFY] suite={suite} seed={seed} method={kind} K={k} "
    f"ablation={ablation} episodes={len(rows)} SR={100.0 * float(summary['success_rate']):.2f}%"
)
PY
}

write_eval_contract() {
    local root="$1" row_id="$2" suite="$3" seed="$4" kind="$5" k="$6" ablation="$7" base="$8" adapter="$9"
    printf '%s\n' \
        "ROW_ID=${row_id}" \
        "SUITE=${suite}" \
        "EVAL_SEED=${seed}" \
        "METHOD=${kind}" \
        "QUERY_INTERVAL=${k}" \
        "ABLATION_MODE=${ablation}" \
        "EPISODES_PER_TASK=${EPISODES_PER_TASK}" \
        "NUM_TASKS=${NUM_TASKS}" \
        "RENDERER=egl" \
        "BASELINE_CKPT=${base}" \
        "ADAPTER_CKPT=${adapter}" \
        > "${root}/row_contract.env"
}

run_eval_row() {
    local row_id="$1" suite="$2" seed="$3" kind="$4" k="$5" baseline="$6" adapter="$7" ablation="${8:-stepwise}"
    local root="${PAPER_ROOT}/eval/${row_id}"
    local logfile="${PAPER_ROOT}/logs/${row_id}.log"
    local run_baseline=0 intervals="${k}"
    ROW_INDEX=$((ROW_INDEX + 1))
    CURRENT_STAGE="eval_${row_id}"
    if [[ -d "${root}" ]] && validate_eval_row "${root}" "${suite}" "${seed}" "${kind}" "${k}" "${ablation}" >/dev/null 2>&1; then
        echo "[SKIP] verified complete row: ${row_id}"
        return
    fi
    if [[ -e "${root}" ]]; then
        local quarantine="${root}.incomplete.$(date +%Y%m%d_%H%M%S)"
        echo "[RESUME] preserving incomplete row at ${quarantine}"
        mv "${root}" "${quarantine}"
    fi
    verify_source_lock
    require_file "${baseline}"
    require_file "${adapter}"
    if [[ "${kind}" == "baseline" ]]; then
        run_baseline=1
        intervals=""
    fi
    local port=$((EVAL_MASTER_PORT_BASE + ROW_INDEX))
    mkdir -p "$(dirname "${logfile}")"
    echo "[EVAL START] row=${row_id} suite=${suite} seed=${seed} method=${kind} K=${k} ablation=${ablation} port=${port}"
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
        EVAL_SUITE="${suite}" \
        EVAL_SEED="${seed}" \
        EVAL_NUM_EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
        EVAL_NUM_TASKS="${NUM_TASKS}" \
        EVAL_CONTROL_HZ=20 \
        LIBERO_EVAL_MAX_STEPS=600 \
        EVAL_LIBERO_ENSEMBLING=1 \
        BASELINE_CKPT="${baseline}" \
        BASELINE_CKPT_ID="$(basename "${baseline}" .pth)" \
        BASELINE_NAME="seer_${suite}_teacher" \
        OURS_CKPT="${adapter}" \
        OURS_CKPT_ID="$(basename "${adapter}" .pth)" \
        OURS_NAME="latentloop_${suite}" \
        METHOD_TAG=latentloop_v0 \
        LRNODE_EVAL_BASE_CKPT="${baseline}" \
        LRNODE_TRAIN_PROTOCOL=adapter \
        LRNODE_FREEZE_SEER_FOR_ADAPTER=1 \
        LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1 \
        LRNODE_EVAL_STEP_LOG=1 \
        LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=1 \
        LRNODE_EVAL_SHADOW_FULL_FORWARD=0 \
        LRNODE_GATE_INIT_BIAS=-4.0 \
        LRNODE_EVAL_ABLATION_MODE="${ablation}" \
        LRNODE_NO_DELTA_MODE=zero \
        LRNODE_CHUNK_TOKEN_POLICY=skip_only \
        RUN_BASELINE="${run_baseline}" \
        RUN_OURS_FULL=0 \
        LRNODE_QUERY_INTERVALS_STR="${intervals}" \
        NODE_NUM=4 \
        MASTER_PORT="${port}" \
        SAVE_VIDEO=0 \
        SAVE_VIDEO_SUCC=0 \
        SAVE_VIDEO_FAIL=0 \
        SAVE_VIDEO_ALL_RANKS=0 \
        EXPERIMENT_NAME=seer_egl50 \
        EXPERIMENT_TAG="${row_id}" \
        RESULT_ROOT="${root}" \
        bash "${EVAL_SCRIPT}" 2>&1 | tee -a "${logfile}"
    rc=${PIPESTATUS[0]}
    set -e
    if ! validate_eval_row "${root}" "${suite}" "${seed}" "${kind}" "${k}" "${ablation}"; then
        fail "incomplete or invalid eval row: ${row_id}; rc=${rc}; log=${logfile}"
    fi
    if (( rc != 0 )); then
        echo "[WARN] eval wrapper rc=${rc}, but all 500 episodes and required artifacts passed validation"
    fi
    write_eval_contract "${root}" "${row_id}" "${suite}" "${seed}" "${kind}" "${k}" "${ablation}" "${baseline}" "${adapter}"
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
        "episodes": sum(int(x.get("num_episodes", 0)) for x in summary.get("task_results", [])),
        "success_rate_pct": 100.0 * float(summary.get("success_rate", 0.0)),
        "env_steps": int(query.get("num_env_steps", 0)),
        "full_forward_calls": int(query.get("num_full_forward_calls", 0)),
        "latentloop_update_calls": int(query.get("num_lrnode_update_calls", 0)),
        "query_reduction_pct": 100.0 * float(query.get("full_query_reduction_ratio", 0.0)),
        "avg_policy_step_ms": 1000.0 * float(lrnode.get("avg_policy_step_latency_sec", 0.0)),
        "renderer": summary.get("renderer_backend", {}).get("effective_backend"),
    })
output = root / "campaign_rows.csv"
if records:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
print(f"[CAMPAIGN TABLE] complete_rows={len(records)} path={output}")
PY
}

run_long_main_seed() {
    local seed="$1"
    run_eval_row "long_seed${seed}_baseline_k1" libero_10 "${seed}" baseline 1 "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" stepwise
    run_eval_row "long_seed${seed}_latentloop_k4" libero_10 "${seed}" latentloop 4 "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" stepwise
    run_eval_row "long_seed${seed}_latentloop_k8" libero_10 "${seed}" latentloop 8 "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" stepwise
}

run_suite_main_seed() {
    local seed="$1" suite base adapter
    for suite in libero_spatial libero_object libero_goal; do
        base="$(suite_baseline_ckpt "${suite}")"
        adapter="$(suite_adapter_ckpt "${suite}")"
        run_eval_row "${suite}_seed${seed}_baseline_k1" "${suite}" "${seed}" baseline 1 "${base}" "${adapter}" stepwise
        run_eval_row "${suite}_seed${seed}_latentloop_k4" "${suite}" "${seed}" latentloop 4 "${base}" "${adapter}" stepwise
    done
}

run_direct_mechanisms() {
    local seed
    for seed in "${EVAL_SEEDS[@]}"; do
        run_eval_row "long_seed${seed}_no_observation_k4" libero_10 "${seed}" latentloop 4 "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" no_delta
        run_eval_row "long_seed${seed}_predicted_horizon_replay_k4" libero_10 "${seed}" latentloop 4 "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" seer_token_chunk
    done
    run_eval_row "long_seed42_hold_latent_k4" libero_10 42 latentloop 4 "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" hold_latent
    run_eval_row "long_seed42_hold_action_k4" libero_10 42 latentloop 4 "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" hold_action
}

run_k_curve() {
    local k seed
    for k in 2 3 5 6 7 9 10 11 12 13 14 15 16; do
        run_eval_row "long_seed42_latentloop_k${k}" libero_10 42 latentloop "${k}" "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" stepwise
    done
    for seed in 43 44; do
        for k in 2 12 16; do
            run_eval_row "long_seed${seed}_latentloop_k${k}" libero_10 "${seed}" latentloop "${k}" "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}" stepwise
        done
    done
}

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || fail "expected host ${EXPECTED_HOST}, got $(hostname)"
[[ "$(readlink -f "${REPO_ROOT}")" == "$(readlink -f "${EXPECTED_REPO}")" ]] \
    || fail "unexpected source tree: ${REPO_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "seer_libero" ]] || fail "activate conda environment seer_libero first"
[[ "${GPU_LIST}" == "4,5,6,7" ]] || fail "sd1 paper protocol requires CUDA_VISIBLE_DEVICES=4,5,6,7"
[[ "${EPISODES_PER_TASK}" == "50" && "${NUM_TASKS}" == "10" ]] \
    || fail "paper protocol is fixed to 50 episodes/task and 10 tasks"
[[ "${EVAL_SEEDS_STR}" == "42 43 44" ]] || fail "paper protocol is fixed to execution seeds 42 43 44"
is_bool "${REPORT_TO_WANDB}" || fail "REPORT_TO_WANDB must be 0 or 1"
is_bool "${RUN_DIRECT_MECHANISMS}" || fail "RUN_DIRECT_MECHANISMS must be 0 or 1"
is_bool "${RUN_K_CURVE}" || fail "RUN_K_CURVE must be 0 or 1"
is_bool "${PREFLIGHT_ONLY}" || fail "PREFLIGHT_ONLY must be 0 or 1"
for port in "${TRAIN_MASTER_PORT_BASE}" "${EVAL_MASTER_PORT_BASE}"; do
    [[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 1024 && port <= 64000 )) \
        || fail "invalid master port base: ${port}"
done

require_dir "${LIBERO_PATH}"
require_file "${SCRATCH_SCRIPT}"
require_file "${DISTILL_SCRIPT}"
require_file "${EVAL_SCRIPT}"
require_sha256 "public Seer 33" "${PUBLIC33_CKPT}" "${PUBLIC33_SHA256}"
require_sha256 "public33 adapter39" "${PUBLIC33_ADAPTER}" "${PUBLIC33_ADAPTER_SHA256}"
require_sha256 "MAE ViT" "${VIT_CHECKPOINT_PATH}" "${VIT_SHA256}"
for suite in libero_spatial libero_object libero_goal; do
    verify_converted_suite "${suite}"
done

mkdir -p "${PAPER_ROOT}/logs" "${PAPER_ROOT}/eval" "${PAPER_ROOT}/parity"
exec 9>"${PAPER_ROOT}/queue.lock"
flock -n 9 || fail "another queue process already owns ${PAPER_ROOT}/queue.lock"
rm -f "${PAPER_ROOT}/queue_failed.env" "${PAPER_ROOT}/queue_complete.env"
exec > >(tee -a "${PAPER_ROOT}/sequential_run.log") 2>&1
trap on_exit EXIT

cat > "${PAPER_ROOT}/campaign_contract.env" <<EOF
PROTOCOL=seer_public33_egl50_main_v1
HOST=${EXPECTED_HOST}
SOURCE_REPO=${REPO_ROOT}
GPU_LIST=${GPU_LIST}
RENDERER=egl
EPISODES_PER_TASK=${EPISODES_PER_TASK}
NUM_TASKS=${NUM_TASKS}
EVAL_SEEDS=${EVAL_SEEDS_STR}
PUBLIC33_CKPT=${PUBLIC33_CKPT}
PUBLIC33_SHA256=${PUBLIC33_SHA256}
PUBLIC33_ADAPTER=${PUBLIC33_ADAPTER}
PUBLIC33_ADAPTER_SHA256=${PUBLIC33_ADAPTER_SHA256}
SUITE_TRAIN_ROOT=${SUITE_TRAIN_ROOT}
RUN_DIRECT_MECHANISMS=${RUN_DIRECT_MECHANISMS}
RUN_K_CURVE=${RUN_K_CURVE}
EOF
git -C "${REPO_ROOT}" rev-parse HEAD > "${PAPER_ROOT}/git_commit.txt"
git -C "${REPO_ROOT}" status --short > "${PAPER_ROOT}/git_status_short.txt"
initialize_source_lock

python - <<'PY'
import torch

if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected four visible CUDA devices, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(4)]
if not all("RTX 3090" in name for name in names):
    raise RuntimeError(f"sd1 paper latency/eval GPUs must all be RTX 3090: {names}")
print(f"[VERIFY][OK] visible CUDA devices: {names}")
PY

echo "============================================================"
echo "[QUEUE] public33 EGL-50 paper campaign"
echo "[QUEUE] waits for Object teacher39, then runs sequentially on physical GPUs 4-7"
echo "[QUEUE] primary rows: Long K1/K4/K8 and Spatial/Object/Goal K1/K4, seeds 42/43/44"
echo "[QUEUE] direct mechanisms=${RUN_DIRECT_MECHANISMS}; dense/anchor K curve=${RUN_K_CURVE}"
echo "[QUEUE] result_root=${PAPER_ROOT}"
echo "============================================================"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    CURRENT_STAGE="preflight_only"
    verify_checkpoint_pair public33_teacher_adapter "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}"
    python "${UPSTREAM_DIR}/scripts/debug/check_lrnode_parity.py" \
        | tee "${PAPER_ROOT}/parity/synthetic_full_path_parity.log"
    echo "[PREFLIGHT][PASS] no training or evaluation was launched"
    exit 0
fi

wait_for_object_training
retire_legacy_wrapper
wait_for_selected_gpus

CURRENT_STAGE="checkpoint_parity_public33"
verify_checkpoint_pair public33_teacher_adapter "${PUBLIC33_CKPT}" "${PUBLIC33_ADAPTER}"
python "${UPSTREAM_DIR}/scripts/debug/check_lrnode_parity.py" \
    | tee "${PAPER_ROOT}/parity/synthetic_full_path_parity.log"

# Get the primary Long operating points first.
run_long_main_seed 42
write_campaign_table

# Finish suite-specific teachers/adapters without running the legacy OSMesa rows.
suite_index=0
for suite in libero_spatial libero_object libero_goal; do
    train_suite_baseline "${suite}" "${suite_index}"
    suite_index=$((suite_index + 1))
done
suite_index=0
for suite in libero_spatial libero_object libero_goal; do
    train_suite_adapter "${suite}" "${suite_index}"
    suite_index=$((suite_index + 1))
done

run_suite_main_seed 42
write_campaign_table

# Additional execution seeds reuse exactly the same checkpoints and init-state IDs 0-49.
for seed in 43 44; do
    run_long_main_seed "${seed}"
    run_suite_main_seed "${seed}"
    write_campaign_table
done

if [[ "${RUN_DIRECT_MECHANISMS}" == "1" ]]; then
    run_direct_mechanisms
    write_campaign_table
fi

if [[ "${RUN_K_CURVE}" == "1" ]]; then
    run_k_curve
    write_campaign_table
fi

CURRENT_STAGE="complete"
printf '%s\n' "$(date --iso-8601=seconds)" > "${PAPER_ROOT}/sequential_run_complete.txt"
echo "[DONE] EGL-50 queue complete: ${PAPER_ROOT}"
