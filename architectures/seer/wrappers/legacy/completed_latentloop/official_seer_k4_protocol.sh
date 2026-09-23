#!/usr/bin/env bash

set -euo pipefail

# Reproducible protocol for evaluating LR-NODE on released Seer LIBERO-LONG
# checkpoints. Each official checkpoint gets its own frozen-teacher adapter.
# Reusing the local ckpt33 adapter is intentionally unsupported because it was
# distilled in a different latent/action-head coordinate system.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
WRAPPER_DIR="$(dirname "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/../../../.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
OFFICIAL_UPSTREAM_ROOT="${OFFICIAL_UPSTREAM_ROOT:-${REPO_ROOT}/.cache/official_upstreams}"
OFFICIAL_SEER_REPO="${OFFICIAL_SEER_REPO:-${OFFICIAL_UPSTREAM_ROOT}/Seer}"
OFFICIAL_SIMVLA_REPO="${OFFICIAL_SIMVLA_REPO:-${OFFICIAL_UPSTREAM_ROOT}/SimVLA}"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-${REPO_ROOT}}"
OFFICIAL_CKPT_ROOT="${OFFICIAL_CKPT_ROOT:-${SHARED_SEER_ROOT}/checkpoints/Seer_LIBERO_LONG/Seer}"
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH:-${SHARED_SEER_ROOT}/checkpoints/vit_mae/mae_pretrain_vit_base.pth}"
VIT_CHECKPOINT_SHA256="${VIT_CHECKPOINT_SHA256:-aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d}"
LIBERO_TRAIN_ROOT="${LIBERO_TRAIN_ROOT:-${SHARED_SEER_ROOT}/LIBERO_DATASETS/libero_10_converted}"
LIBERO_TRAIN_DATASET="${LIBERO_TRAIN_ROOT}/libero_10_converted"
DATA_INFO_PATH="${DATA_INFO_PATH:-${UPSTREAM_DIR}/data_info/libero_10_converted.json}"
DATA_INFO_SHA256="${DATA_INFO_SHA256:-4b8241c1dd39b62c56aa6bbd7dca1afb397a4f9862e74f7718a3b00ca4679120}"

PROTOCOL_NAME="${PROTOCOL_NAME:-official_seer_libero_k4_v1}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${REPO_ROOT}/results/seer/lrnode/${PROTOCOL_NAME}}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${PROTOCOL_ROOT}/train/adapters}"
CHECKPOINT_IDS_STR="${CHECKPOINT_IDS_STR:-33 36 38}"
ADAPTER_CKPT_ID="${ADAPTER_CKPT_ID:-39}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
# train.py saves when epoch > start_save_checkpoint. Therefore 25 makes
# 26.pth the first checkpoint while retaining the final 39.pth adapter.
START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT:-25}"
STAGES="${STAGES:-audit}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NODE_NUM="${NODE_NUM:-4}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export VIDEO_FPS="${VIDEO_FPS:-20}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

read -r -a CHECKPOINT_IDS <<< "${CHECKPOINT_IDS_STR}"
read -r -a REQUESTED_STAGES <<< "${STAGES}"

has_stage() {
    local wanted="$1"
    local stage
    for stage in "${REQUESTED_STAGES[@]}"; do
        [[ "${stage}" == "${wanted}" || "${stage}" == "all" ]] && return 0
    done
    return 1
}

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

activate_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "seer_libero" ]]; then
        return
    fi
    command -v conda >/dev/null 2>&1 || fail "Activate conda env seer_libero before running."
    eval "$(conda shell.bash hook)"
    conda activate seer_libero
}

expected_sha() {
    case "$1" in
        33) echo "a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646" ;;
        36) echo "4fce785bc3b6bad9cc12061a97d251dfb405363ddb23978cc94e0654c0b9b1d2" ;;
        38) echo "6e6835c2c5f02a97d820766029ff191a901579714739fa788491358eebd29ee2" ;;
        *) fail "No pinned SHA256 for official checkpoint $1" ;;
    esac
}

expected_sr() {
    case "$1" in
        33) echo "0.885" ;;
        36) echo "0.870" ;;
        38) echo "0.875" ;;
        *) fail "No official SR guard for checkpoint $1" ;;
    esac
}

expected_tasks() {
    case "$1" in
        33) echo "19 18 19 20 19 19 16 17 15 15" ;;
        36) echo "18 18 20 20 18 19 17 18 10 16" ;;
        38) echo "18 18 20 20 18 18 18 18 12 15" ;;
        *) fail "No official task guard for checkpoint $1" ;;
    esac
}

adapter_run_name() {
    echo "official_seer_ckpt${1}_lrnode_adapter_v1"
}

official_ckpt() {
    echo "${OFFICIAL_CKPT_ROOT}/${1}.pth"
}

adapter_ckpt() {
    echo "${ADAPTER_ROOT}/$(adapter_run_name "$1")/${ADAPTER_CKPT_ID}.pth"
}

verify_sha256() {
    local path="$1"
    local expected="$2"
    [[ -f "${path}" ]] || fail "Missing file: ${path}"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || fail "SHA256 mismatch: ${path}; expected=${expected}, actual=${actual}"
}

verify_training_dataset() {
    LIBERO_TRAIN_DATASET="${LIBERO_TRAIN_DATASET}" DATA_INFO_PATH="${DATA_INFO_PATH}" python - <<'PY'
import json
import os
from pathlib import Path

import h5py

dataset = Path(os.environ["LIBERO_TRAIN_DATASET"])
data_info_path = Path(os.environ["DATA_INFO_PATH"])
rows = json.loads(data_info_path.read_text(encoding="utf-8"))
episodes = dataset / "episodes"
expected_ids = [row[0] for row in rows]
actual_ids = sorted(path.name for path in episodes.iterdir() if path.is_dir())
checks = {
    "dataset_dir": dataset.is_dir(),
    "dataset_meta": (dataset / "meta_info.h5").is_file(),
    "episode_count": len(rows) == len(actual_ids) == 500,
    "episode_ids": actual_ids == expected_ids,
    "total_steps": sum(int(row[1]) for row in rows) == 138090,
}
length_errors = []
for episode_id, expected_length in rows:
    meta_path = episodes / episode_id / "meta_info.h5"
    if not meta_path.is_file():
        length_errors.append(f"{episode_id}: missing meta_info.h5")
        continue
    with h5py.File(meta_path, "r") as handle:
        actual_length = int(handle["length"][()])
    if actual_length != int(expected_length):
        length_errors.append(
            f"{episode_id}: expected={expected_length}, actual={actual_length}"
        )
checks["episode_lengths"] = not length_errors
for episode_id in (rows[0][0], rows[-1][0]):
    step = episodes / episode_id / "steps" / "0000"
    checks[f"episode_{episode_id}_first_step"] = all(
        (step / name).is_file()
        for name in ("image_primary.jpg", "image_wrist.jpg", "other.h5")
    )
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(
        f"[VERIFY][FAIL] shared LIBERO training dataset: {failed}; "
        f"length_errors={length_errors[:10]}"
    )
print(
    f"[VERIFY][OK] shared LIBERO training dataset: {dataset}; "
    "500 episodes, 138090 steps"
)
PY
}

validate_ids() {
    [[ "${#CHECKPOINT_IDS[@]}" -gt 0 ]] || fail "CHECKPOINT_IDS_STR is empty."
    [[ "${NUM_EPOCHS}" =~ ^[0-9]+$ ]] || fail "NUM_EPOCHS must be a non-negative integer; got ${NUM_EPOCHS}."
    [[ "${START_SAVE_CHECKPOINT}" =~ ^[0-9]+$ ]] || fail "START_SAVE_CHECKPOINT must be a non-negative integer; got ${START_SAVE_CHECKPOINT}."
    [[ "${ADAPTER_CKPT_ID}" =~ ^[0-9]+$ ]] || fail "ADAPTER_CKPT_ID must be a non-negative integer; got ${ADAPTER_CKPT_ID}."
    (( ADAPTER_CKPT_ID < NUM_EPOCHS )) || fail "ADAPTER_CKPT_ID=${ADAPTER_CKPT_ID} is outside NUM_EPOCHS=${NUM_EPOCHS}."
    (( ADAPTER_CKPT_ID > START_SAVE_CHECKPOINT )) || fail "ADAPTER_CKPT_ID=${ADAPTER_CKPT_ID} will not be saved because train.py requires epoch > START_SAVE_CHECKPOINT=${START_SAVE_CHECKPOINT}."
    local id
    for id in "${CHECKPOINT_IDS[@]}"; do
        [[ "${id}" =~ ^(33|36|38)$ ]] || fail "Supported official checkpoint IDs are 33, 36, 38; got ${id}."
    done
}

run_audit() {
    echo "[STAGE] audit"
    [[ -d "${OFFICIAL_SEER_REPO}/.git" ]] || fail "Missing clean Seer clone: ${OFFICIAL_SEER_REPO}"
    [[ -d "${OFFICIAL_SIMVLA_REPO}/.git" ]] || fail "Missing clean SimVLA clone: ${OFFICIAL_SIMVLA_REPO}"
    [[ -f "${VIT_CHECKPOINT_PATH}" ]] || fail "Missing ViT checkpoint: ${VIT_CHECKPOINT_PATH}"
    [[ -d "${LIBERO_TRAIN_ROOT}" ]] || fail "Missing LIBERO training root: ${LIBERO_TRAIN_ROOT}"
    [[ -d "${LIBERO_TRAIN_DATASET}" ]] || fail "Missing LIBERO training dataset: ${LIBERO_TRAIN_DATASET}"
    [[ -f "${DATA_INFO_PATH}" ]] || fail "Missing Seer LIBERO-10 data_info: ${DATA_INFO_PATH}"

    [[ -z "$(git -C "${OFFICIAL_SEER_REPO}" status --porcelain=v1)" ]] || fail "Official Seer clone is dirty."
    [[ -z "$(git -C "${OFFICIAL_SIMVLA_REPO}" status --porcelain=v1)" ]] || fail "Official SimVLA clone is dirty."
    [[ "$(git -C "${OFFICIAL_SEER_REPO}" rev-parse HEAD)" == "ee228e110e098beefa23f94d558dfa8492259e6e" ]] || fail "Unexpected official Seer commit."
    [[ "$(git -C "${OFFICIAL_SIMVLA_REPO}" rev-parse HEAD)" == "32700d0ad8991996e123e4b685abe370ce6e9aab" ]] || fail "Unexpected official SimVLA commit."

    local id
    for id in "${CHECKPOINT_IDS[@]}"; do
        verify_sha256 "$(official_ckpt "${id}")" "$(expected_sha "${id}")"
    done
    verify_sha256 "${VIT_CHECKPOINT_PATH}" "${VIT_CHECKPOINT_SHA256}"
    verify_sha256 "${DATA_INFO_PATH}" "${DATA_INFO_SHA256}"
    verify_training_dataset

    VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" python - <<'PY'
import os

import torch

checkpoint = torch.load(os.environ["VIT_CHECKPOINT_PATH"], map_location="cpu")
if set(checkpoint) != {"model"}:
    raise RuntimeError(f"Unexpected MAE checkpoint keys: {sorted(checkpoint)}")
state = checkpoint["model"]
if len(state) != 150:
    raise RuntimeError(f"Unexpected MAE model tensor count: {len(state)}")
elements = sum(value.numel() for value in state.values() if torch.is_tensor(value))
if elements != 85_798_656:
    raise RuntimeError(f"Unexpected MAE model element count: {elements}")
print("[VERIFY][OK] official MAE ViT checkpoint: 150 tensors, 85,798,656 elements")
PY

    mkdir -p "${PROTOCOL_ROOT}/audit"
    local source_lock="${PROTOCOL_ROOT}/audit/source_sha256.lock"
    local source_current="${PROTOCOL_ROOT}/audit/source_sha256.current"
    {
        find "${UPSTREAM_DIR}" -type f \
            \( -name '*.py' -o -name '*.sh' \) \
            -not -path '*/__pycache__/*' \
            -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
        sha256sum "${SCRIPT_PATH}"
    } > "${source_current}"
    if [[ -f "${source_lock}" ]]; then
        if ! cmp -s "${source_lock}" "${source_current}"; then
            diff -u "${source_lock}" "${source_current}" || true
            fail "Core source changed after protocol lock: ${source_lock}"
        fi
        rm -f "${source_current}"
        echo "[VERIFY][OK] core source matches protocol lock"
    else
        mv "${source_current}" "${source_lock}"
        echo "[VERIFY][OK] created core source lock: ${source_lock}"
    fi

    OFFICIAL_CKPT_ROOT="${OFFICIAL_CKPT_ROOT}" \
    CHECKPOINT_IDS_STR="${CHECKPOINT_IDS_STR}" \
    AUDIT_OUTPUT="${PROTOCOL_ROOT}/audit/checkpoint_audit.json" \
    python - <<'PY'
import gc
import json
import os
from pathlib import Path

import torch

root = Path(os.environ["OFFICIAL_CKPT_ROOT"])
ids = [int(x) for x in os.environ["CHECKPOINT_IDS_STR"].split()]
expected_sha = {
    33: "a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646",
    36: "4fce785bc3b6bad9cc12061a97d251dfb405363ddb23978cc94e0654c0b9b1d2",
    38: "6e6835c2c5f02a97d820766029ff191a901579714739fa788491358eebd29ee2",
}
rows = []
reference_shapes = None
for checkpoint_id in ids:
    path = root / f"{checkpoint_id}.pth"
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    shapes = {key: tuple(value.shape) for key, value in state.items()}
    if reference_shapes is None:
        reference_shapes = shapes
    missing = sorted(set(reference_shapes) - set(shapes))
    unexpected = sorted(set(shapes) - set(reference_shapes))
    mismatch = sorted(key for key in set(reference_shapes) & set(shapes) if reference_shapes[key] != shapes[key])
    checks = {
        "epoch": checkpoint.get("epoch") == checkpoint_id,
        "top_level": set(checkpoint) == {"epoch", "model_state_dict", "optimizer_state_dict", "lr_scheduler_state_dict"},
        "tensor_count": len(state) == 400,
        "numel": sum(value.numel() for value in state.values()) == 67688462,
        "module_prefix": all(key.startswith("module.") for key in state),
        "no_lrnode": not any("lrnode" in key.lower() for key in state),
        "same_keys_and_shapes": not missing and not unexpected and not mismatch,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"[VERIFY][FAIL] official ckpt{checkpoint_id}: {failed}")
    rows.append({
        "checkpoint_id": checkpoint_id,
        "path": str(path),
        "sha256": expected_sha[checkpoint_id],
        "state_tensor_count": len(state),
        "state_numel": sum(value.numel() for value in state.values()),
        "lrnode_tensor_count": 0,
        "checks": checks,
    })
    del checkpoint, state
    gc.collect()

payload = {"status": "passed", "checkpoints": rows}
Path(os.environ["AUDIT_OUTPUT"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("[VERIFY][OK]", json.dumps(payload, indent=2))
PY

    cat > "${PROTOCOL_ROOT}/audit/source_manifest.env" <<EOF
OFFICIAL_SEER_REPO=${OFFICIAL_SEER_REPO}
OFFICIAL_SEER_COMMIT=$(git -C "${OFFICIAL_SEER_REPO}" rev-parse HEAD)
OFFICIAL_SIMVLA_REPO=${OFFICIAL_SIMVLA_REPO}
OFFICIAL_SIMVLA_COMMIT=$(git -C "${OFFICIAL_SIMVLA_REPO}" rev-parse HEAD)
GNAROSHI_REPO=${REPO_ROOT}
GNAROSHI_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)
OFFICIAL_CKPT_ROOT=${OFFICIAL_CKPT_ROOT}
CHECKPOINT_IDS=${CHECKPOINT_IDS_STR}
VIT_CHECKPOINT_PATH=${VIT_CHECKPOINT_PATH}
VIT_CHECKPOINT_SHA256=${VIT_CHECKPOINT_SHA256}
SHARED_SEER_ROOT=${SHARED_SEER_ROOT}
LIBERO_TRAIN_ROOT=${LIBERO_TRAIN_ROOT}
LIBERO_TRAIN_DATASET=${LIBERO_TRAIN_DATASET}
DATA_INFO_PATH=${DATA_INFO_PATH}
DATA_INFO_SHA256=${DATA_INFO_SHA256}
EOF
    git -C "${REPO_ROOT}" status --short > "${PROTOCOL_ROOT}/audit/gnaroshi_git_status.txt"
    echo "[VERIFY][OK] audit artifacts: ${PROTOCOL_ROOT}/audit"
}

record_official_baseline_reference() {
    local id="$1"
    local result_root="$2"
    local expected_task_vector
    expected_task_vector="$(expected_tasks "${id}")"
    CHECKPOINT_ID="${id}" EXPECTED_SR="$(expected_sr "${id}")" \
    EXPECTED_TASKS="${expected_task_vector}" RESULT_ROOT="${result_root}" \
    python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ["RESULT_ROOT"])
paths = list(root.glob("baseline_*/analysis/eval_summary.json"))
if len(paths) != 1:
    raise SystemExit(f"[VERIFY][FAIL] expected one baseline summary, found {len(paths)}")
summary_path = paths[0]
summary = json.loads(summary_path.read_text())
episode_path = summary_path.parent / "eval_episode_metrics.csv"
with episode_path.open(newline="") as handle:
    episodes = list(csv.DictReader(handle))
expected_tasks = [int(x) for x in os.environ["EXPECTED_TASKS"].split()]
actual_tasks = [sum(int(row["success"]) for row in episodes if int(row["task_id"]) == task) for task in range(10)]
integrity_checks = {
    "episodes_200": len(episodes) == 200,
    "lrnode_disabled": summary["lrnode"]["enabled"] is False,
    "full_calls_only": summary["query_reduction"]["num_lrnode_update_calls"] == 0,
}
reference_checks = {
    "success_rate": math.isclose(summary["success_rate"], float(os.environ["EXPECTED_SR"]), abs_tol=1e-12),
    "task_success_vector": actual_tasks == expected_tasks,
}
failed_integrity = [name for name, passed in integrity_checks.items() if not passed]
payload = {
    "checkpoint_id": int(os.environ["CHECKPOINT_ID"]),
    "status": "recorded" if not failed_integrity else "invalid",
    "reference_match": all(reference_checks.values()),
    "expected_success_rate": float(os.environ["EXPECTED_SR"]),
    "actual_success_rate": summary["success_rate"],
    "expected_task_successes": expected_tasks,
    "actual_task_successes": actual_tasks,
    "integrity_checks": integrity_checks,
    "reference_checks": reference_checks,
}
(root / "official_baseline_observation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
if failed_integrity:
    raise SystemExit(f"[VERIFY][FAIL] official ckpt{os.environ['CHECKPOINT_ID']} baseline integrity: {payload}")
if payload["reference_match"]:
    print("[REFERENCE][MATCH]", json.dumps(payload, indent=2))
else:
    print("[REFERENCE][DIFF][NON-BLOCKING]", json.dumps(payload, indent=2))
PY
}

run_baselines() {
    echo "[STAGE] baseline"
    local id result_root
    for id in "${CHECKPOINT_IDS[@]}"; do
        result_root="${PROTOCOL_ROOT}/eval/baseline_reference/ckpt_${id}"
        if [[ -f "${result_root}/official_baseline_observation.json" && "${REUSE_COMPLETED:-1}" == "1" ]]; then
            echo "[SKIP] completed official baseline ckpt${id}: ${result_root}"
            record_official_baseline_reference "${id}" "${result_root}"
            continue
        fi
        [[ ! -e "${result_root}" ]] || fail "Incomplete baseline result exists: ${result_root}. Move it or set a new PROTOCOL_ROOT."
        (
            unset OURS_CKPT OURS_RUN_NAME OURS_CKPT_ROOT OURS_ENV
            cd "${UPSTREAM_DIR}"
            BASELINE_CKPT="$(official_ckpt "${id}")" \
            BASELINE_CKPT_ID="${id}" \
            BASELINE_RUN_NAME="official_seer_libero_long" \
            BASELINE_NAME="official_seer_ckpt${id}" \
            BASELINE_CKPT_TAG="official_seer_ckpt${id}" \
            RUN_BASELINE=1 \
            RUN_OURS_FULL=0 \
            LRNODE_QUERY_INTERVALS_STR="" \
            EXPERIMENT_NAME="official_seer_baseline_reference" \
            EXPERIMENT_TAG="ckpt${id}" \
            RESULT_ROOT="${result_root}" \
            LRNODE_PROTOCOL_ROOT="${PROTOCOL_ROOT}" \
            VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
            MASTER_PORT="$((13100 + id))" \
            bash scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh
        )
        record_official_baseline_reference "${id}" "${result_root}"
    done
}

verify_adapter_checkpoint() {
    local id="$1"
    local path
    path="$(adapter_ckpt "${id}")"
    [[ -f "${path}" ]] || fail "Missing adapter checkpoint: ${path}"
    ADAPTER_PATH="${path}" BASE_PATH="$(official_ckpt "${id}")" CHECKPOINT_ID="${id}" \
    OUTPUT_PATH="$(dirname "${path}")/adapter_guard.json" python - <<'PY'
import json
import os
from pathlib import Path

import torch

adapter_path = Path(os.environ["ADAPTER_PATH"])
base_path = Path(os.environ["BASE_PATH"])
adapter = torch.load(adapter_path, map_location="cpu")["model_state_dict"]
base = torch.load(base_path, map_location="cpu")["model_state_dict"]
allowed = ("module.lrnode_delta_encoder.", "module.lrnode_dynamics.")
checks = {
    "adapter_tensor_count": len(adapter) == 30,
    "adapter_numel": sum(value.numel() for value in adapter.values()) == 470146,
    "adapter_only_lrnode": all(key.startswith(allowed) for key in adapter),
    "base_has_no_lrnode": not any("lrnode" in key.lower() for key in base),
    "no_key_overlap": not (set(adapter) & set(base)),
}
failed = [name for name, passed in checks.items() if not passed]
payload = {
    "checkpoint_id": int(os.environ["CHECKPOINT_ID"]),
    "base_path": str(base_path),
    "adapter_path": str(adapter_path),
    "status": "passed" if not failed else "failed",
    "checks": checks,
}
Path(os.environ["OUTPUT_PATH"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
if failed:
    raise SystemExit(f"[VERIFY][FAIL] adapter checkpoint: {payload}")
print("[VERIFY][OK]", json.dumps(payload, indent=2))
PY
}

run_training() {
    echo "[STAGE] train"
    local id run_name final_ckpt
    verify_training_dataset
    for id in "${CHECKPOINT_IDS[@]}"; do
        [[ -f "${PROTOCOL_ROOT}/eval/baseline_reference/ckpt_${id}/official_baseline_observation.json" ]] || \
            fail "Run STAGES=baseline before training ckpt${id}."
        run_name="$(adapter_run_name "${id}")"
        final_ckpt="$(adapter_ckpt "${id}")"
        if [[ -f "${final_ckpt}" && "${REUSE_COMPLETED:-1}" == "1" ]]; then
            echo "[SKIP] completed adapter ckpt${id}: ${final_ckpt}"
            verify_adapter_checkpoint "${id}"
            continue
        fi
        [[ ! -e "${ADAPTER_ROOT}/${run_name}" ]] || fail "Adapter directory already exists without final checkpoint: ${ADAPTER_ROOT}/${run_name}"
        mkdir -p "${PROTOCOL_ROOT}/train/logs"
        (
            cd "${UPSTREAM_DIR}"
            BASELINE_CKPT="$(official_ckpt "${id}")" \
            BASELINE_CKPT_ID="${id}" \
            BASELINE_RUN_NAME="official_seer_libero_long" \
            METHOD_TAG="official_seer_ckpt${id}_lrnode_adapter_v1" \
            RUN_NAME="${run_name}" \
            EXPERIMENT_TAG="official_ckpt${id}_v1" \
            LRNODE_PROTOCOL_ROOT="${PROTOCOL_ROOT}" \
            SAVE_CHECKPOINT_PATH="${ADAPTER_ROOT}" \
            ROOT_DIR="${LIBERO_TRAIN_ROOT}" \
            VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
            NUM_EPOCHS="${NUM_EPOCHS}" \
            START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT}" \
            MASTER_PORT="$((13200 + id))" \
            bash scripts/LIBERO_LONG/Seer/distill_node.sh 2>&1 | \
                tee "${PROTOCOL_ROOT}/train/logs/official_seer_ckpt${id}_adapter.log"
        )
        verify_adapter_checkpoint "${id}"
    done
}

verify_k4_result() {
    local id="$1"
    local baseline_root="${PROTOCOL_ROOT}/eval/baseline_reference/ckpt_${id}"
    local result_root="${PROTOCOL_ROOT}/eval/k4/ckpt_${id}"
    CHECKPOINT_ID="${id}" BASELINE_ROOT="${baseline_root}" RESULT_ROOT="${result_root}" \
    python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

checkpoint_id = int(os.environ["CHECKPOINT_ID"])
baseline_root = Path(os.environ["BASELINE_ROOT"])
result_root = Path(os.environ["RESULT_ROOT"])
baseline_paths = list(baseline_root.glob("baseline_*/analysis/eval_summary.json"))
full_paths = list(result_root.glob("ours_*_full_K1_*/analysis/eval_summary.json"))
k4_paths = list(result_root.glob("ours_*_skip_K4_*/analysis/eval_summary.json"))
if not (len(baseline_paths) == len(full_paths) == len(k4_paths) == 1):
    raise SystemExit(
        f"[VERIFY][FAIL] expected baseline/full/K4 summaries, got "
        f"{len(baseline_paths)}/{len(full_paths)}/{len(k4_paths)}"
    )

def load(path):
    summary = json.loads(path.read_text())
    with (path.parent / "eval_episode_metrics.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    keys = ["task_id", "episode_id", "seed", "success", "num_steps"]
    signatures = [tuple(row[key] for key in keys) for row in episodes]
    return summary, episodes, signatures

baseline, baseline_episodes, baseline_sig = load(baseline_paths[0])
full, full_episodes, full_sig = load(full_paths[0])
k4, k4_episodes, _ = load(k4_paths[0])
q = k4["query_reduction"]
checks = {
    "baseline_episodes_200": len(baseline_episodes) == 200,
    "full_episodes_200": len(full_episodes) == 200,
    "k4_episodes_200": len(k4_episodes) == 200,
    "full_k1_matches_baseline": full_sig == baseline_sig,
    "full_k1_sr_matches": math.isclose(full["success_rate"], baseline["success_rate"], abs_tol=1e-12),
    "k4_lrnode_enabled": k4["lrnode"]["enabled"] is True,
    "k4_skip_enabled": k4["lrnode"]["eval_skip_full_forward"] is True,
    "k4_interval": int(k4["lrnode"]["query_interval"]) == 4,
    "calls_partition_steps": q["num_full_forward_calls"] + q["num_lrnode_update_calls"] == q["num_env_steps"],
    "fast_equals_updates": q["num_fast_encoder_calls"] == q["num_lrnode_update_calls"],
    "head_equals_updates": q["num_action_head_calls"] == q["num_lrnode_update_calls"],
    "no_fallback": q["num_fallback_full_calls"] == 0,
    "no_ablation": all(q[name] == 0 for name in (
        "num_hold_action_steps", "num_hold_latent_steps", "num_chunk_token_steps", "num_no_delta_steps"
    )),
}
failed = [name for name, passed in checks.items() if not passed]
payload = {
    "checkpoint_id": checkpoint_id,
    "status": "passed" if not failed else "failed",
    "baseline_sr": baseline["success_rate"],
    "ours_full_k1_sr": full["success_rate"],
    "ours_k4_sr": k4["success_rate"],
    "k4_delta_pp": 100.0 * (k4["success_rate"] - baseline["success_rate"]),
    "k4_full_query_reduction": q["full_query_reduction_ratio"],
    "checks": checks,
}
(result_root / "k4_protocol_guard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
if failed:
    raise SystemExit(f"[VERIFY][FAIL] ckpt{checkpoint_id} K4 protocol: {payload}")
print("[VERIFY][OK]", json.dumps(payload, indent=2))
PY
}

run_evaluation() {
    echo "[STAGE] eval"
    local id result_root run_name
    for id in "${CHECKPOINT_IDS[@]}"; do
        [[ -f "${PROTOCOL_ROOT}/eval/baseline_reference/ckpt_${id}/official_baseline_observation.json" ]] || \
            fail "Missing baseline reference observation for ckpt${id}."
        verify_adapter_checkpoint "${id}"
        result_root="${PROTOCOL_ROOT}/eval/k4/ckpt_${id}"
        if [[ -f "${result_root}/k4_protocol_guard.json" && "${REUSE_COMPLETED:-1}" == "1" ]]; then
            echo "[SKIP] completed K4 evaluation ckpt${id}: ${result_root}"
            verify_k4_result "${id}"
            continue
        fi
        [[ ! -e "${result_root}" ]] || fail "Incomplete K4 result exists: ${result_root}. Move it or set a new PROTOCOL_ROOT."
        run_name="$(adapter_run_name "${id}")"
        (
            cd "${UPSTREAM_DIR}"
            BASELINE_CKPT="$(official_ckpt "${id}")" \
            BASELINE_CKPT_ID="${id}" \
            BASELINE_RUN_NAME="official_seer_libero_long" \
            BASELINE_NAME="official_seer_ckpt${id}" \
            BASELINE_CKPT_TAG="official_seer_ckpt${id}" \
            OURS_CKPT="$(adapter_ckpt "${id}")" \
            OURS_CKPT_ID="${ADAPTER_CKPT_ID}" \
            OURS_RUN_NAME="${run_name}" \
            OURS_NAME="official_seer_ckpt${id}_lrnode" \
            OURS_CKPT_TAG="official_seer_ckpt${id}_lrnode_adapter_ckpt${ADAPTER_CKPT_ID}" \
            METHOD_TAG="official_seer_ckpt${id}_lrnode_adapter_v1" \
            LRNODE_EVAL_BASE_CKPT="$(official_ckpt "${id}")" \
            LRNODE_TRAIN_PROTOCOL="adapter" \
            LRNODE_FREEZE_SEER_FOR_ADAPTER=1 \
            LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1 \
            RUN_BASELINE=0 \
            RUN_OURS_FULL=1 \
            LRNODE_QUERY_INTERVALS_STR="4" \
            EXPERIMENT_NAME="official_seer_lrnode_k4" \
            EXPERIMENT_TAG="ckpt${id}" \
            RESULT_ROOT="${result_root}" \
            LRNODE_PROTOCOL_ROOT="${PROTOCOL_ROOT}" \
            VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
            MASTER_PORT="$((13300 + id))" \
            bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
        )
        verify_k4_result "${id}"
    done

    PROTOCOL_ROOT="${PROTOCOL_ROOT}" CHECKPOINT_IDS_STR="${CHECKPOINT_IDS_STR}" \
    python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["PROTOCOL_ROOT"])
rows = []
for checkpoint_id in [int(x) for x in os.environ["CHECKPOINT_IDS_STR"].split()]:
    guard = json.loads((root / "eval" / "k4" / f"ckpt_{checkpoint_id}" / "k4_protocol_guard.json").read_text())
    k4_path = next((root / "eval" / "k4" / f"ckpt_{checkpoint_id}").glob("ours_*_skip_K4_*/analysis/eval_summary.json"))
    k4 = json.loads(k4_path.read_text())
    rows.append({
        "official_checkpoint": checkpoint_id,
        "baseline_sr_pct": 100.0 * guard["baseline_sr"],
        "ours_full_k1_sr_pct": 100.0 * guard["ours_full_k1_sr"],
        "ours_k4_sr_pct": 100.0 * guard["ours_k4_sr"],
        "k4_delta_pp": guard["k4_delta_pp"],
        "k4_query_reduction_pct": 100.0 * guard["k4_full_query_reduction"],
        "k4_avg_full_forward_ms": k4["avg_full_forward_latency_ms"],
        "k4_avg_lrnode_ms": k4["avg_lrnode_latency_ms"],
        "k4_avg_policy_step_ms": k4["avg_policy_step_latency_ms"],
    })
output = root / "official_seer_k4_summary.csv"
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"[SUMMARY] {output}")
for row in rows:
    print(row)
PY
}

validate_ids
activate_env
mkdir -p "${PROTOCOL_ROOT}"

cat > "${PROTOCOL_ROOT}/protocol_config.env" <<EOF
SCRIPT=architectures/seer/wrappers/lrnode/official_seer_k4_protocol.sh
REPO_ROOT=${REPO_ROOT}
OFFICIAL_UPSTREAM_ROOT=${OFFICIAL_UPSTREAM_ROOT}
OFFICIAL_CKPT_ROOT=${OFFICIAL_CKPT_ROOT}
VIT_CHECKPOINT_PATH=${VIT_CHECKPOINT_PATH}
VIT_CHECKPOINT_SHA256=${VIT_CHECKPOINT_SHA256}
SHARED_SEER_ROOT=${SHARED_SEER_ROOT}
LIBERO_TRAIN_ROOT=${LIBERO_TRAIN_ROOT}
LIBERO_TRAIN_DATASET=${LIBERO_TRAIN_DATASET}
DATA_INFO_PATH=${DATA_INFO_PATH}
DATA_INFO_SHA256=${DATA_INFO_SHA256}
CHECKPOINT_IDS=${CHECKPOINT_IDS_STR}
ADAPTER_CKPT_ID=${ADAPTER_CKPT_ID}
NUM_EPOCHS=${NUM_EPOCHS}
START_SAVE_CHECKPOINT=${START_SAVE_CHECKPOINT}
FIRST_SAVED_CHECKPOINT=$((START_SAVE_CHECKPOINT + 1))
PROTOCOL_ROOT=${PROTOCOL_ROOT}
ADAPTER_ROOT=${ADAPTER_ROOT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
NODE_NUM=${NODE_NUM}
STAGES=${STAGES}
EOF

# Audit is cheap and is always rerun before any long stage.
run_audit
has_stage baseline && run_baselines
has_stage train && run_training
has_stage eval && run_evaluation

echo "[DONE] stages=${STAGES} protocol_root=${PROTOCOL_ROOT}"
