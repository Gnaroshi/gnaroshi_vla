#!/usr/bin/env bash

set -Eeuo pipefail

# Evaluate the existing LIBERO-Object Seer teacher and LatentLoop adapter with
# the MuJoCo 3.3.2 rendering distribution. No training is performed here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
EVAL_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh"
RENDER_CHECKER="${REPO_ROOT}/tools/seer/verify_libero_object_render.py"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
EXPECTED_REPO="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
CONDA_SH="${CONDA_SH:-/home/mingyujung/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-seer_libero_mojoco332}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer}"
SHARED_GNAROSHI_ROOT="${SHARED_GNAROSHI_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla}"
PAPER_CHECKPOINT_ROOT="${PAPER_CHECKPOINT_ROOT:-${SHARED_GNAROSHI_ROOT}/artifacts/checkpoints/seer/paper}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-seer_object_mujoco332_egl50_v1}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${SHARED_GNAROSHI_ROOT}/results/seer/latentloop/reproductions/${CAMPAIGN_TAG}}"
LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
# Do not inherit a stale VIT_CHECKPOINT_PATH from a previous tmux experiment.
# This protocol can be overridden only through its namespaced variable.
VIT_CHECKPOINT_PATH="${OBJECT_VIT_CHECKPOINT_PATH:-${SHARED_SEER_ROOT}/vit_mae/mae_pretrain_vit_base.pth}"
BASELINE_CKPT="${BASELINE_CKPT:-${PAPER_CHECKPOINT_ROOT}/libero_object/teacher_scratch39.pth}"
ADAPTER_CKPT="${ADAPTER_CKPT:-${PAPER_CHECKPOINT_ROOT}/libero_object/latentloop_adapter39.pth}"

EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-42 43 44}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS_STR}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
NUM_TASKS="${NUM_TASKS:-10}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-17600}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

validate_row() {
    local root="$1" method="$2" seed="$3"
    python - "${root}" "${method}" "${seed}" "${EPISODES_PER_TASK}" "${NUM_TASKS}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
method = sys.argv[2]
seed = int(sys.argv[3])
expected = int(sys.argv[4]) * int(sys.argv[5])
paths = sorted(root.glob("*/analysis/eval_summary.json"))
if len(paths) != 1:
    raise RuntimeError(f"expected one eval summary under {root}, found {len(paths)}")
payload = json.loads(paths[0].read_text(encoding="utf-8"))
episodes = sum(int(item.get("num_episodes", 0)) for item in payload.get("task_results", []))
if payload.get("suite") != "libero_object" or episodes != expected:
    raise RuntimeError(
        f"invalid Object result: suite={payload.get('suite')}, episodes={episodes}, expected={expected}"
    )
renderer = payload.get("renderer_backend", {})
if renderer.get("effective_backend") != "egl" or renderer.get("backend_classification") != "hardware_egl":
    raise RuntimeError(f"invalid renderer metadata: {renderer}")
lrnode = payload.get("lrnode", {})
if method == "baseline":
    if bool(lrnode.get("enabled")) or bool(lrnode.get("eval_skip_full_forward")):
        raise RuntimeError(f"baseline unexpectedly enabled LatentLoop: {lrnode}")
else:
    if not bool(lrnode.get("enabled")) or not bool(lrnode.get("eval_skip_full_forward")):
        raise RuntimeError(f"LatentLoop row did not use skip path: {lrnode}")
    if int(lrnode.get("query_interval", -1)) != 4:
        raise RuntimeError(f"expected K=4, got {lrnode.get('query_interval')}")
args_paths = sorted(root.glob("*/analysis/args_snapshot.json"))
if len(args_paths) != 1:
    raise RuntimeError(f"expected one args snapshot under {root}, found {len(args_paths)}")
args = json.loads(args_paths[0].read_text(encoding="utf-8"))
if int(args.get("seed", -1)) != seed:
    raise RuntimeError(f"seed mismatch: expected={seed}, actual={args.get('seed')}")
print(
    f"[VERIFY][PASS] method={method} seed={seed} episodes={episodes} "
    f"SR={100.0 * float(payload.get('success_rate', 0.0)):.2f}%"
)
PY
}

write_runtime_contract() {
    local output="$1"
    REPO_ROOT="${REPO_ROOT}" \
    BASELINE_CKPT="${BASELINE_CKPT}" \
    ADAPTER_CKPT="${ADAPTER_CKPT}" \
    VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
    LIBERO_PATH="${LIBERO_PATH}" \
    GPU_LIST="${GPU_LIST}" \
    python - "${output}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


repo = pathlib.Path(os.environ["REPO_ROOT"])
payload = {
    "status": "LOCKED",
    "host": platform.node(),
    "python_executable": sys.executable,
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("mujoco", "robosuite", "libero", "torch", "numpy", "PyOpenGL")
    },
    "renderer": "egl",
    "cuda_visible_devices": os.environ["GPU_LIST"],
    "suite": "libero_object",
    "episodes_per_task": int(os.environ.get("EPISODES_PER_TASK", "50")),
    "num_tasks": int(os.environ.get("NUM_TASKS", "10")),
    "checkpoints": {
        "baseline": {
            "path": os.environ["BASELINE_CKPT"],
            "sha256": sha256(os.environ["BASELINE_CKPT"]),
        },
        "adapter": {
            "path": os.environ["ADAPTER_CKPT"],
            "sha256": sha256(os.environ["ADAPTER_CKPT"]),
        },
        "vit_mae": {
            "path": os.environ["VIT_CHECKPOINT_PATH"],
            "sha256": sha256(os.environ["VIT_CHECKPOINT_PATH"]),
        },
    },
    "libero_path": os.environ["LIBERO_PATH"],
    "git_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_status": subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines(),
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[RUNTIME CONTRACT] {path}")
PY
}

run_row() {
    local method="$1" seed="$2" port="$3"
    local root="${CAMPAIGN_ROOT}/eval/seed${seed}_${method}"
    local log="${CAMPAIGN_ROOT}/logs/seed${seed}_${method}.log"
    local run_baseline intervals
    if [[ "${method}" == "baseline_k1" ]]; then
        run_baseline=1
        intervals=""
    else
        run_baseline=0
        intervals="4"
    fi

    if validate_row "${root}" "${method%%_*}" "${seed}" >/dev/null 2>&1; then
        validate_row "${root}" "${method%%_*}" "${seed}"
        echo "[SKIP] verified complete row: seed${seed}_${method}"
        return
    fi
    if [[ -e "${root}" ]]; then
        local quarantine="${root}.incomplete.$(date +%Y%m%d_%H%M%S)"
        echo "[RESUME] preserving incomplete row at ${quarantine}"
        mv "${root}" "${quarantine}"
    fi
    if ! python - "${port}" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
    then
        fail "master port already in use: ${port}"
    fi

    echo "[EVAL START] suite=libero_object method=${method} seed=${seed} port=${port}"
    set +e
    env \
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        LIBERO_GL_BACKEND=egl \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        LIBERO_GL_REQUIRE_ACTUAL=1 \
        LIBERO_PATH="${LIBERO_PATH}" \
        VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
        LRNODE_PROTOCOL_ROOT="${CAMPAIGN_ROOT}" \
        SAVE_CHECKPOINT_PATH="${CAMPAIGN_ROOT}/eval_checkpoints" \
        RESULT_ROOT="${root}" \
        EVAL_SUITE=libero_object \
        EVAL_SEED="${seed}" \
        EVAL_NUM_EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
        EVAL_NUM_TASKS="${NUM_TASKS}" \
        EVAL_CONTROL_HZ=20 \
        LIBERO_EVAL_MAX_STEPS=600 \
        EVAL_LIBERO_ENSEMBLING=1 \
        BASELINE_CKPT="${BASELINE_CKPT}" \
        BASELINE_CKPT_ID=39 \
        BASELINE_NAME=seer_libero_object_teacher39_mujoco332 \
        OURS_CKPT="${ADAPTER_CKPT}" \
        OURS_CKPT_ID=39 \
        OURS_NAME=latentloop_libero_object_adapter39_mujoco332 \
        METHOD_TAG=latentloop_v0 \
        LRNODE_EVAL_BASE_CKPT="${BASELINE_CKPT}" \
        LRNODE_TRAIN_PROTOCOL=adapter \
        LRNODE_FREEZE_SEER_FOR_ADAPTER=1 \
        LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1 \
        LRNODE_GATE_INIT_BIAS=-4.0 \
        LRNODE_EVAL_STEP_LOG=1 \
        LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=1 \
        LRNODE_EVAL_SHADOW_FULL_FORWARD=0 \
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
        EXPERIMENT_NAME=seer_object_mujoco332 \
        EXPERIMENT_TAG="seed${seed}_${method}" \
        bash "${EVAL_SCRIPT}" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    set -e
    (( rc == 0 )) || fail "evaluation failed rc=${rc}: seed${seed}_${method}; log=${log}"
    validate_row "${root}" "${method%%_*}" "${seed}"
}

[[ "$(hostname -s)" == "${EXPECTED_HOST}" ]] \
    || fail "expected host ${EXPECTED_HOST}, got $(hostname -s)"
[[ "${REPO_ROOT}" == "${EXPECTED_REPO}" ]] \
    || fail "expected repo ${EXPECTED_REPO}, got ${REPO_ROOT}"
[[ -f "${CONDA_SH}" ]] || fail "missing Conda activation script: ${CONDA_SH}"
[[ "${PREFLIGHT_ONLY}" == "0" || "${PREFLIGHT_ONLY}" == "1" ]] \
    || fail "PREFLIGHT_ONLY must be 0 or 1"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export PYTHONPATH="${REPO_ROOT}:${UPSTREAM_DIR}:${LIBERO_PATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}"
export EPISODES_PER_TASK NUM_TASKS

require_file "${EVAL_SCRIPT}"
require_file "${RENDER_CHECKER}"
require_file "${BASELINE_CKPT}"
require_file "${ADAPTER_CKPT}"
require_file "${VIT_CHECKPOINT_PATH}"
[[ -d "${LIBERO_PATH}" ]] || fail "missing LIBERO repository: ${LIBERO_PATH}"
mkdir -p "${CAMPAIGN_ROOT}/logs" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

python - "${GPU_LIST}" <<'PY'
import importlib.metadata
import sys
import torch

devices = [int(item) for item in sys.argv[1].split(",")]
if sorted(devices) != [4, 5, 6, 7] or len(set(devices)) != 4:
    raise RuntimeError(f"sd1 Seer evaluation requires physical GPUs 4,5,6,7; got {devices}")
expected = {
    "mujoco": "3.3.2",
    "robosuite": "1.4.0",
    "libero": "0.1.0",
    "torch": "2.2.0+cu121",
    "numpy": "1.23.1",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"runtime mismatch: expected={expected}, actual={actual}")
if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected four visible GPUs, got {torch.cuda.device_count()}")
print(f"[PREFLIGHT][PASS] versions={actual} visible_gpus={devices}")
PY

RUNTIME_CONTRACT="${CAMPAIGN_ROOT}/runtime_contract.json"
if [[ ! -s "${RUNTIME_CONTRACT}" ]]; then
    write_runtime_contract "${RUNTIME_CONTRACT}"
fi

RENDER_AUDIT="${CAMPAIGN_ROOT}/render_audit"
if [[ ! -s "${RENDER_AUDIT}/render_metrics.json" ]]; then
    [[ ! -e "${RENDER_AUDIT}" ]] || fail "partial render audit exists: ${RENDER_AUDIT}"
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    LIBERO_GL_BACKEND=egl MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python "${RENDER_CHECKER}" \
        --libero-path "${LIBERO_PATH}" \
        --output-dir "${RENDER_AUDIT}" \
        --render-gpu-device-id "${GPU_LIST%%,*}" \
        --expected-mujoco 3.3.2 \
        --minimum-primary-mean 125
else
    python - "${RENDER_AUDIT}/render_metrics.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "PASS" or payload.get("versions", {}).get("mujoco") != "3.3.2":
    raise RuntimeError(f"invalid existing render audit: {payload}")
print(
    "[RENDER AUDIT][PASS] primary_mean="
    f"{payload['aggregate']['primary_mean_intensity']:.3f}"
)
PY
fi

echo "============================================================"
echo "[OBJECT EVAL] environment=${CONDA_ENV}, mujoco=3.3.2, renderer=egl"
echo "[OBJECT EVAL] teacher=${BASELINE_CKPT}"
echo "[OBJECT EVAL] adapter=${ADAPTER_CKPT}"
echo "[OBJECT EVAL] seeds=${EVAL_SEEDS[*]}, episodes=${EPISODES_PER_TASK}/task x ${NUM_TASKS} tasks"
echo "[OBJECT EVAL] rows per seed: baseline K1, LatentLoop K4"
echo "[OBJECT EVAL] result=${CAMPAIGN_ROOT}"
echo "============================================================"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[DONE] preflight only"
    exit 0
fi

row_index=0
for seed in "${EVAL_SEEDS[@]}"; do
    run_row baseline_k1 "${seed}" "$((MASTER_PORT_BASE + row_index))"
    row_index=$((row_index + 1))
    run_row latentloop_k4 "${seed}" "$((MASTER_PORT_BASE + row_index))"
    row_index=$((row_index + 1))
done

CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
EXPECTED_SEEDS="${#EVAL_SEEDS[@]}" \
python - <<'PY'
import csv
import json
import os
import pathlib
import statistics

root = pathlib.Path(os.environ["CAMPAIGN_ROOT"])
rows = []
for path in sorted(root.glob("eval/seed*_*/*/analysis/eval_summary.json")):
    row_id = path.parents[2].name
    seed = int(row_id.split("_", 1)[0].removeprefix("seed"))
    method = row_id.split("_", 1)[1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"seed": seed, "method": method, "success_rate": float(payload["success_rate"])})
expected_rows = 2 * int(os.environ["EXPECTED_SEEDS"])
if len(rows) != expected_rows:
    raise RuntimeError(f"expected {expected_rows} complete rows, found {len(rows)}")
csv_path = root / "object_mujoco332_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=("seed", "method", "success_rate"))
    writer.writeheader()
    writer.writerows(rows)
aggregate = {}
for method in ("baseline_k1", "latentloop_k4"):
    values = [row["success_rate"] for row in rows if row["method"] == method]
    aggregate[method] = {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
payload = {"rows": rows, "aggregate": aggregate}
(root / "object_mujoco332_summary.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY

date --iso-8601=seconds > "${CAMPAIGN_ROOT}/campaign_complete.txt"
echo "[DONE] ${CAMPAIGN_ROOT}/object_mujoco332_summary.json"
