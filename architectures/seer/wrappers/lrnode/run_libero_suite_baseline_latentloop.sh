#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
CONVERTER="${REPO_ROOT}/tools/seer/convert_libero_suite.py"
SUMMARIZER="${REPO_ROOT}/tools/seer/summarize_libero_suite_campaign.py"
SCRATCH_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/scratch.sh"
DISTILL_SCRIPT="${SCRIPT_DIR}/distill_node.sh"
EVAL_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
EXPECTED_REPO="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
SHARED_ROOT="${SHARED_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/libero_suite_study}"
SHARED_RESULT_ROOT="${SHARED_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/seer/latentloop/suite_training}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-spatial_object_goal_v1}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${SHARED_RESULT_ROOT}/${CAMPAIGN_TAG}}"
RAW_DATASET_ROOT="${RAW_DATASET_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/datasets/LIBERO/datasets}"
CONVERTED_ROOT="${CONVERTED_ROOT:-${SHARED_ROOT}/datasets}"
LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
SUITES_STR="${SUITES:-libero_spatial libero_object libero_goal}"
read -r -a SUITES_ARRAY <<< "${SUITES_STR}"

TRAIN_SEED="${TRAIN_SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-40}"
ADAPTER_EPOCHS="${ADAPTER_EPOCHS:-40}"
BASELINE_CKPT_ID="${BASELINE_CKPT_ID:-39}"
ADAPTER_CKPT_ID="${ADAPTER_CKPT_ID:-39}"
START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT:-29}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
NUM_TASKS="${NUM_TASKS:-10}"
CONVERT_WORKERS="${CONVERT_WORKERS:-8}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-16600}"
RUN_ADAPTER_K1_PARITY="${RUN_ADAPTER_K1_PARITY:-1}"
REPORT_TO_WANDB="${REPORT_TO_WANDB:-1}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "missing file: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "missing directory: $1"
}

run_logged() {
    local label="$1"
    local logfile="$2"
    shift 2
    mkdir -p "$(dirname "${logfile}")"
    echo "[STAGE START] ${label} $(date --iso-8601=seconds)"
    local started
    started=$(date +%s)
    set +e
    "$@" 2>&1 | tee "${logfile}"
    local rc=${PIPESTATUS[0]}
    set -e
    local elapsed=$(( $(date +%s) - started ))
    if (( rc != 0 )); then
        fail "stage failed rc=${rc}: ${label}; log=${logfile}"
    fi
    echo "[STAGE DONE] ${label} elapsed=$((elapsed / 3600))h$(((elapsed % 3600) / 60))m"
}

find_eval_summary() {
    local root="$1"
    find "${root}" -path '*/analysis/eval_summary.json' -type f -print 2>/dev/null
}

require_complete_eval() {
    local root="$1"
    local expected=$((EPISODES_PER_TASK * NUM_TASKS))
    python - "${root}" "${expected}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
paths = sorted(root.glob("*/analysis/eval_summary.json"))
if len(paths) != 1:
    raise RuntimeError(f"expected one eval summary under {root}, found {len(paths)}")
payload = json.loads(paths[0].read_text())
episodes = sum(int(item.get("num_episodes", 0)) for item in payload.get("task_results", []))
if episodes != expected:
    raise RuntimeError(f"expected {expected} episodes, found {episodes}: {paths[0]}")
print(f"[EVAL VERIFY] suite={payload.get('suite')} episodes={episodes} SR={100.0 * float(payload.get('success_rate', 0.0)):.2f}%")
PY
}

require_complete_conversion() {
    local suite="$1"
    local target="${CONVERTED_ROOT}/${suite}_converted"
    python - "${target}" "${suite}" <<'PY'
import json
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
suite = sys.argv[2]
manifest_path = target / "conversion_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "status": "complete",
    "suite": suite,
    "num_tasks": 10,
    "num_episodes": 500,
}
actual = {key: manifest.get(key) for key in expected}
if actual != expected:
    raise RuntimeError(
        f"invalid converted dataset manifest: expected={expected}, actual={actual}"
    )
for required in ("meta_info.h5", "data_info.json"):
    if not (target / required).is_file():
        raise FileNotFoundError(target / required)
print(
    f"[CONVERT VERIFY] suite={suite} tasks={manifest['num_tasks']} "
    f"episodes={manifest['num_episodes']} steps={manifest['num_steps']}"
)
PY
}

require_k1_parity() {
    local suite="$1"
    local baseline_root="${CAMPAIGN_ROOT}/eval/${suite}/baseline"
    local adapter_root="${CAMPAIGN_ROOT}/eval/${suite}/adapter_k1"
    python - "${baseline_root}" "${adapter_root}" "${suite}" <<'PY'
import json
import pathlib
import sys


def load_one(root: pathlib.Path) -> dict:
    paths = sorted(root.glob("*/analysis/eval_summary.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one eval summary under {root}, found {len(paths)}")
    return json.loads(paths[0].read_text(encoding="utf-8"))


def task_success_counts(payload: dict) -> list[tuple[int, int, int]]:
    rows = []
    for item in payload.get("task_results", []):
        episodes = int(item.get("num_episodes", 0))
        successes = int(round(float(item.get("success_rate", 0.0)) * episodes))
        rows.append((int(item["task_id"]), successes, episodes))
    return rows


baseline = load_one(pathlib.Path(sys.argv[1]))
adapter = load_one(pathlib.Path(sys.argv[2]))
suite = sys.argv[3]
baseline_counts = task_success_counts(baseline)
adapter_counts = task_success_counts(adapter)
if baseline.get("suite") != suite or adapter.get("suite") != suite:
    raise RuntimeError(
        f"suite mismatch: expected={suite}, baseline={baseline.get('suite')}, "
        f"adapter={adapter.get('suite')}"
    )
if baseline_counts != adapter_counts:
    raise RuntimeError(
        "K=1 parity failed; refusing to run K=4. "
        f"baseline={baseline_counts}, adapter_loaded={adapter_counts}"
    )
print(
    f"[K1 PARITY][PASS] suite={suite} "
    f"SR={100.0 * float(baseline.get('success_rate', 0.0)):.2f}%"
)
PY
}

run_eval_row() {
    local suite="$1"
    local method="$2"
    local result_root="$3"
    local baseline_ckpt="$4"
    local adapter_ckpt="$5"
    local port="$6"
    local run_baseline="$7"
    local run_ours_full="$8"
    local intervals="$9"

    if [[ -n "$(find_eval_summary "${result_root}")" ]]; then
        echo "[SKIP] complete eval result exists: ${suite}/${method}"
        require_complete_eval "${result_root}"
        return
    fi
    [[ ! -e "${result_root}" ]] || fail "partial eval root exists: ${result_root}"

    run_logged "eval ${suite} ${method}" "${CAMPAIGN_ROOT}/logs/${suite}_${method}.log" \
        env \
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
            LIBERO_GL_BACKEND=osmesa \
            LIBERO_GL_REQUIRE_ACTUAL=1 \
            LIBERO_PATH="${LIBERO_PATH}" \
            VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
            LRNODE_PROTOCOL_ROOT="${CAMPAIGN_ROOT}" \
            SAVE_CHECKPOINT_PATH="${CAMPAIGN_ROOT}/eval_checkpoints" \
            EVAL_SUITE="${suite}" \
            EVAL_SEED="${EVAL_SEED}" \
            EVAL_NUM_EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
            EVAL_NUM_TASKS="${NUM_TASKS}" \
            EVAL_CONTROL_HZ=20 \
            LIBERO_EVAL_MAX_STEPS=600 \
            EVAL_LIBERO_ENSEMBLING=1 \
            BASELINE_CKPT="${baseline_ckpt}" \
            BASELINE_CKPT_ID="${BASELINE_CKPT_ID}" \
            BASELINE_NAME="seer_${suite}_scratch" \
            OURS_CKPT="${adapter_ckpt}" \
            OURS_CKPT_ID="${ADAPTER_CKPT_ID}" \
            OURS_NAME="latentloop_${suite}_adapter" \
            METHOD_TAG=latentloop_v0 \
            LRNODE_EVAL_BASE_CKPT="${baseline_ckpt}" \
            LRNODE_TRAIN_PROTOCOL=adapter \
            LRNODE_FREEZE_SEER_FOR_ADAPTER=1 \
            LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1 \
            LRNODE_EVAL_STEP_LOG=1 \
            LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=1 \
            LRNODE_GATE_INIT_BIAS=-4.0 \
            RUN_BASELINE="${run_baseline}" \
            RUN_OURS_FULL="${run_ours_full}" \
            LRNODE_QUERY_INTERVALS_STR="${intervals}" \
            NODE_NUM=4 \
            MASTER_PORT="${port}" \
            SAVE_VIDEO=0 \
            SAVE_VIDEO_SUCC=0 \
            SAVE_VIDEO_FAIL=0 \
            SAVE_VIDEO_ALL_RANKS=0 \
            RESULT_ROOT="${result_root}" \
            bash "${EVAL_SCRIPT}"
    require_complete_eval "${result_root}"
    python "${SUMMARIZER}" --campaign-root "${CAMPAIGN_ROOT}" \
        --suites "${SUITES_ARRAY[@]}" --allow-incomplete
}

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || fail "expected host ${EXPECTED_HOST}, got $(hostname)"
[[ "$(readlink -f "${REPO_ROOT}")" == "$(readlink -f "${EXPECTED_REPO}")" ]] \
    || fail "unexpected source tree: ${REPO_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "seer_libero" ]] \
    || fail "activate conda environment seer_libero first"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "4,5,6,7" ]] \
    || fail "set CUDA_VISIBLE_DEVICES=4,5,6,7"
[[ "${#SUITES_ARRAY[@]}" -eq 3 ]] || fail "expected exactly three suites"
for value in "${BASELINE_EPOCHS}" "${ADAPTER_EPOCHS}" "${EPISODES_PER_TASK}" \
    "${NUM_TASKS}" "${CONVERT_WORKERS}" "${MASTER_PORT_BASE}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "positive integer required, got ${value}"
done
for value in "${BASELINE_CKPT_ID}" "${ADAPTER_CKPT_ID}" "${START_SAVE_CHECKPOINT}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "non-negative integer required, got ${value}"
done
(( BASELINE_CKPT_ID < BASELINE_EPOCHS )) \
    || fail "BASELINE_CKPT_ID must be lower than BASELINE_EPOCHS"
(( ADAPTER_CKPT_ID < ADAPTER_EPOCHS )) \
    || fail "ADAPTER_CKPT_ID must be lower than ADAPTER_EPOCHS"
(( START_SAVE_CHECKPOINT < BASELINE_CKPT_ID )) \
    || fail "START_SAVE_CHECKPOINT must be lower than BASELINE_CKPT_ID"
(( START_SAVE_CHECKPOINT < ADAPTER_CKPT_ID )) \
    || fail "START_SAVE_CHECKPOINT must be lower than ADAPTER_CKPT_ID"
(( EPISODES_PER_TASK <= 50 )) \
    || fail "LIBERO provides 50 initial states/task; EPISODES_PER_TASK must be <= 50"
(( NUM_TASKS == 10 )) || fail "this campaign requires all 10 tasks in each suite"
[[ "${RUN_ADAPTER_K1_PARITY}" == "0" || "${RUN_ADAPTER_K1_PARITY}" == "1" ]] \
    || fail "RUN_ADAPTER_K1_PARITY must be 0 or 1"
[[ "${REPORT_TO_WANDB}" == "0" || "${REPORT_TO_WANDB}" == "1" ]] \
    || fail "REPORT_TO_WANDB must be 0 or 1"

require_dir "${LIBERO_PATH}"
require_dir "${RAW_DATASET_ROOT}"
require_file "${VIT_CHECKPOINT_PATH}"
require_file "${CONVERTER}"
require_file "${SUMMARIZER}"
require_file "${SCRATCH_SCRIPT}"
require_file "${DISTILL_SCRIPT}"
require_file "${EVAL_SCRIPT}"

export PYTHONPATH="${REPO_ROOT}:${UPSTREAM_DIR}:${LIBERO_PATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}"
mkdir -p "${CAMPAIGN_ROOT}/logs" "${CONVERTED_ROOT}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

python - <<'PY'
import torch
if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected 4 visible CUDA devices, got {torch.cuda.device_count()}")
print("[PREFLIGHT] visible CUDA devices=4")
PY

echo "============================================================"
echo "[CAMPAIGN] suite-level training: one model per 10-task suite"
echo "[CAMPAIGN] suites=${SUITES_ARRAY[*]}"
echo "[CAMPAIGN] baseline=${BASELINE_EPOCHS} epochs; checkpoint=${BASELINE_CKPT_ID}"
echo "[CAMPAIGN] LatentLoop adapter=${ADAPTER_EPOCHS} epochs; checkpoint=${ADAPTER_CKPT_ID}"
echo "[CAMPAIGN] evaluation=${EPISODES_PER_TASK} episodes/task x ${NUM_TASKS} tasks"
echo "[CAMPAIGN] rows=baseline K=1, adapter-loaded K=1 parity, LatentLoop K=4"
echo "[CAMPAIGN] horizon=600, temporal ensembling=on, renderer=osmesa, seed=${EVAL_SEED}"
echo "[CAMPAIGN] estimated wall time on four sd1 GPUs: about 75-90 hours"
echo "[CAMPAIGN] training tqdm shows loss/progress/ETA; eval prints live SR/remaining/ETA"
echo "[CAMPAIGN] root=${CAMPAIGN_ROOT}"
echo "============================================================"

for suite in "${SUITES_ARRAY[@]}"; do
    case "${suite}" in
        libero_spatial|libero_object|libero_goal) ;;
        *) fail "unsupported campaign suite: ${suite}" ;;
    esac
    run_logged "convert ${suite}" "${CAMPAIGN_ROOT}/logs/${suite}_convert.log" \
        python "${CONVERTER}" \
            --suite "${suite}" \
            --source-root "${RAW_DATASET_ROOT}" \
            --output-root "${CONVERTED_ROOT}" \
            --workers "${CONVERT_WORKERS}"
    require_complete_conversion "${suite}"
done

suite_index=0
for suite in "${SUITES_ARRAY[@]}"; do
    dataset="${suite}_converted"
    root_dir="${CONVERTED_ROOT}"
    dataset_info="${CONVERTED_ROOT}/${dataset}/data_info.json"
    baseline_save_root="${CAMPAIGN_ROOT}/train/${suite}/baseline"
    baseline_run="seer_${suite}_scratch_seed${TRAIN_SEED}"
    baseline_run_dir="${baseline_save_root}/${baseline_run}"
    baseline_ckpt="${baseline_run_dir}/${BASELINE_CKPT_ID}.pth"
    port=$((MASTER_PORT_BASE + suite_index * 20))

    if [[ -f "${baseline_ckpt}" ]]; then
        echo "[SKIP] baseline checkpoint exists: ${baseline_ckpt}"
    else
        [[ ! -e "${baseline_run_dir}" ]] \
            || fail "partial baseline training directory exists: ${baseline_run_dir}"
        run_logged "train baseline ${suite}" "${CAMPAIGN_ROOT}/logs/${suite}_baseline_train.log" \
            env \
                CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
                LRNODE_PROTOCOL_ROOT="${CAMPAIGN_ROOT}" \
                DATASET="${dataset}" \
                ROOT_DIR="${root_dir}" \
                LIBERO_DATASET_INFO_PATH="${dataset_info}" \
                VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
                LIBERO_PATH="${LIBERO_PATH}" \
                SAVE_CHECKPOINT_PATH="${baseline_save_root}" \
                RUN_NAME="${baseline_run}" \
                METHOD_TAG="seer_${suite}_scratch" \
                EXPERIMENT_TAG="${CAMPAIGN_TAG}_${suite}" \
                NUM_EPOCHS="${BASELINE_EPOCHS}" \
                START_SAVE_CHECKPOINT="${START_SAVE_CHECKPOINT}" \
                SEED="${TRAIN_SEED}" \
                LEARNING_RATE=1e-3 \
                REPORT_TO_WANDB="${REPORT_TO_WANDB}" \
                WANDB_PROJECT=seer_libero_suite \
                NODE_NUM=4 \
                MASTER_PORT="${port}" \
                bash "${SCRATCH_SCRIPT}"
        require_file "${baseline_ckpt}"
    fi

    run_eval_row "${suite}" baseline_k1 \
        "${CAMPAIGN_ROOT}/eval/${suite}/baseline" \
        "${baseline_ckpt}" "${baseline_ckpt}" \
        "$((port + 1))" 1 0 ""
    suite_index=$((suite_index + 1))
done

suite_index=0
for suite in "${SUITES_ARRAY[@]}"; do
    dataset="${suite}_converted"
    root_dir="${CONVERTED_ROOT}"
    dataset_info="${CONVERTED_ROOT}/${dataset}/data_info.json"
    baseline_save_root="${CAMPAIGN_ROOT}/train/${suite}/baseline"
    baseline_run="seer_${suite}_scratch_seed${TRAIN_SEED}"
    baseline_ckpt="${baseline_save_root}/${baseline_run}/${BASELINE_CKPT_ID}.pth"
    adapter_save_root="${CAMPAIGN_ROOT}/train/${suite}/adapter"
    adapter_run="latentloop_${suite}_teacher${BASELINE_CKPT_ID}_seed${TRAIN_SEED}"
    adapter_run_dir="${adapter_save_root}/${adapter_run}"
    adapter_ckpt="${adapter_run_dir}/${ADAPTER_CKPT_ID}.pth"
    port=$((MASTER_PORT_BASE + suite_index * 20 + 10))

    if [[ -f "${adapter_ckpt}" ]]; then
        echo "[SKIP] adapter checkpoint exists: ${adapter_ckpt}"
    else
        [[ ! -e "${adapter_run_dir}" ]] \
            || fail "partial adapter training directory exists: ${adapter_run_dir}"
        run_logged "train LatentLoop adapter ${suite}" "${CAMPAIGN_ROOT}/logs/${suite}_adapter_train.log" \
            env \
                CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
                LRNODE_PROTOCOL_ROOT="${CAMPAIGN_ROOT}" \
                DATASET="${dataset}" \
                ROOT_DIR="${root_dir}" \
                LIBERO_DATASET_INFO_PATH="${dataset_info}" \
                VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
                LIBERO_PATH="${LIBERO_PATH}" \
                BASELINE_CKPT="${baseline_ckpt}" \
                BASELINE_CKPT_ID="${BASELINE_CKPT_ID}" \
                SAVE_CHECKPOINT_PATH="${adapter_save_root}" \
                RUN_NAME="${adapter_run}" \
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
                MASTER_PORT="${port}" \
                bash "${DISTILL_SCRIPT}"
        require_file "${adapter_ckpt}"
    fi

    if [[ "${RUN_ADAPTER_K1_PARITY}" == "1" ]]; then
        run_eval_row "${suite}" adapter_loaded_k1 \
            "${CAMPAIGN_ROOT}/eval/${suite}/adapter_k1" \
            "${baseline_ckpt}" "${adapter_ckpt}" "$((port + 1))" 0 1 ""
        require_k1_parity "${suite}"
    fi
    run_eval_row "${suite}" latentloop_k4 \
        "${CAMPAIGN_ROOT}/eval/${suite}/latentloop_k4" \
        "${baseline_ckpt}" "${adapter_ckpt}" "$((port + 2))" 0 0 "4"
    suite_index=$((suite_index + 1))
done

python "${SUMMARIZER}" --campaign-root "${CAMPAIGN_ROOT}" \
    --suites "${SUITES_ARRAY[@]}"
printf '%s\n' "$(date --iso-8601=seconds)" > "${CAMPAIGN_ROOT}/campaign_complete.txt"
echo "[DONE] complete cross-suite campaign: ${CAMPAIGN_ROOT}"
