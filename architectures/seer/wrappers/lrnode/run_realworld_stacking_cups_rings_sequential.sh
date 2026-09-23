#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PREPARE_SCRIPT="${REPO_ROOT}/tools/seer/prepare_sd1_realworld_stacking_rings_assets.sh"
VERIFY_SCRIPT="${REPO_ROOT}/tools/seer/verify_realworld_latentloop_assets.py"

STORAGE_ROOT="${SEER_REALWORLD_STORAGE_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla}"
ENV_PREFIX="${SEER_ENV_PREFIX:-/home/mingyujung/miniconda3/envs/seer_libero}"
CHECKPOINT_ROOT="${STORAGE_ROOT}/artifacts/checkpoints/seer"
DATASET_ROOT="${STORAGE_ROOT}/artifacts/datasets/seer/realworld"
RESULT_ROOT="${STORAGE_ROOT}/results/seer/latentloop/realworld/droid38"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"

if [[ "$(hostname -s)" != "jbrserver1" ]]; then
    echo "[ERROR] this launcher is locked to sd1/jbrserver1" >&2
    exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3" ]]; then
    echo "[ERROR] set CUDA_VISIBLE_DEVICES=0,1,2,3 exactly" >&2
    exit 1
fi
if [[ ! -x "${PREPARE_SCRIPT}" ]]; then
    echo "[ERROR] missing executable asset preparation script: ${PREPARE_SCRIPT}" >&2
    exit 1
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "[ERROR] missing Seer Python environment: ${ENV_PREFIX}/bin/python" >&2
    exit 1
fi

export SEER_REALWORLD_STORAGE_ROOT="${STORAGE_ROOT}"
export SEER_ENV_PREFIX="${ENV_PREFIX}"
export RUN_ID="${RUN_ID:-r1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-1}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[ERROR] RUN_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 1
fi

"${PREPARE_SCRIPT}"

verify_task() {
    local task="$1" dataset="$2" teacher_sha="$3" data_info_sha="$4"
    local frames="$5" windows="$6" instruction="$7"
    "${ENV_PREFIX}/bin/python" "${VERIFY_SCRIPT}" \
        --task "${task}" \
        --teacher-checkpoint "${CHECKPOINT_ROOT}/realworld/droid38/${task}/38.pth" \
        --teacher-sha256 "${teacher_sha}" \
        --vit-checkpoint "${CHECKPOINT_ROOT}/vit_mae/mae_pretrain_vit_base.pth" \
        --vit-sha256 "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d" \
        --clip-checkpoint "${CHECKPOINT_ROOT}/clip/ViT-B-32.pt" \
        --clip-sha256 "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af" \
        --dataset-root "${DATASET_ROOT}/${dataset}" \
        --dataset-name "${dataset}" \
        --data-info "${UPSTREAM_DIR}/data_info/${dataset}.json" \
        --data-info-sha256 "${data_info_sha}" \
        --expected-instruction "${instruction}" \
        --expected-episodes 40 \
        --expected-total-frames "${frames}" \
        --expected-train-windows "${windows}" \
        --window-size 10
}

echo "[PREFLIGHT] validating both tasks before starting either training run"
verify_task \
    stacking_cups stacking_cups_filtered_40p \
    53cd9d1647dd1be42d7761c38952f9c4c43ecd336807dff25e0ac9738aa0c98a \
    5cca7a31471ee64801fdae0d899e2d7367573d897c2019356095467970f104ca \
    18439 18039 \
    "Stack the orange cup on top of the blue cup, then stack the yellow cup on top of the orange cup"
verify_task \
    rings rings_filtered_40p \
    1755e91255b405d56e81223634fdc373e1502a92d5cda4be40cc5d56ce2cb219 \
    22f7a64cb2de12181db7542a4fd35e18040fe9005627a93ba057cb711aff8fc9 \
    17371 16971 \
    "Put the blue ring on the wooden stand, then put the pink ring on the wooden stand"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[PREFLIGHT][PASS] both tasks are ready; no training was started"
    exit 0
fi

run_task() {
    local task="$1" script="$2" port="$3"
    local run_dir="${RESULT_ROOT}/${task}/real_${task}_latentloop_droid38_adapter_${RUN_ID}"
    if [[ -s "${run_dir}/training_complete.txt" && \
          "$(tr -d '[:space:]' < "${run_dir}/exit_code.txt" 2>/dev/null || true)" == "0" ]]; then
        echo "[SKIP] completed task=${task}; output=${run_dir}"
        return
    fi
    if [[ -e "${run_dir}" ]]; then
        echo "[ERROR] incomplete existing run requires inspection: ${run_dir}" >&2
        echo "[ERROR] preserve it and choose a new RUN_ID before retrying" >&2
        exit 1
    fi
    MASTER_PORT="${port}" "${script}"
}

echo "[1/2] training Stacking Cups LatentLoop adapter"
run_task \
    stacking_cups \
    "${SCRIPT_DIR}/train_realworld_stacking_cups_latentloop.sh" \
    "${STACKING_CUPS_MASTER_PORT:-18200}"

echo "[2/2] training Rings LatentLoop adapter"
run_task \
    rings \
    "${SCRIPT_DIR}/train_realworld_rings_latentloop.sh" \
    "${RINGS_MASTER_PORT:-18210}"

echo "[DONE] Stacking Cups and Rings LatentLoop adapter training completed"
