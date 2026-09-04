#!/usr/bin/env bash

set -euo pipefail

LATENT_BRIDGE_EFFICIENT_WRAPPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LATENT_BRIDGE_RESULT_ROOT="${LATENT_BRIDGE_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/latent_bridge/public33_compute_matched_full_egl50_seed42}"
export LATENT_BRIDGE_RENDERER="${LATENT_BRIDGE_RENDERER:-egl}"
source "${LATENT_BRIDGE_EFFICIENT_WRAPPER_DIR}/../latent_bridge/common.sh"

LATENT_BRIDGE_EFFICIENT_SYNC_STAGE="${LATENT_BRIDGE_EFFICIENT_SYNC_STAGE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/latent_bridge/public33_full_egl50_seed42/continuity_and_sync}"

latent_bridge_require_runtime() {
    [[ "${CONDA_DEFAULT_ENV:-}" == "seer_libero" ]] || \
        latent_bridge_fail "activate conda environment seer_libero first"
    [[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3" ]] || \
        latent_bridge_fail "efficient sd1 run requires CUDA_VISIBLE_DEVICES=0,1,2,3"
    [[ "${NODE_NUM:-4}" == "4" ]] || \
        latent_bridge_fail "compute-matched contract requires NODE_NUM=4"
    [[ -s "${LATENT_BRIDGE_PUBLIC33}" ]] || \
        latent_bridge_fail "missing public33: ${LATENT_BRIDGE_PUBLIC33}"
    [[ -s "${LATENT_BRIDGE_VIT}" ]] || \
        latent_bridge_fail "missing ViT-MAE: ${LATENT_BRIDGE_VIT}"
    [[ -s "${LATENT_BRIDGE_DATASET_ROOT}/libero_10_converted/meta_info.h5" ]] || \
        latent_bridge_fail "invalid converted dataset root: ${LATENT_BRIDGE_DATASET_ROOT}"
    [[ -d "${LATENT_BRIDGE_LIBERO_PATH}/libero/libero" ]] || \
        latent_bridge_fail "invalid LIBERO path: ${LATENT_BRIDGE_LIBERO_PATH}"
    [[ -e "${LATENT_BRIDGE_OFFICIAL_SOURCE}/.git" ]] || \
        latent_bridge_fail "official Latent Bridge source is absent"
    [[ -s "${LATENT_BRIDGE_EFFICIENT_SYNC_STAGE}/COMPLETE" ]] || \
        latent_bridge_fail "reusable continuity/sync source is incomplete"
    [[ "${LATENT_BRIDGE_RESULT_ROOT}" != "${LATENT_BRIDGE_EFFICIENT_SYNC_STAGE}"* ]] || \
        latent_bridge_fail "result root overlaps the read-only sync source"
}

latent_bridge_efficient_head() {
    git -C "${LATENT_BRIDGE_REPO_ROOT}" rev-parse HEAD
}

latent_bridge_efficient_require_clean_source() {
    [[ -z "$(git -C "${LATENT_BRIDGE_REPO_ROOT}" status --porcelain --untracked-files=no)" ]] || \
        latent_bridge_fail "efficient worktree has tracked modifications; commit before launch"
}

latent_bridge_efficient_initialize_source_lock() {
    latent_bridge_efficient_require_clean_source
    local lock_path="${LATENT_BRIDGE_RESULT_ROOT}/SOURCE_LOCK"
    local current_head
    current_head="$(latent_bridge_efficient_head)"
    if [[ -s "${lock_path}" ]]; then
        [[ "$(<"${lock_path}")" == "${current_head}" ]] || \
            latent_bridge_fail "result root is locked to a different source commit"
    else
        printf '%s\n' "${current_head}" > "${lock_path}"
    fi
}

latent_bridge_efficient_verify_source_lock() {
    latent_bridge_efficient_require_clean_source
    local lock_path="${LATENT_BRIDGE_RESULT_ROOT}/SOURCE_LOCK"
    [[ -s "${lock_path}" ]] || latent_bridge_fail "missing efficient SOURCE_LOCK"
    [[ "$(<"${lock_path}")" == "$(latent_bridge_efficient_head)" ]] || \
        latent_bridge_fail "source commit changed after pipeline launch"
}

latent_bridge_efficient_load_profile() {
    local profile="${LATENT_BRIDGE_RESULT_ROOT}/preflight/training_profile.env"
    [[ -s "${profile}" ]] || latent_bridge_fail "missing validated training profile: ${profile}"
    set -a
    source "${profile}"
    set +a
    [[ "$((TRAIN_PER_RANK_BATCH * 4 * TRAIN_GRADIENT_ACCUMULATION_STEPS))" -eq 64 ]] || \
        latent_bridge_fail "validated profile does not preserve effective batch 64"
}
