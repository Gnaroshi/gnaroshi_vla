#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST="$(hostname -s)"

case "${HOST}" in
    jbrserver1)
        DATASET_SOURCE="${SEER_DATASET_SOURCE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/LIBERO_DATASETS}"
        VIT_SOURCE="${SEER_VIT_SOURCE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
        ;;
    jbrserver5)
        DATASET_SOURCE="${SEER_DATASET_SOURCE:-/var/shared/hdd_ext/nvme4000/mingyujung/FlowVLA/seer/LIBERO_DATASETS}"
        VIT_SOURCE="${SEER_VIT_SOURCE:-/var/shared/hdd_ext/nvme4000/mingyujung/FlowVLA/seer/checkpoints_seer_official/vit_mae/mae_pretrain_vit_base.pth}"
        ;;
    jbrserver18)
        DATASET_SOURCE="${SEER_DATASET_SOURCE:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/datasets/LIBERO_DATASETS}"
        VIT_SOURCE="${SEER_VIT_SOURCE:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/artifacts/seer/vit_mae/mae_pretrain_vit_base.pth}"
        ;;
    *)
        : "${SEER_DATASET_SOURCE:?Set SEER_DATASET_SOURCE on host ${HOST}}"
        : "${SEER_VIT_SOURCE:?Set SEER_VIT_SOURCE on host ${HOST}}"
        DATASET_SOURCE="${SEER_DATASET_SOURCE}"
        VIT_SOURCE="${SEER_VIT_SOURCE}"
        ;;
esac

UPSTREAM="${REPO_ROOT}/architectures/seer/upstream"
LOCAL_SEER_ROOT="${REPO_ROOT}/data/seer"
DATASET_RESOLVER="${LOCAL_SEER_ROOT}/LIBERO_DATASETS"
DATASET_LINK="${UPSTREAM}/LIBERO_DATASETS"
DATASET_LINK_TARGET="../../../data/seer/LIBERO_DATASETS"
VIT_LINK="${UPSTREAM}/checkpoints/vit_mae/mae_pretrain_vit_base.pth"
LIBERO_PATH="${LIBERO_PATH:-$(dirname "${REPO_ROOT}")/LIBERO}"
EXPECTED_VIT_SHA256="aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"

require_dir() {
    [[ -d "$1" ]] || { echo "[ERROR] missing directory: $1" >&2; exit 1; }
}

require_file() {
    [[ -f "$1" ]] || { echo "[ERROR] missing file: $1" >&2; exit 1; }
}

ensure_link() {
    local link_path="$1" target="$2"
    mkdir -p "$(dirname "${link_path}")"
    if [[ -L "${link_path}" ]]; then
        if [[ "$(readlink -f "${link_path}" 2>/dev/null || true)" != "$(readlink -f "${target}")" ]]; then
            echo "[RELINK] ${link_path}: $(readlink "${link_path}") -> ${target}"
            ln -sfn "${target}" "${link_path}"
        fi
    elif [[ -e "${link_path}" ]]; then
        echo "[ERROR] refusing to replace non-symlink path: ${link_path}" >&2
        exit 1
    else
        ln -s "${target}" "${link_path}"
    fi
}

ensure_dataset_interface() {
    ensure_link "${DATASET_RESOLVER}" "${DATASET_SOURCE}"
    if [[ -L "${DATASET_LINK}" ]]; then
        if [[ "$(readlink "${DATASET_LINK}")" != "${DATASET_LINK_TARGET}" ]]; then
            echo "[RELINK] ${DATASET_LINK}: $(readlink "${DATASET_LINK}") -> ${DATASET_LINK_TARGET}"
            ln -sfn "${DATASET_LINK_TARGET}" "${DATASET_LINK}"
        fi
    elif [[ -e "${DATASET_LINK}" ]]; then
        echo "[ERROR] refusing to replace non-symlink path: ${DATASET_LINK}" >&2
        exit 1
    else
        ln -s "${DATASET_LINK_TARGET}" "${DATASET_LINK}"
    fi
}

require_dir "${REPO_ROOT}"
require_dir "${DATASET_SOURCE}"
require_file "${DATASET_SOURCE}/libero_10_converted/libero_10_converted/meta_info.h5"
require_file "${VIT_SOURCE}"
require_dir "${LIBERO_PATH}"

actual_vit_sha256="$(sha256sum "${VIT_SOURCE}" | awk '{print $1}')"
[[ "${actual_vit_sha256}" == "${EXPECTED_VIT_SHA256}" ]] || {
    echo "[ERROR] ViT SHA-256 mismatch: ${actual_vit_sha256}" >&2
    exit 1
}

ensure_dataset_interface
ensure_link "${VIT_LINK}" "${VIT_SOURCE}"

require_file "${DATASET_LINK}/libero_10_converted/libero_10_converted/meta_info.h5"
require_file "${VIT_LINK}"

echo "[OK] host=${HOST}"
echo "[OK] repo_root=${REPO_ROOT}"
echo "[OK] dataset=${DATASET_LINK} -> $(readlink -f "${DATASET_LINK}")"
echo "[OK] vit=${VIT_LINK} -> $(readlink -f "${VIT_LINK}")"
echo "[OK] vit_sha256=${actual_vit_sha256}"
echo "[OK] libero=${LIBERO_PATH}"
