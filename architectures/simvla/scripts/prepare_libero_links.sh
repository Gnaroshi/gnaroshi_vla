#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIMVLA_DIR="${SIMVLA_DIR:-${ARCH_DIR}/SimVLA}"
DEFAULT_LIBERO_ROOT="/home/mingyujung/shared/nvme1/mingyujung/datasets/robotics/LIBERO"
LIBERO_ROOT="${LIBERO_ROOT:-${DEFAULT_LIBERO_ROOT}}"

if [[ -d "${LIBERO_ROOT}/datasets/libero_10" ]]; then
    LIBERO_DATA_ROOT="${LIBERO_ROOT}/datasets"
elif [[ -d "${LIBERO_ROOT}/libero_10" ]]; then
    LIBERO_DATA_ROOT="${LIBERO_ROOT}"
else
    echo "[ERROR] Could not find LIBERO subset directories under: ${LIBERO_ROOT}" >&2
    echo "[ERROR] Expected either LIBERO_ROOT/datasets/libero_10 or LIBERO_ROOT/libero_10" >&2
    exit 2
fi

METAS_DIR="${SIMVLA_DIR}/datasets/metas"
mkdir -p "${METAS_DIR}"

subsets=(libero_10 libero_goal libero_object libero_spatial libero_90)

echo "[SIMVLA] upstream_dir=${SIMVLA_DIR}"
echo "[SIMVLA] libero_data_root=${LIBERO_DATA_ROOT}"
echo "[SIMVLA] metas_dir=${METAS_DIR}"

for subset in "${subsets[@]}"; do
    target="${LIBERO_DATA_ROOT}/${subset}"
    dest="${METAS_DIR}/${subset}"

    if [[ ! -d "${target}" ]]; then
        echo "[ERROR] Missing LIBERO subset: ${target}" >&2
        exit 2
    fi

    if [[ -e "${dest}" && ! -L "${dest}" ]]; then
        echo "[ERROR] Refusing to replace non-symlink path: ${dest}" >&2
        exit 2
    fi

    ln -sfnT "${target}" "${dest}"
    count="$(find "${target}" -maxdepth 1 -type f -name '*.hdf5' | wc -l)"
    echo "[OK] ${dest} -> ${target} (${count} hdf5 files)"
done

