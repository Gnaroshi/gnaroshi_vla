#!/usr/bin/env bash

set -Eeuo pipefail

SOURCE_REPO="${SOURCE_REPO:-/home/mingyujung/private/gnaroshi_vla}"
S4_HOST="${S4_HOST:-s4}"
S4_CODE_ROOT="${S4_CODE_ROOT:-/home/mingyujung/private/gnaroshi_vla}"
S4_STORAGE_ROOT="${S4_STORAGE_ROOT:-/var/shared/hdd_ext/ssd8000/mingyujung/gnaroshi_vla}"
SD1_ENV_PREFIX="${SD1_ENV_PREFIX:-/home/mingyujung/miniconda3/envs/seer_libero}"
SD1_VIT="${SD1_VIT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
SD1_CLIP="${SD1_CLIP:-/home/mingyujung/.cache/clip/ViT-B-32.pt}"
LOCAL_STAGE="${LOCAL_STAGE:-/tmp/seer_realworld_droid38_stage}"
ENV_ARCHIVE="${ENV_ARCHIVE:-/tmp/seer_libero_s4.tar.gz}"
TRANSFER_LOG="${TRANSFER_LOG:-/tmp/seer_s4_realworld_assets_transfer.log}"

INFERENCE_HOST="jbr@210.107.197.121"
INFERENCE_PORT=9000
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=20)
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=20"

DOLL_SHA256="481a8dff0b0425b30fb9a589bbc20f89944e9e27240e1384bce0ccebfb5610b9"
CABINET_SHA256="fe642965f46a35ad6810f155bf7f62c1219e0a78fbe32fd2c3e1b5b95e563996"
VIT_SHA256="aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
CLIP_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

DOLL_SOURCE="/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/doll/seer_baseline_real-world_ft_40p_task-doll_droidpt/38.pth"
CABINET_SOURCE="/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/cabinet/seer_baseline_real-world_ft_40p_task-cabinet_droidpt/38.pth"

mkdir -p "$(dirname "${TRANSFER_LOG}")"
exec > >(tee -a "${TRANSFER_LOG}") 2>&1

on_error() {
    local exit_code=$?
    echo "[ERROR] sd1-to-s4 transfer failed at line ${BASH_LINENO[0]} with exit code ${exit_code}" >&2
    echo "[ERROR] log preserved at ${TRANSFER_LOG}" >&2
    exit "${exit_code}"
}
trap on_error ERR

fail() {
    echo "[ERROR] $*" >&2
    return 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

verify_local_hash() {
    local file_path="$1" expected="$2" label="$3" actual
    require_file "${file_path}"
    actual="$(sha256sum "${file_path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || fail "${label} SHA-256 mismatch: ${actual}"
    echo "[VERIFY][OK] ${label}: ${actual}"
}

remote_hash() {
    local remote_path="$1"
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" "sha256sum '${remote_path}' | awk '{print \$1}'"
}

push_verified() {
    local source_file="$1" destination="$2" expected="$3" label="$4" actual

    if ssh "${SSH_OPTS[@]}" "${S4_HOST}" "test -e '${destination}'"; then
        ssh "${SSH_OPTS[@]}" "${S4_HOST}" "test -s '${destination}'" \
            || fail "existing s4 ${label} is empty: ${destination}"
        actual="$(remote_hash "${destination}")"
        [[ "${actual}" == "${expected}" ]] \
            || fail "existing s4 ${label} SHA-256 mismatch: ${actual}"
        echo "[SKIP][VERIFIED] s4 already has ${label}"
        return
    fi

    ssh "${SSH_OPTS[@]}" "${S4_HOST}" "mkdir -p '$(dirname "${destination}")'"
    rsync -a --partial --info=progress2 -e "${RSYNC_RSH}" \
        "${source_file}" "${S4_HOST}:${destination}.partial"
    actual="$(remote_hash "${destination}.partial")"
    [[ "${actual}" == "${expected}" ]] \
        || fail "transferred s4 ${label} SHA-256 mismatch: ${actual}"
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "test ! -e '${destination}' && mv '${destination}.partial' '${destination}'"
    echo "[VERIFY][OK] s4 ${label}: ${actual}"
}

pull_teacher() {
    local source_file="$1" destination="$2" expected="$3" label="$4"
    if [[ -s "${destination}" ]]; then
        verify_local_hash "${destination}" "${expected}" "${label} local staging"
        return
    fi

    mkdir -p "$(dirname "${destination}")"
    rsync -a --partial --info=progress2 \
        -e "ssh -p ${INFERENCE_PORT} -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=20" \
        "${INFERENCE_HOST}:${source_file}" "${destination}"
    verify_local_hash "${destination}" "${expected}" "${label} local staging"
}

sync_source() {
    local metadata_dir
    metadata_dir="$(mktemp -d /tmp/s4_source_metadata.XXXXXX)"
    trap 'rm -rf "${metadata_dir}"' RETURN

    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "mkdir -p '${S4_CODE_ROOT}/architectures' '${S4_CODE_ROOT}/methods' '${S4_CODE_ROOT}/tools' '${S4_STORAGE_ROOT}/manifests/source'"

    rsync -a \
        --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
        --exclude='upstream/LIBERO_DATASETS' --exclude='upstream/checkpoints' \
        --exclude='upstream/wandb/' --exclude='upstream/runs/' \
        -e "${RSYNC_RSH}" \
        "${SOURCE_REPO}/architectures/seer/" "${S4_HOST}:${S4_CODE_ROOT}/architectures/seer/"
    rsync -a --exclude='__pycache__/' --exclude='*.pyc' -e "${RSYNC_RSH}" \
        "${SOURCE_REPO}/methods/" "${S4_HOST}:${S4_CODE_ROOT}/methods/"
    rsync -a --exclude='__pycache__/' --exclude='*.pyc' -e "${RSYNC_RSH}" \
        "${SOURCE_REPO}/tools/seer/" "${S4_HOST}:${S4_CODE_ROOT}/tools/seer/"
    rsync -a -e "${RSYNC_RSH}" \
        "${SOURCE_REPO}/AGENTS.md" "${SOURCE_REPO}/README.md" "${S4_HOST}:${S4_CODE_ROOT}/"

    git -C "${SOURCE_REPO}" rev-parse HEAD > "${metadata_dir}/origin_commit.txt"
    git -C "${SOURCE_REPO}" rev-parse --abbrev-ref HEAD > "${metadata_dir}/origin_branch.txt"
    git -C "${SOURCE_REPO}" status --short > "${metadata_dir}/origin_git_status.txt"
    date --iso-8601=seconds > "${metadata_dir}/captured_at.txt"
    rsync -a -e "${RSYNC_RSH}" "${metadata_dir}/" \
        "${S4_HOST}:${S4_STORAGE_ROOT}/manifests/source/"

    rm -rf "${metadata_dir}"
    trap - RETURN
    echo "[VERIFY][OK] minimal Seer source synchronized"
}

pack_environment() {
    local building_archive="${ENV_ARCHIVE}.building" archive_sha
    if [[ -s "${ENV_ARCHIVE}" ]]; then
        tar -tzf "${ENV_ARCHIVE}" >/dev/null \
            || fail "existing environment archive is invalid: ${ENV_ARCHIVE}"
        echo "[SKIP][VERIFIED] valid local environment archive already exists"
    else
        rm -f "${building_archive}"
        echo "[PACK] creating relocatable seer_libero environment"
        /home/mingyujung/miniconda3/bin/conda-pack \
            -p "${SD1_ENV_PREFIX}" -o "${building_archive}" \
            --format tar.gz --compress-level 4 --ignore-editable-packages
        tar -tzf "${building_archive}" >/dev/null
        mv "${building_archive}" "${ENV_ARCHIVE}"
    fi

    archive_sha="$(sha256sum "${ENV_ARCHIVE}" | awk '{print $1}')"
    /home/mingyujung/miniconda3/bin/conda list -p "${SD1_ENV_PREFIX}" --explicit \
        > /tmp/seer_libero_conda_explicit.txt
    "${SD1_ENV_PREFIX}/bin/python" -m pip freeze > /tmp/seer_libero_pip_freeze.txt

    push_verified "${ENV_ARCHIVE}" \
        "${S4_STORAGE_ROOT}/staging/seer_libero.tar.gz" \
        "${archive_sha}" "packed seer_libero environment"
    printf '%s  seer_libero.tar.gz\n' "${archive_sha}" > /tmp/seer_libero_archive.sha256
    rsync -a -e "${RSYNC_RSH}" \
        /tmp/seer_libero_archive.sha256 \
        /tmp/seer_libero_conda_explicit.txt \
        /tmp/seer_libero_pip_freeze.txt \
        "${S4_HOST}:${S4_STORAGE_ROOT}/manifests/environment/"
}

echo "[START] $(date --iso-8601=seconds)"
[[ "$(hostname)" == "jbrserver1" ]] || fail "run this script on sd1/jbrserver1"
[[ -d "${SOURCE_REPO}/architectures/seer/upstream" ]] \
    || fail "invalid SOURCE_REPO: ${SOURCE_REPO}"
require_file /home/mingyujung/miniconda3/bin/conda-pack
require_file "${SD1_ENV_PREFIX}/bin/python"
verify_local_hash "${SD1_VIT}" "${VIT_SHA256}" "ViT-MAE"
verify_local_hash "${SD1_CLIP}" "${CLIP_SHA256}" "CLIP ViT-B/32"

ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
    "test -d '${S4_STORAGE_ROOT}' && test -w '${S4_STORAGE_ROOT}'"
ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
    "mkdir -p '${S4_STORAGE_ROOT}/manifests/environment' '${S4_STORAGE_ROOT}/staging'"

sync_source
pull_teacher "${DOLL_SOURCE}" "${LOCAL_STAGE}/doll/38.pth" \
    "${DOLL_SHA256}" "Doll teacher 38"
pull_teacher "${CABINET_SOURCE}" "${LOCAL_STAGE}/cabinet/38.pth" \
    "${CABINET_SHA256}" "Cabinet teacher 38"

push_verified "${LOCAL_STAGE}/doll/38.pth" \
    "${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/realworld/droid38/doll/38.pth" \
    "${DOLL_SHA256}" "Doll teacher 38"
push_verified "${LOCAL_STAGE}/cabinet/38.pth" \
    "${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/realworld/droid38/cabinet/38.pth" \
    "${CABINET_SHA256}" "Cabinet teacher 38"
push_verified "${SD1_VIT}" \
    "${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/vit_mae/mae_pretrain_vit_base.pth" \
    "${VIT_SHA256}" "ViT-MAE"
push_verified "${SD1_CLIP}" \
    "${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/clip/ViT-B-32.pt" \
    "${CLIP_SHA256}" "CLIP ViT-B/32"

pack_environment

echo "[DONE] $(date --iso-8601=seconds)"
echo "[DONE] source, teachers, ViT-MAE, CLIP, and environment archive are verified on s4"
echo "[DONE] log=${TRANSFER_LOG}"
