#!/usr/bin/env bash

set -Eeuo pipefail

CODE_ROOT="${CODE_ROOT:-/home/mingyujung/private/gnaroshi_vla}"
STORAGE_ROOT="${SEER_REALWORLD_STORAGE_ROOT:-/var/shared/hdd_ext/ssd8000/mingyujung/gnaroshi_vla}"
ENV_PREFIX="${SEER_ENV_PREFIX:-${STORAGE_ROOT}/envs/seer_libero}"
ENV_ARCHIVE="${STORAGE_ROOT}/staging/seer_libero.tar.gz"
FINALIZE_LOG="${FINALIZE_LOG:-/tmp/seer_s4_realworld_finalize.log}"

DOLL_SHA256="481a8dff0b0425b30fb9a589bbc20f89944e9e27240e1384bce0ccebfb5610b9"
CABINET_SHA256="fe642965f46a35ad6810f155bf7f62c1219e0a78fbe32fd2c3e1b5b95e563996"
VIT_SHA256="aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
CLIP_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

mkdir -p "$(dirname "${FINALIZE_LOG}")"
exec > >(tee -a "${FINALIZE_LOG}") 2>&1

on_error() {
    local exit_code=$?
    echo "[ERROR] s4 finalization failed at line ${BASH_LINENO[0]} with exit code ${exit_code}" >&2
    echo "[ERROR] log preserved at ${FINALIZE_LOG}" >&2
    exit "${exit_code}"
}
trap on_error ERR

fail() {
    echo "[ERROR] $*" >&2
    return 1
}

check_hash() {
    local file_path="$1" expected="$2" label="$3" actual
    [[ -s "${file_path}" ]] || fail "missing or empty ${label}: ${file_path}"
    actual="$(sha256sum "${file_path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || fail "${label} SHA-256 mismatch: ${actual}"
    echo "[VERIFY][OK] ${label}: ${actual}"
}

install_environment() {
    local incoming="${ENV_PREFIX}.incoming"
    if [[ -s "${ENV_PREFIX}/.s4_relocation_complete" ]]; then
        echo "[SKIP][VERIFIED] relocated environment already installed"
        return
    fi
    if [[ -d "${ENV_PREFIX}" ]]; then
        [[ -x "${ENV_PREFIX}/bin/python" && -x "${ENV_PREFIX}/bin/conda-unpack" ]] \
            || fail "incomplete relocated environment cannot be resumed: ${ENV_PREFIX}"
        echo "[RESUME] completing previously extracted environment: ${ENV_PREFIX}"
    else
        [[ -s "${ENV_ARCHIVE}" ]] || fail "missing environment archive: ${ENV_ARCHIVE}"
        (
            cd "${STORAGE_ROOT}/staging"
            sha256sum --quiet -c "${STORAGE_ROOT}/manifests/environment/seer_libero_archive.sha256"
        )

        rm -rf "${incoming}"
        mkdir -p "${incoming}"
        tar -xzf "${ENV_ARCHIVE}" -C "${incoming}"
        mv "${incoming}" "${ENV_PREFIX}"
    fi
    PATH="${ENV_PREFIX}/bin:${PATH}" "${ENV_PREFIX}/bin/conda-unpack"
    date --iso-8601=seconds > "${ENV_PREFIX}/.s4_relocation_complete"
    echo "[VERIFY][OK] relocated environment installed: ${ENV_PREFIX}"
}

echo "[START] $(date --iso-8601=seconds)"
[[ "$(hostname)" == "jbrserver4" ]] || fail "run this script on s4/jbrserver4"
[[ -d "${CODE_ROOT}/architectures/seer/upstream" ]] || fail "invalid CODE_ROOT: ${CODE_ROOT}"
[[ -d "${STORAGE_ROOT}" && -w "${STORAGE_ROOT}" ]] \
    || fail "storage root is missing or not writable: ${STORAGE_ROOT}"

check_hash "${STORAGE_ROOT}/artifacts/checkpoints/seer/realworld/droid38/doll/38.pth" \
    "${DOLL_SHA256}" "Doll teacher 38"
check_hash "${STORAGE_ROOT}/artifacts/checkpoints/seer/realworld/droid38/cabinet/38.pth" \
    "${CABINET_SHA256}" "Cabinet teacher 38"
check_hash "${STORAGE_ROOT}/artifacts/checkpoints/seer/vit_mae/mae_pretrain_vit_base.pth" \
    "${VIT_SHA256}" "ViT-MAE"
check_hash "${STORAGE_ROOT}/artifacts/checkpoints/seer/clip/ViT-B-32.pt" \
    "${CLIP_SHA256}" "CLIP ViT-B/32"

DATASET_ROOT="${STORAGE_ROOT}/artifacts/datasets/seer/realworld"
for dataset_name in doll_filtered_40p cabinet_filtered_40p; do
    [[ -s "${DATASET_ROOT}/${dataset_name}/.transfer_verified" ]] \
        || fail "dataset transfer marker is missing: ${dataset_name}"
    [[ -s "${STORAGE_ROOT}/manifests/datasets/${dataset_name}.sha256" ]] \
        || fail "dataset checksum manifest is missing: ${dataset_name}"
    (
        cd "${DATASET_ROOT}"
        sha256sum --quiet -c "${STORAGE_ROOT}/manifests/datasets/${dataset_name}.sha256"
    )
    echo "[VERIFY][OK] dataset manifest: ${dataset_name}"
done

install_environment

mkdir -p "${CODE_ROOT}/architectures/seer/upstream/checkpoints/clip" \
    "${CODE_ROOT}/architectures/seer/upstream/checkpoints/vit_mae"
ln -sfn "${STORAGE_ROOT}/artifacts/checkpoints/seer/clip/ViT-B-32.pt" \
    "${CODE_ROOT}/architectures/seer/upstream/checkpoints/clip/ViT-B-32.pt"
ln -sfn "${STORAGE_ROOT}/artifacts/checkpoints/seer/vit_mae/mae_pretrain_vit_base.pth" \
    "${CODE_ROOT}/architectures/seer/upstream/checkpoints/vit_mae/mae_pretrain_vit_base.pth"

chmod +x \
    "${CODE_ROOT}/architectures/seer/wrappers/lrnode/train_realworld_latentloop_adapter.sh" \
    "${CODE_ROOT}/architectures/seer/wrappers/lrnode/train_realworld_doll_latentloop.sh" \
    "${CODE_ROOT}/architectures/seer/wrappers/lrnode/train_realworld_cabinet_latentloop.sh"

"${ENV_PREFIX}/bin/python" - <<'PY'
import cv2
import h5py
import numpy
import torch
import torchvision
import transformers

print("[VERIFY][OK] runtime imports")
print("torch=", torch.__version__, "cuda=", torch.version.cuda)
print("physical_cuda_devices=", torch.cuda.device_count())
PY

cd "${CODE_ROOT}"
CUDA_VISIBLE_DEVICES=0,1,2,3 PREFLIGHT_ONLY=1 REPORT_TO_WANDB=0 \
    bash architectures/seer/wrappers/lrnode/train_realworld_doll_latentloop.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 PREFLIGHT_ONLY=1 REPORT_TO_WANDB=0 \
    bash architectures/seer/wrappers/lrnode/train_realworld_cabinet_latentloop.sh

rm -f "${ENV_ARCHIVE}"
echo "[DONE] $(date --iso-8601=seconds)"
echo "[DONE] s4 environment, assets, datasets, and both 4-GPU launchers verified"
echo "[DONE] log=${FINALIZE_LOG}"
