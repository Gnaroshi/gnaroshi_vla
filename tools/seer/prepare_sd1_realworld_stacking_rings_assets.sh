#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STORAGE_ROOT="${SEER_REALWORLD_STORAGE_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla}"
CHECKPOINT_ROOT="${STORAGE_ROOT}/artifacts/checkpoints/seer"
DATASET_ROOT="${STORAGE_ROOT}/artifacts/datasets/seer/realworld"
MANIFEST_ROOT="${STORAGE_ROOT}/manifests/seer/realworld/droid38"

INFERENCE_HOST="${INFERENCE_HOST:-jbr@210.107.197.121}"
INFERENCE_PORT="${INFERENCE_PORT:-9000}"
S5_HOST="${S5_HOST:-s5}"
SSH_OPTIONS="-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
INFERENCE_RSH="ssh -p ${INFERENCE_PORT} ${SSH_OPTIONS}"
S5_RSH="ssh ${SSH_OPTIONS}"

VIT_SOURCE="/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth"
VIT_SHA256="aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
CLIP_SOURCE="/home/mingyujung/.cache/clip/ViT-B-32.pt"
CLIP_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

require_sd1() {
    local host
    host="$(hostname -s)"
    if [[ "${host}" != "jbrserver1" ]]; then
        echo "[ERROR] this asset preparation is locked to sd1/jbrserver1; got ${host}" >&2
        exit 1
    fi
}

verify_sha256() {
    local label="$1" path="$2" expected="$3" actual
    if [[ ! -s "${path}" ]]; then
        echo "[ERROR] missing or empty ${label}: ${path}" >&2
        exit 1
    fi
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[ERROR] ${label} SHA-256 mismatch: expected=${expected}, actual=${actual}" >&2
        exit 1
    fi
    echo "[VERIFY][OK] ${label} sha256=${actual}"
}

link_vit() {
    local destination="${CHECKPOINT_ROOT}/vit_mae/mae_pretrain_vit_base.pth"
    verify_sha256 "canonical ViT-MAE" "${VIT_SOURCE}" "${VIT_SHA256}"
    mkdir -p "$(dirname "${destination}")"
    if [[ ! -e "${destination}" && ! -L "${destination}" ]]; then
        ln -s "${VIT_SOURCE}" "${destination}"
    fi
    verify_sha256 "shared ViT-MAE" "${destination}" "${VIT_SHA256}"
}

copy_local_file() {
    local label="$1" source="$2" destination="$3" expected="$4"
    verify_sha256 "source ${label}" "${source}" "${expected}"
    mkdir -p "$(dirname "${destination}")"
    if [[ ! -e "${destination}" ]]; then
        cp --reflink=auto "${source}" "${destination}.partial"
        mv "${destination}.partial" "${destination}"
    fi
    verify_sha256 "shared ${label}" "${destination}" "${expected}"
}

fetch_teacher() {
    local task="$1" source="$2" expected="$3"
    local destination="${CHECKPOINT_ROOT}/realworld/droid38/${task}/38.pth"
    mkdir -p "$(dirname "${destination}")"
    if [[ ! -s "${destination}" ]]; then
        echo "[TRANSFER] ${task} teacher38 from inference computer"
        rsync -ah --partial --append-verify --info=progress2 \
            -e "${INFERENCE_RSH}" \
            "${INFERENCE_HOST}:${source}" "${destination}.partial"
        mv "${destination}.partial" "${destination}"
    fi
    verify_sha256 "${task} teacher38" "${destination}" "${expected}"
}

fetch_dataset() {
    local task="$1" dataset_name="$2"
    local source="/var/shared/hdd_ext/nvme4000/mingyujung/FlowVLA_backups/dataset/${dataset_name}"
    local destination="${DATASET_ROOT}/${dataset_name}"
    mkdir -p "${destination}"
    echo "[TRANSFER] ${task} dataset from s5 (existing files are resumed)"
    rsync -ah --partial --info=progress2 -e "${S5_RSH}" \
        "${S5_HOST}:${source}/" "${destination}/"
    local delta
    delta="$(rsync -rcn --delete --itemize-changes -e "${S5_RSH}" \
        "${S5_HOST}:${source}/" "${destination}/")"
    if [[ -n "${delta}" ]]; then
        echo "[ERROR] ${task} dataset differs after transfer:" >&2
        printf '%s\n' "${delta}" >&2
        exit 1
    fi
    if [[ ! -d "${destination}/${dataset_name}/0000" ]]; then
        echo "[ERROR] nested ${task} dataset is missing after transfer" >&2
        exit 1
    fi
    echo "[VERIFY][OK] ${task} dataset matches s5 source"
}

require_sd1
command -v rsync >/dev/null
command -v ssh >/dev/null
command -v sha256sum >/dev/null
mkdir -p "${MANIFEST_ROOT}"

link_vit
copy_local_file "CLIP ViT-B/32" "${CLIP_SOURCE}" \
    "${CHECKPOINT_ROOT}/clip/ViT-B-32.pt" "${CLIP_SHA256}"

fetch_teacher \
    stacking_cups \
    "/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/stacking_cups/seer_baseline_real-world_ft_40p_task-stacking_cups_droidpt/38.pth" \
    "53cd9d1647dd1be42d7761c38952f9c4c43ecd336807dff25e0ac9738aa0c98a"
fetch_teacher \
    rings \
    "/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/rings/seer_baseline_real-world_ft_40p_task-rings_droidpt/38.pth" \
    "1755e91255b405d56e81223634fdc373e1502a92d5cda4be40cc5d56ce2cb219"

fetch_dataset stacking_cups stacking_cups_filtered_40p
fetch_dataset rings rings_filtered_40p

cat > "${MANIFEST_ROOT}/sd1_stacking_cups_rings_sources.env" <<EOF
storage_root=${STORAGE_ROOT}
stacking_cups_teacher_source=${INFERENCE_HOST}:/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/stacking_cups/seer_baseline_real-world_ft_40p_task-stacking_cups_droidpt/38.pth
stacking_cups_teacher_sha256=53cd9d1647dd1be42d7761c38952f9c4c43ecd336807dff25e0ac9738aa0c98a
rings_teacher_source=${INFERENCE_HOST}:/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/rings/seer_baseline_real-world_ft_40p_task-rings_droidpt/38.pth
rings_teacher_sha256=1755e91255b405d56e81223634fdc373e1502a92d5cda4be40cc5d56ce2cb219
stacking_cups_dataset_source=${S5_HOST}:/var/shared/hdd_ext/nvme4000/mingyujung/FlowVLA_backups/dataset/stacking_cups_filtered_40p
rings_dataset_source=${S5_HOST}:/var/shared/hdd_ext/nvme4000/mingyujung/FlowVLA_backups/dataset/rings_filtered_40p
vit_source=${VIT_SOURCE}
vit_sha256=${VIT_SHA256}
clip_source=${CLIP_SOURCE}
clip_sha256=${CLIP_SHA256}
EOF

echo "[DONE] real-world Stacking Cups and Rings assets are ready under ${STORAGE_ROOT}"
