#!/usr/bin/env bash

set -Eeuo pipefail

SOURCE_REPO="${SOURCE_REPO:-/home/mingyujung/private/gnaroshi_vla}"
S4_HOST="${S4_HOST:-s4}"
S5_HOST="${S5_HOST:-s5}"
S4_CODE_ROOT="${S4_CODE_ROOT:-/home/mingyujung/private/gnaroshi_vla}"
S4_STORAGE_ROOT="${S4_STORAGE_ROOT:-/var/shared/hdd_ext/ssd8000/mingyujung/gnaroshi_vla}"
SD1_ENV_PREFIX="${SD1_ENV_PREFIX:-/home/mingyujung/miniconda3/envs/seer_libero}"
CONDA_PACK="${CONDA_PACK:-/home/mingyujung/miniconda3/bin/conda-pack}"
SD1_VIT="${SD1_VIT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
SD1_CLIP="${SD1_CLIP:-/home/mingyujung/.cache/clip/ViT-B-32.pt}"

S5_DATASET_BASE="/var/shared/hdd_ext/nvme4000/mingyujung/FlowVLA_backups/dataset"
INFERENCE_HOST="jbr@210.107.197.121"
INFERENCE_PORT=9000

VIT_SHA256="aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
CLIP_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
DOLL_TEACHER_SHA256="481a8dff0b0425b30fb9a589bbc20f89944e9e27240e1384bce0ccebfb5610b9"
CABINET_TEACHER_SHA256="fe642965f46a35ad6810f155bf7f62c1219e0a78fbe32fd2c3e1b5b95e563996"

DOLL_TEACHER_SOURCE="/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/doll/seer_baseline_real-world_ft_40p_task-doll_droidpt/38.pth"
CABINET_TEACHER_SOURCE="/home/jbr/bc_data/3dflow/checkpoints_seer_baseline/Real-World/Droid_Pre-trained/cabinet/seer_baseline_real-world_ft_40p_task-cabinet_droidpt/38.pth"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=20)
TEMP_PATHS=()

cleanup() {
    local path
    for path in "${TEMP_PATHS[@]:-}"; do
        [[ -n "${path}" ]] && rm -rf "${path}"
    done
}
trap cleanup EXIT

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

verify_local_hash() {
    local path="$1" expected="$2" label="$3" actual
    require_file "${path}"
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || fail "${label} SHA-256 mismatch: ${actual}"
    echo "[VERIFY][OK] ${label} sha256=${actual}"
}

verify_s4_hash() {
    local path="$1" expected="$2" label="$3" actual
    actual="$(ssh "${SSH_OPTS[@]}" "${S4_HOST}" "sha256sum '$path' | awk '{print \$1}'")"
    [[ "${actual}" == "${expected}" ]] || fail "s4 ${label} SHA-256 mismatch: ${actual}"
    echo "[VERIFY][OK] s4 ${label} sha256=${actual}"
}

copy_local_file_to_s4() {
    local source="$1" destination="$2" expected="$3" label="$4"
    if ssh "${SSH_OPTS[@]}" "${S4_HOST}" "test -s '$destination'"; then
        verify_s4_hash "${destination}" "${expected}" "${label}"
        return
    fi
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" "mkdir -p '$(dirname "${destination}")'"
    rsync -a --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
        "${source}" "${S4_HOST}:${destination}.partial"
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" "mv '${destination}.partial' '${destination}'"
    verify_s4_hash "${destination}" "${expected}" "${label}"
}

stream_remote_file_to_s4() {
    local source="$1" destination="$2" expected="$3" label="$4"
    if ssh "${SSH_OPTS[@]}" "${S4_HOST}" "test -s '$destination'"; then
        verify_s4_hash "${destination}" "${expected}" "${label}"
        return
    fi
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "mkdir -p '$(dirname "${destination}")'; rm -f '${destination}.partial'"
    ssh -p "${INFERENCE_PORT}" "${SSH_OPTS[@]}" "${INFERENCE_HOST}" "cat '${source}'" \
        | ssh "${SSH_OPTS[@]}" "${S4_HOST}" "cat > '${destination}.partial'"
    verify_s4_hash "${destination}.partial" "${expected}" "${label} partial"
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" "mv '${destination}.partial' '${destination}'"
    verify_s4_hash "${destination}" "${expected}" "${label}"
}

transfer_dataset() {
    local dataset_name="$1"
    local destination_base="${S4_STORAGE_ROOT}/artifacts/datasets/seer/realworld"
    local destination="${destination_base}/${dataset_name}"
    local marker="${destination}/.transfer_verified"
    local staging="${destination_base}/.incoming_${dataset_name}"
    local manifest

    if ssh "${SSH_OPTS[@]}" "${S4_HOST}" "test -s '${marker}'"; then
        echo "[SKIP] verified dataset already present: ${destination}"
        return
    fi
    if ssh "${SSH_OPTS[@]}" "${S4_HOST}" "test -e '${destination}'"; then
        fail "unverified dataset destination already exists on s4: ${destination}"
    fi

    manifest="$(mktemp "/tmp/${dataset_name}.manifest.XXXXXX")"
    TEMP_PATHS+=("${manifest}")
    echo "[TRANSFER] building source checksum manifest for ${dataset_name}"
    ssh "${SSH_OPTS[@]}" "${S5_HOST}" \
        "cd '${S5_DATASET_BASE}' && find '${dataset_name}' -type f -print0 | sort -z | xargs -0 sha256sum" \
        > "${manifest}"
    [[ -s "${manifest}" ]] || fail "empty source manifest for ${dataset_name}"

    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "mkdir -p '${destination_base}'; rm -rf '${staging}'; mkdir -p '${staging}'"
    echo "[TRANSFER] streaming ${dataset_name} from s5 to s4"
    ssh "${SSH_OPTS[@]}" "${S5_HOST}" "tar -C '${S5_DATASET_BASE}' -cf - '${dataset_name}'" \
        | ssh "${SSH_OPTS[@]}" "${S4_HOST}" "tar -C '${staging}' -xf -"

    rsync -a -e "ssh ${SSH_OPTS[*]}" "${manifest}" \
        "${S4_HOST}:${S4_STORAGE_ROOT}/manifests/datasets/${dataset_name}.sha256"
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "cd '${staging}' && sha256sum --quiet -c '${S4_STORAGE_ROOT}/manifests/datasets/${dataset_name}.sha256'"
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "test ! -e '${destination}' && mv '${staging}/${dataset_name}' '${destination}' && rmdir '${staging}' && date --iso-8601=seconds > '${marker}'"
    echo "[VERIFY][OK] dataset transferred and checksummed: ${dataset_name}"
}

setup_environment() {
    local destination="${S4_STORAGE_ROOT}/envs/seer_libero"
    local marker="${destination}/.s4_relocation_complete"
    local archive remote_archive explicit pip_freeze

    if ssh "${SSH_OPTS[@]}" "${S4_HOST}" "test -s '${marker}'"; then
        echo "[SKIP] relocated Seer environment already present: ${destination}"
        return
    fi
    if ssh "${SSH_OPTS[@]}" "${S4_HOST}" "test -e '${destination}'"; then
        fail "unverified environment destination already exists on s4: ${destination}"
    fi

    archive="$(mktemp /tmp/seer_libero_s4.XXXXXX.tar.gz)"
    explicit="$(mktemp /tmp/seer_libero_explicit.XXXXXX.txt)"
    pip_freeze="$(mktemp /tmp/seer_libero_pip.XXXXXX.txt)"
    TEMP_PATHS+=("${archive}" "${explicit}" "${pip_freeze}")
    remote_archive="${S4_STORAGE_ROOT}/staging/seer_libero.tar.gz"

    echo "[ENV] packing sd1 seer_libero; the editable LIBERO package is intentionally excluded"
    "${CONDA_PACK}" -p "${SD1_ENV_PREFIX}" -o "${archive}" \
        --format tar.gz --compress-level 4 --ignore-editable-packages --force
    /home/mingyujung/miniconda3/bin/conda list -p "${SD1_ENV_PREFIX}" --explicit > "${explicit}"
    "${SD1_ENV_PREFIX}/bin/python" -m pip freeze > "${pip_freeze}"

    ssh "${SSH_OPTS[@]}" "${S4_HOST}" "mkdir -p '${S4_STORAGE_ROOT}/staging' '${S4_STORAGE_ROOT}/manifests/environment'"
    rsync -a --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
        "${archive}" "${S4_HOST}:${remote_archive}.partial"
    ssh "${SSH_OPTS[@]}" "${S4_HOST}" "mv '${remote_archive}.partial' '${remote_archive}'"
    rsync -a -e "ssh ${SSH_OPTS[*]}" "${explicit}" \
        "${S4_HOST}:${S4_STORAGE_ROOT}/manifests/environment/conda_explicit.txt"
    rsync -a -e "ssh ${SSH_OPTS[*]}" "${pip_freeze}" \
        "${S4_HOST}:${S4_STORAGE_ROOT}/manifests/environment/pip_freeze.txt"

    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "rm -rf '${destination}.incoming'; mkdir -p '${destination}.incoming'; tar -xzf '${remote_archive}' -C '${destination}.incoming'; '${destination}.incoming/bin/conda-unpack'; mv '${destination}.incoming' '${destination}'; rm -f '${remote_archive}'; date --iso-8601=seconds > '${marker}'"
    echo "[VERIFY][OK] relocated environment: ${destination}"
}

sync_source() {
    local origin_dir status_file diff_file
    origin_dir="$(mktemp -d /tmp/s4_source_origin.XXXXXX)"
    TEMP_PATHS+=("${origin_dir}")

    git -C "${SOURCE_REPO}" rev-parse HEAD > "${origin_dir}/origin_commit.txt"
    git -C "${SOURCE_REPO}" rev-parse --abbrev-ref HEAD > "${origin_dir}/origin_branch.txt"
    status_file="${origin_dir}/origin_git_status.txt"
    diff_file="${origin_dir}/origin_tracked_diff.patch"
    git -C "${SOURCE_REPO}" status --short > "${status_file}"
    git -C "${SOURCE_REPO}" diff --binary > "${diff_file}"
    date --iso-8601=seconds > "${origin_dir}/captured_at.txt"

    ssh "${SSH_OPTS[@]}" "${S4_HOST}" "mkdir -p '${S4_CODE_ROOT}/architectures' '${S4_CODE_ROOT}/methods' '${S4_CODE_ROOT}/tools'"
    rsync -a \
        --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
        --exclude='upstream/LIBERO_DATASETS' --exclude='upstream/checkpoints' \
        --exclude='upstream/wandb/' --exclude='upstream/runs/' \
        -e "ssh ${SSH_OPTS[*]}" \
        "${SOURCE_REPO}/architectures/seer/" "${S4_HOST}:${S4_CODE_ROOT}/architectures/seer/"

    local method
    for method in \
        decoder_guided_latent_dynamics \
        joint_latent_action_surrogate \
        latent_prediction_correction \
        latentloop_comparison \
        latentloop_horizon_regeneration \
        latentloop_plan_continuation \
        latentloop_segment_grid; do
        rsync -a --exclude='__pycache__/' --exclude='*.pyc' -e "ssh ${SSH_OPTS[*]}" \
            "${SOURCE_REPO}/methods/${method}/" "${S4_HOST}:${S4_CODE_ROOT}/methods/${method}/"
    done
    rsync -a --exclude='__pycache__/' --exclude='*.pyc' -e "ssh ${SSH_OPTS[*]}" \
        "${SOURCE_REPO}/tools/seer/" "${S4_HOST}:${S4_CODE_ROOT}/tools/seer/"
    rsync -a -e "ssh ${SSH_OPTS[*]}" \
        "${SOURCE_REPO}/AGENTS.md" "${SOURCE_REPO}/README.md" "${origin_dir}/" \
        "${S4_HOST}:${S4_CODE_ROOT}/"

    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "mkdir -p '${S4_CODE_ROOT}/architectures/seer/upstream/checkpoints/clip' '${S4_CODE_ROOT}/architectures/seer/upstream/checkpoints/vit_mae'; ln -sfn '${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/clip/ViT-B-32.pt' '${S4_CODE_ROOT}/architectures/seer/upstream/checkpoints/clip/ViT-B-32.pt'; ln -sfn '${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/vit_mae/mae_pretrain_vit_base.pth' '${S4_CODE_ROOT}/architectures/seer/upstream/checkpoints/vit_mae/mae_pretrain_vit_base.pth'"

    ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
        "cd '${S4_CODE_ROOT}' && find AGENTS.md README.md architectures/seer methods tools/seer -type f -print0 | sort -z | xargs -0 sha256sum > SOURCE_SNAPSHOT.sha256 && if test ! -d .git; then git init -q -b s4-realworld-latentloop; fi && git add AGENTS.md README.md SOURCE_SNAPSHOT.sha256 origin_commit.txt origin_branch.txt origin_git_status.txt origin_tracked_diff.patch captured_at.txt architectures/seer methods tools/seer && git -c user.name='Gnaroshi source snapshot' -c user.email='snapshot@local.invalid' commit -q --allow-empty -m 'Snapshot Seer LatentLoop real-world training source'"
    echo "[VERIFY][OK] minimal Seer source synced: ${S4_HOST}:${S4_CODE_ROOT}"
}

[[ "$(hostname)" == "jbrserver1" ]] || fail "run this setup script on sd1/jbrserver1"
[[ -d "${SOURCE_REPO}/architectures/seer/upstream" ]] || fail "invalid SOURCE_REPO: ${SOURCE_REPO}"
require_file "${CONDA_PACK}"
verify_local_hash "${SD1_VIT}" "${VIT_SHA256}" "ViT-MAE"
verify_local_hash "${SD1_CLIP}" "${CLIP_SHA256}" "CLIP"

ssh "${SSH_OPTS[@]}" "${S4_HOST}" true
if ! ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
    "test -d '${S4_STORAGE_ROOT}' && test -w '${S4_STORAGE_ROOT}'"; then
    cat >&2 <<EOF
[ERROR] s4 storage root is not writable: ${S4_STORAGE_ROOT}
Run this once on s4, then rerun this setup script on sd1:

  sudo mkdir -p ${S4_STORAGE_ROOT}
  sudo chown -R mingyujung:mingyujung ${S4_STORAGE_ROOT}
EOF
    exit 1
fi

ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
    "mkdir -p '${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/realworld/droid38/doll' '${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/realworld/droid38/cabinet' '${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/vit_mae' '${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/clip' '${S4_STORAGE_ROOT}/artifacts/datasets/seer/realworld' '${S4_STORAGE_ROOT}/results/seer/latentloop/realworld/droid38' '${S4_STORAGE_ROOT}/manifests/datasets'"

sync_source
copy_local_file_to_s4 "${SD1_VIT}" \
    "${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/vit_mae/mae_pretrain_vit_base.pth" \
    "${VIT_SHA256}" "ViT-MAE"
copy_local_file_to_s4 "${SD1_CLIP}" \
    "${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/clip/ViT-B-32.pt" \
    "${CLIP_SHA256}" "CLIP"

stream_remote_file_to_s4 "${DOLL_TEACHER_SOURCE}" \
    "${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/realworld/droid38/doll/38.pth" \
    "${DOLL_TEACHER_SHA256}" "Doll teacher 38"
stream_remote_file_to_s4 "${CABINET_TEACHER_SOURCE}" \
    "${S4_STORAGE_ROOT}/artifacts/checkpoints/seer/realworld/droid38/cabinet/38.pth" \
    "${CABINET_TEACHER_SHA256}" "Cabinet teacher 38"

transfer_dataset doll_filtered_40p
transfer_dataset cabinet_filtered_40p
setup_environment

ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
    "chmod +x '${S4_CODE_ROOT}/architectures/seer/wrappers/lrnode/train_realworld_latentloop_adapter.sh' '${S4_CODE_ROOT}/architectures/seer/wrappers/lrnode/train_realworld_doll_latentloop.sh' '${S4_CODE_ROOT}/architectures/seer/wrappers/lrnode/train_realworld_cabinet_latentloop.sh' '${S4_CODE_ROOT}/tools/seer/verify_realworld_latentloop_assets.py'"

ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
    "cd '${S4_CODE_ROOT}' && CUDA_VISIBLE_DEVICES=0,1,2,3 PREFLIGHT_ONLY=1 REPORT_TO_WANDB=0 bash architectures/seer/wrappers/lrnode/train_realworld_doll_latentloop.sh"
ssh "${SSH_OPTS[@]}" "${S4_HOST}" \
    "cd '${S4_CODE_ROOT}' && CUDA_VISIBLE_DEVICES=4,5,6,7 PREFLIGHT_ONLY=1 REPORT_TO_WANDB=0 bash architectures/seer/wrappers/lrnode/train_realworld_cabinet_latentloop.sh"

echo "[DONE] s4 real-world LatentLoop environment and both task inputs are verified"
echo "[DONE] code=${S4_HOST}:${S4_CODE_ROOT}"
echo "[DONE] storage=${S4_HOST}:${S4_STORAGE_ROOT}"
