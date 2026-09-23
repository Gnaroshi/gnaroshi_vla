#!/usr/bin/env bash

set -euo pipefail

wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${wrapper_dir}/../../../.." && pwd)"
expected_repo="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latent_bridge_paper}"
[[ "$(hostname)" == "${EXPECTED_HOST:-jbrserver1}" ]] || {
    echo "[ERROR] this launcher is locked to sd1" >&2
    exit 1
}
[[ "$(readlink -f "${repo_root}")" == "$(readlink -f "${expected_repo}")" ]] || {
    echo "[ERROR] unexpected source: ${repo_root}" >&2
    exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3" ]] || {
    echo "[ERROR] set CUDA_VISIBLE_DEVICES=0,1,2,3" >&2
    exit 1
}

export LATENT_BRIDGE_EXPECTED_CONDA_ENV=seer_libero
export LATENT_BRIDGE_EXPECTED_CUDA_VISIBLE_DEVICES=0,1,2,3
paper_checkpoint_root=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/artifacts/checkpoints/seer/paper
export LATENT_BRIDGE_BASE_CHECKPOINT="${paper_checkpoint_root}/libero_long/teacher_public33.pth"
export LATENT_BRIDGE_BASE_CHECKPOINT_SHA256=a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646
export LATENT_BRIDGE_VIT=/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth
export LATENT_BRIDGE_DATASET_ROOT=/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/LIBERO_DATASETS/libero_10_converted
export LATENT_BRIDGE_DATASET_NAME=libero_10_converted
export LATENT_BRIDGE_DATASET_INFO="${repo_root}/architectures/seer/upstream/data_info/libero_10_converted.json"
export LATENT_BRIDGE_LIBERO_PATH=/home/mingyujung/private/LIBERO
export LATENT_BRIDGE_RESULT_ROOT="${LATENT_BRIDGE_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/seer/latent_bridge/libero_long_public33_v1}"
export LATENT_BRIDGE_R1_CHECKPOINT="${paper_checkpoint_root}/libero_long/latent_bridge_large_best.pt"
export LATENT_BRIDGE_RENDERER=egl
export LATENT_BRIDGE_SUITE=libero_10
export LATENT_BRIDGE_RUN_LABEL=seer_public33_latent_bridge
export SEER_LATENT_BRIDGE_DETERMINISTIC=0
export EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}"
export REFRESH_PERIODS="${REFRESH_PERIODS:-3 4}"
export EVAL_EPISODES_PER_TASK=50
export EVAL_NUM_TASKS=10
export RUN_BASELINE=1
export RUN_COMPONENT_LATENCY=1
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-18800}"
export NODE_NUM=4

source "${wrapper_dir}/../latent_bridge/common.sh"
latent_bridge_require_runtime
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=no)" ]] || \
    latent_bridge_fail "source worktree has tracked modifications"

mkdir -p "${LATENT_BRIDGE_RESULT_ROOT}"
exec 9>"${LATENT_BRIDGE_RESULT_ROOT}/pipeline.lock"
flock -n 9 || latent_bridge_fail "another Long evaluation owns the result root"
exec > >(tee -a "${LATENT_BRIDGE_RESULT_ROOT}/pipeline.log") 2>&1

preflight="${LATENT_BRIDGE_RESULT_ROOT}/eval_preflight"
if [[ ! -s "${preflight}/COMPLETE" ]]; then
    [[ ! -e "${preflight}" ]] || latent_bridge_archive_partial "${preflight}"
    mkdir -p "${preflight}"
    python "${repo_root}/tools/seer_latent_bridge/renderer_smoke.py" \
        --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" --renderer egl \
        --suite libero_10 --image-size 128 --output "${preflight}/renderer_smoke.json"
    python - "${LATENT_BRIDGE_R1_CHECKPOINT}" <<'PY'
from architectures.seer.adapters.latent_bridge.checkpoint import (
    load_bridge_checkpoint,
    validate_bridge_runtime_provenance,
)
import sys
_, payload = load_bridge_checkpoint(sys.argv[1], map_location="cpu")
print(validate_bridge_runtime_provenance(payload))
PY
    printf 'SEER_LATENT_BRIDGE_LONG_EVAL_PREFLIGHT_COMPLETE\n' > "${preflight}/COMPLETE"
fi

bash "${wrapper_dir}/evaluate.sh"
printf 'SEER_LATENT_BRIDGE_LONG_PAPER_RUN_COMPLETE\n' > \
    "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_COMPLETE"
echo "[DONE] Long paper evaluation: ${LATENT_BRIDGE_RESULT_ROOT}"
