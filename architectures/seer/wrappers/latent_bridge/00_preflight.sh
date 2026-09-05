#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime

stage="${LATENT_BRIDGE_RESULT_ROOT}/preflight"
if ! latent_bridge_prepare_stage "${stage}"; then exit 0; fi

python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/preflight.py" \
    --repo-root "${LATENT_BRIDGE_REPO_ROOT}" \
    --official-source "${LATENT_BRIDGE_OFFICIAL_SOURCE}" \
    --checkpoint "${LATENT_BRIDGE_BASE_CHECKPOINT}" \
    --checkpoint-sha256 "${LATENT_BRIDGE_BASE_CHECKPOINT_SHA256}" \
    --vit-checkpoint "${LATENT_BRIDGE_VIT}" \
    --dataset-root "${LATENT_BRIDGE_DATASET_ROOT}" \
    --dataset-name "${LATENT_BRIDGE_DATASET_NAME}" \
    --dataset-info "${LATENT_BRIDGE_DATASET_INFO}" \
    --suite "${LATENT_BRIDGE_SUITE}" \
    --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" \
    --renderer "${LATENT_BRIDGE_RENDERER}" \
    --visible-devices "${CUDA_VISIBLE_DEVICES}" \
    --output "${stage}/source_and_contract.json"

python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/verify_condition_hook.py" \
    --checkpoint "${LATENT_BRIDGE_BASE_CHECKPOINT}" \
    --checkpoint-sha256 "${LATENT_BRIDGE_BASE_CHECKPOINT_SHA256}" \
    --vit-checkpoint "${LATENT_BRIDGE_VIT}" \
    --dataset-root "${LATENT_BRIDGE_DATASET_ROOT}" \
    --dataset-name "${LATENT_BRIDGE_DATASET_NAME}" \
    --dataset-info "${LATENT_BRIDGE_DATASET_INFO}" \
    --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" \
    --output-dir "${stage}/hook_equivalence" --device cuda:0

python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/bridge_smoke.py" \
    --preset full --stable-seq-len 14 --batch-size 2 --device cuda:0 --compile \
    --output "${stage}/full_bridge_gpu_smoke.json"
python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/bridge_smoke.py" \
    --preset small --stable-seq-len 14 --batch-size 2 --device cuda:0 --compile \
    --output "${stage}/small_bridge_gpu_compile_smoke.json"
python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/renderer_smoke.py" \
    --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" \
    --renderer "${LATENT_BRIDGE_RENDERER}" \
    --suite "${LATENT_BRIDGE_SUITE}" \
    --output "${stage}/renderer_smoke.json"

PYTHONDONTWRITEBYTECODE=1 pytest -q "${LATENT_BRIDGE_REPO_ROOT}/tests/seer_latent_bridge" | \
    tee "${stage}/pytest.log"
printf 'SEER_LATENT_BRIDGE_PREFLIGHT_COMPLETE\n' > "${stage}/COMPLETE"
