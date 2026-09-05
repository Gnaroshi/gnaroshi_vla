#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
latent_bridge_efficient_verify_source_lock

stage="${LATENT_BRIDGE_EFFICIENT_PREFLIGHT_ROOT}"
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

profile_batch=16
profile_accumulation=1
if ! torchrun --nnodes=1 --nproc_per_node=4 \
    --master_port="$(( ${MASTER_PORT_BASE:-18200} + 1 ))" \
    "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/ddp_training_smoke.py" \
    --preset full --stable-seq-len 14 --batch-size 16 \
    --output "${stage}/full_bridge_batch16_ddp_smoke.json" 2>&1 | \
    tee "${stage}/full_bridge_batch16_ddp_smoke.log"; then
    echo "[PROFILE] batch 16 failed; validating batch 8 with accumulation 2"
    profile_batch=8
    profile_accumulation=2
    torchrun --nnodes=1 --nproc_per_node=4 \
        --master_port="$(( ${MASTER_PORT_BASE:-18200} + 2 ))" \
        "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/ddp_training_smoke.py" \
        --preset full --stable-seq-len 14 --batch-size 8 \
        --output "${stage}/full_bridge_batch8_ddp_smoke.json" 2>&1 | \
        tee "${stage}/full_bridge_batch8_ddp_smoke.log"
fi

printf 'TRAIN_PER_RANK_BATCH=%s\nTRAIN_GRADIENT_ACCUMULATION_STEPS=%s\n' \
    "${profile_batch}" "${profile_accumulation}" > "${stage}/training_profile.env"

python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/preflight_efficient.py" \
    --sync-stage "${LATENT_BRIDGE_EFFICIENT_SYNC_STAGE}" \
    --result-root "${LATENT_BRIDGE_RESULT_ROOT}" \
    --world-size 4 --per-rank-batch "${profile_batch}" \
    --gradient-accumulation-steps "${profile_accumulation}" \
    --r0-optimizer-steps "${R0_OPTIMIZER_STEPS:-50200}" \
    --r1-optimizer-steps "${R1_OPTIMIZER_STEPS:-44000}" \
    --output "${stage}/efficient_contract.json"

PYTHONDONTWRITEBYTECODE=1 pytest -q \
    "${LATENT_BRIDGE_REPO_ROOT}/tests/seer_latent_bridge" | tee "${stage}/pytest.log"
printf 'SEER_LATENT_BRIDGE_EFFICIENT_PREFLIGHT_COMPLETE\n' > "${stage}/COMPLETE"
