#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
latent_bridge_efficient_verify_source_lock
latent_bridge_efficient_load_profile

source_stage="${LATENT_BRIDGE_EFFICIENT_SYNC_STAGE}"
stage="${LATENT_BRIDGE_RESULT_ROOT}/r0_${BRIDGE_PRESET:-full}"
if ! latent_bridge_prepare_stage "${stage}" resume; then exit 0; fi
readarray -t decision < "${source_stage}/stable_context.txt"
master_port="${MASTER_PORT:-$(( ${MASTER_PORT_BASE:-18200} + 20 ))}"

torchrun --nnodes=1 --nproc_per_node=4 --master_port="${master_port}" \
    --module architectures.seer.adapters.latent_bridge.train_efficient \
    --stage R0 --sync-root "${source_stage}/sync_300/shards" \
    --output-dir "${stage}/training" --official-source "${LATENT_BRIDGE_OFFICIAL_SOURCE}" \
    --preset "${BRIDGE_PRESET:-full}" --stable-layer "${decision[0]}" \
    --stable-token-group "${decision[1]}" --stable-seq-len "${decision[2]}" \
    --optimizer-steps "${R0_OPTIMIZER_STEPS:-50200}" --learning-rate 3e-4 \
    --per-rank-batch "${TRAIN_PER_RANK_BATCH}" \
    --gradient-accumulation-steps "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" \
    --validation-batch-size "${VALIDATION_BATCH_SIZE:-32}" \
    --validation-every-data-epochs "${VALIDATION_EVERY_DATA_EPOCHS:-4}" \
    --checkpoint-every-data-epochs "${CHECKPOINT_EVERY_DATA_EPOCHS:-4}" \
    --log-every-updates "${LOG_EVERY_UPDATES:-100}" --precision bf16 \
    --preload-dataset --resume-checkpoint "${stage}/training/resume.pt" 2>&1 | \
    tee -a "${stage}/train.log"
[[ -s "${stage}/training/COMPLETE" ]] || latent_bridge_fail "efficient R0 did not complete"
printf 'SEER_LATENT_BRIDGE_EFFICIENT_R0_COMPLETE\n' > "${stage}/COMPLETE"
