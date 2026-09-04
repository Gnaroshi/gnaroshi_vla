#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
latent_bridge_efficient_verify_source_lock
latent_bridge_efficient_load_profile

preset="${BRIDGE_PRESET:-full}"
r0="${LATENT_BRIDGE_RESULT_ROOT}/r0_${preset}/training/best.pt"
dagger="${LATENT_BRIDGE_RESULT_ROOT}/dagger_${preset}_f3"
sync="${LATENT_BRIDGE_EFFICIENT_SYNC_STAGE}/sync_300/shards"
[[ -s "${dagger}/COMPLETE" ]] || latent_bridge_fail "efficient DAgger stage is incomplete"
stage="${LATENT_BRIDGE_RESULT_ROOT}/r1_${preset}"
if ! latent_bridge_prepare_stage "${stage}" resume; then exit 0; fi
master_port="${MASTER_PORT:-$(( ${MASTER_PORT_BASE:-18200} + 40 ))}"

torchrun --nnodes=1 --nproc_per_node=4 --master_port="${master_port}" \
    --module architectures.seer.adapters.latent_bridge.train_efficient \
    --stage R1 --sync-root "${sync}" --dagger-root "${dagger}/shards" \
    --initial-checkpoint "${r0}" --output-dir "${stage}/training" \
    --official-source "${LATENT_BRIDGE_OFFICIAL_SOURCE}" --preset "${preset}" \
    --optimizer-steps "${R1_OPTIMIZER_STEPS:-44000}" --learning-rate 3e-5 \
    --per-rank-batch "${TRAIN_PER_RANK_BATCH}" \
    --gradient-accumulation-steps "${TRAIN_GRADIENT_ACCUMULATION_STEPS}" \
    --validation-batch-size "${VALIDATION_BATCH_SIZE:-32}" \
    --validation-every-data-epochs "${VALIDATION_EVERY_DATA_EPOCHS:-4}" \
    --checkpoint-every-data-epochs "${CHECKPOINT_EVERY_DATA_EPOCHS:-4}" \
    --log-every-updates "${LOG_EVERY_UPDATES:-100}" --precision bf16 \
    --preload-dataset --resume-checkpoint "${stage}/training/resume.pt" 2>&1 | \
    tee -a "${stage}/train.log"
[[ -s "${stage}/training/COMPLETE" ]] || latent_bridge_fail "efficient R1 did not complete"
printf 'SEER_LATENT_BRIDGE_EFFICIENT_R1_COMPLETE\n' > "${stage}/COMPLETE"
