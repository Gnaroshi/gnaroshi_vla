#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
source_stage="${LATENT_BRIDGE_RESULT_ROOT}/continuity_and_sync"
[[ -s "${source_stage}/COMPLETE" ]] || latent_bridge_fail "continuity/sync stage is incomplete"
stage="${LATENT_BRIDGE_RESULT_ROOT}/r0_${BRIDGE_PRESET:-full}"
if ! latent_bridge_prepare_stage "${stage}" resume; then exit 0; fi
readarray -t decision < "${source_stage}/stable_context.txt"
base_port="${MASTER_PORT_BASE:-18100}"
master_port="${MASTER_PORT:-$((base_port + 20))}"

torchrun --nnodes=1 --nproc_per_node=4 --master_port="${master_port}" \
    --module architectures.seer.adapters.latent_bridge.train \
    --stage R0 --sync-root "${source_stage}/sync_300/shards" \
    --output-dir "${stage}/training" --official-source "${LATENT_BRIDGE_OFFICIAL_SOURCE}" \
    --preset "${BRIDGE_PRESET:-full}" --stable-layer "${decision[0]}" \
    --stable-token-group "${decision[1]}" --stable-seq-len "${decision[2]}" \
    --epochs 200 --learning-rate 3e-4 --per-rank-batch 2 \
    --gradient-accumulation-steps 8 --workers "${TRAIN_WORKERS:-4}" --precision bf16 \
    --resume-checkpoint "${stage}/training/resume.pt" \
    --checkpoint-every-epochs "${TRAIN_CHECKPOINT_EVERY_EPOCHS:-1}" 2>&1 | \
    tee -a "${stage}/train.log"
[[ -s "${stage}/training/COMPLETE" ]] || latent_bridge_fail "R0 trainer did not complete"
printf 'SEER_LATENT_BRIDGE_R0_COMPLETE\n' > "${stage}/COMPLETE"
