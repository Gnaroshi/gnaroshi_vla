#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
preset="${BRIDGE_PRESET:-full}"
r0="${LATENT_BRIDGE_RESULT_ROOT}/r0_${preset}/training/best.pt"
dagger="${LATENT_BRIDGE_RESULT_ROOT}/dagger_${preset}_f3"
sync="${LATENT_BRIDGE_RESULT_ROOT}/continuity_and_sync/sync_300/shards"
[[ -s "${dagger}/COMPLETE" ]] || latent_bridge_fail "DAgger stage is incomplete"
stage="${LATENT_BRIDGE_RESULT_ROOT}/r1_${preset}"
if ! latent_bridge_prepare_stage "${stage}"; then exit 0; fi
base_port="${MASTER_PORT_BASE:-18100}"
master_port="${MASTER_PORT:-$((base_port + 40))}"
torchrun --nnodes=1 --nproc_per_node=4 --master_port="${master_port}" \
    --module architectures.seer.adapters.latent_bridge.train \
    --stage R1 --sync-root "${sync}" --dagger-root "${dagger}/shards" \
    --initial-checkpoint "${r0}" --output-dir "${stage}/training" \
    --official-source "${LATENT_BRIDGE_OFFICIAL_SOURCE}" --preset "${preset}" \
    --epochs 100 --learning-rate 3e-5 --per-rank-batch 2 \
    --gradient-accumulation-steps 8 --workers "${TRAIN_WORKERS:-4}" --precision bf16 2>&1 | \
    tee "${stage}/train.log"
[[ -s "${stage}/training/COMPLETE" ]] || latent_bridge_fail "R1 trainer did not complete"
printf 'SEER_LATENT_BRIDGE_R1_COMPLETE\n' > "${stage}/COMPLETE"
