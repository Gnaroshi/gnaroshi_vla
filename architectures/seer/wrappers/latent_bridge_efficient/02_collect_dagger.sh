#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
latent_bridge_efficient_verify_source_lock

r0="${LATENT_BRIDGE_RESULT_ROOT}/r0_${BRIDGE_PRESET:-full}/training/best.pt"
[[ -s "${r0}" ]] || latent_bridge_fail "efficient R0 best checkpoint is absent: ${r0}"
stage="${LATENT_BRIDGE_RESULT_ROOT}/dagger_${BRIDGE_PRESET:-full}_f3"
if [[ -s "${stage}/COMPLETE" ]]; then
    echo "[SKIP] completed stage: ${stage}"
    exit 0
fi
if ! latent_bridge_eval_row_is_complete "${stage}" 42 30 10; then
    latent_bridge_prepare_eval_row "${stage}" 42 30 10
    export SEER_LATENT_BRIDGE_CHECKPOINT="${r0}"
    export SEER_LATENT_BRIDGE_REFRESH_PERIOD=3
    export SEER_LATENT_BRIDGE_PRECISION=bf16
    export SEER_LATENT_BRIDGE_COMPILE=1
    export SEER_LATENT_BRIDGE_DAGGER_OUTPUT="${stage}/shards"
    master_port="${MASTER_PORT:-$(( ${MASTER_PORT_BASE:-18200} + 30 ))}"
    latent_bridge_run_eval \
        architectures.seer.adapters.latent_bridge.evaluation_entry \
        "${stage}" dagger_f3_public33_compute_matched 42 30 10 "${master_port}"
fi
find "${stage}/shards" -name 'dagger_transitions_rank*.h5.manifest.json' -type f | sort > \
    "${stage}/transition_manifests.txt"
[[ "$(wc -l < "${stage}/transition_manifests.txt")" -eq 4 ]] || \
    latent_bridge_fail "expected four DAgger transition manifests"
printf 'SEER_LATENT_BRIDGE_EFFICIENT_DAGGER_COMPLETE\n' > "${stage}/COMPLETE"
