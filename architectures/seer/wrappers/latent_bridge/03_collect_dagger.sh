#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
r0="${LATENT_BRIDGE_RESULT_ROOT}/r0_${BRIDGE_PRESET:-full}/training/best.pt"
[[ -s "${r0}" ]] || latent_bridge_fail "R0 best checkpoint is absent: ${r0}"
stage="${LATENT_BRIDGE_RESULT_ROOT}/dagger_${BRIDGE_PRESET:-full}_f3"
if ! latent_bridge_prepare_stage "${stage}"; then exit 0; fi
export SEER_LATENT_BRIDGE_CHECKPOINT="${r0}"
export SEER_LATENT_BRIDGE_REFRESH_PERIOD=3
export SEER_LATENT_BRIDGE_PRECISION=bf16
export SEER_LATENT_BRIDGE_COMPILE=1
export SEER_LATENT_BRIDGE_DAGGER_OUTPUT="${stage}/shards"
base_port="${MASTER_PORT_BASE:-18100}"
master_port="${MASTER_PORT:-$((base_port + 30))}"
latent_bridge_run_eval \
    architectures.seer.adapters.latent_bridge.evaluation_entry \
    "${stage}" dagger_f3_public33 42 30 10 "${master_port}"
find "${stage}/shards" -name 'dagger_transitions_rank*.h5.manifest.json' -type f | sort > \
    "${stage}/transition_manifests.txt"
[[ "$(wc -l < "${stage}/transition_manifests.txt")" -eq 4 ]] || \
    latent_bridge_fail "expected four DAgger transition manifests"
printf 'SEER_LATENT_BRIDGE_DAGGER_COMPLETE\n' > "${stage}/COMPLETE"
