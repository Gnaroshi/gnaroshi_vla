#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
[[ -s "${LATENT_BRIDGE_RESULT_ROOT}/preflight/COMPLETE" ]] || \
    latent_bridge_fail "preflight must complete first"

stage="${LATENT_BRIDGE_RESULT_ROOT}/continuity_and_sync"
if ! latent_bridge_prepare_stage "${stage}"; then exit 0; fi

calibration="${stage}/continuity"
mkdir -p "${calibration}"
export LATENT_BRIDGE_CONTINUITY_OUTPUT="${calibration}/shards"
export LATENT_BRIDGE_CONTINUITY_LAYERS="0,3,7,11,15,19,23"
export LATENT_BRIDGE_CONTINUITY_GROUPS="text,state,primary_resampled,wrist_resampled,primary_cls,wrist_cls,visual,multimodal_context,observation_prediction,action,all"
export LATENT_BRIDGE_CONTINUITY_MAX_OFFSET=4
export LATENT_BRIDGE_CONTINUITY_METRICS=1
export LATENT_BRIDGE_COLLECT_TRANSITIONS=0
export LATENT_BRIDGE_STABLE_LAYER=block_00
export LATENT_BRIDGE_STABLE_GROUP=visual
latent_bridge_run_eval \
    architectures.seer.adapters.latent_bridge.continuity_entry \
    "${calibration}" continuity_public33 42 2 10 "${MASTER_PORT_BASE:-18100}"
python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/aggregate_continuity.py" \
    --input-dir "${calibration}/shards"

readarray -t decision < <(python - "${calibration}/shards/seer_latent_continuity_summary.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
d=p["selected_stable_context"]
s=p["token_layout"]["per_timestep_slices"][d["token_group"]]
print(d["layer"])
print(d["token_group"])
print(s[1]-s[0])
PY
)
stable_layer="${decision[0]}"
stable_group="${decision[1]}"
stable_len="${decision[2]}"
printf '%s\n%s\n%s\n' "${stable_layer}" "${stable_group}" "${stable_len}" > "${stage}/stable_context.txt"

sync="${stage}/sync_300"
mkdir -p "${sync}"
export LATENT_BRIDGE_CONTINUITY_OUTPUT="${sync}/shards"
export LATENT_BRIDGE_CONTINUITY_LAYERS=""
export LATENT_BRIDGE_CONTINUITY_GROUPS=""
export LATENT_BRIDGE_CONTINUITY_METRICS=0
export LATENT_BRIDGE_COLLECT_TRANSITIONS=1
export LATENT_BRIDGE_STABLE_LAYER="${stable_layer}"
export LATENT_BRIDGE_STABLE_GROUP="${stable_group}"
latent_bridge_run_eval \
    architectures.seer.adapters.latent_bridge.continuity_entry \
    "${sync}" sync_public33 42 30 10 "$(( ${MASTER_PORT_BASE:-18100} + 10 ))"
find "${sync}/shards" -name 'sync_transitions_rank*.h5.manifest.json' -type f | sort > \
    "${sync}/transition_manifests.txt"
[[ "$(wc -l < "${sync}/transition_manifests.txt")" -eq 4 ]] || \
    latent_bridge_fail "expected four synchronized transition manifests"
printf 'SEER_LATENT_BRIDGE_CONTINUITY_AND_SYNC_COMPLETE\n' > "${stage}/COMPLETE"
