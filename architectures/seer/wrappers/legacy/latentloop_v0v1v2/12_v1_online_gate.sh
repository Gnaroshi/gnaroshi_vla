#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
OUTPUT="${CAMPAIGN_ROOT}/v1/gates/v1_online_gate.json"
[[ -f "${CAMPAIGN_ROOT}/v1/eval/fixed_k4_200/analysis/eval_summary.json" ]] || fail "V1 K4 evaluation is incomplete"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/check_latentloop_v1_online_gate.py" \
  --v0-root "${CAMPAIGN_ROOT}/v0/reproduction/v0_k4" \
  --v1-root "${CAMPAIGN_ROOT}/v1/eval/fixed_k4_200" --output "${OUTPUT}"
