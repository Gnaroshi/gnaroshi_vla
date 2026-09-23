#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_gate "${CAMPAIGN_ROOT}/v2/gates/v2_online_gate.json" V2_TARGET_K4_PASS
OUTPUT="${CAMPAIGN_ROOT}/analysis/final_v0v1v2_results.json"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/aggregate_s18_v0v1v2.py" \
  --campaign-root "${CAMPAIGN_ROOT}" --output "${OUTPUT}"
