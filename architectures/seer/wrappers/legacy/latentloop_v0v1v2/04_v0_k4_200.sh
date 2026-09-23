#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_four_gpu_lane
require_gate "${CAMPAIGN_ROOT}/gates/full_k1_200_complete.json" CANONICAL_EVAL_ROW_COMPLETE
RESULT="${CAMPAIGN_ROOT}/v0/reproduction/v0_k4"
MARKER="${CAMPAIGN_ROOT}/gates/v0_k4_200_complete.json"
refuse_existing "${MARKER}"
run_v0_eval "${RESULT}" 20 10 0 0 "4" "${MASTER_PORT:-16040}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/record_canonical_eval_row.py" \
  --row-root "${RESULT}" --row-id v0_k4 --expected-episodes 200 \
  --expected-query-interval 4 --expected-lrnode-enabled 1 --output "${MARKER}"
