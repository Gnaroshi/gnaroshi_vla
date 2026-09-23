#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_four_gpu_lane
require_gate "${K4_GATE}" K4_CALL_PATH_PASS
RESULT="${CAMPAIGN_ROOT}/v0/reproduction/full_k1"
MARKER="${CAMPAIGN_ROOT}/gates/full_k1_200_complete.json"
refuse_existing "${MARKER}"
run_v0_eval "${RESULT}" 20 10 1 0 "" "${MASTER_PORT:-16030}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/record_canonical_eval_row.py" \
  --row-root "${RESULT}" --row-id full_k1 --expected-episodes 200 \
  --expected-query-interval 1 --expected-lrnode-enabled 0 --output "${MARKER}"
