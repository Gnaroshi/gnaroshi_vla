#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_four_gpu_lane
require_gate "${K1_GATE}" K1_PARITY_PASS
RESULT="${CAMPAIGN_ROOT}/v0/gate1_k4_call_path_smoke"
refuse_existing "${K4_GATE}"
run_v0_eval "${RESULT}" 2 1 0 0 "4" "${MASTER_PORT:-16020}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/audit_k4_call_path.py" \
  --row-root "${RESULT}" --repo-root "${REPO_ROOT}" --audited-episodes 2 \
  --output "${K4_GATE}"
