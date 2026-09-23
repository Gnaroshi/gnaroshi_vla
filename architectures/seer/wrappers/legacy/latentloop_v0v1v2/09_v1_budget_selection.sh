#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
E20="${CAMPAIGN_ROOT}/v1/train/e20/validation_metrics.json"
E40="${CAMPAIGN_ROOT}/v1/train/e40/validation_metrics.json"
OUTPUT="${CAMPAIGN_ROOT}/v1/selection/v1_budget_selection.json"
[[ -f "${E20}" && -f "${E40}" ]] || fail "both independent validation metrics are required"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/select_latentloop_v1_budget.py" \
  --e20 "${E20}" --e40 "${E40}" --output "${OUTPUT}"
