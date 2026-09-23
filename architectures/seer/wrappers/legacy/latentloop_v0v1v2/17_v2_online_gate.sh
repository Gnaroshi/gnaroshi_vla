#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_gate "${CAMPAIGN_ROOT}/v2/calibration/v2_threshold_lock.json" V2_THRESHOLD_LOCKED
[[ -f "${CAMPAIGN_ROOT}/v1/eval/fixed_k4_200/analysis/eval_summary.json" ]] || fail "V1 fixed-K4 row must finish first"
[[ -f "${CAMPAIGN_ROOT}/v2/eval/target_k4_200/analysis/eval_summary.json" ]] || fail "V2 target-K4 row must finish first"
[[ -f "${CAMPAIGN_ROOT}/v2/eval/random_matched_budget_200/analysis/eval_summary.json" ]] || fail "matched-random row must finish first"
OUTPUT="${CAMPAIGN_ROOT}/v2/gates/v2_online_gate.json"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/check_latentloop_v2_online_gate.py" \
  --v1-root "${CAMPAIGN_ROOT}/v1/eval/fixed_k4_200" \
  --v2-root "${CAMPAIGN_ROOT}/v2/eval/target_k4_200" \
  --random-root "${CAMPAIGN_ROOT}/v2/eval/random_matched_budget_200" --output "${OUTPUT}"
