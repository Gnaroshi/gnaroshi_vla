#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_gate "${CAMPAIGN_ROOT}/gates/full_k1_200_complete.json" CANONICAL_EVAL_ROW_COMPLETE
require_gate "${CAMPAIGN_ROOT}/gates/v0_k4_200_complete.json" CANONICAL_EVAL_ROW_COMPLETE
ANALYSIS="${CAMPAIGN_ROOT}/gates/canonical_reproduction_analysis.json"
refuse_existing "${ANALYSIS}"
refuse_existing "${REPRO_DECISION}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/analyze_canonical_reproduction.py" \
  --baseline-root "${CAMPAIGN_ROOT}/v0/reproduction/full_k1" \
  --v0-root "${CAMPAIGN_ROOT}/v0/reproduction/v0_k4" \
  --canonical-v0-episodes "${REPO_ROOT}/.canonical/reference/v0_k4_eval_episode_metrics.csv" \
  --bootstrap-iterations 10000 --bootstrap-seed 42 --output "${ANALYSIS}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/decide_canonical_reproduction.py" \
  --analysis "${ANALYSIS}" --output "${REPRO_DECISION}"
