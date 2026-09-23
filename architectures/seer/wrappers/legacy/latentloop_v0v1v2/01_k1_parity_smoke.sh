#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_four_gpu_lane
require_gate "${PREFLIGHT_ARTIFACT}" S18_CANONICAL_PREFLIGHT_PASS
RESULT="${CAMPAIGN_ROOT}/v0/gate0_k1_parity_smoke"
refuse_existing "${K1_GATE}"
run_v0_eval "${RESULT}" 4 1 1 1 "" "${MASTER_PORT:-16010}"
shopt -s nullglob
BASE=("${RESULT}"/baseline_*_full_K1_*)
CAND=("${RESULT}"/ours_*_full_K1_*)
[[ ${#BASE[@]} -eq 1 && ${#CAND[@]} -eq 1 ]] || fail "unexpected K1 output layout"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/audit_canonical_k1_parity.py" \
  --baseline-root "${BASE[0]}" --candidate-root "${CAND[0]}" \
  --tolerance 1e-6 --output "${K1_GATE}"
