#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_gate "${CAMPAIGN_ROOT}/v1/gates/v1_online_gate.json" V1_ONLINE_PASS
TRACE="${CAMPAIGN_ROOT}/v2/defect/heldout_defect_trace.jsonl"
OUTPUT="${CAMPAIGN_ROOT}/v2/gates/defect_signal.json"
SELECTION="${CAMPAIGN_ROOT}/v1/selection/v1_budget_selection.json"
BUDGET="$("${S18_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_budget_epochs"])' "${SELECTION}")"
CHECKPOINT="${CAMPAIGN_ROOT}/v1/train/e${BUDGET}/selected_checkpoint.pth"
[[ -f "${TRACE}" ]] || fail "reviewed disjoint V2 defect trace is missing: ${TRACE}"
[[ -f "${CHECKPOINT}" ]] || fail "selected V1 checkpoint is missing"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/fit_validate_latentloop_v2_defect.py" \
  --trace "${TRACE}" \
  --split-manifest "${CAMPAIGN_ROOT}/v1/splits/v1_episode_disjoint_split.json" \
  --source-lock "${REPO_ROOT}/.canonical/source_lock/source_lock_manifest.json" \
  --checkpoint "${CHECKPOINT}" --high-error-quantile 0.90 --output "${OUTPUT}"
