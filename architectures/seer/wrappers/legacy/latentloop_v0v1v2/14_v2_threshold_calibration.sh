#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_gate "${CAMPAIGN_ROOT}/v2/gates/defect_signal.json" DEFECT_SIGNAL_PASS
CANDIDATES="${CAMPAIGN_ROOT}/v2/calibration/scheduler_candidates.json"
OUTPUT="${CAMPAIGN_ROOT}/v2/calibration/v2_threshold_lock.json"
SELECTION="${CAMPAIGN_ROOT}/v1/selection/v1_budget_selection.json"
BUDGET="$("${S18_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_budget_epochs"])' "${SELECTION}")"
CHECKPOINT="${CAMPAIGN_ROOT}/v1/train/e${BUDGET}/selected_checkpoint.pth"
[[ -f "${CANDIDATES}" ]] || fail "complete scheduler-calibration episode simulations are missing"
[[ -f "${CHECKPOINT}" ]] || fail "selected V1 checkpoint is missing"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/calibrate_latentloop_v2_scheduler.py" \
  --candidates "${CANDIDATES}" \
  --split-manifest "${CAMPAIGN_ROOT}/v1/splits/v1_episode_disjoint_split.json" \
  --source-lock "${REPO_ROOT}/.canonical/source_lock/source_lock_manifest.json" \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT}"
