#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
SELECTION="${CAMPAIGN_ROOT}/v1/selection/v1_budget_selection.json"
[[ -f "${SELECTION}" ]] || fail "V1 budget selection is missing"
BUDGET="$("${S18_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_budget_epochs"])' "${SELECTION}")"
METRICS="${CAMPAIGN_ROOT}/v1/train/e${BUDGET}/offline_gate_metrics.json"
CHECKPOINT="${CAMPAIGN_ROOT}/v1/train/e${BUDGET}/selected_checkpoint.pth"
OUTPUT="${CAMPAIGN_ROOT}/v1/gates/v1_offline_gate.json"
[[ -f "${METRICS}" ]] || fail "selected V1 offline metrics are missing: ${METRICS}"
[[ -f "${CHECKPOINT}" ]] || fail "selected V1 checkpoint is missing: ${CHECKPOINT}"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/check_latentloop_v1_offline_gate.py" \
  --metrics "${METRICS}" \
  --source-lock "${REPO_ROOT}/.canonical/source_lock/source_lock_manifest.json" \
  --split-manifest "${CAMPAIGN_ROOT}/v1/splits/v1_episode_disjoint_split.json" \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT}"
