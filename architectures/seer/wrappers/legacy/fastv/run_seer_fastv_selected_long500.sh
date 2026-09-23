#!/usr/bin/env bash

set -Eeuo pipefail

# Run the selected canonical FastV configuration and an unpruned Seer baseline
# for 10 tasks x 50 episodes. The screening gate must pass first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
RUNNER="${SCRIPT_DIR}/run_seer_fastv_libero_long.sh"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer}"
SCREEN_RESULT_ROOT="${SCREEN_RESULT_ROOT:-}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-seer_public33_fastv_selected_long500}"
RESULT_ROOT="${RESULT_ROOT:-${SHARED_SEER_ROOT}/fastv/final/${CAMPAIGN_TAG}}"
EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-42}"
NODE_NUM="${NODE_NUM:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-18100}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -n "${SCREEN_RESULT_ROOT}" ]] \
    || fail "SCREEN_RESULT_ROOT must point to a completed FastV screening campaign"
SELECTION_JSON="${SCREEN_RESULT_ROOT}/screening_selection.json"
[[ -s "${SELECTION_JSON}" ]] || fail "missing screening selection: ${SELECTION_JSON}"

read -r selection_status layer ratio < <(
    python - "${SELECTION_JSON}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = payload["selected"]
print(
    payload["status"],
    int(selected["fastv_prune_layer"]),
    float(selected["fastv_prune_ratio"]),
)
PY
)
[[ "${selection_status}" == "PAPER_CANDIDATE" ]] \
    || fail "screening did not pass the preregistered SR/latency gate: ${selection_status}"

echo "[FINAL 500] score=last_token_at_l L=${layer} R=${ratio} seeds=${EVAL_SEEDS_STR}"
env \
    RESULT_ROOT="${RESULT_ROOT}" \
    EVAL_SEEDS_STR="${EVAL_SEEDS_STR}" \
    FASTV_GRID="${layer}:${ratio}" \
    FASTV_SCORE_MODE=last_token_at_l \
    FASTV_RETENTION_DIAGNOSTICS=0 \
    RUN_BASELINE=1 \
    EPISODES_PER_TASK=50 \
    NUM_TASKS=10 \
    NODE_NUM="${NODE_NUM}" \
    MASTER_PORT_BASE="${MASTER_PORT_BASE}" \
    PREFLIGHT_ONLY="${PREFLIGHT_ONLY}" \
    bash "${RUNNER}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PREFLIGHT][DONE] selected 500-episode contract passed; no evaluation launched"
    exit 0
fi

echo "[DONE] ${RESULT_ROOT}/campaign_summary.md"
