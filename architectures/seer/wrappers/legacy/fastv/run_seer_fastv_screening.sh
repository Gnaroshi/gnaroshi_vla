#!/usr/bin/env bash

set -Eeuo pipefail

# Two-stage Seer FastV audit:
#   1. diagnostic-only retention map for the published L=2, R=0.50 setting;
#   2. latency-clean L/R screening with the official last-query scoring rule.
# The 100-episode screen selects a candidate but never launches the 500-episode
# paper evaluation automatically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
RUNNER="${SCRIPT_DIR}/run_seer_fastv_libero_long.sh"

CAMPAIGN_TAG="${CAMPAIGN_TAG:-seer_public33_fastv_screen}"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer}"
RESULT_ROOT="${RESULT_ROOT:-${SHARED_SEER_ROOT}/fastv/screening/${CAMPAIGN_TAG}}"
SCREEN_SEED="${SCREEN_SEED:-41}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
NUM_TASKS="${NUM_TASKS:-10}"
NODE_NUM="${NODE_NUM:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-17900}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SCREEN_LAYERS="${SCREEN_LAYERS:-2 3 5}"
SCREEN_RATIOS="${SCREEN_RATIOS:-0.10 0.20 0.25}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -s "${RUNNER}" ]] || fail "missing FastV runner: ${RUNNER}"
[[ "${EPISODES_PER_TASK}" == "10" && "${NUM_TASKS}" == "10" ]] \
    || fail "screening contract requires 10 tasks x 10 episodes = 100 episodes per row"
[[ "${PREFLIGHT_ONLY}" == "0" || "${PREFLIGHT_ONLY}" == "1" ]] \
    || fail "PREFLIGHT_ONLY must be 0 or 1"

grid=""
for layer in ${SCREEN_LAYERS}; do
    for ratio in ${SCREEN_RATIOS}; do
        grid+="${layer}:${ratio} "
    done
done
grid="${grid% }"
[[ -n "${grid}" ]] || fail "empty FastV screening grid"

echo "[PHASE A] retention diagnostic: official score, L=2, R=0.50, 100 episodes"
env \
    RESULT_ROOT="${RESULT_ROOT}/retention_l2_r0p50" \
    EVAL_SEEDS_STR="${SCREEN_SEED}" \
    FASTV_GRID="2:0.50" \
    FASTV_SCORE_MODE=last_token_at_l \
    FASTV_RETENTION_DIAGNOSTICS=1 \
    RUN_BASELINE=0 \
    EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
    NUM_TASKS="${NUM_TASKS}" \
    NODE_NUM="${NODE_NUM}" \
    MASTER_PORT_BASE="${MASTER_PORT_BASE}" \
    PREFLIGHT_ONLY="${PREFLIGHT_ONLY}" \
    bash "${RUNNER}"

echo "[PHASE B] latency-clean canonical FastV screen: grid=${grid}"
env \
    RESULT_ROOT="${RESULT_ROOT}/canonical_screen" \
    EVAL_SEEDS_STR="${SCREEN_SEED}" \
    FASTV_GRID="${grid}" \
    FASTV_SCORE_MODE=last_token_at_l \
    FASTV_RETENTION_DIAGNOSTICS=0 \
    RUN_BASELINE=1 \
    EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
    NUM_TASKS="${NUM_TASKS}" \
    NODE_NUM="${NODE_NUM}" \
    MASTER_PORT_BASE="$((MASTER_PORT_BASE + 100))" \
    PREFLIGHT_ONLY="${PREFLIGHT_ONLY}" \
    bash "${RUNNER}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PREFLIGHT][DONE] diagnostic and screening contracts passed"
    exit 0
fi

python - "${RESULT_ROOT}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary_path = root / "canonical_screen" / "campaign_summary.csv"
with summary_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

baseline_rows = [row for row in rows if int(row["fastv_enabled"]) == 0]
candidate_rows = [row for row in rows if int(row["fastv_enabled"]) == 1]
if len(baseline_rows) != 1 or not candidate_rows:
    raise RuntimeError(
        f"expected one baseline and FastV candidates, got {len(baseline_rows)} and "
        f"{len(candidate_rows)}"
    )

baseline = baseline_rows[0]
baseline_sr = float(baseline["success_rate"])
baseline_policy_ms = float(baseline["policy_ms"])
baseline_full_ms = float(baseline["full_forward_ms"])

for row in candidate_rows:
    row["success_rate"] = float(row["success_rate"])
    row["policy_ms"] = float(row["policy_ms"])
    row["full_forward_ms"] = float(row["full_forward_ms"])
    row["latency_improving"] = bool(
        row["policy_ms"] < baseline_policy_ms
        and row["full_forward_ms"] < baseline_full_ms
    )
    row["within_3pp"] = bool(row["success_rate"] >= baseline_sr - 0.03)

eligible = [
    row for row in candidate_rows
    if row["latency_improving"] and row["within_3pp"]
]
selection_pool = eligible or [
    row for row in candidate_rows if row["latency_improving"]
] or candidate_rows
selected = sorted(
    selection_pool,
    key=lambda row: (-row["success_rate"], row["policy_ms"]),
)[0]
status = "PAPER_CANDIDATE" if eligible else "NO_CANDIDATE_PASSED_3PP_AND_LATENCY_GATE"

payload = {
    "status": status,
    "selection_rule": (
        "highest screening SR among candidates with both policy/full-forward latency "
        "below the same-campaign baseline and SR within 3 percentage points; ties use "
        "lower policy latency"
    ),
    "screening_only": True,
    "final_500_episode_result_required": True,
    "baseline": baseline,
    "selected": selected,
    "candidates": candidate_rows,
}
(root / "screening_selection.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "# Seer FastV screening selection",
    "",
    f"- Status: `{status}`",
    "- This is a 100-episode development screen, not a paper result.",
    "- Retention-diagnostic latency is excluded from selection.",
    "- A selected configuration still requires a fresh 500-episode evaluation.",
    "",
    "| Row | L | R | SR (%) | Policy ms | Full-forward ms | Latency improves | Within 3 pp |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in candidate_rows:
    lines.append(
        f"| {row['row_id']} | {row['fastv_prune_layer']} | "
        f"{float(row['fastv_prune_ratio']):.2f} | "
        f"{100.0 * row['success_rate']:.2f} | {row['policy_ms']:.3f} | "
        f"{row['full_forward_ms']:.3f} | {int(row['latency_improving'])} | "
        f"{int(row['within_3pp'])} |"
    )
lines.extend(
    [
        "",
        f"Selected row: `{selected['row_id']}`",
        "",
    ]
)
(root / "screening_selection.md").write_text(
    "\n".join(lines),
    encoding="utf-8",
)
print(f"[SCREENING] status={status} selected={selected['row_id']}")
print(f"[SCREENING] report={root / 'screening_selection.md'}")
PY

echo "[DONE] ${RESULT_ROOT}"
