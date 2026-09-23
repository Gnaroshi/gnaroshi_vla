#!/usr/bin/env bash

set -Eeuo pipefail

# Compare three explicit FastV visual-token scoring contracts under one
# checkpoint, renderer, seed, episode coverage, layer, and pruning ratio.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
ROW_WRAPPER="${SCRIPT_DIR}/run_seer_fastv_libero_long.sh"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-seer_public33_fastv_score_contracts}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${SHARED_SEER_ROOT}/fastv/scoring_contracts/${CAMPAIGN_TAG}}"
CANONICAL_VIT_CHECKPOINT_PATH="/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth"
VIT_SHA256="aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"

EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-42}"
FASTV_GRID="${FASTV_GRID:-2:0.50}"
SCORE_MODES_STR="${SCORE_MODES_STR:-text_mean_first_l last_token_at_l action_mean_first_l}"
RUN_BASELINE="${RUN_BASELINE:-0}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
NUM_TASKS="${NUM_TASKS:-10}"
NODE_NUM="${NODE_NUM:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-17800}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

validate_bool() {
    [[ "$2" == "0" || "$2" == "1" ]] || fail "$1 must be 0 or 1, got $2"
}

source_lock_lines() {
    sha256sum \
        "${REPO_ROOT}/architectures/seer/upstream/models/fastv.py" \
        "${REPO_ROOT}/architectures/seer/upstream/models/gpt2.py" \
        "${REPO_ROOT}/architectures/seer/upstream/models/seer_model.py" \
        "${REPO_ROOT}/architectures/seer/upstream/utils/arguments_utils.py" \
        "${REPO_ROOT}/architectures/seer/upstream/utils/eval_utils_libero.py" \
        "${ROW_WRAPPER}" \
        "${BASH_SOURCE[0]}"
}

verify_source_lock() {
    local lock="${CAMPAIGN_ROOT}/source_sha256.lock" current
    current="$(mktemp /tmp/seer_fastv_contract_source.XXXXXX)"
    source_lock_lines > "${current}"
    if [[ -s "${lock}" ]]; then
        if ! cmp -s "${lock}" "${current}"; then
            diff -u "${lock}" "${current}" || true
            rm -f "${current}"
            fail "source changed relative to scoring-contract campaign lock: ${lock}"
        fi
        rm -f "${current}"
    else
        mv "${current}" "${lock}"
    fi
}

write_combined_summary() {
    python - "${CAMPAIGN_ROOT}" "${SCORE_MODES_STR}" "${EVAL_SEEDS_STR}" \
        "${FASTV_GRID}" "${RUN_BASELINE}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
modes = sys.argv[2].split()
seeds = [int(item) for item in sys.argv[3].split()]
grid = sys.argv[4].split()
run_baseline = bool(int(sys.argv[5]))
contracts = {
    "text_mean_first_l": "all Seer language-conditioning queries, mean over layers [0,L)",
    "last_token_at_l": "final sequence token, attention from layer L-1 only",
    "action_mean_first_l": "final-timestep action queries, mean over layers [0,L)",
}

records = []
for path in sorted((root / "modes").glob("*/*/*/*/analysis/eval_summary.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    fastv = payload.get("fastv", {})
    relative_parts = path.relative_to(root / "modes").parts
    mode = relative_parts[0]
    row_id = relative_parts[2]
    seed_text = row_id.split("_", 1)[0].removeprefix("seed")
    if not seed_text.isdigit():
        raise RuntimeError(f"cannot recover evaluation seed from row_id={row_id!r}")
    records.append(
        {
            "score_contract": mode if fastv.get("enabled") else "baseline",
            "contract_definition": (
                contracts[mode] if fastv.get("enabled") else "FastV disabled"
            ),
            "eval_seed": int(seed_text),
            "prune_layer": int(fastv.get("prune_layer", -1)),
            "prune_ratio": float(fastv.get("prune_ratio", 0.0)),
            "score_query_indices": json.dumps(fastv.get("score_query_indices", [])),
            "tokens_before": int(fastv.get("tokens_before_pruning", 0)),
            "tokens_after": int(fastv.get("tokens_after_pruning", 0)),
            "success_rate": float(payload["success_rate"]),
            "policy_ms": float(payload["avg_policy_step_latency_ms"]),
            "full_forward_ms": float(payload["avg_full_forward_latency_ms"]),
            "runtime_verified": bool(fastv.get("runtime_verified", False)),
            "summary_path": str(path),
        }
    )

expected_fastv = len(modes) * len(seeds) * len(grid)
fastv_records = [row for row in records if row["score_contract"] != "baseline"]
baseline_records = [row for row in records if row["score_contract"] == "baseline"]
if len(fastv_records) != expected_fastv:
    raise RuntimeError(
        f"expected {expected_fastv} FastV rows, found {len(fastv_records)}"
    )
if len(baseline_records) != (len(seeds) if run_baseline else 0):
    raise RuntimeError(
        "baseline row count mismatch: "
        f"expected={len(seeds) if run_baseline else 0}, actual={len(baseline_records)}"
    )
if {row["score_contract"] for row in fastv_records} != set(modes):
    raise RuntimeError("score-contract coverage mismatch")
if not all(row["runtime_verified"] for row in records):
    raise RuntimeError("at least one row failed FastV runtime verification")

fieldnames = list(records[0])
with (root / "combined_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

lines = [
    "# Seer FastV scoring-contract comparison",
    "",
    f"- MAE checkpoint: `{root.joinpath('canonical_vit_checkpoint.txt').read_text().strip()}`",
    "- All FastV rows use identical checkpoint, EGL renderer, episode coverage, L, and R.",
    "",
    "| Contract | Seed | L | R | Queries | Tokens | SR (%) | Policy (ms) | Full forward (ms) |",
    "|---|---:|---:|---:|---|---:|---:|---:|---:|",
]
for row in records:
    lines.append(
        f"| {row['score_contract']} | {row['eval_seed']} | {row['prune_layer']} | "
        f"{row['prune_ratio']:.2f} | `{row['score_query_indices']}` | "
        f"{row['tokens_before']} -> {row['tokens_after']} | "
        f"{100.0 * row['success_rate']:.2f} | {row['policy_ms']:.3f} | "
        f"{row['full_forward_ms']:.3f} |"
    )
(root / "combined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[FASTV CONTRACT SUMMARY] rows={len(records)} path={root / 'combined_summary.md'}")
PY
}

[[ "$(hostname -s)" == "${EXPECTED_HOST}" ]] \
    || fail "expected host ${EXPECTED_HOST}, got $(hostname -s)"
[[ "${GPU_LIST}" == "4,5,6,7" ]] \
    || fail "sd1 protocol requires physical GPUs 4,5,6,7; got ${GPU_LIST}"
[[ "${NODE_NUM}" == "4" ]] || fail "NODE_NUM must be 4"
validate_bool RUN_BASELINE "${RUN_BASELINE}"
validate_bool PREFLIGHT_ONLY "${PREFLIGHT_ONLY}"
require_file "${ROW_WRAPPER}"
require_file "${CANONICAL_VIT_CHECKPOINT_PATH}"
actual_vit_sha="$(sha256sum "${CANONICAL_VIT_CHECKPOINT_PATH}" | awk '{print $1}')"
[[ "${actual_vit_sha}" == "${VIT_SHA256}" ]] \
    || fail "canonical MAE SHA256 mismatch: expected=${VIT_SHA256}, actual=${actual_vit_sha}"

read -r -a SCORE_MODES <<< "${SCORE_MODES_STR}"
[[ "${#SCORE_MODES[@]}" -gt 0 ]] || fail "SCORE_MODES_STR must be non-empty"
python - "${SCORE_MODES_STR}" <<'PY'
import sys

expected = {"text_mean_first_l", "last_token_at_l", "action_mean_first_l"}
modes = sys.argv[1].split()
if len(modes) != len(set(modes)):
    raise RuntimeError(f"duplicate score modes: {modes}")
if set(modes) != expected:
    raise RuntimeError(f"expected exactly {sorted(expected)}, got {modes}")
print("[VERIFY][OK] scoring contracts:", " ".join(modes))
PY

mkdir -p "${CAMPAIGN_ROOT}"
verify_source_lock
printf '%s\n' "${CANONICAL_VIT_CHECKPOINT_PATH}" \
    > "${CAMPAIGN_ROOT}/canonical_vit_checkpoint.txt"
printf '%s\n' "${actual_vit_sha}" > "${CAMPAIGN_ROOT}/canonical_vit_sha256.txt"

echo "[PREFLIGHT] repo=${REPO_ROOT}"
echo "[PREFLIGHT] result_root=${CAMPAIGN_ROOT}"
echo "[PREFLIGHT] vit_mae=${CANONICAL_VIT_CHECKPOINT_PATH}"
echo "[PREFLIGHT] score_modes=${SCORE_MODES[*]}"
echo "[PREFLIGHT] seeds=${EVAL_SEEDS_STR} grid=${FASTV_GRID}"

mode_index=0
for score_mode in "${SCORE_MODES[@]}"; do
    mode_root="${CAMPAIGN_ROOT}/modes/${score_mode}"
    run_baseline_for_mode=0
    if (( mode_index == 0 )) && [[ "${RUN_BASELINE}" == "1" ]]; then
        run_baseline_for_mode=1
    fi
    echo "[$((mode_index + 1))/${#SCORE_MODES[@]}] score_contract=${score_mode}"
    env -u VIT_CHECKPOINT_PATH \
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        FASTV_VIT_CHECKPOINT_PATH="${CANONICAL_VIT_CHECKPOINT_PATH}" \
        CAMPAIGN_TAG="${CAMPAIGN_TAG}_${score_mode}" \
        RESULT_ROOT="${mode_root}" \
        EVAL_SEEDS_STR="${EVAL_SEEDS_STR}" \
        FASTV_GRID="${FASTV_GRID}" \
        FASTV_SCORE_MODE="${score_mode}" \
        RUN_BASELINE="${run_baseline_for_mode}" \
        EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
        NUM_TASKS="${NUM_TASKS}" \
        NODE_NUM="${NODE_NUM}" \
        MASTER_PORT_BASE="$((MASTER_PORT_BASE + mode_index * 100))" \
        PREFLIGHT_ONLY="${PREFLIGHT_ONLY}" \
        bash "${ROW_WRAPPER}"
    mode_index=$((mode_index + 1))
done

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PREFLIGHT][DONE] all three scoring contracts passed"
    exit 0
fi

write_combined_summary
printf 'status=COMPLETE\ntime=%s\n' "$(date --iso-8601=seconds)" \
    > "${CAMPAIGN_ROOT}/campaign_complete.env"
echo "[DONE] ${CAMPAIGN_ROOT}/combined_summary.md"
