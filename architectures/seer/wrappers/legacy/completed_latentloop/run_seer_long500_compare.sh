#!/usr/bin/env bash

set -Eeuo pipefail

# Direct 500-episode LIBERO-Long comparison:
#   1) official Seer checkpoint 33, full inference at every step (K=1)
#   2) the same Seer checkpoint + LatentLoop V0 adapter 39 (K=4)
#
# The launcher runs one EGL evaluation seed over all 10 tasks and 50 initial
# states per task. It validates checkpoint identity, renderer identity, exact
# episode coverage, runtime call partition, and the final paired report.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
EVAL_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
EXPECTED_REPO="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer}"
RUN_ID="${RUN_ID:-seed42_r1}"
RESULT_ROOT="${RESULT_ROOT:-${SHARED_SEER_ROOT}/paper_long500/public33_latentloop_v0/${RUN_ID}}"
LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
INHERITED_VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH:-}"
VIT_CHECKPOINT_PATH="${LONG500_VIT_CHECKPOINT_PATH:-${SHARED_SEER_ROOT}/vit_mae/mae_pretrain_vit_base.pth}"
BASELINE_CKPT="${BASELINE_CKPT:-${SHARED_SEER_ROOT}/checkpoints_Seer_LIBERO_LONG/Seer/33.pth}"
ADAPTER_CKPT="${ADAPTER_CKPT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/seer.incoming_20260817/lrnode/official_seer_libero_k4_v1/train/adapters/official_seer_ckpt33_lrnode_adapter_v1/39.pth}"

BASELINE_SHA256="${BASELINE_SHA256:-a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646}"
ADAPTER_SHA256="${ADAPTER_SHA256:-3f70179ab9b1bae64fc772d71c57a93592b9f82e53b5fcaf1a6beb319c280462}"
VIT_SHA256="${VIT_SHA256:-aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d}"

EVAL_SEED="${EVAL_SEED:-42}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
NUM_TASKS="${NUM_TASKS:-10}"
NODE_NUM="${NODE_NUM:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-17700}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

require_sha256() {
    local label="$1" path="$2" expected="$3" actual
    require_file "${path}"
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] \
        || fail "${label} SHA256 mismatch: expected=${expected}, actual=${actual}, path=${path}"
    echo "[VERIFY][OK] ${label} sha256=${actual}"
}

source_lock_lines() {
    sha256sum \
        "${UPSTREAM_DIR}/eval_libero.py" \
        "${UPSTREAM_DIR}/models/gpt2.py" \
        "${UPSTREAM_DIR}/models/seer_model.py" \
        "${UPSTREAM_DIR}/models/lrnode_modules.py" \
        "${UPSTREAM_DIR}/utils/arguments_utils.py" \
        "${UPSTREAM_DIR}/utils/eval_utils_libero.py" \
        "${EVAL_SCRIPT}" \
        "${BASH_SOURCE[0]}"
}

verify_source_lock() {
    local lock="${RESULT_ROOT}/source_sha256.lock" current
    current="$(mktemp /tmp/seer_long500_source.XXXXXX)"
    source_lock_lines > "${current}"
    if [[ -s "${lock}" ]]; then
        if ! cmp -s "${lock}" "${current}"; then
            diff -u "${lock}" "${current}" || true
            rm -f "${current}"
            fail "source changed relative to this comparison run: ${lock}"
        fi
        rm -f "${current}"
    else
        mv "${current}" "${lock}"
    fi
}

verify_checkpoint_composition() {
    BASELINE_CKPT="${BASELINE_CKPT}" ADAPTER_CKPT="${ADAPTER_CKPT}" python - <<'PY'
import os
import torch

base = torch.load(os.environ["BASELINE_CKPT"], map_location="cpu")["model_state_dict"]
adapter = torch.load(os.environ["ADAPTER_CKPT"], map_location="cpu")["model_state_dict"]
allowed = ("module.lrnode_delta_encoder.", "module.lrnode_dynamics.")
unexpected = sorted(key for key in adapter if not key.startswith(allowed))
overlap = sorted(set(base) & set(adapter))
if unexpected:
    raise RuntimeError(f"adapter contains non-LatentLoop tensors: {unexpected[:8]}")
if overlap:
    raise RuntimeError(f"adapter overwrites Seer tensors: {overlap[:8]}")
if any("lrnode" in key.lower() for key in base):
    raise RuntimeError("baseline checkpoint unexpectedly contains LatentLoop tensors")
print(
    "[VERIFY][OK] checkpoint composition: "
    f"base_tensors={len(base)}, adapter_tensors={len(adapter)}, shared_overwrites=0"
)
PY
}

validate_row() {
    local row_root="$1" kind="$2" expected_k="$3"
    BASELINE_CKPT="${BASELINE_CKPT}" ADAPTER_CKPT="${ADAPTER_CKPT}" \
    python - "${row_root}" "${kind}" "${expected_k}" "${EVAL_SEED}" \
        "${EPISODES_PER_TASK}" "${NUM_TASKS}" <<'PY'
import csv
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
kind, expected_k = sys.argv[2], int(sys.argv[3])
seed, episodes_per_task, num_tasks = map(int, sys.argv[4:7])
summaries = list(root.glob("*/analysis/eval_summary.json"))
if len(summaries) != 1:
    raise RuntimeError(f"expected exactly one summary under {root}, found {len(summaries)}")
summary_path = summaries[0]
analysis = summary_path.parent
summary = json.loads(summary_path.read_text(encoding="utf-8"))
csv_path = analysis / "eval_episode_metrics.csv"
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

expected_count = episodes_per_task * num_tasks
if len(rows) != expected_count:
    raise RuntimeError(f"expected {expected_count} episodes, found {len(rows)}")
coverage = {(int(row["task_id"]), int(row["episode_id"])) for row in rows}
expected_coverage = {
    (task_id, episode_id)
    for task_id in range(num_tasks)
    for episode_id in range(episodes_per_task)
}
if coverage != expected_coverage:
    raise RuntimeError("task/initial-state coverage is not exactly 10 x 50")
if {int(float(row["seed"])) for row in rows} != {seed}:
    raise RuntimeError("evaluation seed mismatch")
if summary.get("suite") != "libero_10":
    raise RuntimeError(f"suite mismatch: {summary.get('suite')}")

renderer = summary.get("renderer_backend", {})
if renderer.get("requested_backend") != "egl" or renderer.get("effective_backend") != "egl":
    raise RuntimeError(f"renderer mismatch: {renderer}")
rank_contexts = renderer.get("rank_contexts", [])
if not renderer.get("all_ranks_actual_context_verified") or len(rank_contexts) != 4:
    raise RuntimeError(f"four-rank EGL context verification failed: {renderer}")
if not all(item.get("actual_context_verified") for item in rank_contexts):
    raise RuntimeError("at least one rank did not verify a hardware EGL context")

snapshots = list(analysis.glob("args_snapshot_*.json"))
if len(snapshots) != 1:
    raise RuntimeError(f"expected one args snapshot, found {len(snapshots)}")
args = json.loads(snapshots[0].read_text(encoding="utf-8"))
common_expected = {
    "seed": seed,
    "finetune_type": "libero_10",
    "action_pred_steps": 3,
    "future_steps": 3,
    "eval_libero_ensembling": True,
    "fastv_enabled": 0,
    "lrnode_eval_ablation_mode": "stepwise",
    "lrnode_every_step_filter_mode": "off",
    "lrnode_counterfactual_mode": "standard",
    "latentloop_plan_adapter_mode": "off",
}
for key, expected in common_expected.items():
    if args.get(key) != expected:
        raise RuntimeError(f"argument mismatch {key}: expected={expected!r}, actual={args.get(key)!r}")

lrnode = summary.get("lrnode", {})
query = summary.get("query_reduction", {})
env_steps = int(query.get("num_env_steps", -1))
full_calls = int(query.get("num_full_forward_calls", -1))
update_calls = int(query.get("num_lrnode_update_calls", -1))
fallback_calls = int(query.get("num_fallback_full_calls", -1))
if int(lrnode.get("query_interval", -1)) != expected_k:
    raise RuntimeError(f"runtime K mismatch: {lrnode.get('query_interval')} != {expected_k}")
if fallback_calls != 0:
    raise RuntimeError(f"fallback full calls must be zero, got {fallback_calls}")

base_path = str(pathlib.Path(os.environ["BASELINE_CKPT"]).resolve())
adapter_path = str(pathlib.Path(os.environ["ADAPTER_CKPT"]).resolve())
resume = str(pathlib.Path(args["resume_from_checkpoint"]).resolve())
if kind == "baseline":
    if bool(lrnode.get("enabled")) or bool(lrnode.get("eval_skip_full_forward")):
        raise RuntimeError("baseline unexpectedly enabled LatentLoop")
    if update_calls != 0 or full_calls != env_steps:
        raise RuntimeError("baseline did not execute one full Seer call per environment step")
    if resume != base_path or args.get("finetune_from_pretrained_ckpt") is not None:
        raise RuntimeError("baseline checkpoint load path mismatch")
else:
    if not bool(lrnode.get("enabled")) or not bool(lrnode.get("eval_skip_full_forward")):
        raise RuntimeError("LatentLoop skip path is not enabled")
    if full_calls + update_calls != env_steps:
        raise RuntimeError("full-forward and LatentLoop calls do not partition environment steps")
    if int(lrnode.get("observation_conditioned_update_calls", -1)) != update_calls:
        raise RuntimeError("not every skipped step used a fresh observation-conditioned update")
    if not bool(lrnode.get("query_reduction_claim_allowed")):
        raise RuntimeError("runtime disallowed the query-reduction claim")
    base_arg = str(pathlib.Path(args["finetune_from_pretrained_ckpt"]).resolve())
    if resume != adapter_path or base_arg != base_path:
        raise RuntimeError("LatentLoop base/adapter checkpoint load order mismatch")

for required in ("eval_progress.json", "eval_latency_profile.json"):
    if not (analysis / required).is_file():
        raise FileNotFoundError(analysis / required)
print(
    f"[ROW VERIFY][PASS] kind={kind} K={expected_k} episodes={len(rows)} "
    f"success={sum(int(row['success']) for row in rows)}/{len(rows)} "
    f"full={full_calls} updates={update_calls}"
)
PY
}

run_row() {
    local row_id="$1" kind="$2" k="$3" port="$4"
    local row_root="${RESULT_ROOT}/eval/${row_id}"
    local log_file="${RESULT_ROOT}/logs/${row_id}.log"
    local run_baseline=0 intervals="${k}"

    if [[ -d "${row_root}" ]] && validate_row "${row_root}" "${kind}" "${k}" >/dev/null 2>&1; then
        echo "[SKIP] verified complete row: ${row_id}"
        return
    fi
    if [[ -e "${row_root}" ]]; then
        local quarantine="${row_root}.incomplete.$(date +%Y%m%d_%H%M%S)"
        echo "[RESUME] preserving incomplete row: ${quarantine}"
        mv "${row_root}" "${quarantine}"
    fi
    verify_source_lock
    if [[ "${kind}" == "baseline" ]]; then
        run_baseline=1
        intervals=""
    fi

    mkdir -p "$(dirname "${log_file}")"
    echo "[START] row=${row_id} seed=${EVAL_SEED} episodes=500 K=${k} port=${port}"
    local rc
    set +e
    env \
        CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
        LIBERO_GL_BACKEND=egl \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        LIBERO_GL_REQUIRE_ACTUAL=1 \
        LIBERO_PATH="${LIBERO_PATH}" \
        VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
        LRNODE_PROTOCOL_ROOT="${RESULT_ROOT}" \
        SAVE_CHECKPOINT_PATH="${RESULT_ROOT}/eval_checkpoints" \
        EVAL_SUITE=libero_10 \
        EVAL_SEED="${EVAL_SEED}" \
        EVAL_NUM_EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
        EVAL_NUM_TASKS="${NUM_TASKS}" \
        EVAL_CONTROL_HZ=20 \
        LIBERO_EVAL_MAX_STEPS=600 \
        EVAL_LIBERO_ENSEMBLING=1 \
        BASELINE_CKPT="${BASELINE_CKPT}" \
        BASELINE_CKPT_ID=33 \
        BASELINE_NAME=seer_public33 \
        OURS_CKPT="${ADAPTER_CKPT}" \
        OURS_CKPT_ID=39 \
        OURS_NAME=latentloop_v0_adapter39 \
        METHOD_TAG=latentloop_v0 \
        LRNODE_EVAL_BASE_CKPT="${BASELINE_CKPT}" \
        LRNODE_TRAIN_PROTOCOL=adapter \
        LRNODE_FREEZE_SEER_FOR_ADAPTER=1 \
        LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1 \
        LRNODE_EVAL_STEP_LOG=1 \
        LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=1 \
        LRNODE_EVAL_SHADOW_FULL_FORWARD=0 \
        LRNODE_EVAL_REFRESH_POLICY=periodic \
        LRNODE_EVAL_ABLATION_MODE=stepwise \
        LRNODE_COUNTERFACTUAL_MODE=standard \
        LRNODE_EVERY_STEP_FILTER_MODE=off \
        LATENTLOOP_SEGMENT_GRID_ENABLE=0 \
        LATENTLOOP_PLAN_ADAPTER_MODE=off \
        LATENTLOOP_COMPARISON_PROTOCOL=0 \
        FASTV_ENABLED=0 \
        RUN_BASELINE="${run_baseline}" \
        RUN_OURS_FULL=0 \
        LRNODE_QUERY_INTERVALS_STR="${intervals}" \
        NODE_NUM="${NODE_NUM}" \
        MASTER_PORT="${port}" \
        SAVE_VIDEO=0 \
        SAVE_VIDEO_SUCC=0 \
        SAVE_VIDEO_FAIL=0 \
        SAVE_VIDEO_ALL_RANKS=0 \
        EXPERIMENT_NAME=seer_long500 \
        EXPERIMENT_TAG="${row_id}" \
        RESULT_ROOT="${row_root}" \
        bash "${EVAL_SCRIPT}" 2>&1 | tee -a "${log_file}"
    rc=${PIPESTATUS[0]}
    set -e
    validate_row "${row_root}" "${kind}" "${k}" \
        || fail "invalid row=${row_id}; wrapper_rc=${rc}; log=${log_file}"
    if (( rc != 0 )); then
        echo "[WARN] wrapper rc=${rc}, but the complete 500-episode row passed validation"
    fi
}

write_comparison() {
    python - "${RESULT_ROOT}" <<'PY'
import csv
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])

def load(row_id):
    summaries = list((root / "eval" / row_id).glob("*/analysis/eval_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"missing validated summary for {row_id}")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    with (summaries[0].parent / "eval_episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    return summary, episodes, summaries[0]

baseline, baseline_rows, baseline_path = load("seer_k1")
ours, ours_rows, ours_path = load("latentloop_k4")
key = lambda row: (int(row["task_id"]), int(row["episode_id"]))
base_by_key = {key(row): row for row in baseline_rows}
ours_by_key = {key(row): row for row in ours_rows}
if set(base_by_key) != set(ours_by_key):
    raise RuntimeError("baseline/ours episode coverage mismatch")

wins = losses = both_success = both_failure = 0
common_success_base_steps = []
common_success_ours_steps = []
for episode_key in sorted(base_by_key):
    b = bool(int(base_by_key[episode_key]["success"]))
    o = bool(int(ours_by_key[episode_key]["success"]))
    if o and not b:
        wins += 1
    elif b and not o:
        losses += 1
    elif b:
        both_success += 1
        common_success_base_steps.append(int(base_by_key[episode_key]["num_steps"]))
        common_success_ours_steps.append(int(ours_by_key[episode_key]["num_steps"]))
    else:
        both_failure += 1

def row(name, k, payload, episodes):
    query = payload["query_reduction"]
    return {
        "method": name,
        "k": k,
        "episodes": len(episodes),
        "successes": sum(int(item["success"]) for item in episodes),
        "success_rate_pct": 100.0 * float(payload["success_rate"]),
        "env_steps": int(query["num_env_steps"]),
        "full_forward_calls": int(query["num_full_forward_calls"]),
        "latentloop_update_calls": int(query["num_lrnode_update_calls"]),
        "query_reduction_pct": 100.0 * float(query["full_query_reduction_ratio"]),
        "policy_step_ms": float(payload["avg_policy_step_latency_ms"]),
        "full_forward_ms": float(payload["avg_full_forward_latency_ms"]),
    }

table = [row("Seer", 1, baseline, baseline_rows), row("Seer + LatentLoop", 4, ours, ours_rows)]
comparison = {
    "schema_version": 1,
    "status": "LONG500_COMPARISON_COMPLETE",
    "suite": "libero_10",
    "seed": int(baseline_rows[0]["seed"]),
    "episodes_per_method": 500,
    "rows": table,
    "success_rate_gain_pp": table[1]["success_rate_pct"] - table[0]["success_rate_pct"],
    "policy_speedup": table[0]["policy_step_ms"] / table[1]["policy_step_ms"],
    "paired_outcomes": {
        "ours_only_success": wins,
        "baseline_only_success": losses,
        "both_success": both_success,
        "both_failure": both_failure,
    },
    "common_success": {
        "episodes": both_success,
        "baseline_mean_env_steps": statistics.fmean(common_success_base_steps) if common_success_base_steps else None,
        "ours_mean_env_steps": statistics.fmean(common_success_ours_steps) if common_success_ours_steps else None,
    },
    "source_summaries": [str(baseline_path), str(ours_path)],
}
(root / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
with (root / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)

report = f"""# Seer vs. Seer + LatentLoop: LIBERO-Long 500-Episode Comparison

| Method | K | Success | SR | Full calls | Query reduction | Policy latency |
|---|---:|---:|---:|---:|---:|---:|
| Seer | 1 | {table[0]['successes']}/500 | {table[0]['success_rate_pct']:.2f}% | {table[0]['full_forward_calls']} | {table[0]['query_reduction_pct']:.2f}% | {table[0]['policy_step_ms']:.3f} ms |
| Seer + LatentLoop | 4 | {table[1]['successes']}/500 | {table[1]['success_rate_pct']:.2f}% | {table[1]['full_forward_calls']} | {table[1]['query_reduction_pct']:.2f}% | {table[1]['policy_step_ms']:.3f} ms |

- SR gain: {comparison['success_rate_gain_pp']:+.2f} percentage points
- Policy speedup: {comparison['policy_speedup']:.2f}x
- Paired outcomes: ours-only success {wins}, baseline-only success {losses}, both success {both_success}, both failure {both_failure}
- Common-success mean env steps: Seer {comparison['common_success']['baseline_mean_env_steps']:.2f}, LatentLoop {comparison['common_success']['ours_mean_env_steps']:.2f}
- Renderer: hardware EGL verified on all four RTX 3090 ranks
- Checkpoint pair: official Seer 33 + adapter-only LatentLoop 39
"""
(root / "comparison_report.md").write_text(report, encoding="utf-8")
print(report)
PY
}

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || fail "expected host ${EXPECTED_HOST}, got $(hostname)"
[[ "$(readlink -f "${REPO_ROOT}")" == "$(readlink -f "${EXPECTED_REPO}")" ]] \
    || fail "unexpected source tree: ${REPO_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == "seer_libero" ]] || fail "activate conda environment seer_libero first"
[[ "${GPU_LIST}" == "4,5,6,7" ]] || fail "sd1 protocol requires CUDA_VISIBLE_DEVICES=4,5,6,7"
[[ "${EPISODES_PER_TASK}" == "50" && "${NUM_TASKS}" == "10" ]] \
    || fail "this protocol is fixed to 10 tasks x 50 episodes = 500 episodes"
[[ "${NODE_NUM}" == "4" ]] || fail "this protocol requires four DDP ranks"
[[ "${EVAL_SEED}" =~ ^[0-9]+$ ]] || fail "EVAL_SEED must be an integer"
[[ "${MASTER_PORT_BASE}" =~ ^[0-9]+$ ]] && (( MASTER_PORT_BASE >= 1024 && MASTER_PORT_BASE <= 63998 )) \
    || fail "invalid MASTER_PORT_BASE=${MASTER_PORT_BASE}"
[[ "${PREFLIGHT_ONLY}" == "0" || "${PREFLIGHT_ONLY}" == "1" ]] \
    || fail "PREFLIGHT_ONLY must be 0 or 1"

[[ -d "${LIBERO_PATH}" ]] || fail "missing LIBERO repository: ${LIBERO_PATH}"
if [[ -n "${INHERITED_VIT_CHECKPOINT_PATH}" && "${INHERITED_VIT_CHECKPOINT_PATH}" != "${VIT_CHECKPOINT_PATH}" ]]; then
    echo "[ENV] ignoring inherited VIT_CHECKPOINT_PATH=${INHERITED_VIT_CHECKPOINT_PATH}"
    echo "[ENV] canonical Long-500 ViT=${VIT_CHECKPOINT_PATH}"
fi
require_file "${EVAL_SCRIPT}"
require_sha256 "official Seer 33" "${BASELINE_CKPT}" "${BASELINE_SHA256}"
require_sha256 "LatentLoop adapter 39" "${ADAPTER_CKPT}" "${ADAPTER_SHA256}"
require_sha256 "MAE ViT" "${VIT_CHECKPOINT_PATH}" "${VIT_SHA256}"
verify_checkpoint_composition

mkdir -p "${RESULT_ROOT}/eval" "${RESULT_ROOT}/logs"
exec 9>"${RESULT_ROOT}/run.lock"
flock -n 9 || fail "another process owns ${RESULT_ROOT}/run.lock"
verify_source_lock

cat > "${RESULT_ROOT}/experiment_contract.env" <<EOF
PROTOCOL=public33_latentloop_v0_long500
HOST=${EXPECTED_HOST}
SOURCE_REPO=${REPO_ROOT}
GPU_LIST=${GPU_LIST}
SUITE=libero_10
RENDERER=egl
EVAL_SEED=${EVAL_SEED}
EPISODES_PER_TASK=${EPISODES_PER_TASK}
NUM_TASKS=${NUM_TASKS}
BASELINE=official_seer_33_k1
OURS=official_seer_33_plus_latentloop_adapter_39_k4
BASELINE_CKPT=${BASELINE_CKPT}
BASELINE_SHA256=${BASELINE_SHA256}
ADAPTER_CKPT=${ADAPTER_CKPT}
ADAPTER_SHA256=${ADAPTER_SHA256}
ACTION_PRED_STEPS=3
TEMPORAL_ENSEMBLING=1
FASTV_ENABLED=0
EOF
git -C "${REPO_ROOT}" rev-parse HEAD > "${RESULT_ROOT}/git_commit.txt"
git -C "${REPO_ROOT}" status --short > "${RESULT_ROOT}/git_status_short.txt"

python - <<'PY'
import torch

if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected four visible GPUs, got {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(4)]
if not all("RTX 3090" in name for name in names):
    raise RuntimeError(f"all four sd1 GPUs must be RTX 3090: {names}")
print(f"[VERIFY][OK] four RTX 3090 GPUs: {names}")
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PREFLIGHT][PASS] configuration, checkpoints, source lock, and GPUs verified"
    exit 0
fi

run_row seer_k1 baseline 1 "${MASTER_PORT_BASE}"
run_row latentloop_k4 latentloop 4 "$((MASTER_PORT_BASE + 1))"
write_comparison
date --iso-8601=seconds > "${RESULT_ROOT}/run_complete.txt"
echo "[DONE] ${RESULT_ROOT}/comparison_report.md"
