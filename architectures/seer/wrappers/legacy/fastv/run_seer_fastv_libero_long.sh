#!/usr/bin/env bash

set -Eeuo pipefail

# Training-free FastV evaluation for the public Seer LIBERO-Long checkpoint.
# Default campaign: baseline and FastV(L=2, R=0.50), one EGL-50 evaluation seed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
EVAL_SCRIPT="${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-seer_public33_fastv_egl50_v1}"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer}"
RESULT_ROOT="${RESULT_ROOT:-${SHARED_SEER_ROOT}/fastv/${CAMPAIGN_TAG}}"
LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
INHERITED_VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH:-}"
VIT_CHECKPOINT_PATH="${FASTV_VIT_CHECKPOINT_PATH:-${SHARED_SEER_ROOT}/vit_mae/mae_pretrain_vit_base.pth}"
PUBLIC33_CKPT="${PUBLIC33_CKPT:-${SHARED_SEER_ROOT}/checkpoints_Seer_LIBERO_LONG/Seer/33.pth}"
PUBLIC33_SHA256="${PUBLIC33_SHA256:-a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646}"
VIT_SHA256="${VIT_SHA256:-aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d}"

EVAL_SEEDS_STR="${EVAL_SEEDS_STR:-42}"
FASTV_GRID="${FASTV_GRID:-2:0.50}"
RUN_BASELINE="${RUN_BASELINE:-1}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
NUM_TASKS="${NUM_TASKS:-10}"
NODE_NUM="${NODE_NUM:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-17600}"
FASTV_SCORE_MODE="${FASTV_SCORE_MODE:-last_token_at_l}"
FASTV_RETENTION_DIAGNOSTICS="${FASTV_RETENTION_DIAGNOSTICS:-0}"
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
        || fail "${label} SHA256 mismatch: expected=${expected}, actual=${actual}"
    echo "[VERIFY][OK] ${label} sha256=${actual}"
}

validate_bool() {
    [[ "$2" == "0" || "$2" == "1" ]] || fail "$1 must be 0 or 1, got $2"
}

source_lock_lines() {
    sha256sum \
        "${UPSTREAM_DIR}/models/fastv.py" \
        "${UPSTREAM_DIR}/models/gpt2.py" \
        "${UPSTREAM_DIR}/models/seer_model.py" \
        "${UPSTREAM_DIR}/utils/arguments_utils.py" \
        "${UPSTREAM_DIR}/utils/eval_utils_libero.py" \
        "${UPSTREAM_DIR}/eval_libero.py" \
        "${EVAL_SCRIPT}" \
        "${BASH_SOURCE[0]}"
}

verify_source_lock() {
    local lock="${RESULT_ROOT}/source_sha256.lock" current
    current="$(mktemp /tmp/seer_fastv_source.XXXXXX)"
    source_lock_lines > "${current}"
    if [[ -s "${lock}" ]]; then
        if ! cmp -s "${lock}" "${current}"; then
            diff -u "${lock}" "${current}" || true
            rm -f "${current}"
            fail "FastV source changed relative to campaign lock: ${lock}"
        fi
        rm -f "${current}"
    else
        mv "${current}" "${lock}"
    fi
}

validate_row() {
    local row_root="$1" seed="$2" enabled="$3" layer="$4" ratio="$5"
    python - "${row_root}" "${seed}" "${enabled}" "${layer}" "${ratio}" \
        "${EPISODES_PER_TASK}" "${NUM_TASKS}" "${FASTV_SCORE_MODE}" \
        "${FASTV_RETENTION_DIAGNOSTICS}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
seed, enabled, layer = int(sys.argv[2]), bool(int(sys.argv[3])), int(sys.argv[4])
ratio = float(sys.argv[5])
episodes_per_task, num_tasks = int(sys.argv[6]), int(sys.argv[7])
score_mode = sys.argv[8]
retention_diagnostics = bool(int(sys.argv[9]))
summaries = list(root.glob("*/analysis/eval_summary.json"))
if len(summaries) != 1:
    raise RuntimeError(f"expected exactly one eval summary under {root}, found {len(summaries)}")
summary = json.loads(summaries[0].read_text(encoding="utf-8"))
analysis = summaries[0].parent
args = json.loads((analysis / "args_snapshot.json").read_text(encoding="utf-8"))
with (analysis / "eval_episode_metrics.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
expected = episodes_per_task * num_tasks
if len(rows) != expected:
    raise RuntimeError(f"expected {expected} episodes, found {len(rows)}")
coverage = {(int(row["task_id"]), int(row["episode_id"])) for row in rows}
expected_coverage = {
    (task, episode)
    for task in range(num_tasks)
    for episode in range(episodes_per_task)
}
if coverage != expected_coverage:
    raise RuntimeError("task/init-state coverage mismatch")
if {int(float(row["seed"])) for row in rows} != {seed}:
    raise RuntimeError("evaluation seed mismatch")
renderer = summary.get("renderer_backend", {})
if renderer.get("effective_backend") != "egl":
    raise RuntimeError(f"expected EGL renderer, got {renderer}")
if not renderer.get("all_ranks_actual_context_verified", False):
    raise RuntimeError("all four ranks did not verify an actual EGL context")
fastv = summary.get("fastv", {})
if bool(fastv.get("enabled")) != enabled:
    raise RuntimeError(f"FastV enable mismatch: expected={enabled}, actual={fastv}")
expected_retention_diagnostics = bool(enabled and retention_diagnostics)
if bool(fastv.get("retention_diagnostics_enabled")) != expected_retention_diagnostics:
    raise RuntimeError(
        "FastV retention-diagnostic mode mismatch: "
        f"expected={expected_retention_diagnostics}, actual={fastv}"
    )
if fastv.get("score_mode") != score_mode:
    raise RuntimeError(f"FastV score-mode mismatch: expected={score_mode}, actual={fastv}")
if int(fastv.get("runtime_mismatches", -1)) != 0 or not fastv.get("runtime_verified"):
    raise RuntimeError(f"FastV runtime verification failed: {fastv}")
sequence_length = int(args["sequence_length"])
visual_per_step = 2 * int(args["num_resampler_query"]) + 2
observation_per_step = (
    2 * int(args["num_obs_token_per_image"]) if bool(args["obs_pred"]) else 0
)
tokens_per_step = 2 + visual_per_step + observation_per_step + int(args["action_pred_steps"])
expected_tokens = sequence_length * tokens_per_step
expected_visual_tokens = sequence_length * visual_per_step
if score_mode == "text_mean_first_l":
    expected_score_queries = [step * tokens_per_step for step in range(sequence_length)]
elif score_mode == "action_mean_first_l":
    action_start = tokens_per_step - int(args["action_pred_steps"])
    final_base = (sequence_length - 1) * tokens_per_step
    expected_score_queries = list(
        range(final_base + action_start, final_base + tokens_per_step)
    )
else:
    expected_score_queries = [expected_tokens - 1]
if list(fastv.get("score_query_indices", [])) != expected_score_queries:
    raise RuntimeError(
        "FastV scoring-query mismatch: "
        f"expected={expected_score_queries}, actual={fastv}"
    )
if int(fastv.get("tokens_before_pruning", -1)) != expected_tokens:
    raise RuntimeError(
        f"unexpected Seer token count: expected={expected_tokens}, actual={fastv}"
    )
if int(fastv.get("visual_tokens_before_pruning", -1)) != expected_visual_tokens:
    raise RuntimeError(
        "unexpected Seer visual-token count: "
        f"expected={expected_visual_tokens}, actual={fastv}"
    )
full_calls = int(summary.get("num_full_forward_calls", -1))
if enabled:
    expected_visual = max(1, round(expected_visual_tokens * (1.0 - ratio)))
    expected_total = expected_tokens - (expected_visual_tokens - expected_visual)
    if int(fastv.get("prune_layer", -1)) != layer:
        raise RuntimeError(f"FastV layer mismatch: {fastv}")
    if int(fastv.get("score_layer_count", -1)) != layer:
        raise RuntimeError(f"FastV score-layer count mismatch: {fastv}")
    if abs(float(fastv.get("prune_ratio", -1.0)) - ratio) > 1e-12:
        raise RuntimeError(f"FastV ratio mismatch: {fastv}")
    if int(fastv.get("runtime_calls", -1)) != full_calls:
        raise RuntimeError(f"not every full forward used FastV: {fastv}")
    if int(fastv.get("tokens_after_pruning", -1)) != expected_total:
        raise RuntimeError(f"post-pruning token count mismatch: {fastv}")
    retention = fastv.get("retention_diagnostics", {})
    retention_csv = analysis / "fastv_retention_by_timestep_camera.csv"
    if retention_diagnostics:
        if retention.get("aggregation") != "all_full_forward_calls_across_all_ranks":
            raise RuntimeError(f"unexpected retention aggregation: {retention}")
        if retention.get("camera_names") != ["primary", "wrist"]:
            raise RuntimeError(f"unexpected FastV camera groups: {retention}")
        if retention.get("timestep_indices") != list(range(sequence_length)):
            raise RuntimeError(f"unexpected FastV timestep groups: {retention}")
        if int(retention.get("calls", -1)) != full_calls:
            raise RuntimeError(
                f"retention diagnostics do not cover every full call: {retention}"
            )
        expected_original = [
            [int(args["num_resampler_query"]) + 1] * 2
            for _ in range(sequence_length)
        ]
        if retention.get("original_tokens_by_timestep_camera") != expected_original:
            raise RuntimeError(f"unexpected original visual-token groups: {retention}")
        retained_sum = retention.get("retained_token_sum_by_timestep_camera", [])
        if len(retained_sum) != sequence_length or any(
            len(row) != 2 for row in retained_sum
        ):
            raise RuntimeError(f"invalid retained-token matrix: {retention}")
        if sum(sum(int(value) for value in row) for row in retained_sum) != (
            full_calls * expected_visual
        ):
            raise RuntimeError(
                "retained-token diagnostics violate the per-call top-k count"
            )
        with retention_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestep",
                    "camera",
                    "original_tokens_per_call",
                    "mean_retained_tokens",
                    "retained_fraction",
                    "zero_retention_call_fraction",
                    "min_retained_tokens",
                    "max_retained_tokens",
                    "full_forward_calls",
                ],
            )
            writer.writeheader()
            for timestep in range(sequence_length):
                for camera_index, camera in enumerate(["primary", "wrist"]):
                    writer.writerow(
                        {
                            "timestep": timestep,
                            "camera": camera,
                            "original_tokens_per_call": expected_original[timestep][camera_index],
                            "mean_retained_tokens": retention[
                                "mean_retained_tokens_by_timestep_camera"
                            ][timestep][camera_index],
                            "retained_fraction": retention[
                                "retained_fraction_by_timestep_camera"
                            ][timestep][camera_index],
                            "zero_retention_call_fraction": retention[
                                "zero_retention_call_fraction_by_timestep_camera"
                            ][timestep][camera_index],
                            "min_retained_tokens": retention[
                                "min_retained_tokens_by_timestep_camera"
                            ][timestep][camera_index],
                            "max_retained_tokens": retention[
                                "max_retained_tokens_by_timestep_camera"
                            ][timestep][camera_index],
                            "full_forward_calls": full_calls,
                        }
                    )
    else:
        if retention.get("aggregation") != "disabled" or int(retention.get("calls", -1)) != 0:
            raise RuntimeError(f"retention diagnostics unexpectedly active: {retention}")
        if retention_csv.exists():
            raise RuntimeError(f"stale retention CSV exists with diagnostics disabled: {retention_csv}")
else:
    if int(fastv.get("runtime_calls", -1)) != 0:
        raise RuntimeError(f"baseline unexpectedly executed FastV: {fastv}")
    if int(fastv.get("tokens_after_pruning", -1)) != expected_tokens:
        raise RuntimeError(f"baseline metadata reports token pruning: {fastv}")
for name in ("eval_progress.json", "eval_latency_profile.json"):
    if not (analysis / name).is_file():
        raise FileNotFoundError(analysis / name)
print(
    f"[FASTV VERIFY] seed={seed} enabled={int(enabled)} layer={layer} ratio={ratio:.2f} "
    f"episodes={len(rows)} SR={100.0 * float(summary['success_rate']):.2f}% "
    f"policy_ms={float(summary['avg_policy_step_latency_ms']):.3f}"
)
PY
}

run_row() {
    local row_id="$1" seed="$2" enabled="$3" layer="$4" ratio="$5" port="$6"
    local row_root="${RESULT_ROOT}/eval/${row_id}"
    local log_file="${RESULT_ROOT}/logs/${row_id}.log"
    if [[ -d "${row_root}" ]] && validate_row "${row_root}" "${seed}" "${enabled}" "${layer}" "${ratio}" >/dev/null 2>&1; then
        echo "[SKIP] verified complete row: ${row_id}"
        return
    fi
    if [[ -e "${row_root}" ]]; then
        local quarantine="${row_root}.incomplete.$(date +%Y%m%d_%H%M%S)"
        echo "[RESUME] preserving incomplete row: ${quarantine}"
        mv "${row_root}" "${quarantine}"
    fi
    verify_source_lock
    mkdir -p "$(dirname "${log_file}")"
    echo "[FASTV START] row=${row_id} seed=${seed} enabled=${enabled} layer=${layer} ratio=${ratio}"
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
        EVAL_SEED="${seed}" \
        EVAL_NUM_EPISODES_PER_TASK="${EPISODES_PER_TASK}" \
        EVAL_NUM_TASKS="${NUM_TASKS}" \
        EVAL_CONTROL_HZ=20 \
        LIBERO_EVAL_MAX_STEPS=600 \
        EVAL_LIBERO_ENSEMBLING=1 \
        BASELINE_CKPT="${PUBLIC33_CKPT}" \
        BASELINE_CKPT_ID=33 \
        BASELINE_NAME=seer_public33 \
        OURS_CKPT= \
        RUN_BASELINE=1 \
        RUN_OURS_FULL=0 \
        LRNODE_QUERY_INTERVALS_STR= \
        FASTV_ENABLED="${enabled}" \
        FASTV_PRUNE_LAYER="${layer}" \
        FASTV_PRUNE_RATIO="${ratio}" \
        FASTV_SCORE_MODE="${FASTV_SCORE_MODE}" \
        FASTV_RETENTION_DIAGNOSTICS="${FASTV_RETENTION_DIAGNOSTICS}" \
        NODE_NUM="${NODE_NUM}" \
        MASTER_PORT="${port}" \
        SAVE_VIDEO=0 \
        SAVE_VIDEO_SUCC=0 \
        SAVE_VIDEO_FAIL=0 \
        SAVE_VIDEO_ALL_RANKS=0 \
        EXPERIMENT_NAME=seer_fastv \
        EXPERIMENT_TAG="${row_id}" \
        RESULT_ROOT="${row_root}" \
        bash "${EVAL_SCRIPT}" 2>&1 | tee -a "${log_file}"
    rc=${PIPESTATUS[0]}
    set -e
    validate_row "${row_root}" "${seed}" "${enabled}" "${layer}" "${ratio}" \
        || fail "invalid FastV row=${row_id}; wrapper_rc=${rc}; log=${log_file}"
    if (( rc != 0 )); then
        echo "[WARN] wrapper rc=${rc}, but the complete row passed artifact validation"
    fi
}

write_summary() {
    python - "${RESULT_ROOT}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = []
for path in sorted((root / "eval").glob("*/*/analysis/eval_summary.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    fastv = payload.get("fastv", {})
    row_id = path.parents[2].name
    seed_text = row_id.split("_", 1)[0].removeprefix("seed")
    if not seed_text.isdigit():
        raise RuntimeError(f"cannot recover evaluation seed from row_id={row_id!r}")
    records.append(
        {
            "row_id": row_id,
            "seed": int(seed_text),
            "fastv_enabled": int(bool(fastv.get("enabled"))),
            "fastv_prune_layer": fastv.get("prune_layer"),
            "fastv_prune_ratio": fastv.get("prune_ratio"),
            "fastv_score_mode": fastv.get("score_mode"),
            "fastv_score_layer_count": fastv.get("score_layer_count"),
            "fastv_retention_diagnostics": int(
                bool(fastv.get("retention_diagnostics_enabled"))
            ),
            "tokens_before": fastv.get("tokens_before_pruning"),
            "tokens_after": fastv.get("tokens_after_pruning"),
            "success_rate": payload.get("success_rate"),
            "policy_ms": payload.get("avg_policy_step_latency_ms"),
            "full_forward_ms": payload.get("avg_full_forward_latency_ms"),
            "runtime_verified": fastv.get("runtime_verified"),
            "summary_path": str(path),
        }
    )
if not records:
    raise RuntimeError("no complete FastV summaries found")
fieldnames = list(records[0])
with (root / "campaign_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
lines = [
    "# Seer FastV LIBERO-Long campaign",
    "",
    "| Row | Seed | FastV | Layer | Ratio | Score mode | Score layers | Tokens | SR (%) | Policy (ms) | Full forward (ms) |",
    "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
]
for row in records:
    lines.append(
        f"| {row['row_id']} | {row['seed']} | {row['fastv_enabled']} | "
        f"{row['fastv_prune_layer']} | {float(row['fastv_prune_ratio']):.2f} | "
        f"{row['fastv_score_mode']} | {row['fastv_score_layer_count']} | "
        f"{row['tokens_before']} -> {row['tokens_after']} | "
        f"{100.0 * float(row['success_rate']):.2f} | {float(row['policy_ms']):.3f} | "
        f"{float(row['full_forward_ms']):.3f} |"
    )
(root / "campaign_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[FASTV SUMMARY] rows={len(records)} path={root / 'campaign_summary.md'}")
PY
}

[[ "$(hostname -s)" == "${EXPECTED_HOST}" ]] \
    || fail "expected host ${EXPECTED_HOST}, got $(hostname -s)"
[[ "${GPU_LIST}" == "4,5,6,7" ]] \
    || fail "paper latency protocol requires physical RTX3090 GPUs 4,5,6,7; got ${GPU_LIST}"
[[ "${NODE_NUM}" == "4" ]] || fail "NODE_NUM must be 4"
[[ "${EPISODES_PER_TASK}" -gt 0 && "${NUM_TASKS}" -eq 10 ]] \
    || fail "expected positive EPISODES_PER_TASK and NUM_TASKS=10"
validate_bool RUN_BASELINE "${RUN_BASELINE}"
validate_bool PREFLIGHT_ONLY "${PREFLIGHT_ONLY}"
validate_bool FASTV_RETENTION_DIAGNOSTICS "${FASTV_RETENTION_DIAGNOSTICS}"
[[ "${FASTV_SCORE_MODE}" == "text_mean_first_l" \
    || "${FASTV_SCORE_MODE}" == "last_token_at_l" \
    || "${FASTV_SCORE_MODE}" == "action_mean_first_l" \
    || "${FASTV_SCORE_MODE}" == "hf_last_action_at_l" ]] \
    || fail "unsupported FASTV_SCORE_MODE=${FASTV_SCORE_MODE}"
require_file "${EVAL_SCRIPT}"
[[ -d "${LIBERO_PATH}" ]] || fail "missing LIBERO repository: ${LIBERO_PATH}"
if [[ -n "${INHERITED_VIT_CHECKPOINT_PATH}" && "${INHERITED_VIT_CHECKPOINT_PATH}" != "${VIT_CHECKPOINT_PATH}" ]]; then
    echo "[ENV] ignoring inherited VIT_CHECKPOINT_PATH=${INHERITED_VIT_CHECKPOINT_PATH}"
    echo "[ENV] canonical FastV ViT=${VIT_CHECKPOINT_PATH}"
fi
require_sha256 public33 "${PUBLIC33_CKPT}" "${PUBLIC33_SHA256}"
require_sha256 vit_mae "${VIT_CHECKPOINT_PATH}" "${VIT_SHA256}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS_STR}"
read -r -a GRID_ROWS <<< "${FASTV_GRID}"
[[ "${#EVAL_SEEDS[@]}" -gt 0 && "${#GRID_ROWS[@]}" -gt 0 ]] \
    || fail "EVAL_SEEDS_STR and FASTV_GRID must be non-empty"

python - "${FASTV_GRID}" <<'PY'
import sys
for item in sys.argv[1].split():
    layer_text, ratio_text = item.split(":", 1)
    layer, ratio = int(layer_text), float(ratio_text)
    if not 1 <= layer < 24:
        raise ValueError(f"FastV layer must be in [1, 23], got {layer}")
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"FastV ratio must be in [0, 1), got {ratio}")
print("[VERIFY][OK] FastV grid", sys.argv[1])
PY

echo "[PREFLIGHT] repo=${REPO_ROOT}"
echo "[PREFLIGHT] checkpoint=${PUBLIC33_CKPT}"
echo "[PREFLIGHT] result_root=${RESULT_ROOT}"
echo "[PREFLIGHT] seeds=${EVAL_SEEDS[*]} grid=${GRID_ROWS[*]}"
echo "[PREFLIGHT] renderer=egl episodes_per_task=${EPISODES_PER_TASK} tasks=${NUM_TASKS}"
echo "[PREFLIGHT] FastV candidates=visual-only selection_scope=global_visual"
echo "[PREFLIGHT] retention_diagnostics=${FASTV_RETENTION_DIAGNOSTICS} (disable for latency rows)"
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PREFLIGHT][DONE] no evaluation launched"
    exit 0
fi

mkdir -p "${RESULT_ROOT}/eval" "${RESULT_ROOT}/logs"
verify_source_lock
row_index=0
for seed in "${EVAL_SEEDS[@]}"; do
    if [[ "${RUN_BASELINE}" == "1" ]]; then
        run_row "seed${seed}_baseline" "${seed}" 0 2 0.0 "$((MASTER_PORT_BASE + row_index))"
        row_index=$((row_index + 1))
    fi
    for spec in "${GRID_ROWS[@]}"; do
        layer="${spec%%:*}"
        ratio="${spec#*:}"
        ratio_tag="${ratio//./p}"
        run_row "seed${seed}_fastv_l${layer}_r${ratio_tag}" "${seed}" 1 "${layer}" "${ratio}" \
            "$((MASTER_PORT_BASE + row_index))"
        row_index=$((row_index + 1))
    done
done
write_summary
printf 'status=COMPLETE\ntime=%s\n' "$(date --iso-8601=seconds)" \
    > "${RESULT_ROOT}/campaign_complete.env"
echo "[DONE] ${RESULT_ROOT}/campaign_summary.md"
