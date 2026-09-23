#!/usr/bin/env bash

set -euo pipefail

# Diagnostic K=4 sweep over frozen LR-NODE adapter checkpoints.
#
# Default matrix:
#   official Seer teachers: 33, 36, 38
#   LR-NODE adapters:       31, 32, 33, 34, 35, 36, 37, 38
#
# K=1 is not repeated for every adapter because the LR-NODE modules are unused
# on the full-forward path. The completed teacher-specific K=1 baseline from
# official_seer_k4_protocol.sh is reused for paired analysis.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/../../../.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-${REPO_ROOT}/results/seer/lrnode/official_seer_libero_k4_v1}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${PROTOCOL_ROOT}/train/adapters}"
SHARED_SEER_ROOT="${SHARED_SEER_ROOT:-${REPO_ROOT}}"
OFFICIAL_CKPT_ROOT="${OFFICIAL_CKPT_ROOT:-${SHARED_SEER_ROOT}/checkpoints/Seer_LIBERO_LONG/Seer}"
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH:-${SHARED_SEER_ROOT}/checkpoints/vit_mae/mae_pretrain_vit_base.pth}"

TEACHER_CKPT_IDS_STR="${TEACHER_CKPT_IDS_STR:-33 36 38}"
ADAPTER_CKPT_IDS_STR="${ADAPTER_CKPT_IDS_STR:-31 32 33 34 35 36 37 38}"
SWEEP_NAME="${SWEEP_NAME:-official_seer_k4_adapter_ckpt31_38_sweep_v1}"
SWEEP_ROOT="${SWEEP_ROOT:-${PROTOCOL_ROOT}/eval/${SWEEP_NAME}}"
MAIN_COMPLETION_MARKER="${MAIN_COMPLETION_MARKER:-${PROTOCOL_ROOT}/official_seer_k4_summary.csv}"
WAIT_FOR_MAIN_PROTOCOL="${WAIT_FOR_MAIN_PROTOCOL:-1}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-172800}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NODE_NUM="${NODE_NUM:-4}"
export EVAL_CONTROL_HZ="${EVAL_CONTROL_HZ:-20}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export VIDEO_FPS="${VIDEO_FPS:-20}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
export LRNODE_EVAL_SHADOW_FULL_FORWARD="${LRNODE_EVAL_SHADOW_FULL_FORWARD:-0}"
export LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD="${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

read -r -a TEACHER_CKPT_IDS <<< "${TEACHER_CKPT_IDS_STR}"
read -r -a ADAPTER_CKPT_IDS <<< "${ADAPTER_CKPT_IDS_STR}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

activate_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "seer_libero" ]]; then
        return
    fi
    command -v conda >/dev/null 2>&1 || fail "Activate conda env seer_libero before running."
    eval "$(conda shell.bash hook)"
    conda activate seer_libero
}

adapter_run_name() {
    echo "official_seer_ckpt${1}_lrnode_adapter_v1"
}

official_ckpt() {
    echo "${OFFICIAL_CKPT_ROOT}/${1}.pth"
}

adapter_ckpt() {
    echo "${ADAPTER_ROOT}/$(adapter_run_name "$1")/${2}.pth"
}

baseline_reference_root() {
    echo "${PROTOCOL_ROOT}/eval/baseline_reference/ckpt_${1}"
}

wait_for_main_protocol() {
    [[ "${WAIT_FOR_MAIN_PROTOCOL}" == "1" ]] || return 0
    local elapsed=0
    while [[ ! -f "${MAIN_COMPLETION_MARKER}" ]]; do
        if ! pgrep -f "[o]fficial_seer_k4_protocol.sh" >/dev/null; then
            fail "Main protocol is not running and completion marker is missing: ${MAIN_COMPLETION_MARKER}"
        fi
        if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
            fail "Timed out waiting ${WAIT_TIMEOUT_SECONDS}s for ${MAIN_COMPLETION_MARKER}"
        fi
        echo "[WAIT] main protocol still running; elapsed=${elapsed}s; next_check=${WAIT_INTERVAL_SECONDS}s"
        sleep "${WAIT_INTERVAL_SECONDS}"
        elapsed=$((elapsed + WAIT_INTERVAL_SECONDS))
    done
    echo "[WAIT][DONE] main protocol completed: ${MAIN_COMPLETION_MARKER}"
}

validate_ids_and_inputs() {
    [[ "${#TEACHER_CKPT_IDS[@]}" -gt 0 ]] || fail "TEACHER_CKPT_IDS_STR is empty."
    [[ "${#ADAPTER_CKPT_IDS[@]}" -gt 0 ]] || fail "ADAPTER_CKPT_IDS_STR is empty."
    local teacher adapter
    for teacher in "${TEACHER_CKPT_IDS[@]}"; do
        [[ "${teacher}" =~ ^(33|36|38)$ ]] || fail "Unsupported teacher checkpoint: ${teacher}"
        [[ -f "$(official_ckpt "${teacher}")" ]] || fail "Missing official checkpoint: $(official_ckpt "${teacher}")"
        [[ -f "$(baseline_reference_root "${teacher}")/official_baseline_observation.json" ]] || \
            fail "Missing baseline reference for teacher ${teacher}."
        for adapter in "${ADAPTER_CKPT_IDS[@]}"; do
            [[ "${adapter}" =~ ^[0-9]+$ ]] || fail "Invalid adapter checkpoint ID: ${adapter}"
            [[ -f "$(adapter_ckpt "${teacher}" "${adapter}")" ]] || \
                fail "Missing adapter checkpoint: $(adapter_ckpt "${teacher}" "${adapter}")"
        done
    done
    [[ -f "${VIT_CHECKPOINT_PATH}" ]] || fail "Missing ViT checkpoint: ${VIT_CHECKPOINT_PATH}"
}

verify_variant() {
    local teacher="$1"
    local adapter="$2"
    local variant_root="$3"
    TEACHER="${teacher}" \
    ADAPTER="${adapter}" \
    VARIANT_ROOT="${variant_root}" \
    BASELINE_ROOT="$(baseline_reference_root "${teacher}")" \
    ADAPTER_PATH="$(adapter_ckpt "${teacher}" "${adapter}")" \
    python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

import torch

teacher = int(os.environ["TEACHER"])
adapter_id = int(os.environ["ADAPTER"])
variant_root = Path(os.environ["VARIANT_ROOT"])
baseline_root = Path(os.environ["BASELINE_ROOT"])
adapter_path = Path(os.environ["ADAPTER_PATH"])

summary_paths = list(variant_root.glob("ours_*_skip_K4_*/analysis/eval_summary.json"))
episode_paths = list(variant_root.glob("ours_*_skip_K4_*/analysis/eval_episode_metrics.csv"))
baseline_summary_paths = list(baseline_root.glob("baseline_*/analysis/eval_summary.json"))
baseline_episode_paths = list(baseline_root.glob("baseline_*/analysis/eval_episode_metrics.csv"))
if not all(len(paths) == 1 for paths in (
    summary_paths, episode_paths, baseline_summary_paths, baseline_episode_paths
)):
    raise SystemExit(
        "[VERIFY][FAIL] expected one K4 and one baseline summary/episode CSV; "
        f"got {len(summary_paths)}/{len(episode_paths)}/"
        f"{len(baseline_summary_paths)}/{len(baseline_episode_paths)}"
    )

summary = json.loads(summary_paths[0].read_text())
baseline = json.loads(baseline_summary_paths[0].read_text())
with episode_paths[0].open(newline="") as handle:
    episodes = list(csv.DictReader(handle))

checkpoint = torch.load(adapter_path, map_location="cpu")
state = checkpoint["model_state_dict"]
allowed = ("module.lrnode_delta_encoder.", "module.lrnode_dynamics.")
q = summary["query_reduction"]
checks = {
    "adapter_epoch": int(checkpoint["epoch"]) == adapter_id,
    "adapter_tensor_count": len(state) == 30,
    "adapter_numel": sum(value.numel() for value in state.values()) == 470146,
    "adapter_only_lrnode": all(key.startswith(allowed) for key in state),
    "episodes_200": len(episodes) == 200,
    "lrnode_enabled": summary["lrnode"]["enabled"] is True,
    "skip_enabled": summary["lrnode"]["eval_skip_full_forward"] is True,
    "query_interval_4": int(summary["lrnode"]["query_interval"]) == 4,
    "calls_partition_steps": q["num_full_forward_calls"] + q["num_lrnode_update_calls"] == q["num_env_steps"],
    "fast_equals_updates": q["num_fast_encoder_calls"] == q["num_lrnode_update_calls"],
    "head_equals_updates": q["num_action_head_calls"] == q["num_lrnode_update_calls"],
    "no_fallback": q["num_fallback_full_calls"] == 0,
    "no_ablation": all(q[name] == 0 for name in (
        "num_hold_action_steps", "num_hold_latent_steps",
        "num_chunk_token_steps", "num_no_delta_steps",
    )),
    "finite_sr": math.isfinite(float(summary["success_rate"])),
}
failed = [name for name, passed in checks.items() if not passed]
payload = {
    "status": "passed" if not failed else "failed",
    "teacher_checkpoint": teacher,
    "adapter_checkpoint": adapter_id,
    "adapter_path": str(adapter_path),
    "baseline_success_rate": baseline["success_rate"],
    "k4_success_rate": summary["success_rate"],
    "k4_delta_pp": 100.0 * (summary["success_rate"] - baseline["success_rate"]),
    "checks": checks,
}
(variant_root / "variant_guard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
if failed:
    raise SystemExit(f"[VERIFY][FAIL] {payload}")
print("[VERIFY][OK]", json.dumps(payload, indent=2))
PY
}

write_summary() {
    SWEEP_ROOT="${SWEEP_ROOT}" \
    PROTOCOL_ROOT="${PROTOCOL_ROOT}" \
    python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ["SWEEP_ROOT"])
protocol_root = Path(os.environ["PROTOCOL_ROOT"])

def read_episodes(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["task_id"]), int(row["episode_id"]), int(row["seed"])): int(row["success"])
        for row in rows
    }

def mcnemar_exact(baseline_only, k4_only):
    count = baseline_only + k4_only
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, k) for k in range(min(baseline_only, k4_only) + 1)) / (2 ** count)
    return min(1.0, 2.0 * tail)

rows = []
for guard_path in sorted(root.glob("teacher_ckpt*/adapter_ckpt*/variant_guard.json")):
    guard = json.loads(guard_path.read_text())
    teacher = int(guard["teacher_checkpoint"])
    adapter = int(guard["adapter_checkpoint"])
    variant_root = guard_path.parent
    summary_path = next(variant_root.glob("ours_*_skip_K4_*/analysis/eval_summary.json"))
    episode_path = next(variant_root.glob("ours_*_skip_K4_*/analysis/eval_episode_metrics.csv"))
    baseline_root = protocol_root / "eval" / "baseline_reference" / f"ckpt_{teacher}"
    baseline_summary_path = next(baseline_root.glob("baseline_*/analysis/eval_summary.json"))
    baseline_episode_path = next(baseline_root.glob("baseline_*/analysis/eval_episode_metrics.csv"))
    summary = json.loads(summary_path.read_text())
    baseline = json.loads(baseline_summary_path.read_text())
    ours_episodes = read_episodes(episode_path)
    baseline_episodes = read_episodes(baseline_episode_path)
    keys = sorted(set(ours_episodes) & set(baseline_episodes))
    baseline_only = sum(baseline_episodes[key] and not ours_episodes[key] for key in keys)
    k4_only = sum(not baseline_episodes[key] and ours_episodes[key] for key in keys)
    q = summary["query_reduction"]
    smooth = summary["action_smoothness"]
    rows.append({
        "teacher_checkpoint": teacher,
        "adapter_checkpoint": adapter,
        "baseline_sr_pct": 100.0 * baseline["success_rate"],
        "k4_sr_pct": 100.0 * summary["success_rate"],
        "k4_delta_pp": 100.0 * (summary["success_rate"] - baseline["success_rate"]),
        "baseline_only_successes": baseline_only,
        "k4_only_successes": k4_only,
        "paired_mcnemar_exact_p": mcnemar_exact(baseline_only, k4_only),
        "full_query_reduction_pct": 100.0 * q["full_query_reduction_ratio"],
        "effective_query_interval": q["effective_query_interval"],
        "num_env_steps": q["num_env_steps"],
        "num_full_forward_calls": q["num_full_forward_calls"],
        "num_lrnode_update_calls": q["num_lrnode_update_calls"],
        "avg_full_forward_ms_per_call": summary["avg_full_forward_latency_ms"],
        "avg_lrnode_ms_per_call": summary["avg_lrnode_latency_ms"],
        "avg_policy_step_ms": summary["avg_policy_step_latency_ms"],
        "action_delta_l2_mean": smooth["action_delta_l2_mean"],
        "action_jerk_l2_mean": smooth["action_jerk_l2_mean"],
        "action_jerk_l2_p95": smooth["action_jerk_l2_p95"],
        "gripper_switch_rate": smooth["gripper_switch_rate"],
        "summary_path": str(summary_path),
    })

rows.sort(key=lambda item: (item["teacher_checkpoint"], item["adapter_checkpoint"]))
csv_path = root / "official_seer_k4_adapter_ckpt_sweep_summary.csv"
json_path = root / "official_seer_k4_adapter_ckpt_sweep_summary.json"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"[SUMMARY] {csv_path}")
print(f"[SUMMARY] {json_path}")
for row in rows:
    print(row)
PY
}

activate_env
wait_for_main_protocol
validate_ids_and_inputs

mkdir -p "${SWEEP_ROOT}/_launch_logs"
cat > "${SWEEP_ROOT}/experiment_config.env" <<EOF
SCRIPT=${SCRIPT_PATH}
PROTOCOL_ROOT=${PROTOCOL_ROOT}
SWEEP_ROOT=${SWEEP_ROOT}
TEACHER_CKPT_IDS_STR=${TEACHER_CKPT_IDS_STR}
ADAPTER_CKPT_IDS_STR=${ADAPTER_CKPT_IDS_STR}
QUERY_INTERVAL=4
RUN_BASELINE=0
RUN_OURS_FULL=0
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
NODE_NUM=${NODE_NUM}
SAVE_VIDEO=${SAVE_VIDEO}
WAIT_FOR_MAIN_PROTOCOL=${WAIT_FOR_MAIN_PROTOCOL}
MAIN_COMPLETION_MARKER=${MAIN_COMPLETION_MARKER}
EOF

echo "[SWEEP] teachers=${TEACHER_CKPT_IDS_STR}"
echo "[SWEEP] adapters=${ADAPTER_CKPT_IDS_STR}"
echo "[SWEEP] variants=$((${#TEACHER_CKPT_IDS[@]} * ${#ADAPTER_CKPT_IDS[@]}))"
echo "[SWEEP] K=4 only; teacher-specific K=1 references are reused"
echo "[SWEEP] root=${SWEEP_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY RUN][OK] all inputs exist; no evaluation was started"
    exit 0
fi

index=0
for teacher in "${TEACHER_CKPT_IDS[@]}"; do
    for adapter in "${ADAPTER_CKPT_IDS[@]}"; do
        variant_root="${SWEEP_ROOT}/teacher_ckpt${teacher}/adapter_ckpt${adapter}"
        variant_guard="${variant_root}/variant_guard.json"
        if [[ -f "${variant_guard}" && "${REUSE_COMPLETED}" == "1" ]]; then
            echo "[SKIP] completed teacher=${teacher} adapter=${adapter}"
            verify_variant "${teacher}" "${adapter}" "${variant_root}"
            index=$((index + 1))
            continue
        fi
        [[ ! -e "${variant_root}" ]] || \
            fail "Incomplete variant directory exists: ${variant_root}. Move it before retrying."

        run_name="$(adapter_run_name "${teacher}")"
        launch_log="${SWEEP_ROOT}/_launch_logs/teacher${teacher}_adapter${adapter}.log"
        master_port="$((14100 + index))"
        echo "[RUN] teacher=${teacher} adapter=${adapter} port=${master_port}"
        (
            cd "${UPSTREAM_DIR}"
            BASELINE_CKPT="$(official_ckpt "${teacher}")" \
            BASELINE_CKPT_ID="${teacher}" \
            BASELINE_RUN_NAME="official_seer_libero_long" \
            BASELINE_NAME="official_seer_ckpt${teacher}" \
            BASELINE_CKPT_TAG="official_seer_ckpt${teacher}" \
            OURS_CKPT="$(adapter_ckpt "${teacher}" "${adapter}")" \
            OURS_CKPT_ID="${adapter}" \
            OURS_RUN_NAME="${run_name}" \
            OURS_NAME="official_seer_ckpt${teacher}_lrnode_adapter${adapter}" \
            OURS_CKPT_TAG="official_seer_ckpt${teacher}_lrnode_adapter_ckpt${adapter}" \
            METHOD_TAG="official_seer_ckpt${teacher}_lrnode_adapter_ckpt${adapter}" \
            LRNODE_EVAL_BASE_CKPT="$(official_ckpt "${teacher}")" \
            LRNODE_TRAIN_PROTOCOL="adapter" \
            LRNODE_FREEZE_SEER_FOR_ADAPTER=1 \
            LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1 \
            RUN_BASELINE=0 \
            RUN_OURS_FULL=0 \
            LRNODE_QUERY_INTERVALS_STR="4" \
            EXPERIMENT_NAME="official_seer_k4_adapter_ckpt_sweep" \
            EXPERIMENT_TAG="teacher${teacher}_adapter${adapter}" \
            RESULT_ROOT="${variant_root}" \
            LRNODE_PROTOCOL_ROOT="${PROTOCOL_ROOT}" \
            VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
            MASTER_PORT="${master_port}" \
            bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
        ) 2>&1 | tee "${launch_log}"
        verify_variant "${teacher}" "${adapter}" "${variant_root}"
        index=$((index + 1))
    done
done

write_summary
echo "[DONE] ${SWEEP_ROOT}"
