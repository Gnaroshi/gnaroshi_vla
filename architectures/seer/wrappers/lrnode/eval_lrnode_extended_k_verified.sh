#!/usr/bin/env bash

set -euo pipefail

# Verified Seer LR-NODE high-K sweep.
#
# Stage A reruns ckpt39 K=4 after code/instrumentation changes and requires the
# exact known behavior (182/200 successes and the same call/smoothness metrics).
# Stage B starts only after that guard passes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
UPSTREAM_DIR="${ROOT_DIR}/architectures/seer/upstream"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "seer_libero" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] conda is unavailable; activate seer_libero before running." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate seer_libero
fi

: "${BASELINE_CKPT:?Set BASELINE_CKPT to the pinned Seer ckpt33 path.}"
: "${OURS_CKPT:?Set OURS_CKPT to the pinned LR-NODE adapter ckpt39 path.}"
: "${VIT_CHECKPOINT_PATH:?Set VIT_CHECKPOINT_PATH to mae_pretrain_vit_base.pth.}"
: "${LIBERO_PATH:?Set LIBERO_PATH to the LIBERO repository.}"
BASELINE_SHA256="${BASELINE_SHA256:-a999bf839acfb6f77beb8b86576933254f1981d2bacd1f0d269da093d7205cc5}"
OURS_SHA256="${OURS_SHA256:-badc74e135003fee91ccc69c76fe4f225aece856f487236ca7f626424504f132}"
EXTENDED_K_LIST_STR="${EXTENDED_K_LIST_STR:-9 10 11 12 13 14 15 16}"

read -r -a EXTENDED_K_LIST <<< "${EXTENDED_K_LIST_STR}"
if [[ "${#EXTENDED_K_LIST[@]}" -eq 0 ]]; then
    echo "[ERROR] EXTENDED_K_LIST_STR is empty." >&2
    exit 1
fi
declare -A SEEN_K=()
for k in "${EXTENDED_K_LIST[@]}"; do
    if [[ ! "${k}" =~ ^[0-9]+$ ]] || (( k <= 8 )); then
        echo "[ERROR] Extended K values must be unique integers greater than 8; got '${k}'." >&2
        exit 1
    fi
    if [[ -n "${SEEN_K[${k}]:-}" ]]; then
        echo "[ERROR] Duplicate K in EXTENDED_K_LIST_STR: ${k}" >&2
        exit 1
    fi
    SEEN_K[${k}]=1
done

verify_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing ${label}: ${path}" >&2
        exit 1
    fi
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[ERROR] ${label} SHA256 mismatch: expected=${expected}, actual=${actual}, path=${path}" >&2
        exit 1
    fi
    echo "[VERIFY][OK] ${label} sha256=${actual}"
}

verify_sha256 "baseline ckpt33" "${BASELINE_CKPT}" "${BASELINE_SHA256}"
verify_sha256 "LR-NODE adapter ckpt39" "${OURS_CKPT}" "${OURS_SHA256}"
if [[ ! -f "${VIT_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] Missing ViT checkpoint: ${VIT_CHECKPOINT_PATH}" >&2
    exit 1
fi

timestamp="${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/seer/lrnode/eval/best91_guard_extended_k_${timestamp}}"
STAGE_A_ROOT="${RESULT_ROOT}/stage_a_k4_behavior_guard"
STAGE_B_ROOT="${RESULT_ROOT}/stage_b_extended_k"
mkdir -p "${RESULT_ROOT}"

BASELINE_CKPT="${BASELINE_CKPT}" OURS_CKPT="${OURS_CKPT}" \
CHECKPOINT_PREFLIGHT_OUT="${RESULT_ROOT}/checkpoint_preflight.json" \
python - <<'PY'
import json
import os
from pathlib import Path

import torch

base_path = Path(os.environ["BASELINE_CKPT"])
adapter_path = Path(os.environ["OURS_CKPT"])
base = torch.load(base_path, map_location="cpu")["model_state_dict"]
adapter = torch.load(adapter_path, map_location="cpu")["model_state_dict"]

allowed_adapter_prefixes = (
    "module.lrnode_delta_encoder.",
    "module.lrnode_dynamics.",
)
bad_adapter_keys = sorted(k for k in adapter if not k.startswith(allowed_adapter_prefixes))
if bad_adapter_keys:
    raise SystemExit(f"[VERIFY][FAIL] non-LR-NODE keys in adapter: {bad_adapter_keys}")
if len(adapter) != 30:
    raise SystemExit(f"[VERIFY][FAIL] expected 30 adapter tensors, found {len(adapter)}")
if sum(v.numel() for v in adapter.values()) != 470146:
    raise SystemExit("[VERIFY][FAIL] adapter parameter count is not 470146")
if any("lrnode_" in k for k in base):
    raise SystemExit("[VERIFY][FAIL] baseline checkpoint unexpectedly contains LR-NODE keys")
if set(base) & set(adapter):
    raise SystemExit("[VERIFY][FAIL] baseline and adapter checkpoint key sets overlap")

required_base_prefixes = (
    "module.transformer_backbone.",
    "module.perceiver_resampler.",
    "module.action_decoder.",
    "module.arm_action_decoder.",
    "module.gripper_action_decoder.",
)
missing_base_groups = [p for p in required_base_prefixes if not any(k.startswith(p) for k in base)]
if missing_base_groups:
    raise SystemExit(f"[VERIFY][FAIL] baseline checkpoint misses groups: {missing_base_groups}")

payload = {
    "baseline_path": str(base_path),
    "baseline_tensor_count": len(base),
    "baseline_numel": sum(v.numel() for v in base.values()),
    "adapter_path": str(adapter_path),
    "adapter_tensor_count": len(adapter),
    "adapter_numel": sum(v.numel() for v in adapter.values()),
    "adapter_prefixes": list(allowed_adapter_prefixes),
    "base_adapter_key_overlap": 0,
    "status": "passed",
}
Path(os.environ["CHECKPOINT_PREFLIGHT_OUT"]).write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print("[VERIFY][OK] checkpoint preflight", payload)
PY

export BASELINE_CKPT OURS_CKPT VIT_CHECKPOINT_PATH
export BASELINE_CKPT_ID=33
export OURS_CKPT_ID=39
export BASELINE_RUN_NAME="sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040"
export OURS_RUN_NAME="sd1_distill_node_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_20260620_202533"
export BASELINE_NAME="seer_scratch_baseline"
export METHOD_TAG="lrnode_distill_ckpt33_to_ckpt39"
export OURS_NAME="lrnode_distill_ckpt39"
export LRNODE_EVAL_BASE_CKPT="${BASELINE_CKPT}"
export EVAL_CONTROL_HZ=20
export NODE_NUM=4
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export SAVE_VIDEO_SUCC="${SAVE_VIDEO_SUCC:-1}"
export SAVE_VIDEO_FAIL="${SAVE_VIDEO_FAIL:-1}"
export SAVE_VIDEO_ALL_RANKS="${SAVE_VIDEO_ALL_RANKS:-1}"
export VIDEO_FPS="${VIDEO_FPS:-20}"
export VIDEO_STRIDE="${VIDEO_STRIDE:-1}"
export LRNODE_EVAL_STEP_LOG=1
export LRNODE_EVAL_SHADOW_FULL_FORWARD=0
export LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=1
export LRNODE_EVAL_REFRESH_POLICY=periodic

cat > "${RESULT_ROOT}/launch_config.env" <<EOF
SCRIPT=architectures/seer/wrappers/lrnode/eval_lrnode_extended_k_verified.sh
RESULT_ROOT=${RESULT_ROOT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
BASELINE_CKPT=${BASELINE_CKPT}
BASELINE_SHA256=${BASELINE_SHA256}
OURS_CKPT=${OURS_CKPT}
OURS_SHA256=${OURS_SHA256}
STAGE_A_K=4
STAGE_B_K="${EXTENDED_K_LIST_STR}"
EVAL_CONTROL_HZ=${EVAL_CONTROL_HZ}
NODE_NUM=${NODE_NUM}
SAVE_VIDEO=${SAVE_VIDEO}
LRNODE_EVAL_STEP_LOG=${LRNODE_EVAL_STEP_LOG}
LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD}
EOF

git -C "${ROOT_DIR}" rev-parse HEAD > "${RESULT_ROOT}/git_commit.txt"
git -C "${ROOT_DIR}" status --short > "${RESULT_ROOT}/git_status_short.txt"
sha256sum \
    "${UPSTREAM_DIR}/eval_libero.py" \
    "${UPSTREAM_DIR}/models/seer_model.py" \
    "${UPSTREAM_DIR}/models/lrnode_modules.py" \
    "${UPSTREAM_DIR}/utils/eval_utils_libero.py" \
    "${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh" \
    > "${RESULT_ROOT}/source_sha256.txt"

cd "${UPSTREAM_DIR}"

echo "[STAGE A] strict K4 behavior guard"
EXPERIMENT_NAME="lrnode_k4_behavior_guard" \
EXPERIMENT_TAG="${timestamp}_stage_a" \
RESULT_ROOT="${STAGE_A_ROOT}" \
RUN_BASELINE=0 \
RUN_OURS_FULL=0 \
LRNODE_QUERY_INTERVALS_STR="4" \
MASTER_PORT="${MASTER_PORT_STAGE_A:-12851}" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh

STAGE_A_ROOT="${STAGE_A_ROOT}" python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

root = Path(os.environ["STAGE_A_ROOT"])
summary_paths = list(root.glob("*/analysis/eval_summary.json"))
if len(summary_paths) != 1:
    raise SystemExit(f"[VERIFY][FAIL] expected one K4 summary, found {len(summary_paths)}")
summary = json.loads(summary_paths[0].read_text())
run_dir = summary_paths[0].parents[1]
with (run_dir / "analysis/eval_episode_metrics.csv").open(newline="") as handle:
    episodes = list(csv.DictReader(handle))

expected_task_successes = [18, 19, 20, 20, 15, 20, 18, 20, 16, 16]
actual_task_successes = []
for task_id in range(10):
    rows = [x for x in episodes if int(x["task_id"]) == task_id]
    if len(rows) != 20:
        raise SystemExit(f"[VERIFY][FAIL] K4 task {task_id} episode count={len(rows)}")
    actual_task_successes.append(sum(int(x["success"]) for x in rows))

q = summary["query_reduction"]
smooth = summary["action_smoothness"]
checks = {
    "lrnode_enabled": summary["lrnode"]["enabled"] is True,
    "skip_enabled": summary["lrnode"]["eval_skip_full_forward"] is True,
    "stepwise_mode": summary["lrnode_eval_ablation_mode"] == "stepwise",
    "K4": int(summary["lrnode_query_interval"]) == 4,
    "episodes_200": len(episodes) == 200,
    "successes_182": sum(int(x["success"]) for x in episodes) == 182,
    "success_rate_091": math.isclose(summary["success_rate"], 0.91, abs_tol=1e-12),
    "task_success_vector": actual_task_successes == expected_task_successes,
    "env_steps": q["num_env_steps"] == 60320,
    "full_calls": q["num_full_forward_calls"] == 15145,
    "update_calls": q["num_lrnode_update_calls"] == 45175,
    "fast_calls": q["num_fast_encoder_calls"] == 45175,
    "head_step_calls": q["num_action_head_calls"] == 45175,
    "no_fallback": q["num_fallback_full_calls"] == 0,
    "no_hold_action": q["num_hold_action_steps"] == 0,
    "no_hold_latent": q["num_hold_latent_steps"] == 0,
    "no_chunk": q["num_chunk_token_steps"] == 0,
    "no_delta_ablation": q["num_no_delta_steps"] == 0,
    "jerk_mean": math.isclose(smooth["action_jerk_l2_mean"], 0.10887144206455103, abs_tol=1e-12),
    "jerk_p95": math.isclose(smooth["action_jerk_l2_p95"], 0.5639644220853222, abs_tol=1e-12),
    "gripper_switch": math.isclose(smooth["gripper_switch_rate"], 0.02030880915901121, abs_tol=1e-12),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(
        f"[VERIFY][FAIL] K4 behavior changed; failed={failed}, "
        f"task_successes={actual_task_successes}. Stage B was not started."
    )
print("[VERIFY][OK] K4 behavior guard passed", checks)
PY

echo "[STAGE B] verified LR-NODE extended K sweep: ${EXTENDED_K_LIST_STR}"
EXPERIMENT_NAME="lrnode_extended_k" \
EXPERIMENT_TAG="${timestamp}_stage_b" \
RESULT_ROOT="${STAGE_B_ROOT}" \
RUN_BASELINE=0 \
RUN_OURS_FULL=0 \
LRNODE_QUERY_INTERVALS_STR="${EXTENDED_K_LIST_STR}" \
MASTER_PORT="${MASTER_PORT_STAGE_B:-12852}" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh

RESULT_ROOT="${RESULT_ROOT}" STAGE_A_ROOT="${STAGE_A_ROOT}" STAGE_B_ROOT="${STAGE_B_ROOT}" \
EXPECTED_K_LIST="${EXTENDED_K_LIST_STR}" python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

result_root = Path(os.environ["RESULT_ROOT"])
stage_a = Path(os.environ["STAGE_A_ROOT"])
stage_b = Path(os.environ["STAGE_B_ROOT"])
expected_k = [int(x) for x in os.environ["EXPECTED_K_LIST"].split()]
summary_paths = sorted(stage_b.glob("*/analysis/eval_summary.json"))
if len(summary_paths) != len(expected_k):
    raise SystemExit(
        f"[VERIFY][FAIL] expected {len(expected_k)} extended summaries, found {len(summary_paths)}"
    )

audited = []
for path in summary_paths:
    summary = json.loads(path.read_text())
    run_dir = path.parents[1]
    k = int(summary["lrnode_query_interval"])
    q = summary["query_reduction"]
    with (run_dir / "analysis/eval_episode_metrics.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    videos = list((run_dir / "eval_videos").glob("**/*.mp4"))
    checks = {
        "expected_k": k in expected_k,
        "lrnode_enabled": summary["lrnode"]["enabled"] is True,
        "skip_enabled": summary["lrnode"]["eval_skip_full_forward"] is True,
        "stepwise": summary["lrnode_eval_ablation_mode"] == "stepwise",
        "episodes_200": len(episodes) == 200,
        "episode_coverage": {
            (int(x["task_id"]), int(x["episode_id"])) for x in episodes
        } == {(task, episode) for task in range(10) for episode in range(20)},
        "sr_matches": math.isclose(
            summary["success_rate"],
            sum(int(x["success"]) for x in episodes) / 200.0,
            abs_tol=1e-12,
        ),
        "calls_partition_steps": (
            q["num_full_forward_calls"] + q["num_lrnode_update_calls"] == q["num_env_steps"]
        ),
        "fast_equals_updates": q["num_fast_encoder_calls"] == q["num_lrnode_update_calls"],
        "head_steps_equal_updates": q["num_action_head_calls"] == q["num_lrnode_update_calls"],
        "no_fallback": q["num_fallback_full_calls"] == 0,
        "no_ablation_calls": all(
            q[name] == 0 for name in (
                "num_hold_action_steps",
                "num_hold_latent_steps",
                "num_chunk_token_steps",
                "num_no_delta_steps",
            )
        ),
        "videos_200": len(videos) == 200 if summary["video"]["enabled"] else True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"[VERIFY][FAIL] K={k} failed={failed}")
    audited.append({
        "k": k,
        "successes": sum(int(x["success"]) for x in episodes),
        "success_rate": summary["success_rate"],
        "env_steps": q["num_env_steps"],
        "full_calls": q["num_full_forward_calls"],
        "lrnode_updates": q["num_lrnode_update_calls"],
        "full_query_reduction": q["full_query_reduction_ratio"],
        "avg_policy_step_latency_ms": summary["avg_policy_step_latency_ms"],
        "avg_lrnode_latency_ms": summary["avg_lrnode_latency_ms"],
        "summary_path": str(path),
    })

if sorted(x["k"] for x in audited) != sorted(expected_k):
    raise SystemExit(f"[VERIFY][FAIL] K coverage mismatch: {audited}")

(result_root / "extended_k_audit.json").write_text(
    json.dumps({"status": "passed", "rows": audited}, indent=2), encoding="utf-8"
)

combined = []
for stage, source in (
    ("k4_behavior_guard", stage_a / "experiment_summary.csv"),
    ("extended_k", stage_b / "experiment_summary.csv"),
):
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            combined.append({"stage": stage, **row})
combined.sort(key=lambda row: int(row["query_interval"]))
output = result_root / "k4_guard_and_extended_k_summary.csv"
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(combined[0].keys()))
    writer.writeheader()
    writer.writerows(combined)

print(f"[VERIFY][OK] extended K audit saved: {result_root / 'extended_k_audit.json'}")
print(f"[SUMMARY] {output}")
for row in combined:
    print(
        f"K={row['query_interval']} SR={row['success_rate_pct']}% "
        f"reduction={row['full_query_reduction_pct']}% policy_ms={row['avg_policy_step_ms']}"
    )
PY

echo "[DONE] ${RESULT_ROOT}"
