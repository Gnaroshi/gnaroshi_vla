#!/usr/bin/env bash

set -euo pipefail

# Reproduce the Seer LR-NODE ckpt39 K=4 result, then extend the same
# env-step periodic-refresh protocol to K=5,6,7,8.
#
# Stage A is a guard: Stage B starts only when K=4 reproduces 182/200 = 91% SR.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
UPSTREAM_DIR="${ROOT_DIR}/architectures/seer/upstream"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "seer_libero" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "[ERROR] conda is unavailable; activate seer_libero before running." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate seer_libero
fi

: "${BASELINE_CKPT:?Set BASELINE_CKPT to the pinned Seer ckpt33 path.}"
: "${OURS_CKPT:?Set OURS_CKPT to the pinned LR-NODE adapter ckpt39 path.}"
: "${VIT_CHECKPOINT_PATH:?Set VIT_CHECKPOINT_PATH to mae_pretrain_vit_base.pth.}"
: "${LIBERO_PATH:?Set LIBERO_PATH to the LIBERO repository.}"

BASELINE_SHA256="${BASELINE_SHA256:-a999bf839acfb6f77beb8b86576933254f1981d2bacd1f0d269da093d7205cc5}"
OURS_SHA256="${OURS_SHA256:-badc74e135003fee91ccc69c76fe4f225aece856f487236ca7f626424504f132}"

verify_file() {
    local label="$1"
    local path="$2"
    local expected_sha256="$3"
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Missing ${label}: ${path}" >&2
        exit 1
    fi
    local actual_sha256
    actual_sha256="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
        echo "[ERROR] ${label} SHA256 mismatch." >&2
        echo "[ERROR] expected=${expected_sha256}" >&2
        echo "[ERROR] actual=${actual_sha256}" >&2
        echo "[ERROR] path=${path}" >&2
        exit 1
    fi
    echo "[VERIFY][OK] ${label} sha256=${actual_sha256}"
}

verify_file "baseline ckpt33" "${BASELINE_CKPT}" "${BASELINE_SHA256}"
verify_file "LR-NODE adapter ckpt39" "${OURS_CKPT}" "${OURS_SHA256}"
if [[ ! -f "${VIT_CHECKPOINT_PATH}" ]]; then
    echo "[ERROR] Missing ViT checkpoint: ${VIT_CHECKPOINT_PATH}" >&2
    exit 1
fi

timestamp="${EXPERIMENT_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/seer/lrnode/eval/best91_repro_k8_${timestamp}}"
STAGE_A_ROOT="${RESULT_ROOT}/stage_a_k4_repro"
STAGE_B_ROOT="${RESULT_ROOT}/stage_b_k5_to_k8"
mkdir -p "${RESULT_ROOT}"

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
export LRNODE_EVAL_STEP_LOG="${LRNODE_EVAL_STEP_LOG:-1}"
export LRNODE_EVAL_SHADOW_FULL_FORWARD=0
export LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD="${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD:-1}"
export LRNODE_EVAL_REFRESH_POLICY=periodic

cat > "${RESULT_ROOT}/launch_config.env" <<EOF
SCRIPT=architectures/seer/wrappers/lrnode/eval_lrnode_best91_repro_k8.sh
ROOT_DIR=${ROOT_DIR}
UPSTREAM_DIR=${UPSTREAM_DIR}
RESULT_ROOT=${RESULT_ROOT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
BASELINE_CKPT=${BASELINE_CKPT}
BASELINE_SHA256=${BASELINE_SHA256}
OURS_CKPT=${OURS_CKPT}
OURS_SHA256=${OURS_SHA256}
VIT_CHECKPOINT_PATH=${VIT_CHECKPOINT_PATH}
STAGE_A_K=4
STAGE_B_K="5 6 7 8"
EVAL_CONTROL_HZ=${EVAL_CONTROL_HZ}
NODE_NUM=${NODE_NUM}
SAVE_VIDEO=${SAVE_VIDEO}
LRNODE_EVAL_STEP_LOG=${LRNODE_EVAL_STEP_LOG}
LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=${LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD}
EOF

git -C "${ROOT_DIR}" rev-parse HEAD > "${RESULT_ROOT}/git_commit.txt"
git -C "${ROOT_DIR}" status --short > "${RESULT_ROOT}/git_status_short.txt"

cd "${UPSTREAM_DIR}"

echo "[STAGE A] baseline K1 + ours full K1 + ours LR-NODE K4"
EXPERIMENT_NAME="lrnode_best91_repro" \
EXPERIMENT_TAG="${timestamp}_stage_a" \
RESULT_ROOT="${STAGE_A_ROOT}" \
RUN_BASELINE=1 \
RUN_OURS_FULL=1 \
LRNODE_QUERY_INTERVALS_STR="4" \
MASTER_PORT="${MASTER_PORT_STAGE_A:-12841}" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh

STAGE_A_ROOT="${STAGE_A_ROOT}" EXPECTED_K4_SR="${EXPECTED_K4_SR:-91.0}" python - <<'PY'
import csv
import os
from pathlib import Path

summary = Path(os.environ["STAGE_A_ROOT"]) / "experiment_summary.csv"
rows = list(csv.DictReader(summary.open(newline="", encoding="utf-8")))
matches = [
    row for row in rows
    if int(row["query_interval"]) == 4
    and row["skip_full_forward"].lower() == "true"
]
if len(matches) != 1:
    raise SystemExit(f"[VERIFY][FAIL] expected one K4 skip row, found {len(matches)}: {summary}")

actual = float(matches[0]["success_rate_pct"])
expected = float(os.environ["EXPECTED_K4_SR"])
if actual != expected:
    raise SystemExit(
        f"[VERIFY][FAIL] K4 SR did not reproduce: expected={expected:.1f}%, actual={actual:.1f}%. "
        "Stage B was not started."
    )
print(f"[VERIFY][OK] K4 reproduced exactly: {actual:.1f}%")
PY

echo "[STAGE B] ours LR-NODE K=5,6,7,8"
EXPERIMENT_NAME="lrnode_k5_to_k8" \
EXPERIMENT_TAG="${timestamp}_stage_b" \
RESULT_ROOT="${STAGE_B_ROOT}" \
RUN_BASELINE=0 \
RUN_OURS_FULL=0 \
LRNODE_QUERY_INTERVALS_STR="5 6 7 8" \
MASTER_PORT="${MASTER_PORT_STAGE_B:-12842}" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh

RESULT_ROOT="${RESULT_ROOT}" STAGE_A_ROOT="${STAGE_A_ROOT}" STAGE_B_ROOT="${STAGE_B_ROOT}" python - <<'PY'
import csv
import os
from pathlib import Path

result_root = Path(os.environ["RESULT_ROOT"])
sources = [
    ("k4_reproduction", Path(os.environ["STAGE_A_ROOT"]) / "experiment_summary.csv"),
    ("k5_to_k8", Path(os.environ["STAGE_B_ROOT"]) / "experiment_summary.csv"),
]
rows = []
for stage, source in sources:
    for row in csv.DictReader(source.open(newline="", encoding="utf-8")):
        rows.append({"stage": stage, **row})

rows.sort(key=lambda row: (int(row["query_interval"]), row["run_dir"]))
output = result_root / "best91_repro_k8_combined_summary.csv"
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"[SUMMARY] {output}")
for row in rows:
    print(
        f"K={row['query_interval']} run={row['run_dir']} "
        f"SR={row['success_rate_pct']}% reduction={row['full_query_reduction_pct']}% "
        f"policy_ms={row['avg_policy_step_ms']}"
    )
PY

echo "[DONE] ${RESULT_ROOT}"
