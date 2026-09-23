#!/bin/bash

set -euo pipefail
SECONDS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEACHER="${LOCAL_TEACHER_CKPT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/33.pth}"
CANONICAL_ADAPTER="${LOCAL_LATENTLOOP_ADAPTER:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616/train/distill_node/sd1_distill_node_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_20260620_202533/39.pth}"
CANONICAL_ROW="${CANONICAL_ROW:-${REPO_ROOT}/results/seer/latentloop/local_best91/local_best91_segment_grid_legacy_parity_20260731_v2/stage_a_ckpt33/ckpt33_L4_dense/segment_grid_row.json}"
DATASET_ROOT="${ROOT_DIR:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/LIBERO_DATASETS/libero_10_converted}"
VIT_CHECKPOINT="${VIT_CHECKPOINT_PATH:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
CALIBRATION_JSON="${LATENTLOOP_ACTION_CALIBRATION_JSON:?Set LATENTLOOP_ACTION_CALIBRATION_JSON}"
STAGE="${STAGE:-smoke}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/results/seer/latentloop/comparison/train/action_correction/${STAGE}_${RUN_TAG}}"

[[ "${STAGE}" == "smoke" || "${STAGE}" == "full" ]] || {
    echo "[ERROR] STAGE must be smoke or full" >&2; exit 1;
}
for path in "${TEACHER}" "${CANONICAL_ADAPTER}" "${CANONICAL_ROW}" "${CALIBRATION_JSON}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] missing file: ${path}" >&2; exit 1; }
done
[[ -d "${DATASET_ROOT}/libero_10_converted/episodes" ]] || {
    echo "[ERROR] invalid ROOT_DIR; expected ${DATASET_ROOT}/libero_10_converted/episodes" >&2; exit 1;
}
[[ -f "${DATASET_ROOT}/libero_10_converted/meta_info.h5" ]] || {
    echo "[ERROR] invalid ROOT_DIR; expected ${DATASET_ROOT}/libero_10_converted/meta_info.h5" >&2; exit 1;
}
[[ ! -e "${RUN_ROOT}" ]] || { echo "[ERROR] output exists: ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

read -r ARM_WEIGHT GRIPPER_WEIGHT EXEC_WEIGHT REG_WEIGHT CALIBRATION_PASS < <(
    python - "${CALIBRATION_JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
w = d["weights"]
print(w["arm"], w["gripper"], w["executed_token"], w["residual_regularization"], int(bool(d["pass"])))
PY
)
[[ "${CALIBRATION_PASS}" == "1" ]] || { echo "[ERROR] calibration did not pass" >&2; exit 1; }

python "${REPO_ROOT}/tools/seer/lock_latentloop_comparison_source.py" \
    --repo-root "${REPO_ROOT}" \
    --output-dir "${RUN_ROOT}/source_lock" \
    --teacher "${TEACHER}" \
    --adapter "${CANONICAL_ADAPTER}" \
    --canonical-row "${CANONICAL_ROW}" \
    --vit-checkpoint "${VIT_CHECKPOINT}" \
    --dataset-root "${DATASET_ROOT}"
python - "${CALIBRATION_JSON}" "${RUN_ROOT}/source_lock/source_lock_manifest.json" <<'PY'
import hashlib, json, sys
calibration = json.load(open(sys.argv[1]))
current = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert calibration["source_lock_sha256"] == current, "calibration/source-lock mismatch"
PY
cp "${CALIBRATION_JSON}" "${RUN_ROOT}/loss_calibration.json"

if [[ "${STAGE}" == "full" ]]; then
    TARGET_MICROBATCHES=83200
    CHECKPOINT_MICROBATCHES=2080
    WARMUP_MICROBATCHES=4160
    NUM_EPOCHS=44
else
    TARGET_MICROBATCHES="${SMOKE_MICROBATCHES:-16}"
    CHECKPOINT_MICROBATCHES="${SMOKE_MICROBATCHES:-16}"
    WARMUP_MICROBATCHES=0
    NUM_EPOCHS=2
fi
(( TARGET_MICROBATCHES % 8 == 0 )) || {
    echo "[ERROR] microbatch budget must be divisible by gradient accumulation 8" >&2; exit 1;
}

cat > "${RUN_ROOT}/training_contract.env" <<EOF
SCIENTIFIC_NAME=matched_action_space_correction
STAGE=${STAGE}
TRAIN_SPLIT=train
VALIDATION_FRACTION=0.05
VALIDATION_SEED=20260805
TARGET_MICROBATCHES=${TARGET_MICROBATCHES}
GRADIENT_ACCUMULATION=8
TARGET_OPTIMIZER_STEPS=$((TARGET_MICROBATCHES / 8))
CHECKPOINT_MICROBATCHES=${CHECKPOINT_MICROBATCHES}
CHECKPOINT_SELECTION=validation_total_loss_only
LIBERO_TEST_SELECTION_FORBIDDEN=1
EOF

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export BASELINE_CKPT="${TEACHER}"
export BASELINE_CKPT_ID=33
export LRNODE_PROTOCOL_ROOT="${RUN_ROOT}/protocol"
export SAVE_CHECKPOINT_PATH="${RUN_ROOT}/checkpoints"
export ROOT_DIR="${DATASET_ROOT}"
export VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT}"
export LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
export MASTER_PORT="${MASTER_PORT:-13700}"
export EXPERIMENT_TAG="${RUN_TAG}"
export NUM_EPOCHS
export START_SAVE_CHECKPOINT=0
export METHOD_TAG="matched_action_space_correction_teacher33_${STAGE}_v1"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-0}"
export LRNODE_TRAIN_LATENT_DISTILL=0
export LATENTLOOP_PLAN_ADAPTER_MODE=action_correction
export LATENTLOOP_COMPARISON_PROTOCOL=1
export LATENTLOOP_COMPARISON_OFFSET_SCHEDULE=adjacent
export LATENTLOOP_COMPARISON_SPLIT_ROLE=train
export LATENTLOOP_COMPARISON_VALIDATION_FRACTION=0.05
export LATENTLOOP_COMPARISON_VALIDATION_SEED=20260805
export LATENTLOOP_COMPARISON_TARGET_MICROBATCHES="${TARGET_MICROBATCHES}"
export LATENTLOOP_COMPARISON_WARMUP_MICROBATCHES="${WARMUP_MICROBATCHES}"
export LATENTLOOP_COMPARISON_CHECKPOINT_MICROBATCHES="${CHECKPOINT_MICROBATCHES}"
export LATENTLOOP_ACTION_ARM_WEIGHT="${ARM_WEIGHT}"
export LATENTLOOP_ACTION_GRIPPER_WEIGHT="${GRIPPER_WEIGHT}"
export LATENTLOOP_ACTION_EXEC_WEIGHT="${EXEC_WEIGHT}"
export LATENTLOOP_ACTION_REG_WEIGHT="${REG_WEIGHT}"
export LATENTLOOP_PLAN_LATENT_WEIGHT=0.0

bash "${SCRIPT_DIR}/distill_node.sh" 2>&1 | tee "${RUN_ROOT}/train.log"
python - "${RUN_ROOT}/training_runtime.json" "${SECONDS}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "schema_version": 1,
    "wall_clock_training_seconds": int(sys.argv[2]),
    "includes_source_lock": True,
    "includes_training": True,
}, indent=2) + "\n")
PY
date --iso-8601=seconds > "${RUN_ROOT}/training_wrapper_complete.txt"
echo "[DONE] ${RUN_ROOT}"
