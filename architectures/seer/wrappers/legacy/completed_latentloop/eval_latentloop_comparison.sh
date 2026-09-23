#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
COMPARE="${REPO_ROOT}/architectures/seer/upstream/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh"
TEACHER="${LOCAL_TEACHER_CKPT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/33.pth}"
LATENTLOOP="${LOCAL_LATENTLOOP_ADAPTER:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616/train/distill_node/sd1_distill_node_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_20260620_202533/39.pth}"
CANONICAL_ROW="${CANONICAL_ROW:-${REPO_ROOT}/results/seer/latentloop/local_best91/local_best91_segment_grid_legacy_parity_20260731_v2/stage_a_ckpt33/ckpt33_L4_dense/segment_grid_row.json}"
DATASET_ROOT="${ROOT_DIR:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/LIBERO_DATASETS/libero_10_converted}"
VIT_CHECKPOINT="${VIT_CHECKPOINT_PATH:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}"
STAGE="${STAGE:-parity_smoke}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/seer/latentloop/comparison/eval/${STAGE}_${RUN_TAG}}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-13800}"

case "${STAGE}" in
    parity_smoke) EPISODES_PER_TASK="${EPISODES_PER_TASK:-4}"; NUM_TASKS="${NUM_TASKS:-1}"; NODE_NUM="${NODE_NUM:-4}" ;;
    screening) EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"; NUM_TASKS="${NUM_TASKS:-10}"; NODE_NUM="${NODE_NUM:-2}" ;;
    confirmation|diagnostic_k2_k8) EPISODES_PER_TASK="${EPISODES_PER_TASK:-20}"; NUM_TASKS="${NUM_TASKS:-10}"; NODE_NUM="${NODE_NUM:-4}" ;;
    robustness36|robustness38)
        [[ "${ENABLE_HELDOUT_ROBUSTNESS:-0}" == "1" ]] || {
            echo "[ERROR] held-out robustness is gated; set ENABLE_HELDOUT_ROBUSTNESS=1 only after the local verdict" >&2; exit 1;
        }
        EPISODES_PER_TASK="${EPISODES_PER_TASK:-20}"; NUM_TASKS="${NUM_TASKS:-10}"; NODE_NUM="${NODE_NUM:-4}" ;;
    *) echo "[ERROR] unknown STAGE=${STAGE}" >&2; exit 1 ;;
esac

for path in "${TEACHER}" "${LATENTLOOP}" "${CANONICAL_ROW}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] missing file: ${path}" >&2; exit 1; }
done
[[ -d "${DATASET_ROOT}/libero_10_converted/episodes" ]] || {
    echo "[ERROR] invalid ROOT_DIR; expected ${DATASET_ROOT}/libero_10_converted/episodes" >&2; exit 1;
}
[[ -f "${DATASET_ROOT}/libero_10_converted/meta_info.h5" ]] || {
    echo "[ERROR] invalid ROOT_DIR; expected ${DATASET_ROOT}/libero_10_converted/meta_info.h5" >&2; exit 1;
}
if [[ "${STAGE}" != robustness* ]]; then
    : "${ACTION_CORRECTION_CKPT:?Set ACTION_CORRECTION_CKPT selected on validation only}"
    : "${NONRECURRENT_CKPT:?Set NONRECURRENT_CKPT selected on validation only}"
    : "${ACTION_SELECTION_JSON:?Set ACTION_SELECTION_JSON from validation-only selection}"
    : "${NONRECURRENT_SELECTION_JSON:?Set NONRECURRENT_SELECTION_JSON from validation-only selection}"
    [[ -f "${ACTION_CORRECTION_CKPT}" ]] || { echo "[ERROR] missing action checkpoint" >&2; exit 1; }
    [[ -f "${NONRECURRENT_CKPT}" ]] || { echo "[ERROR] missing nonrecurrent checkpoint" >&2; exit 1; }
    python - "${ACTION_SELECTION_JSON}" "${ACTION_CORRECTION_CKPT}" action_correction <<'PY'
import json, pathlib, sys
d = json.load(open(sys.argv[1]))
assert d["mode"] == sys.argv[3] and not d["uses_libero_test_success"]
assert pathlib.Path(d["selected_checkpoint"]).resolve() == pathlib.Path(sys.argv[2]).resolve()
PY
    python - "${NONRECURRENT_SELECTION_JSON}" "${NONRECURRENT_CKPT}" anchor_bridge <<'PY'
import json, pathlib, sys
d = json.load(open(sys.argv[1]))
assert d["mode"] == sys.argv[3] and not d["uses_libero_test_success"]
assert pathlib.Path(d["selected_checkpoint"]).resolve() == pathlib.Path(sys.argv[2]).resolve()
PY
fi
ACTION_CONFIRM=1
NONRECURRENT_CONFIRM=1
if [[ "${STAGE}" == "confirmation" ]]; then
    : "${SCREENING_GATE_JSON:?Set SCREENING_GATE_JSON from the 100-episode aggregate}"
    [[ -f "${SCREENING_GATE_JSON}" ]] || { echo "[ERROR] missing screening gate" >&2; exit 1; }
    read -r ACTION_CONFIRM NONRECURRENT_CONFIRM < <(
        python - "${SCREENING_GATE_JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["protocol"] == "predeclared_100_episode_screening_v1"
rows = d["rows"]
print(
    int(rows["matched_action_correction_k4"]["proceed_to_confirmation"]),
    int(rows["nonrecurrent_latent_k4"]["proceed_to_confirmation"]),
)
PY
    )
fi
[[ ! -e "${RESULT_ROOT}" ]] || { echo "[ERROR] output exists: ${RESULT_ROOT}" >&2; exit 1; }
mkdir -p "${RESULT_ROOT}"

python "${REPO_ROOT}/tools/seer/lock_latentloop_comparison_source.py" \
    --repo-root "${REPO_ROOT}" --output-dir "${RESULT_ROOT}/source_lock" \
    --teacher "${TEACHER}" --adapter "${LATENTLOOP}" --canonical-row "${CANONICAL_ROW}" \
    --vit-checkpoint "${VIT_CHECKPOINT}" --dataset-root "${DATASET_ROOT}"
if [[ "${STAGE}" != robustness* ]]; then
    python - "${ACTION_SELECTION_JSON}" "${ACTION_CORRECTION_CKPT}" \
        "${NONRECURRENT_SELECTION_JSON}" "${NONRECURRENT_CKPT}" \
        "${RESULT_ROOT}/source_lock/source_lock_manifest.json" <<'PY'
import hashlib, json, pathlib, sys

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

current_lock = digest(sys.argv[5])
for selection_path, checkpoint_path in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    selection = json.load(open(selection_path))
    assert selection["source_lock_sha256"] == current_lock, "selection/source-lock mismatch"
    assert pathlib.Path(selection["selected_checkpoint"]).resolve() == pathlib.Path(checkpoint_path).resolve()
    assert selection["selected_checkpoint_sha256"] == digest(checkpoint_path), "selected checkpoint SHA mismatch"
PY
fi
if [[ "${STAGE}" == "confirmation" ]]; then
    python - "${SCREENING_GATE_JSON}" "${RESULT_ROOT}/source_lock/source_lock_manifest.json" <<'PY'
import hashlib, json, sys
gate = json.load(open(sys.argv[1]))
digest = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert gate["source_lock_sha256"] == digest, "screening/current source-lock mismatch"
PY
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export LIBERO_GL_BACKEND=osmesa
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBERO_GL_REQUIRE_ACTUAL=1
export LIBERO_PATH="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
export VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT}"
export EVAL_NUM_EPISODES_PER_TASK="${EPISODES_PER_TASK}"
export EVAL_NUM_TASKS="${NUM_TASKS}"
export NODE_NUM
export EVAL_LIBERO_ENSEMBLING=1
export SAVE_VIDEO="${SAVE_VIDEO:-0}"
export SAVE_VIDEO_SUCC=0 SAVE_VIDEO_FAIL=0 SAVE_VIDEO_ALL_RANKS=0
export LRNODE_EVAL_STEP_LOG=1
export LRNODE_EVAL_SHADOW_FULL_FORWARD=0
export LRNODE_MECHANISM_TRACE=0
export LATENTLOOP_PLAN_TRACE=0
export LATENTLOOP_COMPARISON_PROTOCOL=1
export BASELINE_CKPT="${TEACHER}"
export BASELINE_CKPT_ID=33
export BASELINE_NAME=local_scratch_teacher33
export LRNODE_EVAL_BASE_CKPT="${TEACHER}"
export LRNODE_TRAIN_PROTOCOL=adapter
export LRNODE_FREEZE_SEER_FOR_ADAPTER=1
export LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1

run_row() {
    local row_id=$1 checkpoint=$2 mode=$3 ablation=$4 k=$5 run_baseline=$6 run_full=$7 port=$8
    local row_root="${RESULT_ROOT}/${row_id}"
    [[ ! -e "${row_root}" ]] || { echo "[ERROR] row exists: ${row_root}" >&2; exit 1; }
    (
        export RESULT_ROOT="${row_root}"
        export EXPERIMENT_NAME=latentloop_comparison
        export EXPERIMENT_TAG="${RUN_TAG}_${row_id}"
        export MASTER_PORT="${port}"
        export LATENTLOOP_PLAN_ADAPTER_MODE="${mode}"
        export LATENTLOOP_FEEDBACK_SOURCE=current
        export LRNODE_EVAL_ABLATION_MODE="${ablation}"
        export RUN_BASELINE="${run_baseline}"
        export RUN_OURS_FULL="${run_full}"
        export LRNODE_QUERY_INTERVALS_STR="${k}"
        export OURS_CKPT="${checkpoint}"
        export OURS_CKPT_ID="$(basename "${checkpoint}" .pth)"
        export OURS_NAME="${row_id}"
        bash "${COMPARE}"
    )
    mkdir -p "${row_root}/comparison_contract"
    sha256sum "${TEACHER}" "${checkpoint}" > "${row_root}/comparison_contract/checkpoints.sha256"
    cp "${RESULT_ROOT}/source_lock/source_lock_manifest.json" "${row_root}/comparison_contract/source_lock_manifest.json"
    cat > "${row_root}/comparison_contract/row.env" <<EOF
ROW_ID=${row_id}
STAGE=${STAGE}
PLAN_ADAPTER_MODE=${mode}
ABLATION_MODE=${ablation}
QUERY_INTERVAL=${k:-1}
EPISODES_PER_TASK=${EPISODES_PER_TASK}
NUM_TASKS=${NUM_TASKS}
RENDERER=osmesa
EOF
}

run_primary_matrix() {
    run_row full_seer_k1 "${LATENTLOOP}" off stepwise "" 1 0 "$((MASTER_PORT_BASE + 0))"
    run_row canonical_latentloop_k4 "${LATENTLOOP}" off stepwise 4 0 0 "$((MASTER_PORT_BASE + 10))"
    if [[ "${STAGE}" == "screening" || "${ACTION_CONFIRM}" == "1" ]]; then
        run_row matched_action_correction_k4 "${ACTION_CORRECTION_CKPT}" action_correction stepwise 4 0 0 "$((MASTER_PORT_BASE + 20))"
    else
        echo "[SKIP] action correction did not pass the predeclared screening gate"
    fi
    if [[ "${STAGE}" == "screening" || "${NONRECURRENT_CONFIRM}" == "1" ]]; then
        run_row nonrecurrent_latent_k4 "${NONRECURRENT_CKPT}" anchor_bridge stepwise 4 0 0 "$((MASTER_PORT_BASE + 30))"
    else
        echo "[SKIP] nonrecurrent latent did not pass the predeclared screening gate"
    fi
    run_row no_observation_latentloop_k4 "${LATENTLOOP}" off no_delta 4 0 0 "$((MASTER_PORT_BASE + 40))"
    run_row predicted_horizon_replay_k4 "${LATENTLOOP}" off seer_token_chunk 4 0 0 "$((MASTER_PORT_BASE + 50))"
}

case "${STAGE}" in
    parity_smoke)
        run_row full_seer_k1 "${LATENTLOOP}" off stepwise "" 1 0 "$((MASTER_PORT_BASE + 0))"
        run_row canonical_loaded_k1 "${LATENTLOOP}" off stepwise "" 0 1 "$((MASTER_PORT_BASE + 10))"
        run_row action_loaded_k1 "${ACTION_CORRECTION_CKPT}" action_correction stepwise "" 0 1 "$((MASTER_PORT_BASE + 20))"
        run_row nonrecurrent_loaded_k1 "${NONRECURRENT_CKPT}" anchor_bridge stepwise "" 0 1 "$((MASTER_PORT_BASE + 30))"
        ;;
    screening|confirmation)
        run_primary_matrix
        ;;
    diagnostic_k2_k8)
        for k in 2 8; do
            run_row "canonical_latentloop_k${k}" "${LATENTLOOP}" off stepwise "${k}" 0 0 "$((MASTER_PORT_BASE + k))"
            run_row "matched_action_correction_k${k}" "${ACTION_CORRECTION_CKPT}" action_correction stepwise "${k}" 0 0 "$((MASTER_PORT_BASE + 20 + k))"
            run_row "nonrecurrent_latent_k${k}" "${NONRECURRENT_CKPT}" anchor_bridge stepwise "${k}" 0 0 "$((MASTER_PORT_BASE + 40 + k))"
        done
        ;;
    robustness36|robustness38)
        echo "[ERROR] robustness requires base-specific trained adapters; use the disabled commands in commands_to_run.md" >&2
        exit 1
        ;;
esac

date --iso-8601=seconds > "${RESULT_ROOT}/evaluation_wrapper_complete.txt"
echo "[DONE] ${RESULT_ROOT}"
