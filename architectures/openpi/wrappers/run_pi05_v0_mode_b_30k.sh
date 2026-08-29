#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

GPU=${OPENPI_PI05_V0_MODE_B_GPU:-0}
PORT=${OPENPI_PI05_V0_MODE_B_PORT:-8164}
RUN_NAME=${OPENPI_PI05_V0_MODE_B_RUN_NAME:-pi05_v0_mode_b_rb2_seed42_30k}
BASELINE_SUMMARY=${OPENPI_PI05_BASELINE_SUMMARY:-}
BASELINE_CHECKPOINT_SHA256_FILE=${OPENPI_PI05_BASELINE_CHECKPOINT_SHA256_FILE:-}
MIN_FREE_MIB=${OPENPI_PI05_V0_MODE_B_MIN_FREE_MIB:-18000}

if [[ ${OPENPI_PI05_V0_MODE_B_RUN:-0} != 1 ]]; then
  echo "Set OPENPI_PI05_V0_MODE_B_RUN=1 to run the gated Mode B 30K experiment." >&2
  exit 2
fi
[[ ${GPU} =~ ^[0-9]+$ ]] || { echo "GPU must be one physical index" >&2; exit 2; }
[[ ${PORT} =~ ^[0-9]+$ ]] || { echo "PORT must be numeric" >&2; exit 2; }
[[ ${MIN_FREE_MIB} =~ ^[0-9]+$ ]] || { echo "MIN_FREE_MIB must be numeric" >&2; exit 2; }
: "${BASELINE_SUMMARY:?Set OPENPI_PI05_BASELINE_SUMMARY to the completed 2,000-episode baseline summary}"
: "${BASELINE_CHECKPOINT_SHA256_FILE:?Set OPENPI_PI05_BASELINE_CHECKPOINT_SHA256_FILE to the baseline checkpoint hash evidence}"

CONTRACT=${OPENPI_LL_RESULTS}/contracts/${RUN_NAME}
LOSS_LOCK=${CONTRACT}/v0_streaming_loss_weights_equalized.json
PARITY=${OPENPI_LL_RESULTS}/audits/${RUN_NAME}/mode_b_real_window_parity
SMOKE=${OPENPI_LL_RESULTS}/cacheless_streaming/smoke/${RUN_NAME}
TRAIN=${OPENPI_LL_RESULTS}/cacheless_streaming/train/${RUN_NAME}
LOCK=${CONTRACT}/source_lock_v2.json
FINAL_MANIFEST=${CONTRACT}/protocol/pi05_final_evaluation_manifest_v2.json
SPLIT_CONTRACT=${CONTRACT}/protocol/pi05_split_contract_v2.json
K1_GATE=${CONTRACT}/k1_episode/combined_gate/pi05_k1_equivalence.json
FREEZE_GATE=${CONTRACT}/freeze/freeze_gate.json
RAW_LOSS=${CONTRACT}/v0_streaming_raw_loss/raw_loss_calibration.json

mkdir -p "${OPENPI_LL_RESULTS}" "${WANDB_DIR}" "${HF_HOME}" "${HF_LEROBOT_HOME}"
test -d "${HF_LEROBOT_HOME}/physical-intelligence/libero"
"${OPENPI_LL_MAIN_PY}" - "${BASELINE_SUMMARY}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p["complete"] is True
assert p["total_episodes"] == 2000
assert p["total_successes"] == 1937
assert abs(p["micro_success_rate"] - 0.9685) < 1e-12
print("RB2_BASELINE_1937_OF_2000_PASS")
PY
read -r EXPECTED_CHECKPOINT_SHA256 RECORDED_CHECKPOINT < "${BASELINE_CHECKPOINT_SHA256_FILE}"
[[ ${EXPECTED_CHECKPOINT_SHA256} =~ ^[0-9a-fA-F]{64}$ ]] || {
  echo "Baseline checkpoint evidence does not begin with a SHA-256 value." >&2
  exit 1
}
MODEL_PATH=${OPENPI_LL_CHECKPOINT}/model.safetensors
[[ $(realpath -- "${RECORDED_CHECKPOINT}") == $(realpath -- "${MODEL_PATH}") ]] || {
  echo "Baseline checkpoint evidence names another model file." >&2
  exit 1
}
ACTUAL_CHECKPOINT_SHA256=$(sha256sum "${MODEL_PATH}" | awk '{print $1}')
[[ ${ACTUAL_CHECKPOINT_SHA256} == "${EXPECTED_CHECKPOINT_SHA256}" ]] || {
  echo "Baseline checkpoint hash evidence is stale." >&2
  exit 1
}
echo "RB2_BASELINE_CHECKPOINT_HASH_PASS sha256=${ACTUAL_CHECKPOINT_SHA256}"

free_mib=$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if ((free_mib < MIN_FREE_MIB)); then
  echo "GPU ${GPU} has ${free_mib} MiB free; require at least ${MIN_FREE_MIB} MiB." >&2
  exit 1
fi
echo "GPU_PREFLIGHT_PASS gpu=${GPU} free_mib=${free_mib}"

cd "${OPENPI_LL_ROOT}"
CUDA_VISIBLE_DEVICES='' "${OPENPI_LL_MAIN_PY}" -m pytest -q tests/openpi_latentloop

if [[ ! -e ${CONTRACT} ]]; then
  echo "[1/5] Freeze RB2 checkpoint/protocol and calibrate raw Mode B losses."
  OPENPI_LATENTLOOP_STREAMING_ACCEPT_RUN=1 \
  bash architectures/openpi/wrappers/accept_pi05_v0_streaming.sh \
    --output "${CONTRACT}" --gpu "${GPU}" --port "${PORT}" \
    --raw-loss-examples 32 --action-execution-mode B \
    --expected-checkpoint-sha256 "${EXPECTED_CHECKPOINT_SHA256}"
else
  "${OPENPI_LL_MAIN_PY}" tools/openpi/source_lock_v2.py verify --lock "${LOCK}"
  test -f "${RAW_LOSS}"
fi
"${OPENPI_LL_MAIN_PY}" - "${LOCK}" "${EXPECTED_CHECKPOINT_SHA256}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p["checkpoint"]["expected_model_sha256"] == sys.argv[2].lower()
print("RB2_BASELINE_SOURCE_LOCK_BINDING_PASS")
PY

if [[ ! -e ${LOSS_LOCK} ]]; then
  read -r state_weight chunk_weight executed_weight gripper_weight < <(
    "${OPENPI_LL_MAIN_PY}" tools/openpi/derive_pi05_v0_equalized_loss_weights.py \
      --raw-loss-calibration "${RAW_LOSS}" --expected-action-execution-mode B --shell
  )
  "${OPENPI_LL_MAIN_PY}" tools/openpi/freeze_pi05_v0_streaming_loss_weights.py \
    --raw-loss-calibration "${RAW_LOSS}" --source-lock "${LOCK}" \
    --checkpoint "${OPENPI_LL_CHECKPOINT}" --final-evaluation-manifest "${FINAL_MANIFEST}" \
    --split-contract "${SPLIT_CONTRACT}" --output "${LOSS_LOCK}" \
    --state-weight "${state_weight}" --chunk-weight "${chunk_weight}" \
    --executed-weight "${executed_weight}" --gripper-weight "${gripper_weight}" --approve
fi

if [[ ! -e ${PARITY} ]]; then
  echo "[2/5] Verify real-window Mode A/B loss and gradient parity."
  CUDA_VISIBLE_DEVICES=${GPU} OPENPI_LATENTLOOP_MODE_B_AUDIT_RUN=1 \
  "${OPENPI_LL_MAIN_PY}" tools/openpi/audit_pi05_v0_mode_b.py \
    --run --output "${PARITY}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
    --source-lock "${LOCK}" --k1-gate "${K1_GATE}" --freeze-gate "${FREEZE_GATE}" \
    --final-evaluation-manifest "${FINAL_MANIFEST}" --split-contract "${SPLIT_CONTRACT}" \
    --loss-weights-gate "${LOSS_LOCK}" --device cuda
fi
"${OPENPI_LL_MAIN_PY}" - "${PARITY}/mode_b_real_window_parity.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1], encoding="utf-8"))["MODE_B_REAL_WINDOW_PARITY_PASS"] is True
print("MODE_B_PARITY_GATE_PASS")
PY

if [[ ! -e ${SMOKE} ]]; then
  echo "[3/5] Run a 200-step Mode B backward smoke."
  OPENPI_LATENTLOOP_STREAMING_TRAIN_RUN=1 \
  bash architectures/openpi/wrappers/train_pi05_v0_streaming.sh \
    --output "${SMOKE}" --source-lock "${LOCK}" --k1-gate "${K1_GATE}" \
    --freeze-gate "${FREEZE_GATE}" --final-evaluation-manifest "${FINAL_MANIFEST}" \
    --split-contract "${SPLIT_CONTRACT}" --loss-weights-gate "${LOSS_LOCK}" \
    --gpu "${GPU}" --max-steps 200 --validation-interval 100 \
    --validation-examples 32 --save-interval 200 --action-execution-mode B \
    --wandb-mode disabled
fi
"${OPENPI_LL_MAIN_PY}" - "${SMOKE}/run_summary.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
assert p["complete"] and p["steps"] == 200 and p["action_execution_mode"] == "B"
print(f"MODE_B_SMOKE_PASS elapsed_seconds={p['elapsed_seconds']:.3f} projected_30k_hours={p['elapsed_seconds'] * 150 / 3600:.2f}")
PY

if [[ ! -e ${TRAIN} ]]; then
  echo "[4/5] Train one Mode B run to 30K; preserve both 10K and 30K checkpoints."
  OPENPI_LATENTLOOP_STREAMING_TRAIN_RUN=1 \
  bash architectures/openpi/wrappers/train_pi05_v0_streaming.sh \
    --output "${TRAIN}" --source-lock "${LOCK}" --k1-gate "${K1_GATE}" \
    --freeze-gate "${FREEZE_GATE}" --final-evaluation-manifest "${FINAL_MANIFEST}" \
    --split-contract "${SPLIT_CONTRACT}" --loss-weights-gate "${LOSS_LOCK}" \
    --gpu "${GPU}" --max-steps 30000 --validation-interval 1000 \
    --validation-examples 32 --save-interval 5000 --action-execution-mode B \
    --wandb-mode online --wandb-name "${RUN_NAME}"
fi

echo "[5/5] Verify completion and the two requested checkpoints."
"${OPENPI_LL_MAIN_PY}" - "${TRAIN}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
p = json.loads((root / "run_summary.json").read_text())
assert p["complete"] and p["V0_TRAIN_COMPLETE"] and p["steps"] == 30000
assert p["action_execution_mode"] == "B"
for step in (10000, 30000):
    assert (root / "checkpoints" / f"step_{step:06d}.pt").is_file()
print("MODE_B_10K_30K_COMPLETE")
print("checkpoint_10k", root / "checkpoints/step_010000.pt")
print("checkpoint_30k", root / "checkpoints/step_030000.pt")
print("best_checkpoint", root / "checkpoints/best.pt", "best_step", p["best_step"])
PY
