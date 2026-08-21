#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

GPU=${OPENPI_PI05_V0_MODE_B_GPU:-4}
PORT=${OPENPI_PI05_V0_MODE_B_PORT:-8164}
RUN_NAME=${OPENPI_PI05_V0_MODE_B_RUN_NAME:-pi05_v0_mode_b_sd1_seed42_10k}
BASELINE_SUMMARY=${OPENPI_PI05_BASELINE_SUMMARY:-}
BASELINE_CHECKPOINT_SHA256_FILE=${OPENPI_PI05_BASELINE_CHECKPOINT_SHA256_FILE:-}
EXPECTED_BASELINE_SUCCESSES=${OPENPI_PI05_EXPECTED_BASELINE_SUCCESSES:-1944}
EXPECTED_BASELINE_EPISODES=${OPENPI_PI05_EXPECTED_BASELINE_EPISODES:-2000}
MIN_FREE_MIB=${OPENPI_PI05_V0_MODE_B_MIN_FREE_MIB:-18000}

if [[ ${OPENPI_PI05_V0_MODE_B_10K_RUN:-0} != 1 ]]; then
  echo "Set OPENPI_PI05_V0_MODE_B_10K_RUN=1 to run the gated Mode B 10K experiment." >&2
  exit 2
fi
[[ ${GPU} =~ ^[0-9]+$ ]] || { echo "GPU must be one physical index" >&2; exit 2; }
[[ ${PORT} =~ ^[0-9]+$ ]] || { echo "PORT must be numeric" >&2; exit 2; }
[[ ${MIN_FREE_MIB} =~ ^[0-9]+$ ]] || { echo "MIN_FREE_MIB must be numeric" >&2; exit 2; }
[[ ${EXPECTED_BASELINE_SUCCESSES} =~ ^[0-9]+$ ]] || {
  echo "EXPECTED_BASELINE_SUCCESSES must be numeric" >&2
  exit 2
}
[[ ${EXPECTED_BASELINE_EPISODES} =~ ^[1-9][0-9]*$ ]] || {
  echo "EXPECTED_BASELINE_EPISODES must be positive" >&2
  exit 2
}
: "${BASELINE_SUMMARY:?Set OPENPI_PI05_BASELINE_SUMMARY to the completed baseline summary}"
: "${BASELINE_CHECKPOINT_SHA256_FILE:?Set OPENPI_PI05_BASELINE_CHECKPOINT_SHA256_FILE to checkpoint hash evidence}"

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
"${OPENPI_LL_MAIN_PY}" - \
  "${BASELINE_SUMMARY}" "${EXPECTED_BASELINE_SUCCESSES}" "${EXPECTED_BASELINE_EPISODES}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected_successes = int(sys.argv[2])
expected_episodes = int(sys.argv[3])
assert payload["complete"] is True
assert payload["total_episodes"] == expected_episodes
assert payload["total_successes"] == expected_successes
assert abs(payload["micro_success_rate"] - expected_successes / expected_episodes) < 1e-12
print(f"BASELINE_PASS successes={expected_successes} episodes={expected_episodes}")
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
echo "BASELINE_CHECKPOINT_HASH_PASS sha256=${ACTUAL_CHECKPOINT_SHA256}"

free_mib=$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if ((free_mib < MIN_FREE_MIB)); then
  echo "GPU ${GPU} has ${free_mib} MiB free; require at least ${MIN_FREE_MIB} MiB." >&2
  exit 1
fi
echo "GPU_PREFLIGHT_PASS gpu=${GPU} free_mib=${free_mib}"

cd "${OPENPI_LL_ROOT}"
CUDA_VISIBLE_DEVICES='' "${OPENPI_LL_MAIN_PY}" -m pytest -q tests/openpi_latentloop

if [[ ! -e ${CONTRACT} ]]; then
  echo "[1/5] Freeze the SD1 checkpoint/protocol and calibrate raw Mode B losses."
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
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["checkpoint"]["expected_model_sha256"] == sys.argv[2].lower()
print("BASELINE_SOURCE_LOCK_BINDING_PASS")
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
import json
import sys

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
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["complete"] and payload["steps"] == 200
assert payload["action_execution_mode"] == "B"
print(f"MODE_B_SMOKE_PASS elapsed_seconds={payload['elapsed_seconds']:.3f}")
PY

if [[ ! -e ${TRAIN} ]]; then
  echo "[4/5] Train one Mode B run to 10K on GPU ${GPU}."
  OPENPI_LATENTLOOP_STREAMING_TRAIN_RUN=1 \
  bash architectures/openpi/wrappers/train_pi05_v0_streaming.sh \
    --output "${TRAIN}" --source-lock "${LOCK}" --k1-gate "${K1_GATE}" \
    --freeze-gate "${FREEZE_GATE}" --final-evaluation-manifest "${FINAL_MANIFEST}" \
    --split-contract "${SPLIT_CONTRACT}" --loss-weights-gate "${LOSS_LOCK}" \
    --gpu "${GPU}" --max-steps 10000 --validation-interval 1000 \
    --validation-examples 32 --save-interval 2000 --action-execution-mode B \
    --wandb-mode online --wandb-name "${RUN_NAME}"
fi

echo "[5/5] Verify Mode B 10K completion."
"${OPENPI_LL_MAIN_PY}" - "${TRAIN}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = json.loads((root / "run_summary.json").read_text())
assert payload["complete"] and payload["V0_TRAIN_COMPLETE"] and payload["steps"] == 10000
assert payload["action_execution_mode"] == "B"
assert (root / "checkpoints" / "step_010000.pt").is_file()
print("MODE_B_10K_COMPLETE")
print("checkpoint_10k", root / "checkpoints/step_010000.pt")
print("best_checkpoint", root / "checkpoints/best.pt", "best_step", payload["best_step"])
PY
