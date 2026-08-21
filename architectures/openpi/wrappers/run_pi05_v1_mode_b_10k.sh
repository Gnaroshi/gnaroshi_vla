#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

GPU=${OPENPI_PI05_V1_MODE_B_GPU:-5}
PORT=${OPENPI_PI05_V1_MODE_B_PORT:-8165}
RUN_NAME=${OPENPI_PI05_V1_MODE_B_RUN_NAME:-pi05_v1_mode_b_sd1_seed42_10k}
APPROVE_WEIGHTS=${OPENPI_PI05_V1_MODE_B_APPROVE_WEIGHTS:-0}
BASELINE_SUMMARY=${OPENPI_PI05_BASELINE_SUMMARY:-${OPENPI_LL_SHARED}/results/openpi/eval/pi05_base_lora_r16_b16_4gpu_seed42_30k/seed7_official_50/combined_summary.json}
BASELINE_CHECKPOINT_SHA256_FILE=${OPENPI_PI05_BASELINE_CHECKPOINT_SHA256_FILE:-${OPENPI_LL_SHARED}/results/openpi/baseline/pi05_base_lora_r16_b16_4gpu_seed42_30k/metadata/final_checkpoint.sha256}
MIN_FREE_MIB=${OPENPI_PI05_V1_MODE_B_MIN_FREE_MIB:-18000}

if [[ ${OPENPI_PI05_V1_MODE_B_10K_RUN:-0} != 1 ]]; then
  echo "Set OPENPI_PI05_V1_MODE_B_10K_RUN=1 to run V1 preparation or approved training." >&2
  exit 2
fi
[[ ${GPU} =~ ^[0-9]+$ ]] || { echo "GPU must be one physical index" >&2; exit 2; }
[[ ${PORT} =~ ^[0-9]+$ ]] || { echo "PORT must be numeric" >&2; exit 2; }
[[ ${MIN_FREE_MIB} =~ ^[0-9]+$ ]] || { echo "MIN_FREE_MIB must be numeric" >&2; exit 2; }
[[ ${APPROVE_WEIGHTS} == 0 || ${APPROVE_WEIGHTS} == 1 ]] || {
  echo "OPENPI_PI05_V1_MODE_B_APPROVE_WEIGHTS must be 0 or 1" >&2
  exit 2
}

CONTRACT=${OPENPI_LL_RESULTS}/contracts/${RUN_NAME}
RAW_LOSS=${CONTRACT}/v1_streaming_raw_loss/raw_loss_calibration.json
SUGGESTED_WEIGHTS=${CONTRACT}/v1_streaming_equalized_weights_suggested.json
LOSS_LOCK=${CONTRACT}/v1_streaming_loss_weights_equalized.json
PARITY=${OPENPI_LL_RESULTS}/audits/${RUN_NAME}/mode_b_real_window_parity
SMOKE=${OPENPI_LL_RESULTS}/cacheless_streaming/smoke/${RUN_NAME}
TRAIN=${OPENPI_LL_RESULTS}/cacheless_streaming/train/${RUN_NAME}
LOCK=${CONTRACT}/source_lock_v2.json
FINAL_MANIFEST=${CONTRACT}/protocol/pi05_final_evaluation_manifest_v2.json
SPLIT_CONTRACT=${CONTRACT}/protocol/pi05_split_contract_v2.json
K1_GATE=${CONTRACT}/k1_episode/combined_gate/pi05_k1_equivalence.json
FREEZE_GATE=${CONTRACT}/freeze/freeze_gate.json

mkdir -p "${OPENPI_LL_RESULTS}" "${WANDB_DIR}" "${HF_HOME}" "${HF_LEROBOT_HOME}"
test -d "${HF_LEROBOT_HOME}/physical-intelligence/libero"
"${OPENPI_LL_MAIN_PY}" - "${BASELINE_SUMMARY}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["complete"] is True
assert payload["total_episodes"] == 2000
assert payload["total_successes"] == 1944
assert abs(payload["micro_success_rate"] - 0.972) < 1e-12
print("BASELINE_PASS successes=1944 episodes=2000")
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
[[ $(sha256sum "${MODEL_PATH}" | awk '{print $1}') == "${EXPECTED_CHECKPOINT_SHA256}" ]] || {
  echo "Baseline checkpoint hash evidence is stale." >&2
  exit 1
}

free_mib=$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if ((free_mib < MIN_FREE_MIB)); then
  echo "GPU ${GPU} has ${free_mib} MiB free; require at least ${MIN_FREE_MIB} MiB." >&2
  exit 1
fi

cd "${OPENPI_LL_ROOT}"
CUDA_VISIBLE_DEVICES='' "${OPENPI_LL_MAIN_PY}" -m pytest -q tests/openpi_latentloop

if [[ ! -e ${CONTRACT} ]]; then
  echo "[1/6] Freeze V1 source/protocol and measure untrained Mode B raw losses."
  OPENPI_LATENTLOOP_STREAMING_ACCEPT_RUN=1 \
  bash architectures/openpi/wrappers/accept_pi05_v0_streaming.sh \
    --variant v1 --output "${CONTRACT}" --gpu "${GPU}" --port "${PORT}" \
    --raw-loss-examples 32 --action-execution-mode B \
    --expected-checkpoint-sha256 "${EXPECTED_CHECKPOINT_SHA256}"
else
  "${OPENPI_LL_MAIN_PY}" tools/openpi/source_lock_v2.py verify --lock "${LOCK}"
  test -f "${RAW_LOSS}"
fi

if [[ ! -e ${SUGGESTED_WEIGHTS} ]]; then
  "${OPENPI_LL_MAIN_PY}" tools/openpi/derive_pi05_v0_equalized_loss_weights.py \
    --variant v1 --raw-loss-calibration "${RAW_LOSS}" \
    --expected-action-execution-mode B > "${SUGGESTED_WEIGHTS}"
fi
echo "[2/6] Suggested equal-contribution V1 weights:"
cat "${SUGGESTED_WEIGHTS}"

if [[ ${APPROVE_WEIGHTS} != 1 ]]; then
  printf '%s\n' \
    "V1_PREAPPROVAL_COMPLETE" \
    "Inspect ${RAW_LOSS}" \
    "Inspect ${SUGGESTED_WEIGHTS}" \
    "No optimizer step has run." \
    "Re-run with OPENPI_PI05_V1_MODE_B_APPROVE_WEIGHTS=1 to approve these exact weights."
  exit 0
fi

if [[ ! -e ${LOSS_LOCK} ]]; then
  read -r state_weight chunk_weight executed_weight gripper_weight composition_weight < <(
    "${OPENPI_LL_MAIN_PY}" tools/openpi/derive_pi05_v0_equalized_loss_weights.py \
      --variant v1 --raw-loss-calibration "${RAW_LOSS}" \
      --expected-action-execution-mode B --shell
  )
  "${OPENPI_LL_MAIN_PY}" tools/openpi/freeze_pi05_v0_streaming_loss_weights.py \
    --variant v1 --raw-loss-calibration "${RAW_LOSS}" --source-lock "${LOCK}" \
    --checkpoint "${OPENPI_LL_CHECKPOINT}" --final-evaluation-manifest "${FINAL_MANIFEST}" \
    --split-contract "${SPLIT_CONTRACT}" --output "${LOSS_LOCK}" \
    --state-weight "${state_weight}" --chunk-weight "${chunk_weight}" \
    --executed-weight "${executed_weight}" --gripper-weight "${gripper_weight}" \
    --composition-weight "${composition_weight}" --approve
fi

if [[ ! -e ${PARITY} ]]; then
  echo "[3/6] Verify V1 Mode A/B real-window loss and gradient parity at delta_q=3."
  CUDA_VISIBLE_DEVICES=${GPU} OPENPI_LATENTLOOP_MODE_B_AUDIT_RUN=1 \
  "${OPENPI_LL_MAIN_PY}" tools/openpi/audit_pi05_v0_mode_b.py \
    --run --variant v1 --output "${PARITY}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
    --source-lock "${LOCK}" --k1-gate "${K1_GATE}" --freeze-gate "${FREEZE_GATE}" \
    --final-evaluation-manifest "${FINAL_MANIFEST}" --split-contract "${SPLIT_CONTRACT}" \
    --loss-weights-gate "${LOSS_LOCK}" --device cuda
fi
"${OPENPI_LL_MAIN_PY}" - "${PARITY}/mode_b_real_window_parity.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["MODE_B_REAL_WINDOW_PARITY_PASS"] is True
assert payload["variant"] == "v1" and payload["delta_q"] == 3
assert payload["expected_action_expert_calls"] == {"A": 2, "B": 1}
print("V1_MODE_B_PARITY_GATE_PASS")
PY

if [[ ! -e ${SMOKE} ]]; then
  echo "[4/6] Run a 200-step cacheless V1 Mode B backward smoke."
  OPENPI_LATENTLOOP_STREAMING_TRAIN_RUN=1 \
  bash architectures/openpi/wrappers/train_pi05_v0_streaming.sh \
    --variant v1 --output "${SMOKE}" --source-lock "${LOCK}" --k1-gate "${K1_GATE}" \
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
assert payload["V1_STREAMING_SCREEN_TRAIN_COMPLETE"] is True
assert payload["action_execution_mode"] == "B"
print(f"V1_MODE_B_SMOKE_PASS elapsed_seconds={payload['elapsed_seconds']:.3f}")
PY

if [[ ! -e ${TRAIN} ]]; then
  echo "[5/6] Train exploratory V1 Mode B for 10K fixed-LR steps on GPU ${GPU}."
  OPENPI_LATENTLOOP_STREAMING_TRAIN_RUN=1 \
  bash architectures/openpi/wrappers/train_pi05_v0_streaming.sh \
    --variant v1 --output "${TRAIN}" --source-lock "${LOCK}" --k1-gate "${K1_GATE}" \
    --freeze-gate "${FREEZE_GATE}" --final-evaluation-manifest "${FINAL_MANIFEST}" \
    --split-contract "${SPLIT_CONTRACT}" --loss-weights-gate "${LOSS_LOCK}" \
    --gpu "${GPU}" --max-steps 10000 --validation-interval 1000 \
    --validation-examples 32 --save-interval 2000 --action-execution-mode B \
    --wandb-mode online --wandb-name "${RUN_NAME}"
fi

echo "[6/6] Verify exploratory V1 completion without unlocking production V1 gates."
"${OPENPI_LL_MAIN_PY}" - "${TRAIN}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = json.loads((root / "run_summary.json").read_text())
assert payload["complete"] and payload["steps"] == 10000
assert payload["V1_STREAMING_SCREEN_TRAIN_COMPLETE"] is True
assert "V1_TRAIN_COMPLETE" not in payload
assert payload["action_execution_mode"] == "B"
assert (root / "checkpoints" / "step_010000.pt").is_file()
print("V1_MODE_B_10K_SCREEN_COMPLETE")
print("checkpoint_10k", root / "checkpoints/step_010000.pt")
print("best_checkpoint", root / "checkpoints/best.pt", "best_step", payload["best_step"])
PY
