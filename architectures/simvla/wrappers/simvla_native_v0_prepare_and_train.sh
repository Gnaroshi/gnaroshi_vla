#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
PREP=${ROOT}/architectures/simvla/wrappers/simvla_native_v0_prepare.sh
TRAIN=${ROOT}/architectures/simvla/wrappers/simvla_native_v0_train.sh
RESULT_ROOT=${SIMVLA_V0_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/correct_native_v0_seed20260815_v1}
CACHE=${SIMVLA_V0_CACHE:-${RESULT_ROOT}/00_training_cache_libero10_r5}
CHECKPOINT=${SIMVLA_V0_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}
SMOLVLM=${SIMVLA_V0_SMOLVLM:-HuggingFaceTB/SmolVLM-500M-Instruct}
NORM=${SIMVLA_V0_NORM:-${ROOT}/architectures/simvla/adapters/latentloop/assets/libero_norm_official_32700d0.json}

if [[ "${SIMVLA_NATIVE_V0_PIPELINE_RUN:-0}" != "1" ]]; then
  echo "Set SIMVLA_NATIVE_V0_PIPELINE_RUN=1 to enable validation and training." >&2
  exit 2
fi
: "${SIMVLA_GPU_IDS:?Set exactly two comma-separated physical GPU IDs.}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable is missing: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CACHE}/manifest.json" ]]; then
  echo "Compact training cache is missing: ${CACHE}/manifest.json" >&2
  exit 2
fi

cd "${ROOT}"
export SIMVLA_NATIVE_V0_RUN=1
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}:${ROOT}/architectures/simvla/upstream${PYTHONPATH:+:${PYTHONPATH}}"

manifest_gpu_ids=$("${PYTHON}" -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["selected_physical_gpu_ids"])))' "${CACHE}/manifest.json")
selected_gpu_ids=${SIMVLA_GPU_IDS//[[:space:]]/}
if [[ "${selected_gpu_ids}" != "${manifest_gpu_ids}" ]]; then
  echo "GPU source-lock mismatch: cache=${manifest_gpu_ids}, requested=${selected_gpu_ids}" >&2
  exit 2
fi

SOURCE_GATE=${RESULT_ROOT}/01_source_audit/source_audit.json
PARITY_GATE=${RESULT_ROOT}/02_k1_parity/k1_parity.json

SOURCE_HASH=$("${PYTHON}" - "${CHECKPOINT}" "${NORM}" "${CACHE}" "${SOURCE_GATE}" "${PARITY_GATE}" <<'PY'
import sys
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    native_v0_source_manifest,
    require_gate,
)

checkpoint, norm, cache, source_gate, parity_gate = sys.argv[1:]
source = native_v0_source_manifest(checkpoint=checkpoint, norm_stats=norm, cache=cache)
expected = source["combined_sha256"]
require_gate(source_gate, verdicts=("SOURCE_AUDIT_PASS",), source_combined_sha256=expected)
require_gate(parity_gate, verdicts=("K1_HOOK_PARITY_PASS",), source_combined_sha256=expected)
print(expected)
PY
)
echo "PRETRAIN_GATES_PASS source=${SOURCE_HASH}"

gate_ok() {
  "${PYTHON}" - "$1" "$2" "${SOURCE_HASH}" <<'PY'
import json, sys
path, verdicts, expected_source = sys.argv[1:]
try:
    payload = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
if payload.get("verdict") not in verdicts.split(","):
    raise SystemExit(1)
if payload.get("source_combined_sha256") != expected_source:
    raise SystemExit(1)
PY
}

require_absent_or_valid() {
  local output=$1
  local gate=$2
  local verdicts=$3
  if gate_ok "${gate}" "${verdicts}"; then
    return 0
  fi
  if [[ -e "${output}" ]]; then
    echo "Incomplete or stale stage output requires manual review: ${output}" >&2
    exit 2
  fi
  return 1
}

if require_absent_or_valid \
  "${RESULT_ROOT}/03_token_analysis" \
  "${RESULT_ROOT}/03_token_analysis/token_analysis_gate.json" \
  "TOKEN_ANALYSIS_PASS"; then
  echo "[3/8] SKIP: bounded token analysis already passed"
else
  echo "[3/8] Bounded token analysis"
  bash "${PREP}" token-analysis \
    --output "${RESULT_ROOT}/03_token_analysis" \
    --cache "${CACHE}" \
    --checkpoint "${CHECKPOINT}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}" \
    --parity-gate "${PARITY_GATE}" \
    --max-sequences 16 \
    --heldout-fraction 0.2 \
    --split-seed 20260822 \
    --seed 20260815
fi

if require_absent_or_valid \
  "${RESULT_ROOT}/04_parameter_audit" \
  "${RESULT_ROOT}/04_parameter_audit/simvla_v0_parameter_audit.json" \
  "PARAMETER_AUDIT_PASS"; then
  echo "[4/8] SKIP: parameter audit already passed"
else
  echo "[4/8] Parameter audit"
  bash "${PREP}" parameters \
    --output "${RESULT_ROOT}/04_parameter_audit" \
    --cache "${CACHE}" \
    --checkpoint "${CHECKPOINT}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}" \
    --analysis-gate "${RESULT_ROOT}/03_token_analysis/token_analysis_gate.json"
fi

if require_absent_or_valid \
  "${RESULT_ROOT}/05_mode_ab" \
  "${RESULT_ROOT}/05_mode_ab/decision/mode_ab_decision.json" \
  "MODE_B_APPROVED,MODE_A_REQUIRED"; then
  echo "[5/8] SKIP: Mode A/B decision already passed"
else
  echo "[5/8] Two-GPU Mode A/B equivalence"
  bash "${PREP}" mode-ab \
    --output "${RESULT_ROOT}/05_mode_ab" \
    --cache "${CACHE}" \
    --checkpoint "${CHECKPOINT}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}" \
    --parameter-gate "${RESULT_ROOT}/04_parameter_audit/simvla_v0_parameter_audit.json" \
    --sequences 4 \
    --heldout-fraction 0.2 \
    --split-seed 20260822 \
    --seed 20260815
fi

if require_absent_or_valid \
  "${RESULT_ROOT}/06_loss_calibration" \
  "${RESULT_ROOT}/06_loss_calibration/loss_scale_calibration.json" \
  "LOSS_SCALE_CALIBRATION_COMPLETE"; then
  echo "[6/8] SKIP: raw loss calibration already completed"
else
  echo "[6/8] Raw loss-scale calibration"
  bash "${PREP}" calibrate \
    --output "${RESULT_ROOT}/06_loss_calibration" \
    --cache "${CACHE}" \
    --checkpoint "${CHECKPOINT}" \
    --norm-stats "${NORM}" \
    --smolvlm-model "${SMOLVLM}" \
    --mode-gate "${RESULT_ROOT}/05_mode_ab/decision/mode_ab_decision.json" \
    --sequences 16 \
    --heldout-fraction 0.2 \
    --split-seed 20260822 \
    --seed 20260815
fi

CALIBRATION=${RESULT_ROOT}/06_loss_calibration/loss_scale_calibration.json
TEMPLATE=${RESULT_ROOT}/06_loss_calibration/approved_loss_weights.template.json
APPROVED=${RESULT_ROOT}/06_loss_calibration/approved_loss_weights.json

"${PYTHON}" - "${CALIBRATION}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print("\nRaw loss calibration (mean / p95):")
for name, stats in sorted(payload["raw_loss_summary"].items()):
    print(f"  {name:32s} mean={stats['mean']:.10g} p95={stats['p95']:.10g}")
PY

validate_approved_weights() {
  "${PYTHON}" - "${APPROVED}" "${SOURCE_HASH}" <<'PY'
import json, math, sys
path, source = sys.argv[1:]
try:
    payload = json.load(open(path))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
names = ("condition", "first5_action", "full_chunk_action", "continuous_gripper", "update_regularization")
try:
    values = {name: float(payload[name]) for name in names}
except (KeyError, TypeError, ValueError):
    raise SystemExit(1)
passed = (
    payload.get("approved_by_user") is True
    and payload.get("source_combined_sha256") == source
    and all(math.isfinite(value) and value >= 0.0 for value in values.values())
    and values["first5_action"] > 0.0
)
raise SystemExit(0 if passed else 1)
PY
}

while ! validate_approved_weights; do
  if [[ ! -t 0 ]]; then
    echo "Calibration requires an interactive terminal for explicit weight approval." >&2
    exit 2
  fi
  echo
  echo "Enter finite non-negative weights. first5_action must be greater than zero."
  read -r -p "condition: " weight_condition
  read -r -p "first5_action (primary): " weight_first5
  read -r -p "full_chunk_action: " weight_full_chunk
  read -r -p "continuous_gripper: " weight_gripper
  read -r -p "update_regularization: " weight_regularization
  read -r -p "Type APPROVE to start the smoke and final 150K training: " approval
  if [[ "${approval}" != "APPROVE" ]]; then
    echo "Weights were not approved; enter them again or press Ctrl-C to stop." >&2
    continue
  fi

  if ! "${PYTHON}" - "${TEMPLATE}" "${APPROVED}" \
    "${weight_condition}" "${weight_first5}" "${weight_full_chunk}" \
    "${weight_gripper}" "${weight_regularization}" <<'PY'
import json, math, os, sys
template_path, output_path = sys.argv[1:3]
names = (
    "condition",
    "first5_action",
    "full_chunk_action",
    "continuous_gripper",
    "update_regularization",
)
values = [float(value) for value in sys.argv[3:]]
if not all(math.isfinite(value) and value >= 0.0 for value in values):
    raise ValueError("all approved weights must be finite and non-negative")
if values[1] <= 0.0:
    raise ValueError("first5_action must be greater than zero")
payload = json.load(open(template_path))
payload["approved_by_user"] = True
payload.update(dict(zip(names, values)))
temporary = f"{output_path}.tmp-{os.getpid()}"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output_path)
print(f"APPROVED_WEIGHTS_WRITTEN {output_path}")
PY
  then
    echo "Invalid weights; no approval file was written. Enter them again." >&2
  fi
done

port=${SIMVLA_DDP_PORT:-$((29600 + $$ % 500))}
export SIMVLA_DDP_PORT=${port}

if require_absent_or_valid \
  "${RESULT_ROOT}/07_smoke" \
  "${RESULT_ROOT}/07_smoke/run_summary.json" \
  "TWO_GPU_SMOKE_PASS"; then
  echo "[7/8] SKIP: two-GPU smoke already passed"
else
  echo "[7/8] Two-GPU smoke"
  bash "${TRAIN}" \
    --output "${RESULT_ROOT}/07_smoke" \
  --cache "${CACHE}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM}" \
  --smolvlm-model "${SMOLVLM}" \
  --parity-gate "${PARITY_GATE}" \
  --analysis-gate "${RESULT_ROOT}/03_token_analysis/token_analysis_gate.json" \
  --parameter-gate "${RESULT_ROOT}/04_parameter_audit/simvla_v0_parameter_audit.json" \
  --mode-gate "${RESULT_ROOT}/05_mode_ab/decision/mode_ab_decision.json" \
  --calibration-gate "${CALIBRATION}" \
  --approved-weights "${APPROVED}" \
  --smoke \
  --max-steps 5 \
  --logical-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --peak-lr 1e-4 \
  --weight-decay 0 \
  --validation-interval 5 \
  --validation-batches 2 \
  --save-interval 5 \
  --seed 20260822 \
    --split-seed 20260822
fi

if require_absent_or_valid \
  "${RESULT_ROOT}/08_train_150k" \
  "${RESULT_ROOT}/08_train_150k/run_summary.json" \
  "FINAL_150K_TRAINING_COMPLETE"; then
  echo "[8/8] SKIP: final-150K training already completed"
else
  echo "[8/8] Scientific final-150K training"
  export WANDB_MODE=${WANDB_MODE:-online}
  bash "${TRAIN}" \
    --output "${RESULT_ROOT}/08_train_150k" \
  --cache "${CACHE}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM}" \
  --smolvlm-model "${SMOLVLM}" \
  --parity-gate "${PARITY_GATE}" \
  --analysis-gate "${RESULT_ROOT}/03_token_analysis/token_analysis_gate.json" \
  --parameter-gate "${RESULT_ROOT}/04_parameter_audit/simvla_v0_parameter_audit.json" \
  --mode-gate "${RESULT_ROOT}/05_mode_ab/decision/mode_ab_decision.json" \
  --calibration-gate "${CALIBRATION}" \
  --approved-weights "${APPROVED}" \
  --smoke-gate "${RESULT_ROOT}/07_smoke/run_summary.json" \
  --max-steps 150000 \
  --logical-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --peak-lr 1e-4 \
  --weight-decay 0 \
  --log-interval 100 \
  --validation-interval 5000 \
  --validation-batches 8 \
  --save-interval 10000 \
  --wandb-project gnaroshi-simvla-native-v0 \
  --wandb-name correct_native_v0_k4_rank64_150k \
  --seed 20260822 \
    --split-seed 20260822
fi

echo "TRAINING_COMPLETE ${RESULT_ROOT}/08_train_150k/checkpoints/native_v0_step_150000.pt"
