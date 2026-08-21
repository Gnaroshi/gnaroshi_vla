#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

OUTPUT=
SOURCE_LOCK=
K1_GATE=
FREEZE_GATE=
FINAL_MANIFEST=
SPLIT_CONTRACT=
LOSS_WEIGHTS_GATE=
GPU=4
MAX_STEPS=150000
VALIDATION_INTERVAL=1000
VALIDATION_EXAMPLES=32
SAVE_INTERVAL=5000
WANDB_MODE=online
WANDB_NAME=
ACTION_EXECUTION_MODE=A
while (($#)); do
  case "$1" in
    --output) OUTPUT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --k1-gate) K1_GATE=$2; shift 2 ;;
    --freeze-gate) FREEZE_GATE=$2; shift 2 ;;
    --final-evaluation-manifest) FINAL_MANIFEST=$2; shift 2 ;;
    --split-contract) SPLIT_CONTRACT=$2; shift 2 ;;
    --loss-weights-gate) LOSS_WEIGHTS_GATE=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    --validation-interval) VALIDATION_INTERVAL=$2; shift 2 ;;
    --validation-examples) VALIDATION_EXAMPLES=$2; shift 2 ;;
    --save-interval) SAVE_INTERVAL=$2; shift 2 ;;
    --wandb-mode) WANDB_MODE=$2; shift 2 ;;
    --wandb-name) WANDB_NAME=$2; shift 2 ;;
    --action-execution-mode) ACTION_EXECUTION_MODE=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${ACTION_EXECUTION_MODE}" == "A" || "${ACTION_EXECUTION_MODE}" == "B" ]] || {
  echo "--action-execution-mode must be A or B" >&2; exit 2;
}

for required in OUTPUT SOURCE_LOCK K1_GATE FREEZE_GATE FINAL_MANIFEST SPLIT_CONTRACT LOSS_WEIGHTS_GATE; do
  [[ -n "${!required}" ]] || { echo "Missing required argument for ${required}" >&2; exit 2; }
done
if [[ "${OPENPI_LATENTLOOP_STREAMING_TRAIN_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_LATENTLOOP_STREAMING_TRAIN_RUN=1 after approving raw-scale loss weights." >&2
  exit 2
fi
openpi_ll_refuse_nonempty "${OUTPUT}"
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "--gpu must be one physical GPU index" >&2; exit 2; }
for value in MAX_STEPS VALIDATION_INTERVAL VALIDATION_EXAMPLES SAVE_INTERVAL; do
  [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || { echo "${value} must be positive" >&2; exit 2; }
done

extra_wandb=()
if [[ -n "${WANDB_NAME}" ]]; then
  extra_wandb=(--wandb-name "${WANDB_NAME}")
fi
CUDA_VISIBLE_DEVICES=${GPU} OPENPI_LATENTLOOP_STREAMING_RUN=1 \
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/train_pi05_v0_streaming.py" \
  --run --output "${OUTPUT}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
  --source-lock "${SOURCE_LOCK}" --k1-gate "${K1_GATE}" \
  --freeze-gate "${FREEZE_GATE}" \
  --final-evaluation-manifest "${FINAL_MANIFEST}" \
  --split-contract "${SPLIT_CONTRACT}" --loss-weights-gate "${LOSS_WEIGHTS_GATE}" \
  --max-steps "${MAX_STEPS}" --validation-interval "${VALIDATION_INTERVAL}" \
  --validation-examples "${VALIDATION_EXAMPLES}" --save-interval "${SAVE_INTERVAL}" \
  --action-execution-mode "${ACTION_EXECUTION_MODE}" \
  --batch-size 1 --learning-rate 1e-4 --weight-decay 0 \
  --seed 42 --noise-seed-base 20260820 --wandb-mode "${WANDB_MODE}" \
  "${extra_wandb[@]}" --device cuda
