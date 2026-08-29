#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

MODE= CACHE= OUTPUT= SOURCE_LOCK= CACHE_GATE= FINAL_MANIFEST= SPLIT_CONTRACT= INVENTORY=
LOSS_WEIGHTS_GATE=
GPU=${CUDA_VISIBLE_DEVICES:-4}
MAX_STEPS=150000
while (($#)); do
  case "$1" in
    --mode) MODE=$2; shift 2 ;;
    --cache) CACHE=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --cache-gate) CACHE_GATE=$2; shift 2 ;;
    --final-evaluation-manifest) FINAL_MANIFEST=$2; shift 2 ;;
    --split-contract) SPLIT_CONTRACT=$2; shift 2 ;;
    --full-cache-inventory) INVENTORY=$2; shift 2 ;;
    --loss-weights-gate) LOSS_WEIGHTS_GATE=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --max-steps) MAX_STEPS=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
for name in MODE CACHE OUTPUT SOURCE_LOCK CACHE_GATE FINAL_MANIFEST SPLIT_CONTRACT INVENTORY; do
  [[ -n "${!name}" ]] || { echo "Missing required ${name}" >&2; exit 2; }
done
[[ "${MODE}" == "raw-loss" || "${MODE}" == "train" ]] || {
  echo "--mode must be raw-loss or train" >&2; exit 2;
}
[[ "${OPENPI_LATENTLOOP_TRAIN_RUN:-0}" == "1" ]] || {
  echo "Set OPENPI_LATENTLOOP_TRAIN_RUN=1 for the explicitly selected V0 entry." >&2
  exit 2
}
if [[ "${MODE}" == "train" && -z "${LOSS_WEIGHTS_GATE}" ]]; then
  echo "V0 training requires a separately approved --loss-weights-gate." >&2
  exit 2
fi

args=(
  --run --variant v0 --cache "${CACHE}" --output "${OUTPUT}"
  --checkpoint "${OPENPI_LL_CHECKPOINT}" --source-lock "${SOURCE_LOCK}"
  --cache-gate "${CACHE_GATE}" --final-evaluation-manifest "${FINAL_MANIFEST}"
  --split-contract "${SPLIT_CONTRACT}" --full-cache-inventory "${INVENTORY}"
  --max-steps "${MAX_STEPS}" --batch-size 1 --learning-rate 1e-4 --weight-decay 0
  --validation-interval 1000 --save-interval 5000 --log-interval 20
  --wandb-log-interval 100 --seed 42 --device cuda
)
if [[ "${MODE}" == "raw-loss" ]]; then
  args+=(--raw-loss-only --raw-loss-examples 32 --wandb-mode disabled)
else
  args+=(--loss-weights-gate "${LOSS_WEIGHTS_GATE}" --wandb-mode online)
fi

CUDA_VISIBLE_DEVICES=${GPU} "${OPENPI_LL_MAIN_PY}" \
  "${OPENPI_LL_ROOT}/tools/openpi/train_pi05_latentloop.py" "${args[@]}"
