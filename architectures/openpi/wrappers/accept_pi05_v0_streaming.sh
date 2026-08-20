#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

OUTPUT=
GPU=4
PORT=8161
RAW_LOSS_EXAMPLES=32
while (($#)); do
  case "$1" in
    --output) OUTPUT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --raw-loss-examples) RAW_LOSS_EXAMPLES=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${OUTPUT:?--output is required}"
OUTPUT=$(realpath -m -- "${OUTPUT}")
if [[ "${OPENPI_LATENTLOOP_STREAMING_ACCEPT_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_LATENTLOOP_STREAMING_ACCEPT_RUN=1 for bounded streaming acceptance." >&2
  exit 2
fi
openpi_ll_refuse_nonempty "${OUTPUT}"
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "--gpu must be one physical GPU index" >&2; exit 2; }
[[ "${PORT}" =~ ^[0-9]+$ ]] || { echo "--port must be numeric" >&2; exit 2; }
[[ "${RAW_LOSS_EXAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--raw-loss-examples must be positive" >&2
  exit 2
}

CONFIG=${OPENPI_LL_UPSTREAM}/checkpoints/pi05_base_pytorch/config.json
LOCK=${OUTPUT}/source_lock_v2.json
PROTOCOL=${OUTPUT}/protocol
FINAL_MANIFEST=${PROTOCOL}/pi05_final_evaluation_manifest_v2.json
SPLIT_CONTRACT=${PROTOCOL}/pi05_split_contract_v2.json
K1_TENSOR=${OUTPUT}/k1_tensor
FREEZE=${OUTPUT}/freeze
K1_EPISODE=${OUTPUT}/k1_episode
K1_GATE=${K1_EPISODE}/combined_gate/pi05_k1_equivalence.json
RAW_LOSS=${OUTPUT}/v0_streaming_raw_loss

echo "[1/6] Freeze and verify current source/checkpoint/environment identity."
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/source_lock_v2.py" create \
  --output "${LOCK}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
  --config "${CONFIG}" --norm-stats "${OPENPI_LL_NORM}"
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/source_lock_v2.py" verify \
  --lock "${LOCK}"

echo "[2/6] Freeze the final evaluation manifest and disjoint teacher split contract."
OPENPI_LATENTLOOP_MANIFEST_RUN=1 \
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/freeze_pi05_protocol_manifests_v2.py" \
  --run --output "${PROTOCOL}" --source-lock "${LOCK}"
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/verify_pi05_final_manifest_v2.py" \
  --manifest "${FINAL_MANIFEST}" --source-lock "${LOCK}"

echo "[3/6] Check real-batch K=1 tensor equivalence on GPU ${GPU}."
CUDA_VISIBLE_DEVICES=${GPU} OPENPI_LATENTLOOP_K1_RUN=1 \
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/audit_pi05_k1_equivalence.py" \
  --run --output "${K1_TENSOR}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
  --source-lock "${LOCK}" --noise-seed 20260820 --flow-steps 10 --device cuda

echo "[4/6] Check full base freeze and adapter-only optimizer membership on GPU ${GPU}."
CUDA_VISIBLE_DEVICES=${GPU} OPENPI_LATENTLOOP_FREEZE_RUN=1 \
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/verify_pi05_freeze_v2.py" \
  --run --output "${FREEZE}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
  --source-lock "${LOCK}" --device cuda

echo "[5/6] Run only two paired K=1 LIBERO episodes."
OPENPI_LATENTLOOP_EVAL_RUN=1 \
bash "${OPENPI_LL_ROOT}/architectures/openpi/wrappers/eval_pi05_latentloop.sh" \
  --mode k1-smoke --suite libero_10 --output "${K1_EPISODE}" \
  --source-lock "${LOCK}" \
  --k1-tensor-report "${K1_TENSOR}/pi05_k1_equivalence.json" \
  --freeze-gate "${FREEZE}/freeze_gate.json" \
  --gpu "${GPU}" --port "${PORT}"

echo "[6/6] Measure untrained V0 loss on online teacher windows; write no teacher tensors."
CUDA_VISIBLE_DEVICES=${GPU} OPENPI_LATENTLOOP_STREAMING_RUN=1 \
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/train_pi05_v0_streaming.py" \
  --run --raw-loss-only --output "${RAW_LOSS}" \
  --checkpoint "${OPENPI_LL_CHECKPOINT}" --source-lock "${LOCK}" \
  --k1-gate "${K1_GATE}" --freeze-gate "${FREEZE}/freeze_gate.json" \
  --final-evaluation-manifest "${FINAL_MANIFEST}" \
  --split-contract "${SPLIT_CONTRACT}" \
  --raw-loss-examples "${RAW_LOSS_EXAMPLES}" --validation-examples 32 \
  --max-steps 1 --seed 42 --noise-seed-base 20260820 \
  --wandb-mode disabled --device cuda

"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/source_lock_v2.py" verify \
  --lock "${LOCK}"
printf 'STREAMING_V0_PREAPPROVAL_COMPLETE output=%s raw_loss=%s\n' \
  "${OUTPUT}" "${RAW_LOSS}/raw_loss_calibration.json"
printf 'Persistent teacher tensor cache written: 0 bytes. Training was not started.\n'
