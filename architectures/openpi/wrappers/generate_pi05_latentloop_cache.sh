#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

if [[ "${OPENPI_LATENTLOOP_CACHE_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_LATENTLOOP_CACHE_RUN=1 to enable the bounded schema smoke." >&2
  exit 2
fi

OUTPUT= VALIDATION_OUTPUT= SOURCE_LOCK= K1_GATE= FREEZE_GATE= FINAL_MANIFEST= SPLIT_CONTRACT=
MAX_EPISODES=10
MAX_QUERIES_PER_EPISODE=2
GPU=${CUDA_VISIBLE_DEVICES:-4}
while (($#)); do
  case "$1" in
    --output) OUTPUT=$2; shift 2 ;;
    --validation-output) VALIDATION_OUTPUT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --k1-gate) K1_GATE=$2; shift 2 ;;
    --freeze-gate) FREEZE_GATE=$2; shift 2 ;;
    --final-evaluation-manifest) FINAL_MANIFEST=$2; shift 2 ;;
    --split-contract) SPLIT_CONTRACT=$2; shift 2 ;;
    --max-episodes) MAX_EPISODES=$2; shift 2 ;;
    --max-queries-per-episode) MAX_QUERIES_PER_EPISODE=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    full|--full) echo "Full cache generation is disabled pending independent re-audit." >&2; exit 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
for name in OUTPUT VALIDATION_OUTPUT SOURCE_LOCK K1_GATE FREEZE_GATE FINAL_MANIFEST SPLIT_CONTRACT; do
  [[ -n "${!name}" ]] || { echo "Missing required ${name}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing cache output: ${OUTPUT}" >&2; exit 1; }
[[ ! -e "${VALIDATION_OUTPUT}" ]] || { echo "Refusing existing validation output: ${VALIDATION_OUTPUT}" >&2; exit 1; }

# The generator repeats these gates before lazily creating OUTPUT.
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/pi05_stage_gate_v2.py" \
  --stage stage2_cache_smoke --source-lock "${SOURCE_LOCK}" \
  --artifact "${K1_GATE}" --artifact "${FREEZE_GATE}" --output-candidate "${OUTPUT}" >/dev/null
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/verify_pi05_final_manifest_v2.py" \
  --manifest "${FINAL_MANIFEST}" --source-lock "${SOURCE_LOCK}" >/dev/null

CUDA_VISIBLE_DEVICES=${GPU} "${OPENPI_LL_MAIN_PY}" \
  "${OPENPI_LL_ROOT}/tools/openpi/generate_pi05_latentloop_cache_v2.py" \
  --run --output "${OUTPUT}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
  --source-lock "${SOURCE_LOCK}" --k1-gate "${K1_GATE}" --freeze-gate "${FREEZE_GATE}" \
  --final-evaluation-manifest "${FINAL_MANIFEST}" --split-contract "${SPLIT_CONTRACT}" \
  --max-episodes "${MAX_EPISODES}" --max-queries-per-episode "${MAX_QUERIES_PER_EPISODE}" \
  --noise-seed-base 20260820 --device cuda

"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/validate_pi05_cache_v2.py" \
  --cache "${OUTPUT}" --source-lock "${SOURCE_LOCK}" \
  --final-evaluation-manifest "${FINAL_MANIFEST}" --split-contract "${SPLIT_CONTRACT}" \
  --output "${VALIDATION_OUTPUT}"
