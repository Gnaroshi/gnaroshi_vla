#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

OUTPUT_ROOT= SOURCE_LOCK= K1_GATE= FREEZE_GATE= FINAL_MANIFEST= SPLIT_CONTRACT= INVENTORY=
GPU_LIST="4 5 6 7"
while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --k1-gate) K1_GATE=$2; shift 2 ;;
    --freeze-gate) FREEZE_GATE=$2; shift 2 ;;
    --final-evaluation-manifest) FINAL_MANIFEST=$2; shift 2 ;;
    --split-contract) SPLIT_CONTRACT=$2; shift 2 ;;
    --full-cache-inventory) INVENTORY=$2; shift 2 ;;
    --gpus) GPU_LIST=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
for name in OUTPUT_ROOT SOURCE_LOCK K1_GATE FREEZE_GATE FINAL_MANIFEST SPLIT_CONTRACT INVENTORY; do
  [[ -n "${!name}" ]] || { echo "Missing required ${name}" >&2; exit 2; }
done
if [[ "${OPENPI_LATENTLOOP_FULL_CACHE_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_LATENTLOOP_FULL_CACHE_RUN=1 only after independent preparation review." >&2
  exit 2
fi
OUTPUT_ROOT=$(realpath -m -- "${OUTPUT_ROOT}")
openpi_ll_refuse_nonempty "${OUTPUT_ROOT}"
read -r -a GPUS <<< "${GPU_LIST}"
NUM_SHARDS=$("${OPENPI_LL_MAIN_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_shards"])' "${INVENTORY}")
if ((${#GPUS[@]} != NUM_SHARDS)); then
  echo "GPU count ${#GPUS[@]} must equal frozen shard count ${NUM_SHARDS}." >&2
  exit 2
fi

# Fail before result-root creation if source, bounded gates, or inventory are stale.
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/pi05_stage_gate_v2.py" \
  --stage stage2_full_cache --source-lock "${SOURCE_LOCK}" \
  --artifact "${K1_GATE}" --artifact "${FREEZE_GATE}" --artifact "${INVENTORY}" >/dev/null
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/verify_pi05_final_manifest_v2.py" \
  --manifest "${FINAL_MANIFEST}" --source-lock "${SOURCE_LOCK}" >/dev/null

QUERY_COUNT=$("${OPENPI_LL_MAIN_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["statistics"]["queries"])' "${INVENTORY}")
# Bounded real cache measured about 23.75 MB/record. Require 10% headroom at 24 MiB/record.
REQUIRED_BYTES=$((QUERY_COUNT * 24 * 1024 * 1024 * 11 / 10))
STORAGE_PROBE=$(dirname -- "${OUTPUT_ROOT}")
while [[ ! -e "${STORAGE_PROBE}" && "${STORAGE_PROBE}" != "/" ]]; do
  STORAGE_PROBE=$(dirname -- "${STORAGE_PROBE}")
done
AVAILABLE_BYTES=$(df --output=avail -B1 "${STORAGE_PROBE}" | tail -n 1 | tr -d ' ')
if ((AVAILABLE_BYTES < REQUIRED_BYTES)); then
  echo "Insufficient storage: need at least ${REQUIRED_BYTES} bytes, have ${AVAILABLE_BYTES}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"
pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup INT TERM ERR
for ((index=0; index<NUM_SHARDS; index++)); do
  CUDA_VISIBLE_DEVICES=${GPUS[index]} \
  "${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/generate_pi05_latentloop_cache_v2.py" \
    --run --output "${OUTPUT_ROOT}/shard_${index}" --checkpoint "${OPENPI_LL_CHECKPOINT}" \
    --source-lock "${SOURCE_LOCK}" --k1-gate "${K1_GATE}" --freeze-gate "${FREEZE_GATE}" \
    --final-evaluation-manifest "${FINAL_MANIFEST}" --split-contract "${SPLIT_CONTRACT}" \
    --full-cache-inventory "${INVENTORY}" --shard-index "${index}" --num-shards "${NUM_SHARDS}" \
    --noise-seed-base 20260820 --device cuda \
    >"${OUTPUT_ROOT}/logs/shard_${index}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
pids=()
trap - INT TERM ERR

merge_args=()
for ((index=0; index<NUM_SHARDS; index++)); do
  merge_args+=(--shard "${OUTPUT_ROOT}/shard_${index}")
done
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/merge_pi05_latentloop_cache.py" \
  --run --output "${OUTPUT_ROOT}/merged" "${merge_args[@]}" \
  --source-lock "${SOURCE_LOCK}" --final-evaluation-manifest "${FINAL_MANIFEST}" \
  --split-contract "${SPLIT_CONTRACT}" --full-cache-inventory "${INVENTORY}"

printf 'FULL_CACHE_PRODUCER_COMPLETE root=%s merged=%s\n' \
  "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/merged"
printf 'V0 remains blocked until an independent validator emits FULL_CACHE_SCHEMA_V2_PASS.\n'
