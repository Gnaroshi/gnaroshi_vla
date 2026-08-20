#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

OUTPUT=
NUM_SHARDS=4
while (($#)); do
  case "$1" in
    --output) OUTPUT=$2; shift 2 ;;
    --num-shards) NUM_SHARDS=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${OUTPUT:?--output is required}"
OUTPUT=$(realpath -m -- "${OUTPUT}")
if [[ "${OPENPI_LATENTLOOP_PREPARE_RUN:-0}" != "1" ]]; then
  echo "Set OPENPI_LATENTLOOP_PREPARE_RUN=1 to freeze a new source/protocol/cache inventory." >&2
  exit 2
fi
openpi_ll_refuse_nonempty "${OUTPUT}"
[[ "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || { echo "--num-shards must be positive" >&2; exit 2; }

SOURCE_LOCK=${OUTPUT}/source_lock_v2.json
PROTOCOL_ROOT=${OUTPUT}/protocol
FINAL_MANIFEST=${PROTOCOL_ROOT}/pi05_final_evaluation_manifest_v2.json
SPLIT_CONTRACT=${PROTOCOL_ROOT}/pi05_split_contract_v2.json
INVENTORY=${OUTPUT}/pi05_full_cache_inventory_v2.json

# Each tool verifies all inputs before creating its own output.
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/source_lock_v2.py" create \
  --output "${SOURCE_LOCK}" \
  --checkpoint "${OPENPI_LL_CHECKPOINT}" \
  --norm-stats "${OPENPI_LL_NORM}"

OPENPI_LATENTLOOP_MANIFEST_RUN=1 \
"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/freeze_pi05_protocol_manifests_v2.py" \
  --run --output "${PROTOCOL_ROOT}" --source-lock "${SOURCE_LOCK}" \
  --environment-seed 7 --noise-seed-base 7

"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/freeze_pi05_full_cache_inventory_v2.py" \
  --run --output "${INVENTORY}" --source-lock "${SOURCE_LOCK}" \
  --final-evaluation-manifest "${FINAL_MANIFEST}" --split-contract "${SPLIT_CONTRACT}" \
  --num-shards "${NUM_SHARDS}" --noise-seed-base 20260820

printf 'PREPARATION_ONLY_COMPLETE output=%s source_lock=%s inventory=%s\n' \
  "${OUTPUT}" "${SOURCE_LOCK}" "${INVENTORY}"
printf 'No teacher cache, training, fitting, calibration, or LIBERO evaluation was run.\n'
