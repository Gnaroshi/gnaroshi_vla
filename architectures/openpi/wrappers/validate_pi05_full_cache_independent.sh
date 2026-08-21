#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/latentloop_common.sh"

CACHE= OUTPUT= SOURCE_LOCK= FINAL_MANIFEST= SPLIT_CONTRACT= INVENTORY=
while (($#)); do
  case "$1" in
    --cache) CACHE=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --final-evaluation-manifest) FINAL_MANIFEST=$2; shift 2 ;;
    --split-contract) SPLIT_CONTRACT=$2; shift 2 ;;
    --full-cache-inventory) INVENTORY=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
for name in CACHE OUTPUT SOURCE_LOCK FINAL_MANIFEST SPLIT_CONTRACT INVENTORY; do
  [[ -n "${!name}" ]] || { echo "Missing required ${name}" >&2; exit 2; }
done
[[ "${OPENPI_LATENTLOOP_FULL_CACHE_VALIDATE_RUN:-0}" == "1" ]] || {
  echo "Set OPENPI_LATENTLOOP_FULL_CACHE_VALIDATE_RUN=1 for the independent full read." >&2
  exit 2
}
openpi_ll_refuse_nonempty "${OUTPUT}"

"${OPENPI_LL_MAIN_PY}" "${OPENPI_LL_ROOT}/tools/openpi/validate_pi05_cache_v2.py" \
  --cache "${CACHE}" --source-lock "${SOURCE_LOCK}" \
  --final-evaluation-manifest "${FINAL_MANIFEST}" --split-contract "${SPLIT_CONTRACT}" \
  --full-cache-inventory "${INVENTORY}" --require-full --output "${OUTPUT}"
