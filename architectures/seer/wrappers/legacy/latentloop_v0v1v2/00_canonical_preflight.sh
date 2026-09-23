#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_four_gpu_lane
refuse_existing "${CAMPAIGN_ROOT}/gates"
TMP="$(mktemp -d /tmp/s18_runtime_gate.XXXXXX)"
trap 'rm -rf "${TMP}"' EXIT
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/verify_s18_runtime.py" \
  --repo-root "${REPO_ROOT}" --contract "${SOURCE_CONTRACT}" \
  --libero-path "${LIBERO_PATH}" --output "${TMP}/runtime.json"
mkdir -p "${CAMPAIGN_ROOT}/gates"
cp "${TMP}/runtime.json" "${CAMPAIGN_ROOT}/gates/s18_runtime.json"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/record_s18_preflight.py" \
  --source "${SOURCE_GATE_ARTIFACT}" --runtime "${CAMPAIGN_ROOT}/gates/s18_runtime.json" \
  --four-gpu-contract "${GPU_CONTRACT}" --output "${PREFLIGHT_ARTIFACT}"
