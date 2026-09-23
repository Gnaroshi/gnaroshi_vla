#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_reproduction_pass
INPUT="${TRAINING_EPISODE_KEYS:-${CAMPAIGN_ROOT}/v1/splits/locked_dataset_episode_keys.json}"
OUTPUT="${CAMPAIGN_ROOT}/v1/splits/v1_episode_disjoint_split.json"
if [[ ! -f "${INPUT}" ]]; then
  "${S18_PYTHON}" "${REPO_ROOT}/tools/seer/lock_libero_training_episode_keys.py" \
    --data-info "${REPO_ROOT}/architectures/seer/upstream/data_info/libero_10_converted.json" \
    --expected-sha256 4b8241c1dd39b62c56aa6bbd7dca1afb397a4f9862e74f7718a3b00ca4679120 \
    --dataset-name libero_10_converted --output "${INPUT}"
fi
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/create_latentloop_v1v2_splits.py" \
  --training-episode-keys "${INPUT}" \
  --final-episode-manifest "${REPO_ROOT}/.canonical/source_lock/canonical_episode_manifest.csv" \
  --seed 42 --output "${OUTPUT}"
