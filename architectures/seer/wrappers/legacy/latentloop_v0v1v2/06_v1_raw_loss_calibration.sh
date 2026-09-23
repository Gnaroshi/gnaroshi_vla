#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_four_gpu_lane
require_reproduction_pass
require_gate "${REPO_ROOT}/v1_runtime_integration_status.json" V1_RUNTIME_INTEGRATION_PASS
require_gate "${CAMPAIGN_ROOT}/v1/splits/v1_episode_disjoint_split.json" V1_V2_SPLITS_LOCKED
INPUT="${V1_RAW_LOSS_INPUT:-${CAMPAIGN_ROOT}/v1/calibration/raw_loss_samples.json}"
OUTPUT="${CAMPAIGN_ROOT}/v1/calibration/frozen_loss_weights.json"
refuse_existing "${CAMPAIGN_ROOT}/v1/calibration"
"${S18_PYTHON}" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_port="${MASTER_PORT:-16100}" "${REPO_ROOT}/tools/seer/collect_latentloop_v1_raw_losses.py" \
  --integration-status "${REPO_ROOT}/v1_runtime_integration_status.json" \
  --output "${INPUT}" --teacher "${TEACHER}" --adapter-init "${ADAPTER}" \
  --vit-checkpoint "${VIT}" --dataset-root "${DATASET_OUTER_ROOT}" \
  --libero-path "${LIBERO_PATH}" \
  --split-manifest "${CAMPAIGN_ROOT}/v1/splits/v1_episode_disjoint_split.json" \
  --seed 42 --per-gpu-batch 16 --workers "${WORKERS:-8}" \
  --target-microbatches "${V1_RAW_LOSS_MICROBATCHES_PER_RANK:-48}"
[[ -f "${INPUT}" ]] || fail "raw-loss collector did not produce: ${INPUT}"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" "${REPO_ROOT}/tools/seer/calibrate_latentloop_v1_raw_losses.py" \
  --raw-losses "${INPUT}" \
  --split-manifest "${CAMPAIGN_ROOT}/v1/splits/v1_episode_disjoint_split.json" \
  --output "${OUTPUT}"
