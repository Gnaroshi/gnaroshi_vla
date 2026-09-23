#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/architectures/seer/wrappers/latentloop_v0v1v2/_common.sh"
require_base_identity
require_four_gpu_lane
require_reproduction_pass
require_gate "${REPO_ROOT}/v1_runtime_integration_status.json" V1_RUNTIME_INTEGRATION_PASS
require_gate "${CAMPAIGN_ROOT}/v1/calibration/frozen_loss_weights.json" V1_RAW_LOSS_WEIGHTS_LOCKED
OUTPUT="${CAMPAIGN_ROOT}/v1/train/e20"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_port="${MASTER_PORT:-16120}" "${REPO_ROOT}/tools/seer/run_latentloop_v1_training_runtime.py" \
  --integration-status "${REPO_ROOT}/v1_runtime_integration_status.json" \
  --epochs 20 --output-root "${OUTPUT}" --teacher "${TEACHER}" --adapter-init "${ADAPTER}" \
  --vit-checkpoint "${VIT}" --dataset-root "${DATASET_OUTER_ROOT}" \
  --libero-path "${LIBERO_PATH}" \
  --loss-weights "${CAMPAIGN_ROOT}/v1/calibration/frozen_loss_weights.json" \
  --split-manifest "${CAMPAIGN_ROOT}/v1/splits/v1_episode_disjoint_split.json" \
  --seed 42 --per-gpu-batch 16 --workers "${WORKERS:-8}" --gradient-accumulation 8 \
  --learning-rate 0.001 --weight-decay 0.0001 --warmup-fraction 0.05 \
  --precision fp32 --deterministic
