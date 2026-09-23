#!/usr/bin/env bash
set -euo pipefail
[[ "${ENABLE_CONDITIONAL:-0}" == "1" ]] || { echo "[DISABLED] set ENABLE_CONDITIONAL=1 after V1_ONLINE_PASS" >&2; exit 2; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_base_identity
require_four_gpu_lane
require_gate "${CAMPAIGN_ROOT}/v1/gates/v1_online_gate.json" V1_ONLINE_PASS
require_gate "${REPO_ROOT}/v1_runtime_integration_status.json" V1_RUNTIME_INTEGRATION_PASS
BUDGET="$("${S18_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_budget_epochs"])' "${CAMPAIGN_ROOT}/v1/selection/v1_budget_selection.json")"
CHECKPOINT="${CAMPAIGN_ROOT}/v1/train/e${BUDGET}/selected_checkpoint.pth"
OUTPUT="${CAMPAIGN_ROOT}/v1/eval/conditional_k12_200"
[[ -f "${CHECKPOINT}" ]] || fail "selected V1 checkpoint missing"
refuse_existing "${OUTPUT}"
"${S18_PYTHON}" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port="${MASTER_PORT:-16320}" \
  "${REPO_ROOT}/tools/seer/run_latentloop_v1v2_eval_runtime.py" \
  --integration-status "${REPO_ROOT}/v1_runtime_integration_status.json" --mode v1_fixed \
  --checkpoint "${CHECKPOINT}" \
  --teacher "${TEACHER}" --adapter-init "${ADAPTER}" \
  --vit-checkpoint "${VIT}" --libero-path "${LIBERO_PATH}" --output-root "${OUTPUT}" \
  --episode-manifest "${REPO_ROOT}/.canonical/source_lock/canonical_episode_manifest.csv" \
  --query-interval 12 --allow-conditional-interval \
  --seed 42 --tasks 10 --episodes-per-task 20 --max-steps 600 \
  --control-hz 20 --action-prediction-horizon 3 --temporal-ensembling \
  --temporal-ensemble-temperature 0.01 --renderer osmesa --precision fp32 --deterministic
