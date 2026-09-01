#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODE=${1:---all}
PYTHON=${PYTHON:-python}
GPU_ID=${SIMVLA_LATENT_BRIDGE_GPU:-0}
RESULT_ROOT=${SIMVLA_LATENT_BRIDGE_RESULT_ROOT:-}
NORM_STATS=${SIMVLA_NORM_STATS:-"${ROOT}/architectures/simvla/upstream/norm_stats/libero_norm.json"}
CHECKPOINT=${SIMVLA_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}
SMOLVLM_MODEL=${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}

if [[ -z "${RESULT_ROOT}" ]]; then
  echo "Set SIMVLA_LATENT_BRIDGE_RESULT_ROOT to a non-overlay result directory." >&2
  exit 2
fi

export SIMVLA_UPSTREAM_ROOT=${SIMVLA_UPSTREAM_ROOT:-"${ROOT}/architectures/simvla/upstream"}
export LATENT_BRIDGE_UPSTREAM_ROOT=${LATENT_BRIDGE_UPSTREAM_ROOT:-"${ROOT}/architectures/latent_bridge/upstream"}
export LIBERO_ROOT=${LIBERO_ROOT:-"${SIMVLA_UPSTREAM_ROOT}/evaluation/libero/LIBERO"}
export PYTHONPATH="${ROOT}:${SIMVLA_UPSTREAM_ROOT}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

R0_SYNC="${RESULT_ROOT}/r0_sync"
STABLE_PROBE="${RESULT_ROOT}/stable_layer10_probe"
R0_TRAIN="${RESULT_ROOT}/r0_train"
R1_ROLLOUT="${RESULT_ROOT}/r1_dagger_f3"
R1_TRAIN="${RESULT_ROOT}/r1_train"
FINAL_EVAL="${RESULT_ROOT}/final_eval"
SUMMARY="${RESULT_ROOT}/summary"
LOG_ROOT="${RESULT_ROOT}/logs"
STATUS="${RESULT_ROOT}/pipeline.status"
mkdir -p "${LOG_ROOT}"

preflight() {
  for path in "${NORM_STATS}" "${SIMVLA_UPSTREAM_ROOT}/models/modeling_smolvlm_vla.py" "${LIBERO_ROOT}/libero" "${LATENT_BRIDGE_UPSTREAM_ROOT}/qcvla/model/rectified_flow_bridge.py"; do
    if [[ ! -e "${path}" ]]; then
      echo "PREFLIGHT_FAIL missing=${path}" >&2
      return 1
    fi
  done
  "${PYTHON}" - "${LATENT_BRIDGE_UPSTREAM_ROOT}" <<'PY'
import json
import sys
import torch
from architectures.simvla.adapters.latent_bridge.provenance import latent_bridge_source_manifest
from architectures.simvla.adapters.latent_bridge.recipe import scientific_contract
manifest = latent_bridge_source_manifest(sys.argv[1])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print(json.dumps({
    "verdict": "SIMVLA_LATENT_BRIDGE_PREFLIGHT_PASS",
    "gpu": torch.cuda.get_device_name(0),
    "official_commit": manifest["commit"],
    "contract": scientific_contract(),
}, indent=2, sort_keys=True))
PY
}

stage_complete() {
  local stage=$1
  "${PYTHON}" - "${stage}" "${RESULT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

stage, root_text = sys.argv[1:]
root = Path(root_text)

def load(path):
    if not path.is_file():
        return None
    with path.open() as handle:
        return json.load(handle)

ok = False
if stage == "r0_sync":
    data = load(root / "r0_sync/manifest.json")
    ok = bool(
        data
        and data.get("episodes") == 300
        and data.get("trials_per_task") == 30
        and data.get("stable_layer_contract", {}).get("passed") is True
    )
elif stage == "stable_probe":
    data = load(root / "stable_layer10_probe/manifest.json")
    ok = bool(
        data
        and data.get("episodes") == 2
        and data.get("stable_layer_contract", {}).get("passed") is True
    )
elif stage in {"r0_train", "r1_train"}:
    data = load(root / stage / "run_summary.json")
    ok = bool(data and data.get("verdict") == "SIMVLA_LATENT_BRIDGE_TRAINING_COMPLETE")
elif stage == "r1_dagger":
    data = load(root / "r1_dagger_f3/latent_bridge_f3/dagger/manifest.json")
    ok = bool(
        data
        and data.get("episodes") == 300
        and data.get("refresh_every") == 3
    )
elif stage.startswith("seed"):
    data = load(root / f"final_eval/{stage}/comparison_summary.json")
    summaries = (data or {}).get("summaries", {})
    ok = bool(
        data
        and data.get("verdict") == "SIMVLA_LATENT_BRIDGE_EVAL_COMPLETE"
        and set(summaries) == {"baseline_k1", "latent_bridge_f3", "latent_bridge_f4"}
        and all(item.get("episodes") == 200 for item in summaries.values())
    )
elif stage == "summary":
    data = load(root / "summary/three_seed_summary.json")
    ok = bool(data and data.get("verdict") == "SIMVLA_LATENT_BRIDGE_THREE_SEED_SUMMARY_COMPLETE")
raise SystemExit(0 if ok else 1)
PY
}

run_stage() {
  local stage=$1
  shift
  if stage_complete "${stage}" >/dev/null 2>&1; then
    echo "[SKIP] ${stage}: verified complete"
    return 0
  fi
  echo "[START] ${stage} $(date --iso-8601=seconds)"
  set +e
  "$@" 2>&1 | tee "${LOG_ROOT}/${stage}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if (( rc != 0 )); then
    echo "FAILED stage=${stage} rc=${rc} log=${LOG_ROOT}/${stage}.log" | tee "${STATUS}"
    return "${rc}"
  fi
  if ! stage_complete "${stage}"; then
    echo "FAILED stage=${stage} reason=completion_contract_missing log=${LOG_ROOT}/${stage}.log" | tee "${STATUS}"
    return 1
  fi
  echo "[DONE] ${stage} $(date --iso-8601=seconds)"
}

preflight || {
  echo "SIMVLA_LATENT_BRIDGE_PIPELINE_FAILED stage=preflight" | tee "${STATUS}"
  exit 1
}
if [[ "${MODE}" == "--preflight" ]]; then
  exit 0
fi
if [[ "${MODE}" != "--all" ]]; then
  echo "Usage: $0 [--preflight|--all]" >&2
  exit 2
fi
if [[ ${SIMVLA_LATENT_BRIDGE_PIPELINE_RUN:-0} != 1 ]]; then
  echo "Set SIMVLA_LATENT_BRIDGE_PIPELINE_RUN=1 after reviewing this launcher." >&2
  exit 2
fi

set -e
run_stage stable_probe env SIMVLA_LATENT_BRIDGE_COLLECT_RUN=1 \
  bash "${ROOT}/architectures/simvla/wrappers/simvla_latent_bridge_collect_sync.sh" \
  --output "${STABLE_PROBE}" --checkpoint "${CHECKPOINT}" \
  --smolvlm-model "${SMOLVLM_MODEL}" --norm-stats "${NORM_STATS}" \
  --suite libero_10 --max-tasks 1 --num-trials 2 \
  --stable-layer-index 10 --device cuda

run_stage r0_sync env SIMVLA_LATENT_BRIDGE_COLLECT_RUN=1 \
  bash "${ROOT}/architectures/simvla/wrappers/simvla_latent_bridge_collect_sync.sh" \
  --output "${R0_SYNC}" --checkpoint "${CHECKPOINT}" \
  --smolvlm-model "${SMOLVLM_MODEL}" --norm-stats "${NORM_STATS}" \
  --suite libero_10 --num-trials 30 --stable-layer-index 10 --device cuda

run_stage r0_train env SIMVLA_LATENT_BRIDGE_RUN=1 \
  bash "${ROOT}/architectures/simvla/wrappers/simvla_latent_bridge_train.sh" \
  --stage r0 --sync-root "${R0_SYNC}" --output "${R0_TRAIN}" --device cuda

run_stage r1_dagger env SIMVLA_LATENT_BRIDGE_EVAL_RUN=1 \
  bash "${ROOT}/architectures/simvla/wrappers/simvla_latent_bridge_eval.sh" \
  --output "${R1_ROLLOUT}" --checkpoint "${CHECKPOINT}" \
  --smolvlm-model "${SMOLVLM_MODEL}" --norm-stats "${NORM_STATS}" \
  --bridge-checkpoint "${R0_TRAIN}/best.pt" \
  --rows latent_bridge_f3 --suite libero_10 --num-trials 30 \
  --collect-dagger-teacher --seed 0 --device cuda

run_stage r1_train env SIMVLA_LATENT_BRIDGE_RUN=1 \
  bash "${ROOT}/architectures/simvla/wrappers/simvla_latent_bridge_train.sh" \
  --stage r1 --sync-root "${R0_SYNC}" \
  --dagger-root "${R1_ROLLOUT}/latent_bridge_f3/dagger" \
  --resume "${R0_TRAIN}/best.pt" --weights-only-resume \
  --output "${R1_TRAIN}" --device cuda

for seed in 0 1 2; do
  run_stage "seed${seed}" env SIMVLA_LATENT_BRIDGE_EVAL_RUN=1 \
    bash "${ROOT}/architectures/simvla/wrappers/simvla_latent_bridge_eval.sh" \
    --output "${FINAL_EVAL}/seed${seed}" --checkpoint "${CHECKPOINT}" \
    --smolvlm-model "${SMOLVLM_MODEL}" --norm-stats "${NORM_STATS}" \
    --bridge-checkpoint "${R1_TRAIN}/best.pt" \
    --rows baseline_k1 latent_bridge_f3 latent_bridge_f4 \
    --suite libero_10 --num-trials 20 --seed "${seed}" --device cuda
done

run_stage summary "${PYTHON}" -m architectures.simvla.adapters.latent_bridge.summarize \
  --eval-root "${FINAL_EVAL}" --output "${SUMMARY}"

echo "SIMVLA_LATENT_BRIDGE_PIPELINE_COMPLETE" | tee "${STATUS}"
echo "Summary: ${SUMMARY}/three_seed_summary.json"
