#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODE=${1:---all}
PYTHON=${PYTHON:-/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python}
GPU_ID=${SIMVLA_VLA_CACHE_GPU:-0}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
LIBERO=${LIBERO_ROOT:-${UPSTREAM}/evaluation/libero/LIBERO}
RESULT_ROOT=${SIMVLA_VLA_CACHE_RESULT_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/vla_cache/libero_10_paper_compare_v1}
REFERENCE_ROOT=${SIMVLA_REFERENCE_RESULT_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/paper_four_suite_three_seed_v1}
MANIFEST_ROOT=${SIMVLA_EPISODE_MANIFEST_ROOT:-${REFERENCE_ROOT}/manifests/libero_10}
NORM_STATS=${SIMVLA_NORM_STATS:-${ROOT}/architectures/simvla/adapters/latentloop/assets/libero_norm_official_32700d0.json}
CHECKPOINT_REVISION=93dc4d90b0596c652ad2840ad743c62b9c4473fb

export SIMVLA_UPSTREAM_ROOT="${UPSTREAM}"
export LIBERO_ROOT="${LIBERO}"
export PYTHONPATH="${ROOT}:${UPSTREAM}:${LIBERO}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}

LOG_ROOT="${RESULT_ROOT}/logs"
STATUS="${RESULT_ROOT}/pipeline.status"
LOCK="${RESULT_ROOT}/pipeline.lock"
mkdir -p "${LOG_ROOT}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "VLA_CACHE_PIPELINE_ALREADY_RUNNING lock=${LOCK}" >&2
  exit 2
fi

fail() {
  local rc=$?
  echo "SIMVLA_VLA_CACHE_LIBERO_PIPELINE_FAILED rc=${rc} line=${BASH_LINENO[0]} status=${STATUS}" | tee "${STATUS}"
  exit "${rc}"
}
trap fail ERR

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
    return json.loads(path.read_text())

if stage == "checkpoint_smoke":
    data = load(root / "preflight/real_checkpoint_smoke.json")
    ok = bool(data and data.get("verdict") == "SIMVLA_VLA_CACHE_REAL_CHECKPOINT_SMOKE_PASS")
elif stage == "libero_smoke":
    data = load(root / "preflight/libero_smoke/summary.json")
    ok = bool(data and data.get("verdict") == "SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE" and data.get("episodes") == 1)
elif stage == "matched_full_control":
    data = load(root / "matched_full_control/seed01/summary.json")
    ok = bool(data and data.get("verdict") == "SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE" and data.get("episodes") == 500 and data.get("actual_kv_reuse_queries") == 0)
elif stage.startswith("vla_cache_seed"):
    seed = stage.rsplit("seed", 1)[1]
    data = load(root / f"vla_cache/seed{seed}/summary.json")
    ok = bool(data and data.get("verdict") == "SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE" and data.get("episodes") == 500 and data.get("actual_kv_reuse_queries", 0) > 0)
elif stage == "summary":
    data = load(root / "summary/comparison_summary.json")
    ok = bool(data and data.get("verdict") == "SIMVLA_VLA_CACHE_LIBERO_THREE_SEED_COMPLETE")
else:
    ok = False
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
  "$@" 2>&1 | tee "${LOG_ROOT}/${stage}.log"
  if ! stage_complete "${stage}"; then
    echo "stage completion contract missing: ${stage}" >&2
    return 1
  fi
  echo "[DONE] ${stage} $(date --iso-8601=seconds)"
}

preflight() {
  local required=(
    "${PYTHON}"
    "${UPSTREAM}/models/modeling_smolvlm_vla.py"
    "${LIBERO}/libero"
    "${NORM_STATS}"
    "${MANIFEST_ROOT}/seed01/episode_manifest.json"
    "${MANIFEST_ROOT}/seed02/episode_manifest.json"
    "${MANIFEST_ROOT}/seed03/episode_manifest.json"
  )
  for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
      echo "PREFLIGHT_FAIL missing=${path}" >&2
      return 1
    fi
  done
  if [[ $(hostname) != jbr-TRX50 ]]; then
    echo "PREFLIGHT_FAIL this launcher is pinned to rb2/jbr-TRX50" >&2
    return 1
  fi
  "${PYTHON}" - <<'PY'
import json
import torch
from architectures.simvla.adapters.vla_cache.recipe import scientific_contract
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
free, total = torch.cuda.mem_get_info(0)
if free < 24 * 1024**3:
    raise RuntimeError(f"at least 24 GiB free GPU memory required, found {free / 1024**3:.2f} GiB")
print(json.dumps({
    "verdict": "SIMVLA_VLA_CACHE_PREFLIGHT_PASS",
    "gpu": torch.cuda.get_device_name(0),
    "free_memory_gib": free / 1024**3,
    "contract": scientific_contract(),
}, indent=2, sort_keys=True))
PY
  CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m pytest -q "${ROOT}/tests/simvla_vla_cache"
}

preflight
if [[ "${MODE}" == "--preflight" ]]; then
  echo "SIMVLA_VLA_CACHE_PREFLIGHT_COMPLETE"
  exit 0
fi
if [[ "${MODE}" != "--all" ]]; then
  echo "Usage: $0 [--preflight|--all]" >&2
  exit 2
fi
if [[ ${SIMVLA_VLA_CACHE_LIBERO_RUN:-0} != 1 ]]; then
  echo "Set SIMVLA_VLA_CACHE_LIBERO_RUN=1 after reviewing this launcher." >&2
  exit 2
fi

run_stage checkpoint_smoke \
  "${PYTHON}" -m architectures.simvla.adapters.vla_cache.smoke \
    --output "${RESULT_ROOT}/preflight/real_checkpoint_smoke.json" \
    --checkpoint YuankaiLuo/SimVLA-LIBERO \
    --checkpoint-revision "${CHECKPOINT_REVISION}" \
    --norm-stats "${NORM_STATS}" \
    --device cuda

run_stage libero_smoke env PYTHONHASHSEED=20260815 \
  "${PYTHON}" -m architectures.simvla.adapters.vla_cache.eval \
    --output "${RESULT_ROOT}/preflight/libero_smoke" \
    --episode-manifest "${MANIFEST_ROOT}/seed01/episode_manifest.json" \
    --row vla_cache --norm-stats "${NORM_STATS}" --max-episodes 1 --device cuda

run_stage matched_full_control env PYTHONHASHSEED=20260815 \
  "${PYTHON}" -m architectures.simvla.adapters.vla_cache.eval \
    --output "${RESULT_ROOT}/matched_full_control/seed01" \
    --episode-manifest "${MANIFEST_ROOT}/seed01/episode_manifest.json" \
    --row vla_cache_full --norm-stats "${NORM_STATS}" --device cuda

for seed in 01 02 03; do
  case "${seed}" in
    01) python_seed=20260815 ;;
    02) python_seed=20260816 ;;
    03) python_seed=20260817 ;;
  esac
  extra=()
  if [[ "${seed}" == 01 ]]; then
    extra+=(--save-failure-videos --video-stride 2)
  fi
  run_stage "vla_cache_seed${seed}" env PYTHONHASHSEED="${python_seed}" \
    "${PYTHON}" -m architectures.simvla.adapters.vla_cache.eval \
      --output "${RESULT_ROOT}/vla_cache/seed${seed}" \
      --episode-manifest "${MANIFEST_ROOT}/seed${seed}/episode_manifest.json" \
      --row vla_cache --norm-stats "${NORM_STATS}" --device cuda "${extra[@]}"
done

run_stage summary \
  "${PYTHON}" -m architectures.simvla.adapters.vla_cache.summarize \
    --eval-root "${RESULT_ROOT}" \
    --reference-root "${REFERENCE_ROOT}" \
    --output "${RESULT_ROOT}/summary"

echo "SIMVLA_VLA_CACHE_LIBERO_PIPELINE_COMPLETE" | tee "${STATUS}"
echo "Summary: ${RESULT_ROOT}/summary/comparison_summary.md"
