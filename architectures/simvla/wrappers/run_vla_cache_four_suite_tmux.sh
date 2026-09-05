#!/usr/bin/env bash
# Keep failures inside a child shell so the caller's tmux pane stays open.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ ${1:-} != --inner ]]; then
  set +e
  bash "${BASH_SOURCE[0]}" --inner "${1:---all}"
  rc=$?
  if [[ ${rc} -ne 0 ]]; then
    printf 'VLA_CACHE_FOUR_SUITE_STOPPED rc=%s; inspect pipeline.status and logs.\n' "${rc}"
  fi
  exit 0
fi
set -Eeuo pipefail
cd "${ROOT}"
MODE=${2:---all}
case "${MODE}" in --all|--verify|--preflight) ;; *) echo 'Use --all, --verify, or --preflight'; exit 2 ;; esac
STORE=/home/mingyujung/private/gnaroshi_vla_storage
PYTHON=${PYTHON:-${STORE}/envs/simvla/libero_mujoco237/bin/python}
RESULT_ROOT=${SIMVLA_VLA_CACHE_RESULT_ROOT:-${STORE}/results/simvla/vla_cache/four_suite_oft_runtime_v3}
MANIFEST_ROOT=${SIMVLA_EPISODE_MANIFEST_ROOT:-${STORE}/results/simvla/paper_four_suite_three_seed_v1/manifests}
NONLONG_SEED1=${STORE}/results/simvla/paper_nonlong_seed01_primary_v1
NONLONG_SEED23=${STORE}/results/simvla/paper_completion/three_seed_5090_egl_v1/nonlong
NORM=${SIMVLA_NORM_STATS:-${ROOT}/architectures/simvla/adapters/latentloop/assets/libero_norm_official_32700d0.json}
export SIMVLA_UPSTREAM_ROOT=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
export LIBERO_ROOT=${LIBERO_ROOT:-${STORE}/datasets/LIBERO}
export PYTHONPATH="${ROOT}:${SIMVLA_UPSTREAM_ROOT}:${LIBERO_ROOT}"
export CUDA_VISIBLE_DEVICES=${SIMVLA_VLA_CACHE_GPU:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla/.cache/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false USE_TF=0
export PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}
mkdir -p "${RESULT_ROOT}/logs"
STATUS=${RESULT_ROOT}/pipeline.status
exec 9>"${RESULT_ROOT}/pipeline.lock"
flock -n 9 || { echo 'Another four-suite launcher is already running'; exit 2; }
trap 'rc=$?; echo "VLA_CACHE_FOUR_SUITE_FAILED rc=${rc} line=${LINENO}" | tee "${STATUS}"; exit "${rc}"' ERR
trap 'echo VLA_CACHE_FOUR_SUITE_INTERRUPTED | tee "${STATUS}"; exit 130' INT TERM

paper_args=(
  --output "${RESULT_ROOT}" --manifest-root "${MANIFEST_ROOT}"
  --long-reference "${STORE}/results/simvla/paper_followup/three_seed_long500_primary_v1/aggregate/paper_followup_three_seed_summary.json"
  --nonlong-seed1-reference "${NONLONG_SEED1}/summary/selected_matrix_summary.json"
  --nonlong-seed23-reference "${NONLONG_SEED23}/summary/selected_matrix_summary.json"
)

free_gpu() {
  local occupied
  occupied=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader)
  [[ -z "${occupied}" ]] || { echo "GPU occupied by ${occupied}; no process was stopped"; return 1; }
}

[[ $(hostname) == jbr-TRX50 ]] || { echo 'This launcher is for rb2 only'; exit 2; }
free_gpu
"${PYTHON}" - "${ROOT}" "${NORM}" <<'PY'
import sys
from pathlib import Path
import torch
from architectures.simvla.adapters.vla_cache import smolvlm_runtime
from architectures.simvla.adapters.vla_cache.eval import _configure_paths, validate_norm_stats
root = Path(sys.argv[1]).resolve()
assert Path(smolvlm_runtime.__file__).resolve().is_relative_to(root), smolvlm_runtime.__file__
assert hasattr(smolvlm_runtime.IndexedReuseDecoder, "prepare_query")
_configure_paths()
validate_norm_stats(Path(sys.argv[2]))
assert torch.cuda.is_available()
assert "5090" in torch.cuda.get_device_name(0)
free, _ = torch.cuda.mem_get_info(0)
assert free >= 24 * 1024**3, f"At least 24 GiB free required: {free / 1024**3:.2f} GiB"
print("RUNTIME_SOURCE_PASS", smolvlm_runtime.__file__)
PY
"${PYTHON}" -m tools.simvla.vla_cache_paper preflight "${paper_args[@]}" 2>&1 | tee "${RESULT_ROOT}/logs/preflight.log"
CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m pytest -q -p no:cacheprovider "${ROOT}/tests/simvla_vla_cache" 2>&1 | tee "${RESULT_ROOT}/logs/tests.log"
if [[ ${MODE} == --preflight ]]; then
  echo VLA_CACHE_FOUR_SUITE_PREFLIGHT_COMPLETE | tee "${STATUS}"
  exit 0
fi

smoke=${RESULT_ROOT}/preflight/real_checkpoint_smoke.json
if ! "${PYTHON}" - "${smoke}" <<'PY'
import json, sys
from pathlib import Path
from architectures.simvla.adapters.vla_cache.eval import implementation_identity
path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.is_file() else {}
raise SystemExit(0 if data.get("verdict") == "SIMVLA_VLA_CACHE_REAL_CHECKPOINT_SMOKE_PASS" and data.get("implementation_identity") == implementation_identity() else 1)
PY
then
  env PYTHONHASHSEED=20260815 "${PYTHON}" -m architectures.simvla.adapters.vla_cache.smoke \
    --output "${smoke}" --episode-manifest "${MANIFEST_ROOT}/libero_10/seed01/episode_manifest.json" \
    --norm-stats "${NORM}" --device cuda 2>&1 | tee "${RESULT_ROOT}/logs/checkpoint_smoke.log"
fi
if [[ ${MODE} == --verify ]]; then
  echo VLA_CACHE_FOUR_SUITE_VERIFICATION_COMPLETE | tee "${STATUS}"
  exit 0
fi

for suite in libero_10 libero_spatial libero_object libero_goal; do
  for seed in seed01 seed02 seed03; do
    if "${PYTHON}" -m tools.simvla.vla_cache_paper complete "${paper_args[@]}" --suite "${suite}" --seed "${seed}" >/dev/null 2>&1; then
      echo "SKIP ${suite} ${seed}: verified complete"
      continue
    fi
    free_gpu
    seed_number=$((20260814 + 10#${seed#seed}))
    echo "START ${suite} ${seed}: 500 episodes" | tee "${STATUS}"
    extra=()
    [[ ${seed} != seed01 ]] || extra+=(--save-failure-videos --video-stride 2)
    manifest_base=${MANIFEST_ROOT}
    if [[ ${suite} != libero_10 ]]; then
      if [[ ${seed} == seed01 ]]; then
        manifest_base=${NONLONG_SEED1}/manifests
      else
        manifest_base=${NONLONG_SEED23}/manifests
      fi
    fi
    env PYTHONHASHSEED="${seed_number}" "${PYTHON}" -m architectures.simvla.adapters.vla_cache.eval \
      --output "${RESULT_ROOT}/${suite}/vla_cache/${seed}" \
      --episode-manifest "${manifest_base}/${suite}/${seed}/episode_manifest.json" \
      --row vla_cache --norm-stats "${NORM}" --device cuda "${extra[@]}" \
      2>&1 | tee "${RESULT_ROOT}/logs/${suite}_${seed}.log"
    "${PYTHON}" -m tools.simvla.vla_cache_paper complete "${paper_args[@]}" --suite "${suite}" --seed "${seed}"
  done
done
"${PYTHON}" -m tools.simvla.vla_cache_paper summary "${paper_args[@]}" 2>&1 | tee "${RESULT_ROOT}/logs/summary.log"
echo VLA_CACHE_FOUR_SUITE_COMPLETE | tee "${STATUS}"
echo "Paper rows: ${RESULT_ROOT}/summary/main_table_rows.tex"
