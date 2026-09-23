#!/usr/bin/env bash
set -euo pipefail

ROOT="${GNAROSHI_VLA_ROOT:-/home/mingyujung/private/gnaroshi_vla}"
cd "${ROOT}"

has_flag() {
  local flag="$1"
  shift
  for arg in "$@"; do
    [[ "${arg}" == "${flag}" ]] && return 0
  done
  return 1
}

ARGS=("$@")

# Backward-compatible minimal smoke if no cache is supplied.
if [[ "${SIMVLA_DCLD_DEBUG:-0}" == "1" ]] && ! has_flag "--teacher-cache" "${ARGS[@]}" && [[ -z "${SIMVLA_DCLD_TEACHER_CACHE:-}" ]]; then
  OUT="${SIMVLA_DCLD_OUTPUT_DIR:-${ROOT}/codex_outputs/simvla_dcld_train_debug_$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "${OUT}"
  export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
  exec python architectures/simvla/wrappers/simvla_dcld_minimal_sanity.py \
    --output_dir "${OUT}" \
    --checkpoint_id "${SIMVLA_DCLD_TEACHER_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}" \
    --metas_path "${SIMVLA_DCLD_METAS_PATH:-${ROOT}/architectures/simvla/upstream/datasets/metas/libero_train.json}" \
    --norm_stats_path "${SIMVLA_DCLD_NORM_STATS:-${ROOT}/architectures/simvla/upstream/norm_stats/libero_norm.json}" \
    --steps "${SIMVLA_DCLD_ACTION_STEPS:-10}" \
    --image_size "${SIMVLA_DCLD_IMAGE_SIZE:-384}" \
    --device "${SIMVLA_DCLD_DEVICE:-cuda}" \
    --learning_rate "${SIMVLA_DCLD_LR:-1e-4}" \
    --max_batches "${SIMVLA_DCLD_MAX_BATCHES:-2}"
fi

if [[ -n "${SIMVLA_DCLD_TEACHER_CACHE:-}" ]] && ! has_flag "--teacher-cache" "${ARGS[@]}"; then
  ARGS+=(--teacher-cache "${SIMVLA_DCLD_TEACHER_CACHE}")
fi
if [[ -n "${SIMVLA_DCLD_OUTPUT_DIR:-}" ]] && ! has_flag "--output" "${ARGS[@]}"; then
  ARGS+=(--output "${SIMVLA_DCLD_OUTPUT_DIR}")
fi
if ! has_flag "--checkpoint" "${ARGS[@]}"; then
  ARGS+=(--checkpoint "${SIMVLA_DCLD_TEACHER_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}")
fi
if ! has_flag "--norm-stats" "${ARGS[@]}"; then
  ARGS+=(--norm-stats "${SIMVLA_DCLD_NORM_STATS:-${ROOT}/architectures/simvla/upstream/norm_stats/libero_norm.json}")
fi
if ! has_flag "--device" "${ARGS[@]}"; then
  ARGS+=(--device "${SIMVLA_DCLD_DEVICE:-cuda}")
fi
if [[ -n "${SIMVLA_DCLD_MAX_BATCHES:-}" ]] && ! has_flag "--max-batches" "${ARGS[@]}"; then
  ARGS+=(--max-batches "${SIMVLA_DCLD_MAX_BATCHES}")
fi
if [[ "${SIMVLA_DCLD_DEBUG:-0}" == "1" ]] && ! has_flag "--debug" "${ARGS[@]}"; then
  ARGS+=(--debug)
fi

if ! has_flag "--teacher-cache" "${ARGS[@]}"; then
  echo "ERROR: --teacher-cache is required for cache-backed DCLD training." >&2
  exit 2
fi
if ! has_flag "--output" "${ARGS[@]}"; then
  echo "ERROR: --output is required for cache-backed DCLD training." >&2
  exit 2
fi
if [[ "${SIMVLA_DCLD_RUN:-0}" != "1" && "${SIMVLA_DCLD_DEBUG:-0}" != "1" && ! " ${ARGS[*]} " =~ " --debug " ]]; then
  echo "Refusing to launch non-debug DCLD training without SIMVLA_DCLD_RUN=1." >&2
  exit 0
fi

export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
exec python -m architectures.simvla.adapters.dcld.simvla_dcld_distill_trainer "${ARGS[@]}"
