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

if ! has_flag "--checkpoint" "${ARGS[@]}"; then
  ARGS+=(--checkpoint "${SIMVLA_DCLD_TEACHER_CHECKPOINT:-YuankaiLuo/SimVLA-LIBERO}")
fi
if ! has_flag "--suite" "${ARGS[@]}"; then
  ARGS+=(--suite "${SIMVLA_DCLD_SUITE:-libero_10}")
fi
if ! has_flag "--norm-stats" "${ARGS[@]}"; then
  ARGS+=(--norm-stats "${SIMVLA_DCLD_NORM_STATS:-${ROOT}/architectures/simvla/upstream/norm_stats/libero_norm.json}")
fi
if ! has_flag "--output" "${ARGS[@]}"; then
  ARGS+=(--output "${SIMVLA_DCLD_CACHE_OUTPUT:-${ROOT}/results/simvla/dcld_cache/libero_10/simvla_libero_hf}")
fi
if ! has_flag "--device" "${ARGS[@]}"; then
  ARGS+=(--device "${SIMVLA_DCLD_DEVICE:-cuda}")
fi
if [[ -n "${SIMVLA_DCLD_REPORT_DIR:-}" ]] && ! has_flag "--report-dir" "${ARGS[@]}"; then
  ARGS+=(--report-dir "${SIMVLA_DCLD_REPORT_DIR}")
fi

IS_SMOKE=0
if has_flag "--max-samples" "${ARGS[@]}" || has_flag "--max-episodes" "${ARGS[@]}" || has_flag "--raw-rgb-smoke-only" "${ARGS[@]}"; then
  IS_SMOKE=1
fi

if [[ "${SIMVLA_DCLD_RUN:-0}" != "1" && "${IS_SMOKE}" != "1" ]]; then
  cat <<EOF
Refusing to launch full teacher-cache generation without SIMVLA_DCLD_RUN=1.

Smoke examples:
  bash architectures/simvla/wrappers/simvla_dcld_generate_teacher_cache.sh \\
    --raw-rgb-smoke-only --max-episodes 1 --max-samples 2 --report-dir <OUT>

  bash architectures/simvla/wrappers/simvla_dcld_generate_teacher_cache.sh \\
    --suite libero_10 --max-episodes 1 --max-samples 3 --output <CACHE> --report-dir <OUT>
EOF
  exit 0
fi

export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
exec python -m architectures.simvla.adapters.dcld.simvla_teacher_cache "${ARGS[@]}"
