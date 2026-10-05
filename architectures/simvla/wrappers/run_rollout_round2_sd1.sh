#!/usr/bin/env bash
set +e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 0
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 USE_TF=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"
MODULE=tools.simvla.rollout_round2_pipeline
if [ "${1:-}" = "--head-matched" ]; then
  MODULE=tools.simvla.head_matched_pipeline
  shift
fi
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -u -m "$MODULE" "$@"
rc=$?
printf '\nPIPELINE_EXIT=%s; module=%s; tmux pane remains open\n' "$rc" "$MODULE"
exit 0
