#!/usr/bin/env bash
set +e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 0
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 USE_TF=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -u -m tools.simvla.rollout_round2_pipeline "$@"
rc=$?
printf '\nROLLOUT_ROUND2_EXIT=%s; inspect rollout_round2_seed01_v1/pipeline_status.json\n' "$rc"
exit 0
