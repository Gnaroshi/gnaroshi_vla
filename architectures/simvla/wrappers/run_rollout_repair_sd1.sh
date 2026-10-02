#!/usr/bin/env bash
set +e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 0
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 USE_TF=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -u -m tools.simvla.rollout_repair_pipeline "$@"
rc=$?
printf '\nROLLOUT_REPAIR_EXIT=%s; real status is in rollout_state_repair_seed01_v1/pipeline_status.json\n' "$rc"
exit 0
