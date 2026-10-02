#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-online}"
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -u -m tools.simvla.observed_progress_campaign "$@"
rc=$?
printf '\nObserved-progress campaign exit code: %s\n' "$rc"
printf 'Status: shared results/simvla/trend_condition/observed_progress_k8_seed01_v1/pipeline_status.json\n'
exit 0
