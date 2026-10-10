#!/usr/bin/env bash
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_output_split/action_target_alignment_nfe1_seed01_v1
if [[ $(hostname) != jbrserver1 ]]; then
  printf 'This launcher requires sd1, GPU4..7.\n'
  exit 0
fi
cd -- "$ROOT" || exit 0
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="" USE_TF=0 PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -u -m tools.simvla.action_target_alignment "$@"
rc=$?
printf '%s\n' "$rc" > "$OUT/launcher.status"
printf 'ACTION_TARGET_ALIGNMENT_EXIT rc=%s status=%s/launcher.status\n' "$rc" "$OUT"
exit 0
