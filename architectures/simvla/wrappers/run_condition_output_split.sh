#!/usr/bin/env bash
# Keep the caller's pane alive; the status file contains the actual exit code.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
if [[ $(hostname) == jbrserver1 ]]; then
    PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
    MODULE=tools.simvla.condition_output_split_pipeline
    OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_output_split/mixed_k4_k8_seed01_v1
elif [[ $(hostname) == jbr-TRX50 ]]; then
    PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
    MODULE=tools.simvla.condition_output_split_rb2
    OUT=/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/condition_output_split/mixed_k4_k8_compiled_seed01_v1
else
    printf 'Unsupported host: %s\n' "$(hostname)"
    exit 0
fi
mkdir -p "$OUT/logs" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -u -m "$MODULE" "$@" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
printf 'CONDITION_OUTPUT_SPLIT_EXIT=%s status=%s\n' "$rc" "$OUT/logs/launcher.status"
exit 0
