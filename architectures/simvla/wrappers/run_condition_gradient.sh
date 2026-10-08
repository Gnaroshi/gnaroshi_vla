#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
case "$(hostname)" in
  jbrserver1)
    PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
    MODULE=tools.simvla.condition_gradient_pipeline
    OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_output_split/joint_action_gradient_seed01_v1
    ;;
  jbr-TRX50)
    PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
    MODULE=tools.simvla.condition_gradient_rb2
    OUT=/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/condition_output_split/joint_action_gradient_compiled_seed01_v1
    ;;
  *) printf 'Unsupported host\n'; exit 0 ;;
esac
mkdir -p "$OUT/logs" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -u -m "$MODULE" "$@" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
printf 'CONDITION_GRADIENT_EXIT=%s status=%s\n' "$rc" "$OUT/logs/launcher.status"
exit 0
