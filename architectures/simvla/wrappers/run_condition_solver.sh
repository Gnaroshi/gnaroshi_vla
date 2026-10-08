#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
nfe=1
args=()
while (($#)); do
  case "$1" in
    --student-steps)
      if (($# < 2)); then printf 'Missing --student-steps value\n'; exit 0; fi
      nfe=$2; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
case "$nfe" in 1|2) ;; *) printf 'Supported --student-steps: 1 or 2\n'; exit 0 ;; esac
case "$(hostname)" in
  jbrserver1)
    PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
    MODULE=tools.simvla.condition_solver_pipeline
    OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_output_split/solver_matched_nfe${nfe}_seed01_v1
    ;;
  jbr-TRX50)
    PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
    MODULE=tools.simvla.condition_solver_rb2
    OUT=/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/condition_output_split/solver_matched_nfe${nfe}_compiled_seed01_v1
    ;;
  *) printf 'Unsupported host\n'; exit 0 ;;
esac
mkdir -p "$OUT/logs" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -u -m "$MODULE" --student-steps "$nfe" "${args[@]}" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
printf 'CONDITION_SOLVER_EXIT=%s status=%s\n' "$rc" "$OUT/logs/launcher.status"
exit 0
