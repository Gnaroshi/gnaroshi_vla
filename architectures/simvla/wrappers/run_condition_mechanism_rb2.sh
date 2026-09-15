#!/usr/bin/env bash
# Errors are recorded, but the interactive tmux shell is not terminated.
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python}
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20260815
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"${PYTHON}" "${ROOT}/tools/simvla/condition_mechanism_pipeline.py" "$@"
rc=$?
if ((rc != 0)); then
  printf '\n분석이 완료되지 않았습니다(rc=%s). 위 오류와 결과 폴더의 pipeline_status.json을 확인하세요.\n' "$rc"
  printf '같은 명령을 다시 실행하면 완료된 window/평가 분기를 재사용합니다.\n'
fi
if [[ ${SIMVLA_STRICT_EXIT:-0} == 1 ]]; then
  exit "$rc"
fi
exit 0
