#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${ROOT}" || exit 1
case "$(hostname)" in
  jbrserver1) HOST=sd1; PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python ;;
  jbr-TRX50) HOST=rb2; PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python ;;
  *) printf 'Unsupported host\n'; exit 2 ;;
esac
case "${1:---all}" in
  --all) COMMAND=all ;;
  --preflight) COMMAND=preflight ;;
  *) printf 'Usage: bash %s [--all|--preflight]\n' "$0"; exit 2 ;;
esac
export PYTHONPATH="${ROOT}" USE_TF=0 TOKENIZERS_PARALLELISM=false
"${PY}" -u -m tools.simvla.refresh_residual_controls "${COMMAND}" --host "${HOST}"
RC=$?
printf 'REFRESH_RESIDUAL_CONTROLS_EXIT host=%s rc=%s\n' "${HOST}" "${RC}"
exit "${RC}"
