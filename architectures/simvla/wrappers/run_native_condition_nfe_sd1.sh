#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${ROOT}" || exit 1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export USE_TF=0 TOKENIZERS_PARALLELISM=false
MODE=${1:---all}
case "${MODE}" in
  --all) COMMAND=all ;;
  --preflight) COMMAND=preflight ;;
  *) printf 'Usage: bash %s [--all|--preflight]\n' "$0"; exit 2 ;;
esac
/home/mingyujung/miniconda3/envs/simvla_libero/bin/python -u -m tools.simvla.native_condition_nfe "${COMMAND}"
RC=$?
printf 'NATIVE_CONDITION_NFE_EXIT rc=%s\n' "${RC}"
exit "${RC}"
