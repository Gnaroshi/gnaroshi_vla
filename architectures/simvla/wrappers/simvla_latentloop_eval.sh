#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_EVAL_RUN:-0}" != "1" ]]; then
  echo "Refusing evaluation/aggregation: set SIMVLA_LATENTLOOP_EVAL_RUN=1 explicitly." >&2
  exit 2
fi

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 {offline|online|aggregate} [arguments...]" >&2
  exit 2
fi

mode="$1"
shift
case "$mode" in
  offline)
    exec python -m architectures.simvla.adapters.latentloop.offline_evaluator "$@"
    ;;
  online)
    exec python -m architectures.simvla.adapters.latentloop.online_evaluator "$@"
    ;;
  aggregate)
    exec python -m architectures.simvla.adapters.latentloop.result_aggregator "$@"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac
