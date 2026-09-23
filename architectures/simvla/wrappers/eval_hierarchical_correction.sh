#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 {source-lock|parity|smoke|offline|r5-offline|scientific|merge-scientific|aggregate|k8|r5} [arguments...]" >&2
  exit 2
fi

mode="$1"
shift
case "${mode}" in
  source-lock)
    if [[ "${SIMVLA_HIERARCHICAL_AUDIT_RUN:-0}" != "1" ]]; then
      echo "Refusing source-lock audit: set SIMVLA_HIERARCHICAL_AUDIT_RUN=1." >&2
      exit 2
    fi
    exec python tools/simvla/analyze_hierarchical_correction.py source-lock "$@"
    ;;
  parity)
    if [[ "${SIMVLA_HIERARCHICAL_REPLAY_RUN:-0}" != "1" ]]; then
      echo "Refusing endpoint parity: set SIMVLA_HIERARCHICAL_REPLAY_RUN=1." >&2
      exit 2
    fi
    exec python -m architectures.simvla.adapters.hierarchical_correction.offline_replay --mode parity "$@"
    ;;
  offline)
    if [[ "${SIMVLA_HIERARCHICAL_REPLAY_RUN:-0}" != "1" ]]; then
      echo "Refusing offline replay: set SIMVLA_HIERARCHICAL_REPLAY_RUN=1." >&2
      exit 2
    fi
    exec python -m architectures.simvla.adapters.hierarchical_correction.offline_replay --mode gate "$@"
    ;;
  r5-offline)
    if [[ "${SIMVLA_HIERARCHICAL_R5_OFFLINE_RUN:-0}" != "1" ]]; then
      echo "Refusing exact-age R5 gate: set SIMVLA_HIERARCHICAL_R5_OFFLINE_RUN=1." >&2
      exit 2
    fi
    exec python -m architectures.simvla.adapters.hierarchical_correction.r5_regeneration_offline "$@"
    ;;
  smoke)
    if [[ "${SIMVLA_HIERARCHICAL_EVAL_RUN:-0}" != "1" ]]; then
      echo "Refusing LIBERO smoke: set SIMVLA_HIERARCHICAL_EVAL_RUN=1." >&2
      exit 2
    fi
    exec python -m architectures.simvla.adapters.hierarchical_correction.online_evaluator --matrix smoke "$@"
    ;;
  scientific)
    if [[ "${SIMVLA_HIERARCHICAL_EVAL_RUN:-0}" != "1" ]]; then
      echo "Refusing LIBERO diagnostic: set SIMVLA_HIERARCHICAL_EVAL_RUN=1." >&2
      exit 2
    fi
    exec python -m architectures.simvla.adapters.hierarchical_correction.online_evaluator --matrix scientific_r1_k4 "$@"
    ;;
  merge-scientific)
    if [[ "${SIMVLA_HIERARCHICAL_AUDIT_RUN:-0}" != "1" ]]; then
      echo "Refusing shard merge: set SIMVLA_HIERARCHICAL_AUDIT_RUN=1." >&2
      exit 2
    fi
    exec python tools/simvla/analyze_hierarchical_correction.py merge-scientific "$@"
    ;;
  aggregate)
    if [[ "${SIMVLA_HIERARCHICAL_AUDIT_RUN:-0}" != "1" ]]; then
      echo "Refusing aggregation: set SIMVLA_HIERARCHICAL_AUDIT_RUN=1." >&2
      exit 2
    fi
    exec python tools/simvla/analyze_hierarchical_correction.py aggregate "$@"
    ;;
  k8)
    if [[ "${SIMVLA_HIERARCHICAL_EVAL_RUN:-0}" != "1" ]]; then
      echo "Refusing conditional K8: set SIMVLA_HIERARCHICAL_EVAL_RUN=1." >&2
      exit 2
    fi
    exec python -m architectures.simvla.adapters.hierarchical_correction.online_evaluator --matrix conditional_k8 "$@"
    ;;
  r5)
    if [[ "${SIMVLA_HIERARCHICAL_EVAL_RUN:-0}" != "1" ]]; then
      echo "Refusing R5: set SIMVLA_HIERARCHICAL_EVAL_RUN=1." >&2
      exit 2
    fi
    exec python -m architectures.simvla.adapters.hierarchical_correction.online_evaluator --matrix native_r5 "$@"
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac
