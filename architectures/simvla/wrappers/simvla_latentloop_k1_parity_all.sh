#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_K1_PARITY_ALL_RUN:-0}" != "1" ]]; then
  echo "Refusing K1 parity: set SIMVLA_LATENTLOOP_K1_PARITY_ALL_RUN=1 explicitly." >&2
  exit 2
fi

ROOT=/home/mingyujung/private/gnaroshi_vla
ENV_NAME=simvla_libero
LL_TAG=20260804_chunkaware_v3
LL_CHECKPOINT=YuankaiLuo/SimVLA-LIBERO
LL_NORM="$ROOT/architectures/simvla/upstream/norm_stats/libero_norm.json"
LL_RUN_ROOT="$ROOT/results/simvla/latentloop/${LL_TAG}"
EVAL_WRAPPER="$ROOT/architectures/simvla/wrappers/simvla_latentloop_eval.sh"
LOG_ROOT="$LL_RUN_ROOT/online_console_logs"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is unavailable; activate $ENV_NAME first." >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

cd "$ROOT"
mkdir -p "$LOG_ROOT"

export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

test -f "$LL_NORM"
test -x "$EVAL_WRAPPER"

checkpoint_from_pointer() {
  local pointer=$1
  [[ -f "$pointer" ]] || {
    echo "missing checkpoint pointer: $pointer" >&2
    return 2
  }
  local checkpoint
  checkpoint=$(<"$pointer")
  [[ -f "$checkpoint" ]] || {
    echo "checkpoint referenced by $pointer is missing: $checkpoint" >&2
    return 2
  }
  printf '%s\n' "$checkpoint"
}

is_complete() {
  local output=$1
  local execution_horizon=$2
  [[ -f "$output/online_summary.json" ]] || return 1
  [[ -f "$output/episode_metrics.csv" ]] || return 1
  [[ -f "$output/query_trace.jsonl" ]] || return 1
  [[ -f "$output/eval_progress.jsonl" ]] || return 1
  [[ -f "$output/eval_config.json" ]] || return 1
  [[ -f "$output/source_lock.json" ]] || return 1
  python - "$output" "$execution_horizon" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_r = int(sys.argv[2])
summary = json.loads((output / "online_summary.json").read_text())
config = json.loads((output / "eval_config.json").read_text())
parity = summary.get("k1_parity") or {}
passed = (
    summary.get("matrix") == "k1_parity"
    and summary.get("episodes_per_row") == 100
    and set(summary.get("rows", {})) == {"full_k1", "adapter_loaded_full_k1"}
    and config.get("execution_horizon") == expected_r
    and parity.get("K1_PARITY_PASS") is True
    and parity.get("exact_action_chunk_equality") is True
    and parity.get("identical_paired_outcomes") is True
    and parity.get("updater_calls") == 0
    and parity.get("observation_encoder_calls") == 0
    and parity.get("action_encoder_calls") == 0
)
raise SystemExit(0 if passed else 1)
PY
}

run_one() {
  local execution_horizon=$1
  local physical_gpu=$2
  local output=$3
  local checkpoint=$4
  local label="R${execution_horizon}"

  if is_complete "$output" "$execution_horizon"; then
    echo "[$label] completed K1 parity already exists; skipping $output"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "[$label] incomplete K1 parity output exists and cannot be resumed: $output" >&2
    return 2
  fi

  echo "[$label] starting K1 parity on physical GPU $physical_gpu"
  exec env \
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
    SIMVLA_LATENTLOOP_EVAL_RUN=1 \
    bash "$EVAL_WRAPPER" online \
      --matrix k1_parity \
      --output "$output" \
      --checkpoint "$LL_CHECKPOINT" \
      --norm-stats "$LL_NORM" \
      --chunk-aware-checkpoint "$checkpoint" \
      --suite libero_10 \
      --execution-horizon "$execution_horizon" \
      --num-trials 10 \
      --max-tasks 10 \
      --max-env-actions 900 \
      --num-wait-steps 10 \
      --flow-steps 10 \
      --client-resize-size 224 \
      --image-size 384 \
      --resolution 256 \
      --seed 7 \
      --action-noise-seed-base 20260804 \
      --bootstrap-seed 20260804 \
      --task-order official_reverse \
      --tqdm-mininterval 1.0 \
      --device cuda
}

R1_CHECKPOINT=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/chunk_aware_latentloop/latest_checkpoint.txt")
R5_CHECKPOINT=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r5/chunk_aware_latentloop/latest_checkpoint.txt")
R1_OUTPUT="$LL_RUN_ROOT/online/k1_parity_r1_10x10"
R5_OUTPUT="$LL_RUN_ROOT/online/k1_parity_r5_10x10"

run_one 1 4 "$R1_OUTPUT" "$R1_CHECKPOINT" \
  > >(sed -u 's/^/[R1 K1] /' | tee "$LOG_ROOT/k1_parity_r1_10x10.log") 2>&1 &
R1_PID=$!

run_one 5 5 "$R5_OUTPUT" "$R5_CHECKPOINT" \
  > >(sed -u 's/^/[R5 K1] /' | tee "$LOG_ROOT/k1_parity_r5_10x10.log") 2>&1 &
R5_PID=$!

handle_interrupt() {
  trap - INT TERM
  echo "[K1 parity] stopping R1/R5 evaluation." >&2
  kill -INT "$R1_PID" "$R5_PID" 2>/dev/null || true
  wait "$R1_PID" 2>/dev/null || true
  wait "$R5_PID" 2>/dev/null || true
  exit 130
}
trap handle_interrupt INT TERM

set +e
wait "$R1_PID"
R1_STATUS=$?
wait "$R5_PID"
R5_STATUS=$?
set -e
trap - INT TERM

if [[ "$R1_STATUS" -ne 0 || "$R5_STATUS" -ne 0 ]]; then
  echo "[K1 parity] evaluation failed: R1=$R1_STATUS R5=$R5_STATUS" >&2
  exit 1
fi

is_complete "$R1_OUTPUT" 1
is_complete "$R5_OUTPUT" 5
echo "[pass] R1/R5 K1 parity completed. R5 K2 remains blocked by the failed R5 offline gate."
