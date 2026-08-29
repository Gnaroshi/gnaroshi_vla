#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_OFFLINE_ALL_RUN:-0}" != "1" ]]; then
  echo "Refusing offline evaluation: set SIMVLA_LATENTLOOP_OFFLINE_ALL_RUN=1 explicitly." >&2
  exit 2
fi

ROOT=/home/mingyujung/private/gnaroshi_vla
ENV_NAME=simvla_libero
LL_TAG=20260804_chunkaware_v3
LL_CHECKPOINT=YuankaiLuo/SimVLA-LIBERO
LL_NORM="$ROOT/architectures/simvla/upstream/norm_stats/libero_norm.json"
LL_CACHE_ROOT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/${LL_TAG}/cache
LL_RUN_ROOT="$ROOT/results/simvla/latentloop/${LL_TAG}"
EVAL_WRAPPER="$ROOT/architectures/simvla/wrappers/simvla_latentloop_eval.sh"
LOG_ROOT="$LL_RUN_ROOT/offline_console_logs"

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
test -f "$LL_CACHE_ROOT/query_v3_r1_full_10x20/manifest.json"
test -f "$LL_CACHE_ROOT/query_v3_r5_full_10x20/manifest.json"

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
  local expected_records=$3
  [[ -f "$output/offline_metrics.json" ]] || return 1
  [[ -f "$output/offline_gate.json" ]] || return 1
  [[ -f "$output/offline_episode_query_metrics.csv" ]] || return 1
  python - "$output" "$execution_horizon" "$expected_records" <<'PY'
import json
import math
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_r = int(sys.argv[2])
expected_records = int(sys.argv[3])
metrics = json.loads((output / "offline_metrics.json").read_text())
expected_rows = {
    "hold_condition",
    "chunk_aware_latentloop",
    "old_observation_only",
    "no_observation",
    "nonrecurrent_condition",
    "action_chunk_correction",
}
efficiency = metrics.get("efficiency", {})
passed = (
    metrics.get("execution_horizon") == expected_r
    and metrics.get("heldout_records") == expected_records
    and metrics.get("teacher_same_noise_reload_max_abs_diff") == 0.0
    and set(metrics.get("rows", {})) == expected_rows
    and math.isfinite(float(efficiency.get("elapsed_seconds", float("nan"))))
    and math.isfinite(float(efficiency.get("records_per_second", float("nan"))))
    and (output / "eval_progress.jsonl").is_file()
)
raise SystemExit(0 if passed else 1)
PY
}

R1_CHUNK=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/chunk_aware_latentloop/latest_checkpoint.txt")
R1_OLD=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/old_observation_only/latest_checkpoint.txt")
R1_NOOBS=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/no_observation/latest_checkpoint.txt")
R1_NONREC=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/nonrecurrent_condition/latest_checkpoint.txt")
R1_ACTION=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/action_chunk_correction/latest_checkpoint.txt")
R5_CHUNK=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r5/chunk_aware_latentloop/latest_checkpoint.txt")
R5_OLD=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r5/old_observation_only/latest_checkpoint.txt")
R5_NOOBS=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r5/no_observation/latest_checkpoint.txt")
R5_NONREC=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r5/nonrecurrent_condition/latest_checkpoint.txt")
R5_ACTION=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r5/action_chunk_correction/latest_checkpoint.txt")

run_one() {
  local execution_horizon=$1
  local physical_gpu=$2
  local expected_records=$3
  local cache=$4
  local output=$5
  local chunk=$6
  local old=$7
  local noobs=$8
  local nonrec=$9
  local action=${10}
  local label="R${execution_horizon}"

  if is_complete "$output" "$execution_horizon" "$expected_records"; then
    echo "[$label] completed offline evaluation already exists; skipping $output"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "[$label] incomplete offline output exists and cannot be resumed: $output" >&2
    return 2
  fi

  echo "[$label] starting held-out all-row evaluation on physical GPU $physical_gpu"
  exec env \
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
    SIMVLA_LATENTLOOP_EVAL_RUN=1 \
    bash "$EVAL_WRAPPER" offline \
      --cache "$cache" \
      --output "$output" \
      --adapter "chunk_aware_latentloop=$chunk" \
      --adapter "old_observation_only=$old" \
      --adapter "no_observation=$noobs" \
      --adapter "nonrecurrent_condition=$nonrec" \
      --adapter "action_chunk_correction=$action" \
      --execution-horizon "$execution_horizon" \
      --checkpoint "$LL_CHECKPOINT" \
      --norm-stats "$LL_NORM" \
      --flow-steps 10 \
      --heldout-fraction 0.2 \
      --split-seed 20260804 \
      --batch-size 1 \
      --num-workers 2 \
      --progress-interval 100 \
      --tqdm-mininterval 1.0 \
      --device cuda
}

R1_OUTPUT="$LL_RUN_ROOT/offline/t1_r1_all_rows"
R5_OUTPUT="$LL_RUN_ROOT/offline/t1_r5_all_rows"

run_one \
  1 4 19533 \
  "$LL_CACHE_ROOT/query_v3_r1_full_10x20" \
  "$R1_OUTPUT" \
  "$R1_CHUNK" "$R1_OLD" "$R1_NOOBS" "$R1_NONREC" "$R1_ACTION" \
  > >(sed -u 's/^/[R1 offline] /' | tee "$LOG_ROOT/t1_r1_all_rows.log") 2>&1 &
R1_PID=$!

run_one \
  5 5 2550 \
  "$LL_CACHE_ROOT/query_v3_r5_full_10x20" \
  "$R5_OUTPUT" \
  "$R5_CHUNK" "$R5_OLD" "$R5_NOOBS" "$R5_NONREC" "$R5_ACTION" \
  > >(sed -u 's/^/[R5 offline] /' | tee "$LOG_ROOT/t1_r5_all_rows.log") 2>&1 &
R5_PID=$!

handle_interrupt() {
  trap - INT TERM
  echo "[offline] stopping R1/R5 evaluation." >&2
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
  echo "[offline] evaluation failed: R1=$R1_STATUS R5=$R5_STATUS" >&2
  exit 1
fi

is_complete "$R1_OUTPUT" 1 19533
is_complete "$R5_OUTPUT" 5 2550
echo "[pass] R1/R5 held-out all-row offline evaluation completed."
