#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_PRIMARY_RUN:-0}" != "1" ]]; then
  echo "Refusing primary training: set SIMVLA_LATENTLOOP_PRIMARY_RUN=1 explicitly." >&2
  exit 2
fi

ROOT=/home/mingyujung/private/gnaroshi_vla
ENV_NAME=simvla_libero
LL_TAG=20260804_chunkaware_v3
LL_CHECKPOINT=YuankaiLuo/SimVLA-LIBERO
LL_NORM="$ROOT/architectures/simvla/upstream/norm_stats/libero_norm.json"
LL_CACHE_ROOT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/${LL_TAG}/cache
LL_RUN_ROOT="$ROOT/results/simvla/latentloop/${LL_TAG}"
TRAIN_WRAPPER="$ROOT/architectures/simvla/wrappers/simvla_latentloop_train.sh"
LOG_ROOT="$LL_RUN_ROOT/train_console_logs"

LL_R1_W_COND=1.0491880947
LL_R1_W_CHUNK=1.0180370267
LL_R1_W_EXEC=1.0718759559
LL_R1_W_REG=29456.7315621

LL_R5_W_COND=0.0927082299
LL_R5_W_CHUNK=0.2101435974
LL_R5_W_EXEC=0.2154304348
LL_R5_W_REG=1086.0316418

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
export WANDB_MODE="${SIMVLA_WANDB_MODE:-online}"

test -f "$LL_NORM"
test -x "$TRAIN_WRAPPER"
test -f "$LL_CACHE_ROOT/query_v3_r1_full_10x20/manifest.json"
test -f "$LL_CACHE_ROOT/query_v3_r5_full_10x20/manifest.json"

if [[ "$WANDB_MODE" == "online" ]]; then
  if ! wandb login --verify; then
    echo "W&B online authentication failed. Run 'wandb login --verify' and retry." >&2
    exit 2
  fi
fi

is_complete() {
  local output=$1
  local execution_horizon=$2
  [[ -f "$output/run_summary.json" ]] || return 1
  python - "$output" "$execution_horizon" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_r = int(sys.argv[2])
summary = json.loads((output / "run_summary.json").read_text())
checkpoint = Path(summary.get("final_checkpoint") or "")
passed = (
    summary.get("mode") == "training"
    and summary.get("stage") == "t1"
    and summary.get("variant") == "chunk_aware_latentloop"
    and summary.get("execution_horizon") == expected_r
    and summary.get("training_seed") == 20260804
    and summary.get("steps") == 150000
    and summary.get("completed") is True
    and summary.get("interrupted") is False
    and summary.get("teacher_trainable_parameters") == 0
    and summary.get("adapter_trainable_parameters") == 343026
    and checkpoint.is_file()
)
raise SystemExit(0 if passed else 1)
PY
}

run_one() {
  local label=$1
  local execution_horizon=$2
  local physical_gpu=$3
  local cache=$4
  local output=$5
  local condition_weight=$6
  local action_chunk_weight=$7
  local executed_prefix_weight=$8
  local update_regularization_weight=$9
  local resume_args=()

  if is_complete "$output" "$execution_horizon"; then
    echo "[$label] completed run already exists; skipping $output"
    return 0
  fi
  if [[ -e "$output" ]]; then
    if [[ ! -f "$output/latest_checkpoint.txt" ]]; then
      echo "[$label] incomplete output has no resumable checkpoint: $output" >&2
      return 2
    fi
    local resume_checkpoint
    resume_checkpoint=$(<"$output/latest_checkpoint.txt")
    if [[ ! -f "$resume_checkpoint" ]]; then
      echo "[$label] latest checkpoint is missing: $resume_checkpoint" >&2
      return 2
    fi
    resume_args=(--resume-from "$resume_checkpoint")
    echo "[$label] resuming from $resume_checkpoint"
  else
    echo "[$label] starting a new deterministic run"
  fi

  exec env \
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
    SIMVLA_LATENTLOOP_TRAIN_RUN=1 \
    bash "$TRAIN_WRAPPER" \
      --cache "$cache" \
      --output "$output" \
      --variant chunk_aware_latentloop \
      --stage t1 \
      --execution-horizon "$execution_horizon" \
      --checkpoint "$LL_CHECKPOINT" \
      --norm-stats "$LL_NORM" \
      --flow-steps 10 \
      --batch-size 1 \
      --num-workers 2 \
      --heldout-fraction 0.2 \
      --split-seed 20260804 \
      --seed 20260804 \
      --max-steps 150000 \
      --condition-weight "$condition_weight" \
      --action-chunk-weight "$action_chunk_weight" \
      --executed-prefix-weight "$executed_prefix_weight" \
      --update-regularization-weight "$update_regularization_weight" \
      --learning-rate 1e-4 \
      --weight-decay 0 \
      --save-interval 10000 \
      --log-interval 1000 \
      --wandb-project gnaroshi-simvla-latentloop \
      --wandb-name "${LL_TAG}-t1-r${execution_horizon}-chunk-aware" \
      --wandb-log-interval 1000 \
      --device cuda \
      "${resume_args[@]}"
}

R1_OUTPUT="$LL_RUN_ROOT/train/t1_r1/chunk_aware_latentloop"
R5_OUTPUT="$LL_RUN_ROOT/train/t1_r5/chunk_aware_latentloop"

run_one \
  R1 \
  1 \
  4 \
  "$LL_CACHE_ROOT/query_v3_r1_full_10x20" \
  "$R1_OUTPUT" \
  "$LL_R1_W_COND" \
  "$LL_R1_W_CHUNK" \
  "$LL_R1_W_EXEC" \
  "$LL_R1_W_REG" \
  > >(sed -u 's/^/[R1] /' | tee "$LOG_ROOT/t1_r1.log") 2>&1 &
R1_PID=$!

run_one \
  R5 \
  5 \
  5 \
  "$LL_CACHE_ROOT/query_v3_r5_full_10x20" \
  "$R5_OUTPUT" \
  "$LL_R5_W_COND" \
  "$LL_R5_W_CHUNK" \
  "$LL_R5_W_EXEC" \
  "$LL_R5_W_REG" \
  > >(sed -u 's/^/[R5] /' | tee "$LOG_ROOT/t1_r5.log") 2>&1 &
R5_PID=$!

stop_children() {
  trap - INT TERM
  echo "[primary] forwarding interrupt to R1/R5 trainers; wait for checkpoint writes." >&2
  kill -INT "$R1_PID" "$R5_PID" 2>/dev/null || true
  wait "$R1_PID" 2>/dev/null || true
  wait "$R5_PID" 2>/dev/null || true
  exit 130
}
trap stop_children INT TERM

set +e
FINISHED_PID=
wait -n -p FINISHED_PID "$R1_PID" "$R5_PID"
FIRST_STATUS=$?
set -e

if [[ "$FINISHED_PID" == "$R1_PID" ]]; then
  R1_STATUS=$FIRST_STATUS
  REMAINING_PID=$R5_PID
  REMAINING_LABEL=R5
else
  R5_STATUS=$FIRST_STATUS
  REMAINING_PID=$R1_PID
  REMAINING_LABEL=R1
fi

if [[ "$FIRST_STATUS" -ne 0 ]]; then
  echo "[primary] first completed trainer failed; stopping $REMAINING_LABEL safely." >&2
  kill -INT "$REMAINING_PID" 2>/dev/null || true
fi

set +e
wait "$REMAINING_PID"
REMAINING_STATUS=$?
set -e

if [[ "$REMAINING_LABEL" == "R1" ]]; then
  R1_STATUS=$REMAINING_STATUS
else
  R5_STATUS=$REMAINING_STATUS
fi
trap - INT TERM

if [[ "$R1_STATUS" -ne 0 || "$R5_STATUS" -ne 0 ]]; then
  echo "[primary] training did not complete: R1=$R1_STATUS R5=$R5_STATUS" >&2
  echo "[primary] rerun this same command to resume from latest_checkpoint.txt." >&2
  exit 1
fi

is_complete "$R1_OUTPUT" 1
is_complete "$R5_OUTPUT" 5
echo "[pass] deterministic R1/R5 T1 primary training completed."
