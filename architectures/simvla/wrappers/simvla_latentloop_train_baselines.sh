#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_BASELINES_RUN:-0}" != "1" ]]; then
  echo "Refusing baseline training: set SIMVLA_LATENTLOOP_BASELINES_RUN=1 explicitly." >&2
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
LOG_ROOT="$LL_RUN_ROOT/baseline_console_logs"

VARIANTS=(
  old_observation_only
  no_observation
  nonrecurrent_condition
  action_chunk_correction
)
PHYSICAL_GPUS=(4 5 6 7)
EXPECTED_PARAMETERS=(308915 223954 343026 342985)

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
  local variant=$3
  local expected_parameters=$4
  [[ -f "$output/run_summary.json" ]] || return 1
  python - "$output" "$execution_horizon" "$variant" "$expected_parameters" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_r = int(sys.argv[2])
expected_variant = sys.argv[3]
expected_parameters = int(sys.argv[4])
summary = json.loads((output / "run_summary.json").read_text())
checkpoint = Path(summary.get("final_checkpoint") or "")
passed = (
    summary.get("mode") == "training"
    and summary.get("stage") == "t1"
    and summary.get("variant") == expected_variant
    and summary.get("execution_horizon") == expected_r
    and summary.get("training_seed") == 20260804
    and summary.get("steps") == 150000
    and summary.get("completed") is True
    and summary.get("interrupted") is False
    and summary.get("teacher_trainable_parameters") == 0
    and summary.get("adapter_trainable_parameters") == expected_parameters
    and checkpoint.is_file()
)
raise SystemExit(0 if passed else 1)
PY
}

run_one() {
  local execution_horizon=$1
  local physical_gpu=$2
  local variant=$3
  local expected_parameters=$4
  local cache=$5
  local output=$6
  local condition_weight=$7
  local action_chunk_weight=$8
  local executed_prefix_weight=$9
  local update_regularization_weight=${10}
  local label="R${execution_horizon}/${variant}"
  local resume_args=()

  if is_complete "$output" "$execution_horizon" "$variant" "$expected_parameters"; then
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
    echo "[$label] starting a new deterministic run on physical GPU $physical_gpu"
  fi

  exec env \
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
    SIMVLA_LATENTLOOP_TRAIN_RUN=1 \
    bash "$TRAIN_WRAPPER" \
      --cache "$cache" \
      --output "$output" \
      --variant "$variant" \
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
      --wandb-name "${LL_TAG}-t1-r${execution_horizon}-${variant}" \
      --wandb-log-interval 1000 \
      --device cuda \
      "${resume_args[@]}"
}

ACTIVE_PIDS=()

stop_active() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
}

handle_interrupt() {
  trap - INT TERM
  echo "[baselines] forwarding interrupt to active trainers; wait for checkpoint writes." >&2
  stop_active
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit 130
}
trap handle_interrupt INT TERM

run_wave() {
  local execution_horizon=$1
  local cache
  local condition_weight
  local action_chunk_weight
  local executed_prefix_weight
  local update_regularization_weight

  if [[ "$execution_horizon" == "1" ]]; then
    cache="$LL_CACHE_ROOT/query_v3_r1_full_10x20"
    condition_weight=$LL_R1_W_COND
    action_chunk_weight=$LL_R1_W_CHUNK
    executed_prefix_weight=$LL_R1_W_EXEC
    update_regularization_weight=$LL_R1_W_REG
  else
    cache="$LL_CACHE_ROOT/query_v3_r5_full_10x20"
    condition_weight=$LL_R5_W_COND
    action_chunk_weight=$LL_R5_W_CHUNK
    executed_prefix_weight=$LL_R5_W_EXEC
    update_regularization_weight=$LL_R5_W_REG
  fi

  ACTIVE_PIDS=()
  declare -A labels=()
  local index
  for index in "${!VARIANTS[@]}"; do
    local variant=${VARIANTS[$index]}
    local gpu=${PHYSICAL_GPUS[$index]}
    local expected_parameters=${EXPECTED_PARAMETERS[$index]}
    local output="$LL_RUN_ROOT/train/t1_r${execution_horizon}/${variant}"
    local log="$LOG_ROOT/t1_r${execution_horizon}_${variant}.log"
    run_one \
      "$execution_horizon" \
      "$gpu" \
      "$variant" \
      "$expected_parameters" \
      "$cache" \
      "$output" \
      "$condition_weight" \
      "$action_chunk_weight" \
      "$executed_prefix_weight" \
      "$update_regularization_weight" \
      > >(sed -u "s/^/[R${execution_horizon} ${variant}] /" | tee "$log") 2>&1 &
    local pid=$!
    ACTIVE_PIDS+=("$pid")
    labels[$pid]="R${execution_horizon}/${variant}"
  done

  local remaining=("${ACTIVE_PIDS[@]}")
  local failed=0
  while ((${#remaining[@]} > 0)); do
    local finished_pid=
    set +e
    wait -n -p finished_pid "${remaining[@]}"
    local status=$?
    set -e
    echo "[baselines] ${labels[$finished_pid]} exited with status $status"

    local next=()
    local pid
    for pid in "${remaining[@]}"; do
      if [[ "$pid" != "$finished_pid" ]]; then
        next+=("$pid")
      fi
    done
    remaining=("${next[@]}")

    if [[ "$status" -ne 0 && "$failed" -eq 0 ]]; then
      failed=1
      echo "[baselines] stopping the remaining R${execution_horizon} trainers safely." >&2
      for pid in "${remaining[@]}"; do
        kill -INT "$pid" 2>/dev/null || true
      done
    fi
  done
  ACTIVE_PIDS=()
  if [[ "$failed" -ne 0 ]]; then
    echo "[baselines] R${execution_horizon} wave failed; rerun this command to resume." >&2
    return 1
  fi

  for index in "${!VARIANTS[@]}"; do
    local variant=${VARIANTS[$index]}
    is_complete \
      "$LL_RUN_ROOT/train/t1_r${execution_horizon}/${variant}" \
      "$execution_horizon" \
      "$variant" \
      "${EXPECTED_PARAMETERS[$index]}"
  done
  echo "[pass] all R${execution_horizon} comparison-baseline runs completed."
}

run_wave 1
run_wave 5
trap - INT TERM
echo "[pass] all eight deterministic T1 baseline runs completed."
