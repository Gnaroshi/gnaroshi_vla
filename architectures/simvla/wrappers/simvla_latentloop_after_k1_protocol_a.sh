#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_AFTER_K1_RUN:-0}" != "1" ]]; then
  echo "Refusing post-K1 pipeline: set SIMVLA_LATENTLOOP_AFTER_K1_RUN=1 explicitly." >&2
  exit 2
fi

ROOT=/home/mingyujung/private/gnaroshi_vla
ENV_NAME=simvla_libero
LL_TAG=20260804_chunkaware_v3
LL_CHECKPOINT=YuankaiLuo/SimVLA-LIBERO
LL_NORM="$ROOT/architectures/simvla/upstream/norm_stats/libero_norm.json"
LL_RUN_ROOT="$ROOT/results/simvla/latentloop/${LL_TAG}"
EVAL_WRAPPER="$ROOT/architectures/simvla/wrappers/simvla_latentloop_eval.sh"
WAIT_SECONDS=${SIMVLA_LATENTLOOP_WAIT_SECONDS:-60}

R1_K1_RAW="$LL_RUN_ROOT/online/k1_parity_r1_10x10"
R5_K1_RAW="$LL_RUN_ROOT/online/k1_parity_r5_10x10"
R1_K1_REVALIDATED="$LL_RUN_ROOT/online/k1_parity_r1_10x10_revalidated"
R5_K1_REVALIDATED="$LL_RUN_ROOT/online/k1_parity_r5_10x10_revalidated"
OFFLINE_R1="$LL_RUN_ROOT/offline/t1_r1_all_rows/offline_metrics.json"
OFFLINE_R5="$LL_RUN_ROOT/offline/t1_r5_all_rows/offline_metrics.json"
PROTOCOL_ROOT="$LL_RUN_ROOT/online/protocol_a_r1_screening_10x10_parallel"
SHARD_ROOT="$PROTOCOL_ROOT/shards"
MERGED_OUTPUT="$PROTOCOL_ROOT/merged"
LOG_ROOT="$LL_RUN_ROOT/online_console_logs/protocol_a_r1_parallel"

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
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

test -f "$LL_NORM"
test -x "$EVAL_WRAPPER"
test -f "$OFFLINE_R1"
test -f "$OFFLINE_R5"

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

raw_k1_complete() {
  local output=$1
  for name in online_summary.json episode_metrics.csv query_trace.jsonl eval_config.json source_lock.json; do
    [[ -f "$output/$name" ]] || return 1
  done
  python - "$output" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
summary = json.loads((output / "online_summary.json").read_text())
with (output / "episode_metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
passed = (
    summary.get("matrix") == "k1_parity"
    and summary.get("episodes_per_row") == 100
    and set(summary.get("rows", {})) == {"full_k1", "adapter_loaded_full_k1"}
    and len(rows) == 200
    and (output / "query_trace.jsonl").stat().st_size > 0
)
raise SystemExit(0 if passed else 1)
PY
}

show_raw_progress() {
  local label=$1
  local output=$2
  python - "$label" "$output" <<'PY'
import json
import sys
from pathlib import Path

label, raw = sys.argv[1:]
path = Path(raw) / "eval_progress.jsonl"
if not path.is_file():
    print(f"[{label}] waiting for first episode record")
    raise SystemExit
last = None
with path.open() as handle:
    for line in handle:
        if line.strip():
            last = json.loads(line)
if last is None:
    print(f"[{label}] progress file is empty")
else:
    completed = int(last["completed"])
    total = int(last["total"])
    elapsed = float(last["elapsed_seconds"])
    remaining = elapsed / completed * (total - completed) if completed else float("nan")
    print(
        f"[{label}] {completed}/{total} episodes; elapsed={elapsed / 3600:.2f}h; "
        f"rate-based remaining={remaining / 3600:.2f}h"
    )
PY
}

echo "[wait] Raw R1/R5 K1 artifacts must finish before Protocol A is opened."
while ! raw_k1_complete "$R1_K1_RAW" || ! raw_k1_complete "$R5_K1_RAW"; do
  show_raw_progress "R1 K1" "$R1_K1_RAW"
  show_raw_progress "R5 K1" "$R5_K1_RAW"
  sleep "$WAIT_SECONDS"
done
echo "[pass] Raw R1/R5 K1 artifacts are complete."

revalidated_k1_passes() {
  local output=$1
  [[ -f "$output/online_summary.json" ]] || return 1
  python - "$output/online_summary.json" <<'PY'
import json
import sys

parity = json.load(open(sys.argv[1]))["k1_parity"]
passed = (
    parity.get("K1_PARITY_PASS") is True
    and parity.get("exact_action_chunk_equality") is True
    and parity.get("identical_paired_outcomes") is True
    and parity.get("updater_calls") == 0
    and parity.get("observation_encoder_calls") == 0
    and parity.get("action_encoder_calls") == 0
)
raise SystemExit(0 if passed else 1)
PY
}

revalidate_one() {
  local label=$1
  local input=$2
  local output=$3
  if revalidated_k1_passes "$output"; then
    echo "[$label] revalidated K1 pass already exists; skipping."
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "[$label] incomplete or failed revalidation output exists: $output" >&2
    return 2
  fi
  python -m architectures.simvla.adapters.latentloop.k1_parity_revalidator \
    --input "$input" \
    --output "$output"
  revalidated_k1_passes "$output"
}

revalidate_one "R1 K1" "$R1_K1_RAW" "$R1_K1_REVALIDATED"
revalidate_one "R5 K1" "$R5_K1_RAW" "$R5_K1_REVALIDATED"
echo "[pass] K1 bypass parity holds on identical condition/noise inputs for R1 and R5."

python - "$OFFLINE_R1" "$OFFLINE_R5" <<'PY'
import json
import sys

r1 = json.load(open(sys.argv[1]))["gate"]
r5 = json.load(open(sys.argv[2]))["gate"]
if r1.get("OFFLINE_PREFIX_GATE_PASS") is not True:
    raise SystemExit("R1 offline prefix gate is not open")
if r5.get("OFFLINE_PREFIX_GATE_PASS") is not False:
    raise SystemExit("R5 offline gate was expected to remain closed")
print("[pass] R1 offline gate is open; R5 remains blocked and will not run online K>1.")
PY

R1_CHUNK=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/chunk_aware_latentloop/latest_checkpoint.txt")
R1_OLD=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/old_observation_only/latest_checkpoint.txt")
R1_NOOBS=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/no_observation/latest_checkpoint.txt")
R1_NONREC=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/nonrecurrent_condition/latest_checkpoint.txt")
R1_ACTION=$(checkpoint_from_pointer "$LL_RUN_ROOT/train/t1_r1/action_chunk_correction/latest_checkpoint.txt")

shard_complete() {
  local output=$1
  local task_ids=$2
  [[ -f "$output/online_summary.json" ]] || return 1
  [[ -f "$output/episode_metrics.csv" ]] || return 1
  [[ -f "$output/query_trace.jsonl" ]] || return 1
  [[ -f "$output/eval_config.json" ]] || return 1
  [[ -f "$output/source_lock.json" ]] || return 1
  python - "$output" "$task_ids" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected_tasks = [int(value) for value in sys.argv[2].split(",")]
summary = json.loads((output / "online_summary.json").read_text())
config = json.loads((output / "eval_config.json").read_text())
with (output / "episode_metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
passed = (
    summary.get("matrix") == "protocol_a_screening"
    and summary.get("task_ids") == expected_tasks
    and config.get("resolved_task_ids") == expected_tasks
    and len(summary.get("rows", {})) == 13
    and summary.get("episodes_per_row") == len(expected_tasks) * 10
    and len(rows) == 13 * len(expected_tasks) * 10
)
raise SystemExit(0 if passed else 1)
PY
}

run_shard() {
  local physical_gpu=$1
  local task_ids=$2
  local output=$3
  if shard_complete "$output" "$task_ids"; then
    echo "[GPU $physical_gpu tasks $task_ids] complete shard exists; skipping."
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "[GPU $physical_gpu tasks $task_ids] partial output exists: $output" >&2
    return 2
  fi
  export NUMBA_CACHE_DIR="/tmp/numba_cache_simvla_gpu${physical_gpu}"
  exec env \
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
    SIMVLA_LATENTLOOP_EVAL_RUN=1 \
    bash "$EVAL_WRAPPER" online \
      --matrix protocol_a_screening \
      --output "$output" \
      --checkpoint "$LL_CHECKPOINT" \
      --norm-stats "$LL_NORM" \
      --chunk-aware-checkpoint "$R1_CHUNK" \
      --old-observation-checkpoint "$R1_OLD" \
      --no-observation-checkpoint "$R1_NOOBS" \
      --nonrecurrent-checkpoint "$R1_NONREC" \
      --action-correction-checkpoint "$R1_ACTION" \
      --suite libero_10 \
      --execution-horizon 1 \
      --num-trials 10 \
      --max-tasks 10 \
      --task-ids "$task_ids" \
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
      --teacher-tracking \
      --save-video \
      --video-task-id 9 \
      --video-episodes 0,1 \
      --video-fps 10 \
      --video-stride 2 \
      --tqdm-mininterval 1.0 \
      --device cuda
}

SHARD_TASKS=("8,2" "9,5" "3,1,4" "7,6,0")
SHARD_NAMES=("tasks_8_2" "tasks_9_5" "tasks_3_1_4" "tasks_7_6_0")
SHARD_GPUS=(4 5 6 7)
PIDS=()
PID_LABELS=()
PID_SHARD_NAMES=()

for index in "${!SHARD_TASKS[@]}"; do
  tasks=${SHARD_TASKS[$index]}
  name=${SHARD_NAMES[$index]}
  gpu=${SHARD_GPUS[$index]}
  output="$SHARD_ROOT/$name"
  if shard_complete "$output" "$tasks"; then
    echo "[GPU $gpu tasks $tasks] complete shard exists; skipping."
    continue
  fi
  echo "[launch] GPU $gpu <- LIBERO tasks $tasks"
  run_shard "$gpu" "$tasks" "$output" >"$LOG_ROOT/${name}.log" 2>&1 &
  PIDS+=("$!")
  PID_LABELS+=("GPU${gpu}:${tasks}")
  PID_SHARD_NAMES+=("$name")
done

handle_interrupt() {
  trap - INT TERM
  echo "[Protocol A] stopping ${#PIDS[@]} active task shards." >&2
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    kill -INT "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap handle_interrupt INT TERM

while [[ ${#PIDS[@]} -gt 0 ]]; do
  active=0
  for index in "${!PIDS[@]}"; do
    pid=${PIDS[$index]}
    label=${PID_LABELS[$index]}
    if kill -0 "$pid" 2>/dev/null; then
      active=$((active + 1))
      name=${PID_SHARD_NAMES[$index]}
      progress="$SHARD_ROOT/$name/eval_progress.jsonl"
      if [[ -f "$progress" ]]; then
        last=$(tail -n 1 "$progress")
        echo "[$label] $last"
      else
        echo "[$label] loading model/environment"
      fi
    fi
  done
  [[ "$active" -gt 0 ]] || break
  sleep "$WAIT_SECONDS"
done

set +e
FAILED=0
for index in "${!PIDS[@]}"; do
  wait "${PIDS[$index]}"
  status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "[fail] ${PID_LABELS[$index]} exited with status $status" >&2
    FAILED=1
  fi
done
set -e
trap - INT TERM
[[ "$FAILED" -eq 0 ]] || exit 1

for index in "${!SHARD_TASKS[@]}"; do
  shard_complete "$SHARD_ROOT/${SHARD_NAMES[$index]}" "${SHARD_TASKS[$index]}"
done
echo "[pass] Four disjoint Protocol A task shards are complete."

merged_complete() {
  [[ -f "$MERGED_OUTPUT/online_summary.json" ]] || return 1
  [[ -f "$MERGED_OUTPUT/episode_metrics.csv" ]] || return 1
  [[ -f "$MERGED_OUTPUT/shard_manifest.json" ]] || return 1
  python - "$MERGED_OUTPUT/online_summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
passed = (
    summary.get("matrix") == "protocol_a_screening"
    and summary.get("task_ids") == list(range(9, -1, -1))
    and summary.get("episodes_per_row") == 100
    and len(summary.get("rows", {})) == 13
)
raise SystemExit(0 if passed else 1)
PY
}

if merged_complete; then
  echo "[merge] complete merged output already exists; skipping."
elif [[ -e "$MERGED_OUTPUT" ]]; then
  echo "[merge] partial output exists: $MERGED_OUTPUT" >&2
  exit 2
else
  python -m architectures.simvla.adapters.latentloop.task_shard_merger \
    --shard "$SHARD_ROOT/tasks_8_2" \
    --shard "$SHARD_ROOT/tasks_9_5" \
    --shard "$SHARD_ROOT/tasks_3_1_4" \
    --shard "$SHARD_ROOT/tasks_7_6_0" \
    --expected-task-ids 9,8,7,6,5,4,3,2,1,0 \
    --output "$MERGED_OUTPUT"
fi
merged_complete

echo "[done] Protocol A merged summary: $MERGED_OUTPUT/online_summary.json"
echo "[next] Analyze the 13-row screening before approving any confirmation run."
echo "[blocked] R5 online K>1 was not launched because its offline prefix gate is false."
