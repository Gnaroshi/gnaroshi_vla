#!/usr/bin/env bash
# Run one fixed K_C x N_G row over an immutable LIBERO-Long manifest.

set -uo pipefail

if [[ "${SIMVLA_FIXED_2X2_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_FIXED_2X2_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_FIXED_2X2_ROOT:?Set SIMVLA_FIXED_2X2_ROOT}
PYTHON=${SIMVLA_FIXED_2X2_PYTHON:?Set SIMVLA_FIXED_2X2_PYTHON}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:?Set SIMVLA_UPSTREAM_ROOT}
LIBERO_ROOT=${SIMVLA_LIBERO_ROOT:?Set SIMVLA_LIBERO_ROOT}
LIBERO_CONFIG=${LIBERO_CONFIG_PATH:?Set LIBERO_CONFIG_PATH}
ROW=
OUTPUT=
MANIFEST=
MANIFEST_SHA=
BUNDLE=
CONDITION_CHECKPOINT=
SOURCE_LOCK=
CONTROL_MANIFEST=
PARITY_GATE=
GPU=
CLASSIFICATION=RB2_CONFIRMATORY_EGL
INFERENCE_SEED=seed02
TASK_IDS=0,1,2,3,4,5,6,7,8,9
SAVE_FAILURE_VIDEOS=0
EPISODES_PER_TASK_LIMIT=0

while (($#)); do
  case "$1" in
    --row) ROW=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --manifest) MANIFEST=$2; shift 2 ;;
    --manifest-sha256) MANIFEST_SHA=$2; shift 2 ;;
    --bundle-root) BUNDLE=$2; shift 2 ;;
    --condition-checkpoint) CONDITION_CHECKPOINT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --control-manifest) CONTROL_MANIFEST=$2; shift 2 ;;
    --parity-gate) PARITY_GATE=$2; shift 2 ;;
    --physical-gpu-id) GPU=$2; shift 2 ;;
    --classification) CLASSIFICATION=$2; shift 2 ;;
    --inference-seed) INFERENCE_SEED=$2; shift 2 ;;
    --task-ids) TASK_IDS=$2; shift 2 ;;
    --episodes-per-task-limit) EPISODES_PER_TASK_LIMIT=$2; shift 2 ;;
    --save-failure-videos) SAVE_FAILURE_VIDEOS=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$EPISODES_PER_TASK_LIMIT" =~ ^[0-9]+$ ]] || {
  echo "--episodes-per-task-limit must be a non-negative integer" >&2
  exit 2
}

case "$ROW" in
  full_nfe10|generation_ng3|condition_kc2_ng10|condition_kc2_ng3|\
  condition_kc3_ng10|condition_kc3_ng3|condition_kc4_ng10|condition_kc4_ng3|\
  condition_kc2_ng2|condition_kc2_naive_nfe3|condition_kc2_naive_nfe2|\
  condition_kc3_naive_nfe3|mechanical_hold_condition_kc2_ng3|\
  mechanical_native_chunk_replay_kc2_ng3|mechanical_hold_action_kc2_ng3|\
  mechanical_no_observation_kc2_ng3) ;;
  *) echo "Invalid --row: $ROW" >&2; exit 2 ;;
esac
case "$CLASSIFICATION" in
  HOST_LOCAL_EGL_DIAGNOSTIC|RB2_CONFIRMATORY_EGL) ;;
  *) echo "Invalid --classification: $CLASSIFICATION" >&2; exit 2 ;;
esac
case "$INFERENCE_SEED" in
  seed01|seed02|seed03) ;;
  *) echo "Invalid --inference-seed: $INFERENCE_SEED" >&2; exit 2 ;;
esac
for value in "$OUTPUT" "$MANIFEST" "$MANIFEST_SHA" "$BUNDLE" \
  "$CONDITION_CHECKPOINT" "$SOURCE_LOCK" "$PARITY_GATE" "$GPU"; do
  [[ -n "$value" ]] || { echo "Missing required argument" >&2; exit 2; }
done
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 2; }
[[ -d "$LIBERO_ROOT/libero" ]] || {
  echo "Invalid SIMVLA_LIBERO_ROOT: $LIBERO_ROOT" >&2
  exit 2
}
[[ -d "$LIBERO_CONFIG" ]] || {
  echo "Invalid LIBERO_CONFIG_PATH: $LIBERO_CONFIG" >&2
  exit 2
}

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:$UPSTREAM:$LIBERO_ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla_storage/cache/simvla/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset GALLIUM_DRIVER
unset LIBGL_ALWAYS_SOFTWARE

mapfile -t renderer_contract < <(
  "$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
r = d["renderer"]
for name in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTHONHASHSEED", "SIMVLA_RENDER_AXIS"):
    print(r[name])
print(d["suite"])
PY
)
[[ ${#renderer_contract[@]} -eq 5 ]] || { echo "Incomplete renderer contract" >&2; exit 2; }
export CUBLAS_WORKSPACE_CONFIG=${renderer_contract[0]}
export CUDA_DEVICE_MAX_CONNECTIONS=${renderer_contract[1]}
export PYTHONHASHSEED=${renderer_contract[2]}
export SIMVLA_RENDER_AXIS=${renderer_contract[3]}
SUITE=${renderer_contract[4]}

"$PYTHON" - "$SUITE" <<'PY'
import sys
from libero.libero import benchmark

suite = sys.argv[1]
if suite not in benchmark.get_benchmark_dict():
    raise RuntimeError(f"LIBERO suite is unavailable: {suite}")
print(f"LIBERO_RUNTIME_IMPORT_PASS suite={suite}")
PY
import_rc=$?
((import_rc == 0)) || exit "$import_rc"

PREFLIGHT="${OUTPUT}.egl_preflight.json"
[[ ! -e "$PREFLIGHT" ]] || { echo "Refusing existing EGL preflight: $PREFLIGHT" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/simvla/simvla_egl_preflight.py \
  --output "$PREFLIGHT" --gpu-id "$GPU" --suite "$SUITE" || exit $?

mkdir -p "$OUTPUT/logs"
recover_and_merge() {
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.row_postprocess_recovery \
    --row "$ROW" \
    --shard "$OUTPUT/shard_rank0_tasks_0_9" \
    --merged "$OUTPUT/merged" \
    --expected-manifest-sha256 "$MANIFEST_SHA" \
    2>&1 | tee "$OUTPUT/logs/postprocess_recovery.log"
  return "${PIPESTATUS[0]}"
}

control_args=()
if [[ -n "$CONTROL_MANIFEST" ]]; then
  control_args=(--control-manifest "$CONTROL_MANIFEST")
fi
video_args=()
if ((SAVE_FAILURE_VIDEOS == 1)); then
  video_args=(--save-video --video-failures-only --video-stride 2 --video-max-per-task 2)
fi
limit_args=()
if ((EPISODES_PER_TASK_LIMIT > 0)); then
  limit_args=(--episodes-per-task-limit "$EPISODES_PER_TASK_LIMIT")
fi
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval \
  --row "$ROW" \
  --output "$OUTPUT/shard_rank0_tasks_0_9" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --bundle-root "$BUNDLE" \
  --condition-checkpoint "$CONDITION_CHECKPOINT" \
  --fixed-2x2-source-lock "$SOURCE_LOCK" \
  "${control_args[@]}" \
  --fixed-2x2-parity-gate "$PARITY_GATE" \
  --egl-preflight "$PREFLIGHT" \
  --physical-gpu-id "$GPU" \
  --task-ids "$TASK_IDS" \
  "${limit_args[@]}" \
  --classification "$CLASSIFICATION" \
  --inference-seed "$INFERENCE_SEED" \
  "${video_args[@]}" \
  2>&1 | tee "$OUTPUT/logs/evaluate.log"
eval_rc=${PIPESTATUS[0]}
if ((eval_rc != 0)); then
  if ((EPISODES_PER_TASK_LIMIT > 0)); then
    echo "Bounded runtime smoke failed rc=$eval_rc; postprocess recovery is not applicable." >&2
    exit "$eval_rc"
  fi
  echo "Evaluation exited rc=$eval_rc; validating bounded postprocess recovery." >&2
  recover_and_merge || exit "$eval_rc"
  echo "ROW_RECOVERED_AFTER_EVAL_FAILURE row=$ROW output=$OUTPUT/merged"
  exit 0
fi

if ((EPISODES_PER_TASK_LIMIT > 0)); then
  "$PYTHON" - "$OUTPUT/shard_rank0_tasks_0_9/shard_summary.json" \
    "$ROW" "$TASK_IDS" "$EPISODES_PER_TASK_LIMIT" <<'PY'
import json
import sys

path, row, task_ids, limit = sys.argv[1:]
summary = json.load(open(path, encoding="utf-8"))
expected = len(task_ids.split(",")) * int(limit)
assert summary["row"] == row
assert summary["evaluation_scope"] == "bounded_runtime_smoke"
assert int(summary["episodes_per_task_limit"]) == int(limit)
assert int(summary["episodes"]) == expected
assert bool(summary["all_episode_counter_gates_pass"])
assert str(summary["verdict"]).endswith("_SHARD_PASS")
print(f"FIXED_2X2_RUNTIME_SMOKE_PASS row={row} episodes={expected}")
PY
  smoke_rc=$?
  ((smoke_rc == 0)) || exit "$smoke_rc"
  exit 0
fi

CUDA_VISIBLE_DEVICES='' "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate \
  aggregate-row \
  --row "$ROW" \
  --output "$OUTPUT/merged" \
  --shard "$OUTPUT/shard_rank0_tasks_0_9" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  2>&1 | tee "$OUTPUT/logs/aggregate.log"
aggregate_rc=${PIPESTATUS[0]}
if ((aggregate_rc != 0)); then
  echo "Aggregation exited rc=$aggregate_rc; rebuilding validated merged artifacts." >&2
  recover_and_merge || exit "$aggregate_rc"
fi
echo "FIXED_2X2_ROW_PASS row=$ROW output=$OUTPUT/merged"
