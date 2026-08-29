#!/usr/bin/env bash
# Run one SimVLA control method on sd1 GPU 2/3 shards, then aggregate it.

set -uo pipefail

if [[ "${SIMVLA_GENERATION_CONTROL_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_GENERATION_CONTROL_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_CONTROL_ROOT:-/home/mingyujung/private/gnaroshi_vla_simvla_generation_egl_20260824}
PYTHON=${SIMVLA_CONTROL_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}

ROW=
OUTPUT=
MANIFEST=
MANIFEST_SHA=
BUNDLE=
PARITY_GATE=
INFERENCE_SEED=seed01

while (($#)); do
  case "$1" in
    --row) ROW=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --manifest) MANIFEST=$2; shift 2 ;;
    --manifest-sha256) MANIFEST_SHA=$2; shift 2 ;;
    --bundle-root) BUNDLE=$2; shift 2 ;;
    --parity-gate) PARITY_GATE=$2; shift 2 ;;
    --inference-seed) INFERENCE_SEED=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$ROW" in
  full_nfe10|naive_nfe3|generation_ng3) ;;
  *) echo "--row must be full_nfe10, naive_nfe3, or generation_ng3" >&2; exit 2 ;;
esac
for value in "$OUTPUT" "$MANIFEST" "$MANIFEST_SHA" "$BUNDLE" "$PARITY_GATE"; do
  if [[ -z "$value" ]]; then
    echo "Missing required argument" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing result root: $OUTPUT" >&2
  exit 2
fi

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export SIMVLA_UPSTREAM_ROOT="$UPSTREAM"
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset GALLIUM_DRIVER
unset LIBGL_ALWAYS_SOFTWARE
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONHASHSEED=20260815
export SIMVLA_RENDER_AXIS=rb2_egl_long500_v1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

"$PYTHON" - "$PARITY_GATE" <<'PY'
import json, sys
parity = json.load(open(sys.argv[1]))
assert parity["verdict"] == "GENERATION_THREE_ROW_PARITY_PASS", parity
print("PRECEDING_GATES_PASS")
PY
gate_rc=$?
if ((gate_rc != 0)); then
  echo "Wave blocked by parity/EGL gate" >&2
  exit "$gate_rc"
fi

PREFLIGHT_2="${OUTPUT}.egl_gpu2.json"
PREFLIGHT_3="${OUTPUT}.egl_gpu3.json"
for preflight in "$PREFLIGHT_2" "$PREFLIGHT_3"; do
  if [[ -e "$preflight" ]]; then
    echo "Refusing existing per-wave EGL preflight: $preflight" >&2
    exit 2
  fi
done
MUJOCO_EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2 "$PYTHON" tools/simvla/simvla_egl_preflight.py \
  --output "$PREFLIGHT_2" --gpu-id 2 || exit $?
MUJOCO_EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 "$PYTHON" tools/simvla/simvla_egl_preflight.py \
  --output "$PREFLIGHT_3" --gpu-id 3 || exit $?

mkdir -p "$OUTPUT/logs"
common=(
  -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval
  --row "$ROW"
  --manifest "$MANIFEST"
  --expected-manifest-sha256 "$MANIFEST_SHA"
  --bundle-root "$BUNDLE"
  --classification HOST_LOCAL_EGL_DIAGNOSTIC
  --inference-seed "$INFERENCE_SEED"
)

MUJOCO_EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2 "$PYTHON" "${common[@]}" \
  --physical-gpu-id 2 \
  --task-ids 0,1,2,3,4 \
  --egl-preflight "$PREFLIGHT_2" \
  --output "$OUTPUT/gpu2_tasks_0_4" \
  > >(tee "$OUTPUT/logs/gpu2.log") 2>&1 &
pid2=$!

MUJOCO_EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3 "$PYTHON" "${common[@]}" \
  --physical-gpu-id 3 \
  --task-ids 5,6,7,8,9 \
  --egl-preflight "$PREFLIGHT_3" \
  --output "$OUTPUT/gpu3_tasks_5_9" \
  > >(tee "$OUTPUT/logs/gpu3.log") 2>&1 &
pid3=$!

wait "$pid2"; rc2=$?
wait "$pid3"; rc3=$?
printf '%s\n' "gpu2_rc=$rc2" "gpu3_rc=$rc3" > "$OUTPUT/wave.status"
if ((rc2 != 0 || rc3 != 0)); then
  echo "WAVE_FAILED row=$ROW gpu2_rc=$rc2 gpu3_rc=$rc3" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES='' "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate \
  aggregate-row \
  --row "$ROW" \
  --output "$OUTPUT/merged" \
  --shard "$OUTPUT/gpu2_tasks_0_4" \
  --shard "$OUTPUT/gpu3_tasks_5_9" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --expected-episodes 500 \
  2>&1 | tee "$OUTPUT/logs/aggregate.log"
aggregate_rc=${PIPESTATUS[0]}
if ((aggregate_rc != 0)); then
  echo "WAVE_AGGREGATE_FAILED row=$ROW" >&2
  exit "$aggregate_rc"
fi
echo "WAVE_PASS row=$ROW output=$OUTPUT/merged"
