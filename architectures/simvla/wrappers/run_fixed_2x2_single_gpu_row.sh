#!/usr/bin/env bash
# Run one fixed K_C x N_G row over the immutable rb2 Long-500 manifest.

set -uo pipefail

if [[ "${SIMVLA_FIXED_2X2_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_FIXED_2X2_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_FIXED_2X2_ROOT:?Set SIMVLA_FIXED_2X2_ROOT}
PYTHON=${SIMVLA_FIXED_2X2_PYTHON:?Set SIMVLA_FIXED_2X2_PYTHON}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:?Set SIMVLA_UPSTREAM_ROOT}
ROW=
OUTPUT=
MANIFEST=
MANIFEST_SHA=
BUNDLE=
CONDITION_CHECKPOINT=
SOURCE_LOCK=
PARITY_GATE=
GPU=

while (($#)); do
  case "$1" in
    --row) ROW=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --manifest) MANIFEST=$2; shift 2 ;;
    --manifest-sha256) MANIFEST_SHA=$2; shift 2 ;;
    --bundle-root) BUNDLE=$2; shift 2 ;;
    --condition-checkpoint) CONDITION_CHECKPOINT=$2; shift 2 ;;
    --source-lock) SOURCE_LOCK=$2; shift 2 ;;
    --parity-gate) PARITY_GATE=$2; shift 2 ;;
    --physical-gpu-id) GPU=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$ROW" in
  condition_kc2_ng10|condition_kc2_ng3) ;;
  *) echo "Invalid --row: $ROW" >&2; exit 2 ;;
esac
for value in "$OUTPUT" "$MANIFEST" "$MANIFEST_SHA" "$BUNDLE" \
  "$CONDITION_CHECKPOINT" "$SOURCE_LOCK" "$PARITY_GATE" "$GPU"; do
  [[ -n "$value" ]] || { echo "Missing required argument" >&2; exit 2; }
done
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 2; }

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:$UPSTREAM:${PYTHONPATH:-}"
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

PREFLIGHT="${OUTPUT}.egl_preflight.json"
[[ ! -e "$PREFLIGHT" ]] || { echo "Refusing existing EGL preflight: $PREFLIGHT" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/simvla/simvla_egl_preflight.py \
  --output "$PREFLIGHT" --gpu-id "$GPU" --suite "$SUITE" || exit $?

mkdir -p "$OUTPUT/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval \
  --row "$ROW" \
  --output "$OUTPUT/shard_rank0_tasks_0_9" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --bundle-root "$BUNDLE" \
  --condition-checkpoint "$CONDITION_CHECKPOINT" \
  --fixed-2x2-source-lock "$SOURCE_LOCK" \
  --fixed-2x2-parity-gate "$PARITY_GATE" \
  --egl-preflight "$PREFLIGHT" \
  --physical-gpu-id "$GPU" \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --classification RB2_CONFIRMATORY_EGL \
  --inference-seed seed02 \
  2>&1 | tee "$OUTPUT/logs/evaluate.log"
eval_rc=${PIPESTATUS[0]}
((eval_rc == 0)) || exit "$eval_rc"

CUDA_VISIBLE_DEVICES='' "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate \
  aggregate-row \
  --row "$ROW" \
  --output "$OUTPUT/merged" \
  --shard "$OUTPUT/shard_rank0_tasks_0_9" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  2>&1 | tee "$OUTPUT/logs/aggregate.log"
aggregate_rc=${PIPESTATUS[0]}
((aggregate_rc == 0)) || exit "$aggregate_rc"
echo "FIXED_2X2_ROW_PASS row=$ROW output=$OUTPUT/merged"
