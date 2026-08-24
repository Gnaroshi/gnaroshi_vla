#!/usr/bin/env bash
# Run one EGL Generation-control row on one explicitly selected physical GPU.

set -uo pipefail
if [[ "${SIMVLA_GENERATION_CONTROL_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_GENERATION_CONTROL_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_CONTROL_ROOT:?Set SIMVLA_CONTROL_ROOT}
PYTHON=${SIMVLA_CONTROL_PYTHON:?Set SIMVLA_CONTROL_PYTHON}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:?Set SIMVLA_UPSTREAM_ROOT}
CURRENT_RUN_GATE=
PARITY_GATE=
ROW=
OUTPUT=
MANIFEST=
MANIFEST_SHA=
BUNDLE=
GPU=
CLASSIFICATION=RB2_CONFIRMATORY_EGL
INFERENCE_SEED=

while (($#)); do
  case "$1" in
    --current-run-gate) CURRENT_RUN_GATE=$2; shift 2 ;;
    --parity-gate) PARITY_GATE=$2; shift 2 ;;
    --row) ROW=$2; shift 2 ;;
    --output) OUTPUT=$2; shift 2 ;;
    --manifest) MANIFEST=$2; shift 2 ;;
    --manifest-sha256) MANIFEST_SHA=$2; shift 2 ;;
    --bundle-root) BUNDLE=$2; shift 2 ;;
    --physical-gpu-id) GPU=$2; shift 2 ;;
    --classification) CLASSIFICATION=$2; shift 2 ;;
    --inference-seed) INFERENCE_SEED=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$ROW" in
  full_nfe10|naive_nfe3|generation_ng3) ;;
  *) echo "Invalid --row: $ROW" >&2; exit 2 ;;
esac
for value in "$CURRENT_RUN_GATE" "$PARITY_GATE" "$OUTPUT" "$MANIFEST" "$MANIFEST_SHA" "$BUNDLE" "$GPU" "$INFERENCE_SEED"; do
  [[ -n "$value" ]] || { echo "Missing required argument" >&2; exit 2; }
done
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 2; }

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla_storage/cache/simvla/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$GPU"
unset GALLIUM_DRIVER
unset LIBGL_ALWAYS_SOFTWARE
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

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
[[ ${#renderer_contract[@]} -eq 5 ]] || { echo "Manifest renderer contract is incomplete" >&2; exit 2; }
export CUBLAS_WORKSPACE_CONFIG=${renderer_contract[0]}
export CUDA_DEVICE_MAX_CONNECTIONS=${renderer_contract[1]}
export PYTHONHASHSEED=${renderer_contract[2]}
export SIMVLA_RENDER_AXIS=${renderer_contract[3]}
SUITE=${renderer_contract[4]}

"$PYTHON" - "$CURRENT_RUN_GATE" "$PARITY_GATE" "$MANIFEST" "$MANIFEST_SHA" <<'PY'
import json, pathlib, sys
current, parity, manifest_path, expected_manifest = sys.argv[1:]
current_payload = json.load(open(current, encoding="utf-8"))
assert current_payload["verdict"] in {
    "RB2_EGL_GENERATION_NG3_THREE_INFERENCE_SEED_COMPLETE",
    "GENERATION_LOOP_VALUE_CONFIRMED",
}, current_payload
parity_payload = json.load(open(parity, encoding="utf-8"))
assert parity_payload["verdict"] == "GENERATION_THREE_ROW_PARITY_PASS", parity_payload
manifest = json.load(open(manifest_path, encoding="utf-8"))
assert manifest["manifest_sha256"] == expected_manifest, manifest
print("PRECEDING_GATES_PASS")
PY
gate_rc=$?
((gate_rc == 0)) || exit "$gate_rc"

PREFLIGHT="${OUTPUT}.egl_preflight.json"
[[ ! -e "$PREFLIGHT" ]] || { echo "Refusing existing EGL preflight: $PREFLIGHT" >&2; exit 2; }
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/simvla/simvla_egl_preflight.py \
  --output "$PREFLIGHT" --gpu-id "$GPU" --suite "$SUITE" || exit $?

mkdir -p "$OUTPUT/logs"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval \
  --row "$ROW" \
  --output "$OUTPUT/shard_rank0_tasks_0_9" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --bundle-root "$BUNDLE" \
  --egl-preflight "$PREFLIGHT" \
  --physical-gpu-id "$GPU" \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --classification "$CLASSIFICATION" \
  --inference-seed "$INFERENCE_SEED" \
  2>&1 | tee "$OUTPUT/logs/evaluate.log"
eval_rc=${PIPESTATUS[0]}
((eval_rc == 0)) || exit "$eval_rc"

CUDA_VISIBLE_DEVICES='' "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate \
  aggregate-row \
  --row "$ROW" \
  --output "$OUTPUT/merged" \
  --shard "$OUTPUT/shard_rank0_tasks_0_9" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --expected-episodes 500 \
  2>&1 | tee "$OUTPUT/logs/aggregate.log"
aggregate_rc=${PIPESTATUS[0]}
((aggregate_rc == 0)) || exit "$aggregate_rc"
echo "SINGLE_GPU_ROW_PASS row=$ROW seed=$INFERENCE_SEED output=$OUTPUT/merged"
