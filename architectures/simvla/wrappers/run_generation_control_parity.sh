#!/usr/bin/env bash
# Run the bounded one-query parity/counter gate on one explicit physical GPU.

set -uo pipefail
if [[ "${SIMVLA_GENERATION_PARITY_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_GENERATION_PARITY_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_CONTROL_ROOT:-/home/mingyujung/private/gnaroshi_vla_simvla_generation_egl_20260824}
PYTHON=${SIMVLA_CONTROL_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
GPU=${SIMVLA_CONTROL_GPU_ID:-2}
OUTPUT=$1
BUNDLE=$2
MANIFEST=$3
MANIFEST_SHA=$4
PREFLIGHT=$5

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing parity output: $OUTPUT" >&2
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
export MUJOCO_EGL_DEVICE_ID="$GPU"
unset GALLIUM_DRIVER
unset LIBGL_ALWAYS_SOFTWARE
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mapfile -t renderer_contract < <(
  "$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["renderer"]
print(r["CUBLAS_WORKSPACE_CONFIG"])
print(r["CUDA_DEVICE_MAX_CONNECTIONS"])
print(r["PYTHONHASHSEED"])
print(r["SIMVLA_RENDER_AXIS"])
PY
)
[[ ${#renderer_contract[@]} -eq 4 ]] || exit 2
export CUBLAS_WORKSPACE_CONFIG=${renderer_contract[0]}
export CUDA_DEVICE_MAX_CONNECTIONS=${renderer_contract[1]}
export PYTHONHASHSEED=${renderer_contract[2]}
export SIMVLA_RENDER_AXIS=${renderer_contract[3]}

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_parity \
  --output "$OUTPUT" \
  --bundle-root "$BUNDLE" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --egl-preflight "$PREFLIGHT" \
  --physical-gpu-id "$GPU" \
  --classification "${SIMVLA_CONTROL_CLASSIFICATION:-HOST_LOCAL_EGL_DIAGNOSTIC}"
