#!/usr/bin/env bash
# Sequential Full -> naive NFE=3 -> learned N_G=3 Long-500 waves on sd1 GPU 2/3.

set -uo pipefail
if [[ "${SIMVLA_GENERATION_THREE_ROW_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_GENERATION_THREE_ROW_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_CONTROL_ROOT:-/home/mingyujung/private/gnaroshi_vla_simvla_generation_egl_20260824}
RESULT_ROOT=${SIMVLA_GENERATION_RESULT_ROOT:?Set SIMVLA_GENERATION_RESULT_ROOT}
BUNDLE=${SIMVLA_GENERATION_BUNDLE_ROOT:?Set SIMVLA_GENERATION_BUNDLE_ROOT}
MANIFEST=${SIMVLA_GENERATION_MANIFEST:?Set SIMVLA_GENERATION_MANIFEST}
MANIFEST_SHA=${SIMVLA_GENERATION_MANIFEST_SHA256:?Set SIMVLA_GENERATION_MANIFEST_SHA256}
PARITY_GATE=${SIMVLA_GENERATION_PARITY_GATE:?Set SIMVLA_GENERATION_PARITY_GATE}

if [[ -e "$RESULT_ROOT" ]]; then
  echo "Refusing existing three-row root: $RESULT_ROOT" >&2
  exit 2
fi

wave() {
  local row=$1
  SIMVLA_GENERATION_CONTROL_RUN=1 \
    bash "$ROOT/architectures/simvla/wrappers/run_generation_control_wave.sh" \
      --row "$row" \
      --output "$RESULT_ROOT/$row" \
      --manifest "$MANIFEST" \
      --manifest-sha256 "$MANIFEST_SHA" \
      --bundle-root "$BUNDLE" \
      --parity-gate "$PARITY_GATE" \
      --inference-seed seed01
}

wave full_nfe10 || exit $?
wave naive_nfe3 || exit $?
wave generation_ng3 || exit $?

PYTHON=${SIMVLA_CONTROL_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
CUDA_VISIBLE_DEVICES='' "$PYTHON" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate \
  compare-three \
  --output "$RESULT_ROOT/comparison" \
  --full "$RESULT_ROOT/full_nfe10/merged" \
  --naive "$RESULT_ROOT/naive_nfe3/merged" \
  --generation "$RESULT_ROOT/generation_ng3/merged" \
  2>&1 | tee "$RESULT_ROOT/three_row_aggregate.log"
rc=${PIPESTATUS[0]}
if ((rc != 0)); then
  echo "THREE_ROW_AGGREGATE_FAILED rc=$rc" >&2
  exit "$rc"
fi
echo "SD1_THREE_ROW_COMPLETE result=$RESULT_ROOT/comparison/sd1_generation_control_summary.json"
