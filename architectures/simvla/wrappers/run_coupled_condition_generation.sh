#!/usr/bin/env bash
# Train and evaluate the real Condition c_j -> Generation coupling on one GPU.

set -euo pipefail

if [[ "${SIMVLA_COUPLED_RUN:-0}" != "1" ]]; then
  echo "Set SIMVLA_COUPLED_RUN=1 to enable the coupled experiment." >&2
  exit 2
fi

ROOT=${SIMVLA_COUPLED_ROOT:?Set SIMVLA_COUPLED_ROOT}
PYTHON=${SIMVLA_COUPLED_PYTHON:?Set SIMVLA_COUPLED_PYTHON}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:?Set SIMVLA_UPSTREAM_ROOT}
GPU=${SIMVLA_COUPLED_GPU_ID:-0}
CLASSIFICATION=${SIMVLA_COUPLED_CLASSIFICATION:-RB2_CONFIRMATORY_EGL}
INFERENCE_SEED=${SIMVLA_COUPLED_INFERENCE_SEED:-seed02}
BUNDLE=${SIMVLA_COUPLED_BUNDLE:?Set SIMVLA_COUPLED_BUNDLE}
CACHE=${SIMVLA_COUPLED_CACHE:?Set SIMVLA_COUPLED_CACHE}
CONDITION_CHECKPOINT=${SIMVLA_COUPLED_CONDITION_CHECKPOINT:?Set SIMVLA_COUPLED_CONDITION_CHECKPOINT}
BASE_FIXED_SOURCE_LOCK=${SIMVLA_COUPLED_BASE_FIXED_SOURCE_LOCK:?Set SIMVLA_COUPLED_BASE_FIXED_SOURCE_LOCK}
FIXED_PARITY_GATE=${SIMVLA_COUPLED_FIXED_PARITY_GATE:?Set SIMVLA_COUPLED_FIXED_PARITY_GATE}
MANIFEST=${SIMVLA_COUPLED_MANIFEST:?Set SIMVLA_COUPLED_MANIFEST}
UNCOUPLED_ROW=${SIMVLA_COUPLED_UNCOUPLED_ROW:?Set SIMVLA_COUPLED_UNCOUPLED_ROW}
OUTPUT=${SIMVLA_COUPLED_OUTPUT:?Set SIMVLA_COUPLED_OUTPUT}

PARENT_GENERATION=$BUNDLE/checkpoint/generation_step_030000.pt
NORM=$BUNDLE/norm/libero_norm_official_32700d0.json
SMOKE=$OUTPUT/smoke/projection_20
TRAIN=$OUTPUT/train/projection_10k
OFFLINE=$OUTPUT/offline/projection_10k_512
ONLINE=$OUTPUT/online/condition_kc2_ng3_coupled
PROVENANCE=$OUTPUT/provenance
CHECKPOINT=$TRAIN/checkpoints/coupled_generation_step_010000.pt
MANIFEST_SHA=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$MANIFEST")

for path in "$PYTHON" "$PARENT_GENERATION" "$NORM" "$CACHE/manifest.json" \
  "$CONDITION_CHECKPOINT" "$BASE_FIXED_SOURCE_LOCK" "$FIXED_PARITY_GATE" \
  "$MANIFEST" "$UNCOUPLED_ROW/row_summary.json"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 2; }

cd "$ROOT"
export PYTHONPATH="$ROOT:$UPSTREAM:${SIMVLA_LIBERO_ROOT:-}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla_storage/cache/simvla/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=20260825
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SIMVLA_GPU_IDS=$GPU
export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p "$OUTPUT/logs"

echo "[1/7] Condition c_j exposure parity"
$PYTHON -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_parity \
  --output "$OUTPUT/gates/condition_change_code_parity.json" \
  --cache "$CACHE" \
  --condition-checkpoint "$CONDITION_CHECKPOINT" \
  --sequences 16 \
  2>&1 | tee "$OUTPUT/logs/condition_parity.log"

echo "[2/7] Projection-only 20-step smoke"
$PYTHON -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
  --master-port "${SIMVLA_COUPLED_PORT:-29761}" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_generation_train \
  --output "$SMOKE" \
  --cache "$CACHE" \
  --parent-generation-checkpoint "$PARENT_GENERATION" \
  --condition-checkpoint "$CONDITION_CHECKPOINT" \
  --norm-stats "$NORM" \
  --n-g 3 --stop-step 20 --local-batch-size 2 \
  --warmup-steps 2 --save-interval 20 --log-interval 5 \
  2>&1 | tee "$OUTPUT/logs/smoke.log"

echo "[3/7] Projection-only 10K training"
$PYTHON -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
  --master-port "$(( ${SIMVLA_COUPLED_PORT:-29761} + 1 ))" \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_generation_train \
  --output "$TRAIN" \
  --cache "$CACHE" \
  --parent-generation-checkpoint "$PARENT_GENERATION" \
  --condition-checkpoint "$CONDITION_CHECKPOINT" \
  --norm-stats "$NORM" \
  --n-g 3 --stop-step 10000 --local-batch-size 2 \
  --warmup-steps 500 --save-interval 5000 --log-interval 50 \
  --wandb-project "${SIMVLA_COUPLED_WANDB_PROJECT:-gnaroshi-simvla-coupled}" \
  --wandb-name "${SIMVLA_COUPLED_WANDB_NAME:-simvla_kc2_ng3_real_cj_projection10k}" \
  2>&1 | tee "$OUTPUT/logs/train_10k.log"

echo "[4/7] Held-out 512-query coupled integrity/fidelity screen"
$PYTHON -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_generation_offline \
  --output "$OFFLINE" \
  --cache "$CACHE" \
  --parent-generation-checkpoint "$PARENT_GENERATION" \
  --coupled-generation-checkpoint "$CHECKPOINT" \
  --condition-checkpoint "$CONDITION_CHECKPOINT" \
  --norm-stats "$NORM" \
  --queries 512 \
  2>&1 | tee "$OUTPUT/logs/offline.log"

echo "[5/7] Immutable coupled evaluation provenance"
$PYTHON -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock \
  --base-fixed-source-lock "$BASE_FIXED_SOURCE_LOCK" \
  --base-control-manifest "$BUNDLE/transfer_manifest.json" \
  --output "$PROVENANCE" \
  2>&1 | tee "$OUTPUT/logs/provenance.log"

mapfile -t renderer < <(
  $PYTHON - "$MANIFEST" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["renderer"]
for key in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTHONHASHSEED", "SIMVLA_RENDER_AXIS"):
    print(r[key])
PY
)
[[ ${#renderer[@]} -eq 4 ]] || { echo "Incomplete renderer contract" >&2; exit 2; }
export CUBLAS_WORKSPACE_CONFIG=${renderer[0]}
export CUDA_DEVICE_MAX_CONNECTIONS=${renderer[1]}
export PYTHONHASHSEED=${renderer[2]}
export SIMVLA_RENDER_AXIS=${renderer[3]}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$GPU
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset GALLIUM_DRIVER
unset LIBGL_ALWAYS_SOFTWARE

echo "[6/7] Same-manifest 500-episode coupled online evaluation"
$PYTHON tools/simvla/simvla_egl_preflight.py \
  --output "$OUTPUT/gates/coupled_online_egl.json" --gpu-id "$GPU" --suite libero_10
set +e
$PYTHON -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval \
  --row condition_kc2_ng3_coupled \
  --output "$ONLINE/shard_rank0_tasks_0_9" \
  --manifest "$MANIFEST" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --bundle-root "$BUNDLE" \
  --condition-checkpoint "$CONDITION_CHECKPOINT" \
  --coupled-generation-checkpoint "$CHECKPOINT" \
  --fixed-2x2-source-lock "$PROVENANCE/fixed_eval_source_lock.json" \
  --control-manifest "$PROVENANCE/control_manifest.json" \
  --fixed-2x2-parity-gate "$FIXED_PARITY_GATE" \
  --egl-preflight "$OUTPUT/gates/coupled_online_egl.json" \
  --physical-gpu-id "$GPU" \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --classification "$CLASSIFICATION" \
  --inference-seed "$INFERENCE_SEED" \
  --save-video --video-failures-only \
  2>&1 | tee "$OUTPUT/logs/online_500.log"
online_rc=${PIPESTATUS[0]}
set -e
if ((online_rc != 0)); then
  echo "Coupled evaluation exited rc=$online_rc; validating bounded postprocess recovery." >&2
  CUDA_VISIBLE_DEVICES='' $PYTHON \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.row_postprocess_recovery \
    --row condition_kc2_ng3_coupled \
    --shard "$ONLINE/shard_rank0_tasks_0_9" \
    --merged "$ONLINE/merged" \
    --expected-manifest-sha256 "$MANIFEST_SHA" \
    --generation-checkpoint "$CHECKPOINT"
fi

echo "[7/7] Aggregate and paired uncoupled-vs-coupled comparison"
if [[ ! -f "$ONLINE/merged/row_summary.json" ]]; then
  set +e
  CUDA_VISIBLE_DEVICES='' $PYTHON \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate \
    aggregate-row \
    --row condition_kc2_ng3_coupled \
    --output "$ONLINE/merged" \
    --shard "$ONLINE/shard_rank0_tasks_0_9" \
    --expected-manifest-sha256 "$MANIFEST_SHA"
  aggregate_rc=$?
  set -e
  if ((aggregate_rc != 0)); then
    CUDA_VISIBLE_DEVICES='' $PYTHON \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.row_postprocess_recovery \
      --row condition_kc2_ng3_coupled \
      --shard "$ONLINE/shard_rank0_tasks_0_9" \
      --merged "$ONLINE/merged" \
      --expected-manifest-sha256 "$MANIFEST_SHA" \
      --generation-checkpoint "$CHECKPOINT"
  fi
fi
CUDA_VISIBLE_DEVICES='' $PYTHON \
  -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate \
  compare-coupling \
  --output "$OUTPUT/comparison" \
  --uncoupled "$UNCOUPLED_ROW" \
  --coupled "$ONLINE/merged"

echo "COUPLED_CONDITION_GENERATION_COMPLETE output=$OUTPUT"
