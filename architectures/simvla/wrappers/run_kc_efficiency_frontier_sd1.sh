#!/usr/bin/env bash
# Evaluate K_C={3,4} x N_G={10,3} on four sd1 GPUs with one paired manifest.

set -uo pipefail

if [[ "${SIMVLA_KC_FRONTIER_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_KC_FRONTIER_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_KC_FRONTIER_ROOT:?Set SIMVLA_KC_FRONTIER_ROOT}
PYTHON=${SIMVLA_KC_FRONTIER_PYTHON:?Set SIMVLA_KC_FRONTIER_PYTHON}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:?Set SIMVLA_UPSTREAM_ROOT}
MAIN_REPO=${SIMVLA_MAIN_REPO:-/home/mingyujung/private/gnaroshi_vla}
STORAGE=${SIMVLA_KC_FRONTIER_STORAGE:?Set SIMVLA_KC_FRONTIER_STORAGE}
BUNDLE=$STORAGE/artifacts/simvla/generation_eval_bundle_20260824_v1
INPUTS=$STORAGE/artifacts/simvla/fixed_2x2_sd1_v1
CONDITION_CHECKPOINT=$STORAGE/results/simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/checkpoints/native_v0_step_150000.pt
BASE_SOURCE_LOCK=$INPUTS/fixed_2x2_source_lock.json
BASE_CONTROL_MANIFEST=$BUNDLE/transfer_manifest.json
MANIFEST=$INPUTS/manifests/seed02_libero_long500_egl_manifest.json
FIXED_ROOT=$STORAGE/results/simvla/fixed_2x2/sd1_seed01_seed02_v1
PARITY_GATE=$FIXED_ROOT/gates/seed02/fixed_2x2_parity.json
BASELINE=$FIXED_ROOT/seed02/full_nfe10/merged
GENERATION=$FIXED_ROOT/seed02/generation_ng3/merged
OUTPUT=${SIMVLA_KC_FRONTIER_OUTPUT:?Set SIMVLA_KC_FRONTIER_OUTPUT}
EXPECTED_COMMIT=${SIMVLA_KC_FRONTIER_COMMIT:?Set SIMVLA_KC_FRONTIER_COMMIT}
MANIFEST_SHA=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48

gpu_pids() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | sed '/^[[:space:]]*$/d'
}

wait_for_gpus() {
  local gpu busy
  while true; do
    busy=()
    for gpu in 2 3 4 5; do
      [[ -z "$(gpu_pids "$gpu")" ]] || busy+=("$gpu")
    done
    if ((${#busy[@]} == 0)); then
      echo "[$(date --iso-8601=seconds)] GPUs 2,3,4,5 are idle"
      return
    fi
    echo "[$(date --iso-8601=seconds)] waiting=60s busy_gpus=${busy[*]}"
    sleep 60
  done
}

run_row() {
  local row=$1 gpu=$2
  SIMVLA_FIXED_2X2_RUN=1 \
  SIMVLA_FIXED_2X2_ROOT=$ROOT \
  SIMVLA_FIXED_2X2_PYTHON=$PYTHON \
  SIMVLA_UPSTREAM_ROOT=$UPSTREAM \
  bash architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh \
    --row "$row" \
    --output "$OUTPUT/rows/$row" \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA" \
    --bundle-root "$BUNDLE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" \
    --source-lock "$OUTPUT/provenance/fixed_eval_source_lock.json" \
    --control-manifest "$OUTPUT/provenance/control_manifest.json" \
    --parity-gate "$PARITY_GATE" \
    --physical-gpu-id "$gpu" \
    --classification HOST_LOCAL_EGL_DIAGNOSTIC \
    --inference-seed seed02 \
    --task-ids 0,1,2,3,4,5,6,7,8,9
}

run_all() {
  set -euo pipefail
  nvidia-smi -L >/dev/null
  wait_for_gpus
  test -x "$PYTHON"
  test -d "$UPSTREAM/evaluation/libero/LIBERO"
  test -f "$CONDITION_CHECKPOINT"
  test -f "$BASE_SOURCE_LOCK"
  test -f "$BASE_CONTROL_MANIFEST"
  test -f "$MANIFEST"
  test -f "$PARITY_GATE"
  test -f "$BASELINE/row_summary.json"
  test -f "$GENERATION/row_summary.json"
  test "$(git -C "$ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
  test -z "$(git -C "$ROOT" status --short --untracked-files=no)"
  if [[ -f "$OUTPUT/comparison/kc_efficiency_frontier_summary.json" ]]; then
    echo "KC_EFFICIENCY_FRONTIER_ALREADY_COMPLETE output=$OUTPUT"
    return 0
  fi
  [[ ! -e "$OUTPUT" ]] || {
    echo "Refusing partial/existing output: $OUTPUT" >&2
    return 2
  }

  mkdir -p "$OUTPUT/logs"
  cd "$ROOT"
  export PYTHONPATH="$ROOT:$UPSTREAM:$UPSTREAM/evaluation/libero/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HOME=${HF_HOME:-$MAIN_REPO/.cache/huggingface}
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-/home/mingyujung/.libero_orig}
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  unset GALLIUM_DRIVER
  unset LIBGL_ALWAYS_SOFTWARE

  echo "[1/3] Locking K_C frontier source and artifacts"
  "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock \
    --base-fixed-source-lock "$BASE_SOURCE_LOCK" \
    --base-control-manifest "$BASE_CONTROL_MANIFEST" \
    --output "$OUTPUT/provenance"

  rows=(condition_kc3_ng10 condition_kc4_ng10 condition_kc3_ng3 condition_kc4_ng3)
  gpus=(2 3 4 5)
  pids=()
  echo "[2/3] Launching four distinct Long-500 efficiency rows"
  for index in "${!rows[@]}"; do
    row=${rows[$index]}
    gpu=${gpus[$index]}
    run_row "$row" "$gpu" > "$OUTPUT/logs/$row.log" 2>&1 &
    pids+=("$!")
    echo "LAUNCHED row=$row gpu=$gpu pid=${pids[$index]}"
  done

  while true; do
    alive=0
    status=()
    for index in "${!pids[@]}"; do
      if kill -0 "${pids[$index]}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
      progress=$OUTPUT/rows/${rows[$index]}/shard_rank0_tasks_0_9/progress.jsonl
      count=0
      [[ ! -f "$progress" ]] || count=$(wc -l < "$progress")
      status+=("${rows[$index]}=$count/500")
    done
    echo "[$(date --iso-8601=seconds)] ${status[*]}"
    ((alive > 0)) || break
    sleep 60
  done

  failed=0
  for index in "${!pids[@]}"; do
    set +e
    wait "${pids[$index]}"
    rc=$?
    set -e
    printf 'exit_code=%s\n' "$rc" > "$OUTPUT/logs/${rows[$index]}.status"
    if ((rc != 0)); then
      failed=1
      echo "FAILED row=${rows[$index]} gpu=${gpus[$index]} rc=$rc" >&2
      tail -40 "$OUTPUT/logs/${rows[$index]}.log" >&2 || true
    fi
  done
  ((failed == 0)) || return 1

  echo "[3/3] Paired success/latency/call-count frontier aggregation"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_aggregate \
    --output "$OUTPUT/comparison" \
    --baseline "$BASELINE" \
    --generation "$GENERATION" \
    --kc3-ng10 "$OUTPUT/rows/condition_kc3_ng10/merged" \
    --kc4-ng10 "$OUTPUT/rows/condition_kc4_ng10/merged" \
    --kc3-ng3 "$OUTPUT/rows/condition_kc3_ng3/merged" \
    --kc4-ng3 "$OUTPUT/rows/condition_kc4_ng3/merged"
  echo "KC_EFFICIENCY_FRONTIER_COMPLETE output=$OUTPUT"
}

run_all
