#!/usr/bin/env bash
# Run paired learned-versus-naive K_C/NFE mechanism rows on sd1 GPUs 4-7.

set -uo pipefail

if [[ "${SIMVLA_JOINT_NFE_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_JOINT_NFE_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_JOINT_NFE_ROOT:?Set SIMVLA_JOINT_NFE_ROOT}
PYTHON=${SIMVLA_JOINT_NFE_PYTHON:?Set SIMVLA_JOINT_NFE_PYTHON}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:?Set SIMVLA_UPSTREAM_ROOT}
MAIN_REPO=${SIMVLA_MAIN_REPO:-/home/mingyujung/private/gnaroshi_vla}
STORAGE=${SIMVLA_JOINT_NFE_STORAGE:?Set SIMVLA_JOINT_NFE_STORAGE}
OUTPUT=${SIMVLA_JOINT_NFE_OUTPUT:?Set SIMVLA_JOINT_NFE_OUTPUT}
EXPECTED_BRANCH=exp/simvla-joint-nfe-frontier-20260825

BUNDLE=$STORAGE/artifacts/simvla/generation_eval_bundle_20260824_v1
INPUTS=$STORAGE/artifacts/simvla/fixed_2x2_sd1_v1
CONDITION_CHECKPOINT=$STORAGE/results/simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/checkpoints/native_v0_step_150000.pt
MANIFEST=$INPUTS/manifests/seed02_libero_long500_egl_manifest.json
MANIFEST_SHA=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
FIXED_ROOT=$STORAGE/results/simvla/fixed_2x2/sd1_seed01_seed02_v1/seed02
OLD_FRONTIER=$STORAGE/results/simvla/kc_efficiency_frontier/kc3_kc4_ng10_ng3_sd1_seed02_v1
PARITY_GATE=$STORAGE/results/simvla/fixed_2x2/sd1_seed01_seed02_v1/gates/seed02/fixed_2x2_parity.json

rows=(
  condition_kc2_naive_nfe3
  condition_kc2_ng2
  condition_kc2_naive_nfe2
  condition_kc3_naive_nfe3
)
gpus=(4 5 6 7)

gpu_pids() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | sed '/^[[:space:]]*$/d'
}

wait_for_gpus() {
  local gpu busy
  while true; do
    busy=()
    for gpu in "${gpus[@]}"; do
      [[ -z "$(gpu_pids "$gpu")" ]] || busy+=("$gpu")
    done
    if ((${#busy[@]} == 0)); then
      echo "[$(date --iso-8601=seconds)] GPUs ${gpus[*]} are idle"
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] waiting=60s busy_gpus=${busy[*]}"
    sleep 60
  done
}

recover_row_if_complete() {
  local row=$1
  local row_root=$2
  local shard=$row_root/shard_rank0_tasks_0_9
  local count=0
  [[ ! -f "$row_root/merged/row_summary.json" ]] || return 0
  if [[ -f "$shard/episode_metrics.csv" ]]; then
    count=$(($(wc -l < "$shard/episode_metrics.csv") - 1))
  elif [[ -f "$shard/progress.jsonl" ]]; then
    count=$(wc -l < "$shard/progress.jsonl")
  fi
  if ((count != 500)) || [[ ! -f "$shard/action_chunks.npz" ]]; then
    return 1
  fi
  echo "POSTPROCESS_RECOVERY row=$row episodes=$count"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.row_postprocess_recovery \
    --row "$row" \
    --shard "$shard" \
    --merged "$row_root/merged" \
    --expected-manifest-sha256 "$MANIFEST_SHA"
}

recover_all_completed_rows() {
  local row
  for row in "${rows[@]}"; do
    recover_row_if_complete "$row" "$OUTPUT/rows/$row" || true
  done
  for row in condition_kc3_ng10 condition_kc4_ng10 condition_kc3_ng3 condition_kc4_ng3; do
    recover_row_if_complete "$row" "$OLD_FRONTIER/rows/$row" || true
  done
}

on_exit() {
  local rc=$?
  trap - EXIT
  recover_all_completed_rows
  if ((rc != 0)); then
    echo "JOINT_NFE_FRONTIER_FAILED rc=$rc; completed rows received bounded recovery." >&2
  fi
  exit "$rc"
}
trap on_exit EXIT

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
    --task-ids 0,1,2,3,4,5,6,7,8,9 \
    --save-failure-videos
}

wait_for_old_frontier_artifacts() {
  local deadline=$((SECONDS + 21600)) row
  for row in condition_kc3_ng10 condition_kc4_ng10 condition_kc3_ng3 condition_kc4_ng3; do
    while [[ ! -f "$OLD_FRONTIER/rows/$row/shard_rank0_tasks_0_9/action_chunks.npz" ]]; do
      if ((SECONDS >= deadline)); then
        echo "Timed out waiting for existing frontier row=$row" >&2
        return 1
      fi
      echo "[$(date --iso-8601=seconds)] waiting for existing row=$row post-rollout artifacts"
      sleep 60
    done
    recover_row_if_complete "$row" "$OLD_FRONTIER/rows/$row"
  done
}

run_all() {
  set -euo pipefail
  nvidia-smi -L >/dev/null
  test -x "$PYTHON"
  test -d "$UPSTREAM/evaluation/libero/LIBERO"
  test -f "$CONDITION_CHECKPOINT"
  test -f "$INPUTS/fixed_2x2_source_lock.json"
  test -f "$BUNDLE/transfer_manifest.json"
  test -f "$MANIFEST"
  test -f "$PARITY_GATE"
  test "$(git -C "$ROOT" branch --show-current)" = "$EXPECTED_BRANCH"
  test -z "$(git -C "$ROOT" status --short)"

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

  if [[ -f "$OUTPUT/comparison/joint_nfe_frontier_summary.json" ]]; then
    echo "JOINT_NFE_FRONTIER_ALREADY_COMPLETE output=$OUTPUT"
    return 0
  fi
  [[ ! -e "$OUTPUT" ]] || {
    recover_all_completed_rows
    echo "Refusing unresolved partial output: $OUTPUT" >&2
    return 2
  }

  echo "[1/5] CPU contract and postprocess-recovery regression tests"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest -q \
    tests/simvla_fixed_2x2/test_fixed_2x2_contracts.py

  wait_for_gpus
  mkdir -p "$OUTPUT/logs"
  echo "[2/5] Locking exact source and artifacts"
  "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock \
    --base-fixed-source-lock "$INPUTS/fixed_2x2_source_lock.json" \
    --base-control-manifest "$BUNDLE/transfer_manifest.json" \
    --output "$OUTPUT/provenance"

  echo "[3/5] Launching four distinct paired Long-500 mechanism rows"
  local index row gpu alive failed rc progress count
  local -a pids=()
  for index in "${!rows[@]}"; do
    row=${rows[$index]}
    gpu=${gpus[$index]}
    run_row "$row" "$gpu" > "$OUTPUT/logs/$row.log" 2>&1 &
    pids+=("$!")
    echo "LAUNCHED row=$row gpu=$gpu pid=${pids[$index]}"
  done
  while true; do
    alive=0
    local -a status=()
    for index in "${!pids[@]}"; do
      kill -0 "${pids[$index]}" 2>/dev/null && alive=$((alive + 1))
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
    ((rc == 0)) || failed=1
  done
  recover_all_completed_rows
  ((failed == 0)) || return 1

  echo "[4/5] Recovering and aggregating the already-running K_C=3/4 frontier"
  wait_for_old_frontier_artifacts
  if [[ ! -f "$OLD_FRONTIER/comparison/kc_efficiency_frontier_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES='' "$PYTHON" \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_aggregate \
      --output "$OLD_FRONTIER/comparison" \
      --baseline "$FIXED_ROOT/full_nfe10/merged" \
      --generation "$FIXED_ROOT/generation_ng3/merged" \
      --kc3-ng10 "$OLD_FRONTIER/rows/condition_kc3_ng10/merged" \
      --kc4-ng10 "$OLD_FRONTIER/rows/condition_kc4_ng10/merged" \
      --kc3-ng3 "$OLD_FRONTIER/rows/condition_kc3_ng3/merged" \
      --kc4-ng3 "$OLD_FRONTIER/rows/condition_kc4_ng3/merged"
  fi

  echo "[5/5] Paired learned-versus-naive mechanism aggregation"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.joint_nfe_aggregate \
    --output "$OUTPUT/comparison" \
    --baseline "$FIXED_ROOT/full_nfe10/merged" \
    --kc2-ng10 "$FIXED_ROOT/condition_kc2_ng10/merged" \
    --kc2-ng3 "$FIXED_ROOT/condition_kc2_ng3/merged" \
    --kc2-ng2 "$OUTPUT/rows/condition_kc2_ng2/merged" \
    --kc2-naive-nfe3 "$OUTPUT/rows/condition_kc2_naive_nfe3/merged" \
    --kc2-naive-nfe2 "$OUTPUT/rows/condition_kc2_naive_nfe2/merged" \
    --kc3-ng10 "$OLD_FRONTIER/rows/condition_kc3_ng10/merged" \
    --kc3-ng3 "$OLD_FRONTIER/rows/condition_kc3_ng3/merged" \
    --kc3-naive-nfe3 "$OUTPUT/rows/condition_kc3_naive_nfe3/merged"
  echo "JOINT_NFE_FRONTIER_COMPLETE output=$OUTPUT"
}

run_all
