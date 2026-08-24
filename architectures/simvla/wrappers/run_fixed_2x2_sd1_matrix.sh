#!/usr/bin/env bash
# Fill the sd1 K_C x N_G matrix on GPUs 2-7 after those GPUs become idle.

set -uo pipefail

if [[ "${SIMVLA_FIXED_2X2_SD1_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_FIXED_2X2_SD1_RUN=1" >&2
  exit 2
fi

ROOT=/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_fixed_2x2_sd1
PYTHON=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
UPSTREAM=/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream
MAIN_REPO=/home/mingyujung/private/gnaroshi_vla
SHARED=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla
INPUTS=$SHARED/artifacts/simvla/fixed_2x2_sd1_v1
BUNDLE=$SHARED/artifacts/simvla/generation_eval_bundle_20260824_v1
CONDITION_CHECKPOINT=$SHARED/results/simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/checkpoints/native_v0_step_150000.pt
SOURCE_LOCK=$INPUTS/fixed_2x2_source_lock.json
SEED01_MANIFEST=$BUNDLE/manifests/seed01_libero_long500_egl_manifest.json
SEED02_MANIFEST=$INPUTS/manifests/seed02_libero_long500_egl_manifest.json
SEED01_SHA=d1d9bf5a0ff6b20c235eb92dae80189ed3ebdc9eb1591a51fd0d8d572521e74a
SEED02_SHA=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
SEED01_EXISTING=$SHARED/results/simvla/generation_control/20260824_v1/sd1_seed01_three_row_long500
OUT=$SHARED/results/simvla/fixed_2x2/sd1_seed01_seed02_v1
LOG_ROOT=$SHARED/results/simvla/fixed_2x2/logs
LAUNCH_LOG=$LOG_ROOT/sd1_seed01_seed02_v1.log
STATUS=$LOG_ROOT/sd1_seed01_seed02_v1.status

mkdir -p "$LOG_ROOT"

gpu_pids() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | sed '/^[[:space:]]*$/d'
}

wait_for_gpus() {
  local gpu busy
  while true; do
    busy=()
    for gpu in 2 3 4 5 6 7; do
      if [[ -n "$(gpu_pids "$gpu")" ]]; then
        busy+=("$gpu")
      fi
    done
    if ((${#busy[@]} == 0)); then
      echo "[$(date --iso-8601=seconds)] GPUs 2-7 are idle"
      return
    fi
    echo "[$(date --iso-8601=seconds)] waiting for GPUs: ${busy[*]} (GPU 0/1 are ignored)"
    sleep 60
  done
}

activate_manifest_contract() {
  local manifest=$1
  mapfile -t renderer < <(
    "$PYTHON" - "$manifest" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["renderer"]
for key in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTHONHASHSEED", "SIMVLA_RENDER_AXIS"):
    print(r[key])
PY
  )
  [[ ${#renderer[@]} -eq 4 ]] || return 2
  export CUBLAS_WORKSPACE_CONFIG=${renderer[0]}
  export CUDA_DEVICE_MAX_CONNECTIONS=${renderer[1]}
  export PYTHONHASHSEED=${renderer[2]}
  export SIMVLA_RENDER_AXIS=${renderer[3]}
}

run_parity() {
  local seed=$1 manifest=$2 manifest_sha=$3 gpu=$4
  local gate_root=$OUT/gates/$seed
  mkdir -p "$gate_root"
  activate_manifest_contract "$manifest"
  MUJOCO_EGL_DEVICE_ID="$gpu" CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    tools/simvla/simvla_egl_preflight.py \
    --output "$gate_root/egl_preflight.json" --gpu-id "$gpu" --suite libero_10
  MUJOCO_EGL_DEVICE_ID="$gpu" CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_parity \
    --output "$gate_root/fixed_2x2_parity.json" \
    --manifest "$manifest" \
    --expected-manifest-sha256 "$manifest_sha" \
    --bundle-root "$BUNDLE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" \
    --fixed-2x2-source-lock "$SOURCE_LOCK" \
    --egl-preflight "$gate_root/egl_preflight.json" \
    --physical-gpu-id "$gpu" \
    --classification HOST_LOCAL_EGL_DIAGNOSTIC
}

run_row() {
  local label=$1 gpu=$2 seed=$3 row=$4 manifest=$5 manifest_sha=$6
  local row_out=$OUT/$seed/$row
  SIMVLA_FIXED_2X2_RUN=1 \
  SIMVLA_FIXED_2X2_ROOT=$ROOT \
  SIMVLA_FIXED_2X2_PYTHON=$PYTHON \
  SIMVLA_UPSTREAM_ROOT=$UPSTREAM \
  bash architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh \
    --row "$row" \
    --output "$row_out" \
    --manifest "$manifest" \
    --manifest-sha256 "$manifest_sha" \
    --bundle-root "$BUNDLE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" \
    --source-lock "$SOURCE_LOCK" \
    --parity-gate "$OUT/gates/$seed/fixed_2x2_parity.json" \
    --physical-gpu-id "$gpu" \
    --classification HOST_LOCAL_EGL_DIAGNOSTIC \
    --inference-seed "$seed" \
    --task-ids 0,1,2,3,4,5,6,7,8,9 \
    > "$OUT/logs/$label.log" 2>&1
}

compare_seed() {
  local seed=$1 baseline=$2 condition=$3 generation=$4 combined=$5
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate \
    compare \
    --output "$OUT/$seed/comparison" \
    --baseline "$baseline" \
    --condition "$condition" \
    --generation "$generation" \
    --combined "$combined"
}

run_all() {
  set -euo pipefail
  nvidia-smi -L >/dev/null
  wait_for_gpus

  test -x "$PYTHON"
  git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null
  test -d "$UPSTREAM"
  test -d "$UPSTREAM/evaluation/libero/LIBERO"
  test -f "$SOURCE_LOCK"
  test -f "$SEED01_MANIFEST"
  test -f "$SEED02_MANIFEST"
  test -f "$CONDITION_CHECKPOINT"
  test -f "$BUNDLE/checkpoint/generation_step_030000.pt"
  test -f "$SEED01_EXISTING/full_nfe10/merged/row_summary.json"
  test -f "$SEED01_EXISTING/generation_ng3/merged/row_summary.json"
  test ! -e "$OUT"

  expected_commit=$(
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["root_commit"])' \
      "$SOURCE_LOCK"
  )
  test "$(git -C "$ROOT" rev-parse HEAD)" = "$expected_commit"

  mkdir -p "$OUT/logs"
  cd "$ROOT"
  export SIMVLA_UPSTREAM_ROOT=$UPSTREAM
  export PYTHONPATH="$ROOT:$UPSTREAM:$UPSTREAM/evaluation/libero/LIBERO:${PYTHONPATH:-}"
  export HF_HOME=$MAIN_REPO/.cache/huggingface
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  unset GALLIUM_DRIVER
  unset LIBGL_ALWAYS_SOFTWARE

  echo "[$(date --iso-8601=seconds)] bounded parity gates"
  run_parity seed01 "$SEED01_MANIFEST" "$SEED01_SHA" 2
  run_parity seed02 "$SEED02_MANIFEST" "$SEED02_SHA" 4

  labels=(seed01_B seed01_D seed02_A seed02_B seed02_C seed02_D)
  gpus=(2 3 4 5 6 7)
  seeds=(seed01 seed01 seed02 seed02 seed02 seed02)
  rows=(condition_kc2_ng10 condition_kc2_ng3 full_nfe10 condition_kc2_ng10 generation_ng3 condition_kc2_ng3)
  manifests=("$SEED01_MANIFEST" "$SEED01_MANIFEST" "$SEED02_MANIFEST" "$SEED02_MANIFEST" "$SEED02_MANIFEST" "$SEED02_MANIFEST")
  hashes=("$SEED01_SHA" "$SEED01_SHA" "$SEED02_SHA" "$SEED02_SHA" "$SEED02_SHA" "$SEED02_SHA")
  pids=()

  echo "[$(date --iso-8601=seconds)] launching six independent Long-500 rows"
  for index in "${!labels[@]}"; do
    run_row "${labels[$index]}" "${gpus[$index]}" "${seeds[$index]}" \
      "${rows[$index]}" "${manifests[$index]}" "${hashes[$index]}" &
    pids+=("$!")
    echo "LAUNCHED label=${labels[$index]} gpu=${gpus[$index]} pid=${pids[$index]}"
  done

  while true; do
    alive=0
    progress=()
    for index in "${!pids[@]}"; do
      if kill -0 "${pids[$index]}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
      progress_file=$OUT/${seeds[$index]}/${rows[$index]}/shard_rank0_tasks_0_9/progress.jsonl
      if [[ -f "$progress_file" ]]; then
        count=$(wc -l < "$progress_file")
      else
        count=0
      fi
      progress+=("${labels[$index]}=${count}/500")
    done
    echo "[$(date --iso-8601=seconds)] ${progress[*]}"
    ((alive > 0)) || break
    sleep 60
  done

  failed=0
  for index in "${!pids[@]}"; do
    set +e
    wait "${pids[$index]}"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$OUT/logs/${labels[$index]}.status"
    if ((rc != 0)); then
      failed=1
      echo "FAILED label=${labels[$index]} rc=$rc log=$OUT/logs/${labels[$index]}.log" >&2
      tail -40 "$OUT/logs/${labels[$index]}.log" >&2 || true
    fi
  done
  ((failed == 0)) || return 1

  compare_seed seed01 \
    "$SEED01_EXISTING/full_nfe10/merged" \
    "$OUT/seed01/condition_kc2_ng10/merged" \
    "$SEED01_EXISTING/generation_ng3/merged" \
    "$OUT/seed01/condition_kc2_ng3/merged"
  compare_seed seed02 \
    "$OUT/seed02/full_nfe10/merged" \
    "$OUT/seed02/condition_kc2_ng10/merged" \
    "$OUT/seed02/generation_ng3/merged" \
    "$OUT/seed02/condition_kc2_ng3/merged"

  echo "FIXED_2X2_SD1_COMPLETE output=$OUT"
}

set +e
run_all 2>&1 | tee "$LAUNCH_LOG"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$STATUS"
if ((rc == 0)); then
  echo "FIXED_2X2_SD1_PASS status=$STATUS"
else
  echo "FIXED_2X2_SD1_FAILED rc=$rc status=$STATUS log=$LAUNCH_LOG" >&2
fi
echo "Launcher returns 0 so this tmux pane remains available."
exit 0
