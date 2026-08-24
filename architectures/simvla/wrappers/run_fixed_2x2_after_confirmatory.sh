#!/usr/bin/env bash
# Wait for the running rb2 NFE=3 confirmatory job, then fill the two missing 2x2 cells.

set -uo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_fixed_2x2
CONTROL_ROOT=/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_generation_control_egl
PYTHON=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
UPSTREAM=/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream
STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
INPUTS=$STORAGE/artifacts/simvla/fixed_2x2_inputs_v1
BUNDLE=$INPUTS/generation_bundle
CONDITION_CHECKPOINT=$INPUTS/condition/native_v0_step_150000.pt
SOURCE_LOCK=$INPUTS/fixed_2x2_source_lock.json
EXP=$STORAGE/results/simvla/latentloop/generation_loop_ng2_rb2_v1
CURRENT_GATE=$EXP/online/step_030000_long500_egl_three_inference_seeds_v1/three_inference_seed_summary.json
SEED02=$EXP/online/step_030000_long500_egl_seed02_v1
SEED03=$EXP/online/step_030000_long500_egl_seed03_v1
MANIFEST=$SEED02/episode_manifest.json
MANIFEST_SHA=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
BASELINE=$SEED02/baseline_k1
GENERATION=$SEED02/generation_ng3
CONTROL=$STORAGE/results/simvla/generation_control/naive_confirmatory_v1
CURRENT_VERDICT=$CONTROL/confirmatory_verdict/confirmatory_generation_loop_verdict.json
CACHE=$STORAGE/results/simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/03_exact_teacher_cache
RESULT=$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1
LOG=$STORAGE/results/simvla/fixed_2x2/logs/kc2_ng3_seed02_v1.log
STATUS=$STORAGE/results/simvla/fixed_2x2/logs/kc2_ng3_seed02_v1.status

mkdir -p "$(dirname "$LOG")"

aggregate_control_row() {
  local row_root=$1
  local manifest_sha=$2
  if [[ -f "$row_root/merged/row_summary.json" ]]; then
    return
  fi
  test -f "$row_root/shard_rank0_tasks_0_9/shard_summary.json"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate \
    aggregate-row \
    --row naive_nfe3 \
    --output "$row_root/merged" \
    --shard "$row_root/shard_rank0_tasks_0_9" \
    --expected-manifest-sha256 "$manifest_sha" \
    --expected-episodes 500
}

ensure_confirmatory() {
  if [[ -f "$CURRENT_VERDICT" ]]; then
    return
  fi
  cd "$CONTROL_ROOT"
  export SIMVLA_CONTROL_ROOT=$CONTROL_ROOT
  export SIMVLA_CONTROL_PYTHON=$PYTHON
  test -f "$CURRENT_GATE"

  local seed02_sha
  seed02_sha=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$SEED02/episode_manifest.json")
  test "$seed02_sha" = "$MANIFEST_SHA"
  aggregate_control_row "$CONTROL/seed02/naive_nfe3" "$seed02_sha"

  local seed03_manifest=$SEED03/episode_manifest.json
  local seed03_sha
  seed03_sha=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$seed03_manifest")
  mapfile -t seed03_renderer < <(
    "$PYTHON" - "$seed03_manifest" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["renderer"]
for key in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTHONHASHSEED", "SIMVLA_RENDER_AXIS"):
    print(r[key])
PY
  )
  test "${#seed03_renderer[@]}" -eq 4
  export CUBLAS_WORKSPACE_CONFIG=${seed03_renderer[0]}
  export CUDA_DEVICE_MAX_CONNECTIONS=${seed03_renderer[1]}
  export PYTHONHASHSEED=${seed03_renderer[2]}
  export SIMVLA_RENDER_AXIS=${seed03_renderer[3]}
  if [[ ! -f "$CONTROL/seed03/naive_nfe3/merged/row_summary.json" ]]; then
    if [[ -f "$CONTROL/seed03/naive_nfe3/shard_rank0_tasks_0_9/shard_summary.json" ]]; then
      aggregate_control_row "$CONTROL/seed03/naive_nfe3" "$seed03_sha"
    else
      test ! -e "$CONTROL/seed03/naive_nfe3"
      mkdir -p "$CONTROL/gates"
      if [[ ! -f "$CONTROL/gates/seed03_parity/generation_three_row_counter_gate.json" ]]; then
        test ! -e "$CONTROL/gates/seed03_parity"
        test ! -e "$CONTROL/gates/seed03_parity_egl.json"
        MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
          tools/simvla/simvla_egl_preflight.py \
          --output "$CONTROL/gates/seed03_parity_egl.json" --gpu-id 0
        SIMVLA_GENERATION_PARITY_RUN=1 \
        SIMVLA_CONTROL_GPU_ID=0 \
        SIMVLA_CONTROL_CLASSIFICATION=RB2_CONFIRMATORY_EGL \
        bash architectures/simvla/wrappers/run_generation_control_parity.sh \
          "$CONTROL/gates/seed03_parity" \
          "$BUNDLE" "$seed03_manifest" "$seed03_sha" \
          "$CONTROL/gates/seed03_parity_egl.json"
      fi
      SIMVLA_GENERATION_CONTROL_RUN=1 bash \
        architectures/simvla/wrappers/run_generation_control_single_gpu_row.sh \
        --current-run-gate "$CURRENT_GATE" \
        --parity-gate "$CONTROL/gates/seed03_parity/generation_three_row_counter_gate.json" \
        --row naive_nfe3 \
        --output "$CONTROL/seed03/naive_nfe3" \
        --manifest "$seed03_manifest" \
        --manifest-sha256 "$seed03_sha" \
        --bundle-root "$BUNDLE" \
        --physical-gpu-id 0 \
        --classification RB2_CONFIRMATORY_EGL \
        --inference-seed seed03
    fi
  fi

  if [[ ! -f "$CONTROL/offline_naive_nfe3_512/naive_nfe_audit.json" ]]; then
    test ! -e "$CONTROL/offline_naive_nfe3_512"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_offline \
      --output "$CONTROL/offline_naive_nfe3_512" \
      --cache "$CACHE" \
      --source-lock "$BUNDLE/metadata/source_lock.json" \
      --norm-stats "$BUNDLE/norm/libero_norm_official_32700d0.json" \
      --queries 512 \
      --seed 20260824
  fi

  test ! -e "$CONTROL/confirmatory_verdict"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate \
    confirmatory \
    --output "$CONTROL/confirmatory_verdict" \
    --seed02-full "$SEED02/baseline_k1" \
    --seed02-naive "$CONTROL/seed02/naive_nfe3/merged" \
    --seed02-generation "$SEED02/generation_ng3" \
    --seed03-full "$SEED03/baseline_k1" \
    --seed03-naive "$CONTROL/seed03/naive_nfe3/merged" \
    --seed03-generation "$SEED03/generation_ng3" \
    --learned-offline-screen "$EXP/offline/step_030000_ng3_ng2/offline_screen.json" \
    --naive-offline-audit "$CONTROL/offline_naive_nfe3_512/naive_nfe_audit.json"
  test -f "$CURRENT_VERDICT"
}

run_all() {
  set -euo pipefail
  export SIMVLA_UPSTREAM_ROOT=$UPSTREAM
  export LIBERO_CONFIG_PATH=$STORAGE/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config
  export PYTHONPATH="$CONTROL_ROOT:$UPSTREAM:$STORAGE/datasets/LIBERO:${PYTHONPATH:-}"
  export HF_HOME=$STORAGE/cache/simvla/huggingface
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export MUJOCO_EGL_DEVICE_ID=0
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  unset GALLIUM_DRIVER
  unset LIBGL_ALWAYS_SOFTWARE

  echo "[$(date --iso-8601=seconds)] waiting_for=naive_nfe3_confirmatory"
  while pgrep -f '[s]imvla_naive_nfe3_confirmatory_egl.sh' >/dev/null \
    || pgrep -f '[g]eneration_control_eval --row naive_nfe3' >/dev/null; do
    progress=$(find "$STORAGE/results/simvla/generation_control/naive_confirmatory_v1" \
      -name progress.jsonl -type f -exec wc -l {} + 2>/dev/null | \
      awk '$2 != "total" {n += $1} END {print n + 0}')
    echo "[$(date --iso-8601=seconds)] confirmatory_progress_rows=$progress waiting=60s"
    sleep 60
  done

  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  ensure_confirmatory
  "$PYTHON" - "$CURRENT_VERDICT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("verdict") in {
    "GENERATION_LOOP_VALUE_CONFIRMED",
    "NAIVE_NFE3_SUFFICIENT",
    "GENERATION_LOOP_INCONCLUSIVE",
}, d
print("CONFIRMATORY_GATE_PASS", d["verdict"])
PY

  test -x "$PYTHON"
  git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null
  expected_commit=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["root_commit"])' "$SOURCE_LOCK")
  test "$(git -C "$ROOT" rev-parse HEAD)" = "$expected_commit"
  test -d "$UPSTREAM"
  test -f "$BUNDLE/MANIFEST.sha256"
  test -f "$CONDITION_CHECKPOINT"
  test -f "$SOURCE_LOCK"
  test -f "$MANIFEST"
  test -f "$BASELINE/row_summary.json"
  test -f "$GENERATION/row_summary.json"
  test ! -e "$RESULT"

  cd "$ROOT"
  export SIMVLA_FIXED_2X2_ROOT=$ROOT
  export SIMVLA_FIXED_2X2_PYTHON=$PYTHON
  export PYTHONPATH="$ROOT:$UPSTREAM:$STORAGE/datasets/LIBERO:${PYTHONPATH:-}"

  mapfile -t renderer < <(
    "$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))["renderer"]
for key in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_DEVICE_MAX_CONNECTIONS", "PYTHONHASHSEED", "SIMVLA_RENDER_AXIS"):
    print(r[key])
PY
  )
  test "${#renderer[@]}" -eq 4
  export CUBLAS_WORKSPACE_CONFIG=${renderer[0]}
  export CUDA_DEVICE_MAX_CONNECTIONS=${renderer[1]}
  export PYTHONHASHSEED=${renderer[2]}
  export SIMVLA_RENDER_AXIS=${renderer[3]}

  mkdir -p "$RESULT/gates"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" tools/simvla/simvla_egl_preflight.py \
    --output "$RESULT/gates/egl_preflight.json" --gpu-id 0 --suite libero_10

  CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_parity \
    --output "$RESULT/gates/fixed_2x2_parity.json" \
    --manifest "$MANIFEST" \
    --expected-manifest-sha256 "$MANIFEST_SHA" \
    --bundle-root "$BUNDLE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" \
    --fixed-2x2-source-lock "$SOURCE_LOCK" \
    --egl-preflight "$RESULT/gates/egl_preflight.json" \
    --physical-gpu-id 0 \
    --classification RB2_CONFIRMATORY_EGL

  SIMVLA_FIXED_2X2_RUN=1 bash \
    architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh \
    --row condition_kc2_ng10 \
    --output "$RESULT/condition_kc2_ng10" \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA" \
    --bundle-root "$BUNDLE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" \
    --source-lock "$SOURCE_LOCK" \
    --parity-gate "$RESULT/gates/fixed_2x2_parity.json" \
    --physical-gpu-id 0

  SIMVLA_FIXED_2X2_RUN=1 bash \
    architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh \
    --row condition_kc2_ng3 \
    --output "$RESULT/condition_kc2_ng3" \
    --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA" \
    --bundle-root "$BUNDLE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" \
    --source-lock "$SOURCE_LOCK" \
    --parity-gate "$RESULT/gates/fixed_2x2_parity.json" \
    --physical-gpu-id 0

  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate \
    compare \
    --output "$RESULT/comparison" \
    --baseline "$BASELINE" \
    --condition "$RESULT/condition_kc2_ng10/merged" \
    --generation "$GENERATION" \
    --combined "$RESULT/condition_kc2_ng3/merged"
  cp "$CURRENT_VERDICT" "$RESULT/comparison/preceding_generation_verdict.json"
  echo "FIXED_2X2_COMPLETE summary=$RESULT/comparison/fixed_2x2_summary.json"
}

run_all 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
printf 'exit_code=%s\nlog=%s\nresult=%s\n' \
  "$rc" "$LOG" "$RESULT/comparison/fixed_2x2_summary.json" > "$STATUS"
if ((rc == 0)); then
  echo "FIXED_2X2_COMPLETE"
else
  echo "FIXED_2X2_FAILED rc=$rc"
fi
echo "status=$STATUS"

# Keep the interactive tmux pane alive; STATUS contains the real exit code.
exit 0
