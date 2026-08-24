#!/usr/bin/env bash
# Wait for the running rb2 NFE=3 confirmatory job, then fill the two missing 2x2 cells.

set -uo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_fixed_2x2
PYTHON=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
UPSTREAM=/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream
STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
INPUTS=$STORAGE/artifacts/simvla/fixed_2x2_inputs_v1
BUNDLE=$INPUTS/generation_bundle
CONDITION_CHECKPOINT=$INPUTS/condition/native_v0_step_150000.pt
SOURCE_LOCK=$INPUTS/fixed_2x2_source_lock.json
EXP=$STORAGE/results/simvla/latentloop/generation_loop_ng2_rb2_v1
SEED02=$EXP/online/step_030000_long500_egl_seed02_v1
MANIFEST=$SEED02/episode_manifest.json
MANIFEST_SHA=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
BASELINE=$SEED02/baseline_k1
GENERATION=$SEED02/generation_ng3
CURRENT_STATUS=$STORAGE/results/simvla/generation_control/logs/naive_nfe3_confirmatory_egl.status
CURRENT_VERDICT=$STORAGE/results/simvla/generation_control/naive_confirmatory_v1/confirmatory_verdict/confirmatory_generation_loop_verdict.json
RESULT=$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1
LOG=$STORAGE/results/simvla/fixed_2x2/logs/kc2_ng3_seed02_v1.log
STATUS=$STORAGE/results/simvla/fixed_2x2/logs/kc2_ng3_seed02_v1.status

mkdir -p "$(dirname "$LOG")"

run_all() {
  set -euo pipefail
  echo "[$(date --iso-8601=seconds)] waiting_for=naive_nfe3_confirmatory"
  while pgrep -f '[s]imvla_naive_nfe3_confirmatory_egl.sh' >/dev/null; do
    progress=$(find "$STORAGE/results/simvla/generation_control/naive_confirmatory_v1" \
      -name progress.jsonl -type f -exec wc -l {} + 2>/dev/null | \
      awk '$2 != "total" {n += $1} END {print n + 0}')
    echo "[$(date --iso-8601=seconds)] confirmatory_progress_rows=$progress waiting=60s"
    sleep 60
  done

  test -f "$CURRENT_STATUS"
  grep -qx 'exit_code=0' "$CURRENT_STATUS"
  test -f "$CURRENT_VERDICT"
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
  test -d "$UPSTREAM"
  test -f "$BUNDLE/MANIFEST.sha256"
  test -f "$CONDITION_CHECKPOINT"
  test -f "$SOURCE_LOCK"
  test -f "$MANIFEST"
  test -f "$BASELINE/row_summary.json"
  test -f "$GENERATION/row_summary.json"
  test ! -e "$RESULT"
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"

  cd "$ROOT"
  export SIMVLA_FIXED_2X2_ROOT=$ROOT
  export SIMVLA_FIXED_2X2_PYTHON=$PYTHON
  export SIMVLA_UPSTREAM_ROOT=$UPSTREAM
  export LIBERO_CONFIG_PATH=$STORAGE/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config
  export PYTHONPATH="$ROOT:$UPSTREAM:$STORAGE/datasets/LIBERO:${PYTHONPATH:-}"
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
