#!/usr/bin/env bash
# Complete the three-seed LIBERO-Long replay controls on rb2 GPU 0.

set -uo pipefail

MODE=${1:---all}
case "$MODE" in
  --all|--preflight|--smoke) ;;
  *) echo "Usage: $0 [--all|--preflight|--smoke]" >&2; exit 2 ;;
esac

if [[ "${SIMVLA_REPLAY_CONTROL_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_REPLAY_CONTROL_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_REPLAY_CONTROL_ROOT:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_replay_controls}
STORAGE=${SIMVLA_STORAGE_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage}
PYTHON=${SIMVLA_REPLAY_CONTROL_PYTHON:-${STORAGE}/envs/simvla/libero_mujoco237/bin/python}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
EXPECTED_COMMIT=${SIMVLA_REPLAY_CONTROL_COMMIT:?Set SIMVLA_REPLAY_CONTROL_COMMIT}
GPU=${SIMVLA_REPLAY_CONTROL_GPU_ID:-0}
MINIMUM_FREE_MIB=${SIMVLA_MINIMUM_FREE_MIB:-28000}
GPU_WAIT_SECONDS=${SIMVLA_GPU_WAIT_SECONDS:-120}

LIBERO_ROOT=${STORAGE}/datasets/LIBERO
LIBERO_CONFIG=${STORAGE}/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config
BUNDLE=${STORAGE}/artifacts/simvla/generation_eval_bundle_20260824_v1
CONDITION_CHECKPOINT=${STORAGE}/artifacts/simvla/fixed_2x2_inputs_v1/condition/native_v0_step_150000.pt
BASE_SOURCE_LOCK=${STORAGE}/artifacts/simvla/mechanical_controls_kc2_ng3_seed02_v1/source_lock.json
BASE_CONTROL_MANIFEST=${BUNDLE}/transfer_manifest.json
RESULT=${SIMVLA_REPLAY_CONTROL_OUTPUT:-${STORAGE}/results/simvla/replay_controls/three_seed_long500_v1}
PROVENANCE=${RESULT}/provenance
SOURCE_LOCK=${PROVENANCE}/fixed_eval_source_lock.json
CONTROL_MANIFEST=${PROVENANCE}/control_manifest.json
STATUS=${RESULT}/pipeline.status
LOG_ROOT=${RESULT}/logs
FAILED_ROOT=${RESULT}/failed_attempts
LOCK=${STORAGE}/locks/simvla_replay_controls_gpu0.lock
PURE_REPLAY=mechanical_full_nfe10_native_chunk_replay_kc2
LEARNED_REPLAY=mechanical_native_chunk_replay_kc2_ng3
KC2_NAIVE=condition_kc2_naive_nfe3

declare -A MANIFESTS MANIFEST_SHA PARITY_GATES
for seed in seed01 seed02 seed03; do
  MANIFESTS[$seed]=${STORAGE}/results/simvla/paper_four_suite_three_seed_v1/manifests/libero_10/${seed}/episode_manifest.json
  PARITY_GATES[$seed]=${STORAGE}/results/simvla/action_equivalent_refresh/three_seed_long500_v1/gates/${seed}/fixed_2x2_parity.json
done
MANIFEST_SHA[seed01]=d1d9bf5a0ff6b20c235eb92dae80189ed3ebdc9eb1591a51fd0d8d572521e74a
MANIFEST_SHA[seed02]=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
MANIFEST_SHA[seed03]=25c3741fd73034cff2d83640dccb675a9fc526c2dc4b406490209e53fd76c61d

NEW_CELLS=(
  "seed01:${PURE_REPLAY}"
  "seed02:${PURE_REPLAY}"
  "seed03:${PURE_REPLAY}"
  "seed01:${LEARNED_REPLAY}"
  "seed03:${LEARNED_REPLAY}"
  "seed01:${KC2_NAIVE}"
  "seed03:${KC2_NAIVE}"
)

mkdir -p "$RESULT" "$LOG_ROOT" "$FAILED_ROOT" "$STORAGE/locks"
export PYTHONPATH="$ROOT:$UPSTREAM:$LIBERO_ROOT:${PYTHONPATH:-}"
export SIMVLA_UPSTREAM_ROOT="$UPSTREAM"
export SIMVLA_FIXED_2X2_ROOT="$ROOT"
export SIMVLA_FIXED_2X2_PYTHON="$PYTHON"
export SIMVLA_LIBERO_ROOT="$LIBERO_ROOT"
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG"
export HF_HOME=${STORAGE}/cache/simvla/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=0
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER}
export MPLCONFIGDIR=/tmp/matplotlib_${USER}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_ROOT/pipeline.log"; }

write_status() {
  printf 'verdict=%s\nexit_code=%s\nstage=%s\nresult=%s\n' \
    "$1" "$2" "$3" "$RESULT" > "$STATUS"
}

archive_path() {
  local path=$1 label=$2 destination
  [[ -e "$path" ]] || return 0
  destination=${FAILED_ROOT}/${label}_$(date +%Y%m%d_%H%M%S)_$$
  mv "$path" "$destination"
  log "archived label=$label destination=$destination"
}

handle_signal() {
  write_status REPLAY_CONTROL_INTERRUPTED "$1" "$2"
  log "interrupted signal=$2"
  exit "$1"
}
trap 'handle_signal 130 INT' INT
trap 'handle_signal 143 TERM' TERM

static_preflight() {
  local command required seed observed root_commit free_bytes
  for command in bash df flock git nvidia-smi tee; do
    command -v "$command" >/dev/null || {
      log "preflight_fail missing_command=$command"
      return 1
    }
  done
  for required in \
    "$PYTHON" "$ROOT" "$UPSTREAM" "$LIBERO_ROOT" "$LIBERO_CONFIG" \
    "$CONDITION_CHECKPOINT" "$BASE_SOURCE_LOCK" "$BASE_CONTROL_MANIFEST" \
    "$BUNDLE/checkpoint/generation_step_030000.pt"; do
    [[ -e "$required" ]] || { log "preflight_fail missing=$required"; return 1; }
  done
  [[ "$GPU" == "0" ]] || { log "preflight_fail rb2_gpu_must_be_0"; return 1; }
  root_commit=$(git -C "$ROOT" rev-parse HEAD) || return 1
  [[ "$root_commit" == "$EXPECTED_COMMIT" ]] || {
    log "preflight_fail commit=$root_commit expected=$EXPECTED_COMMIT"
    return 1
  }
  [[ -z "$(git -C "$ROOT" status --porcelain | grep -v '^?? MUJOCO_LOG.TXT$')" ]] || {
    git -C "$ROOT" status --short | tee -a "$LOG_ROOT/pipeline.log"
    log "preflight_fail worktree_dirty"
    return 1
  }
  if pgrep -f '[v]3_trajectory_rb2' >/dev/null; then
    log "preflight_fail stale_v3_waiter_present"
    return 1
  fi
  for seed in seed01 seed02 seed03; do
    [[ -f "${MANIFESTS[$seed]}" ]] || return 1
    [[ -f "${PARITY_GATES[$seed]}" ]] || return 1
    observed=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "${MANIFESTS[$seed]}") || return 1
    [[ "$observed" == "${MANIFEST_SHA[$seed]}" ]] || {
      log "preflight_fail manifest=$seed observed=$observed"
      return 1
    }
  done
  free_bytes=$(df -PB1 "$STORAGE" | awk 'NR==2 {print $4}')
  ((free_bytes >= 20 * 1024 * 1024 * 1024)) || {
    log "preflight_fail storage_free_bytes=$free_bytes"
    return 1
  }
  "$PYTHON" - <<'PY' || return 1
import importlib.metadata as metadata
import platform
import torch
assert platform.python_version() == "3.10.20"
assert torch.__version__ == "2.7.1+cu128"
assert torch.version.cuda == "12.8"
assert metadata.version("mujoco") == "2.3.7"
assert metadata.version("transformers") == "4.57.3"
print("REPLAY_RUNTIME_CONTRACT_PASS")
PY
  log "static_preflight_pass commit=$root_commit"
}

prepare_provenance() {
  local observed
  if [[ -f "$SOURCE_LOCK" ]]; then
    observed=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("root_commit", ""))' "$SOURCE_LOCK") || return 1
    if [[ "$observed" != "$EXPECTED_COMMIT" ]]; then
      archive_path "$PROVENANCE" "provenance_commit_${observed:0:12}"
    fi
  fi
  if [[ ! -f "$SOURCE_LOCK" || ! -f "$CONTROL_MANIFEST" ]]; then
    archive_path "$PROVENANCE" provenance_incomplete
    "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock \
      --base-fixed-source-lock "$BASE_SOURCE_LOCK" \
      --base-control-manifest "$BASE_CONTROL_MANIFEST" \
      --output "$PROVENANCE" || return 1
  fi
  "$PYTHON" - "$SOURCE_LOCK" "$CONTROL_MANIFEST" "$EXPECTED_COMMIT" "$ROOT" <<'PY'
import hashlib, json, pathlib, sys
lock_path, control_path, commit, root = sys.argv[1:]
payload = json.load(open(lock_path, encoding="utf-8"))
assert payload["root_commit"] == commit
assert pathlib.Path(control_path).is_file()
for name, expected in payload["file_sha256"].items():
    path = pathlib.Path(root) / name
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == expected, (name, observed, expected)
print("REPLAY_PROVENANCE_PASS")
PY
}

wait_for_gpu() {
  local free_mib pids
  while true; do
    free_mib=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    pids=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && ((free_mib >= MINIMUM_FREE_MIB)) && [[ -z "$pids" ]]; then
      return 0
    fi
    log "gpu_wait free_mib=${free_mib:-unknown} pids=${pids//$'\n'/,} retry_seconds=$GPU_WAIT_SECONDS"
    sleep "$GPU_WAIT_SECONDS"
  done
}

cell_root() { echo "$RESULT/rows/$1/$2"; }

existing_root() {
  local seed=$1 row=$2
  case "$seed:$row" in
    seed01:full_nfe10|seed02:full_nfe10|seed03:full_nfe10)
      echo "$STORAGE/results/simvla/paper_followup/three_seed_long500_primary_v1/reused/$seed/full_nfe10" ;;
    seed01:condition_kc2_ng3_coupled|seed03:condition_kc2_ng3_coupled)
      echo "$STORAGE/results/simvla/paper_followup/three_seed_long500_primary_v1/rows/$seed/condition_kc2_ng3_coupled/merged" ;;
    seed02:condition_kc2_ng3_coupled)
      echo "$STORAGE/results/simvla/coupled_condition_generation/kc2_ng3_real_cj_projection10k_seed02_v1/online/condition_kc2_ng3_coupled/merged" ;;
    seed02:condition_kc2_naive_nfe3)
      echo "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/rows/condition_kc2_naive_nfe3/merged" ;;
    seed02:mechanical_native_chunk_replay_kc2_ng3)
      echo "$STORAGE/results/simvla/mechanical_controls/kc2_ng3_long500_seed02_v1/rows/mechanical_native_chunk_replay_kc2_ng3/merged" ;;
    *) echo "$(cell_root "$seed" "$row")/merged" ;;
  esac
}

cell_complete() {
  local seed=$1 row=$2 root=$3
  "$PYTHON" - "$seed" "$row" "$root" "${MANIFEST_SHA[$seed]}" <<'PY' >/dev/null 2>&1
import csv, json, pathlib, sys
seed, row, root, manifest = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4]
summary = json.load(open(root / "merged" / "row_summary.json", encoding="utf-8"))
with open(root / "merged" / "episode_metrics.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
ids = {(int(float(item["task_id"])), int(float(item["trial_id"]))) for item in rows}
assert summary["row"] == row
assert summary["inference_seed"] == seed
assert summary["manifest_sha256"] == manifest
assert str(summary["verdict"]).endswith("_ROW_PASS")
assert len(rows) == len(ids) == 500
assert ids == {(task, trial) for task in range(10) for trial in range(50)}
PY
}

run_cell() {
  local seed=$1 row=$2 output attempt rc expected
  output=$(cell_root "$seed" "$row")
  expected=${MANIFEST_SHA[$seed]}
  if cell_complete "$seed" "$row" "$output"; then
    log "cell_skip_complete seed=$seed row=$row"
    return 0
  fi
  for attempt in 1 2; do
    archive_path "$output" "${seed}_${row}_attempt${attempt}"
    archive_path "${output}.egl_preflight.json" "${seed}_${row}_preflight_attempt${attempt}"
    wait_for_gpu || return 1
    export SIMVLA_FIXED_2X2_RUN=1
    log "cell_start seed=$seed row=$row attempt=$attempt episodes=500"
    bash "$ROOT/architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh" \
      --row "$row" \
      --output "$output" \
      --manifest "${MANIFESTS[$seed]}" \
      --manifest-sha256 "$expected" \
      --bundle-root "$BUNDLE" \
      --condition-checkpoint "$CONDITION_CHECKPOINT" \
      --source-lock "$SOURCE_LOCK" \
      --control-manifest "$CONTROL_MANIFEST" \
      --parity-gate "${PARITY_GATES[$seed]}" \
      --physical-gpu-id "$GPU" \
      --classification RB2_CONFIRMATORY_EGL \
      --inference-seed "$seed" \
      --task-ids 0,1,2,3,4,5,6,7,8,9 \
      --save-failure-videos \
      2>&1 | tee -a "$LOG_ROOT/${seed}_${row}.log"
    rc=${PIPESTATUS[0]}
    if ((rc == 0)) && cell_complete "$seed" "$row" "$output"; then
      log "cell_complete seed=$seed row=$row attempt=$attempt"
      return 0
    fi
    log "cell_attempt_failed seed=$seed row=$row attempt=$attempt rc=$rc"
  done
  return 1
}

runtime_smoke() {
  local row output rc
  [[ -f "$RESULT/metadata/runtime_smoke.pass" ]] && return 0
  mkdir -p "$RESULT/metadata"
  for row in "$PURE_REPLAY" "$LEARNED_REPLAY" "$KC2_NAIVE"; do
    output=$RESULT/runtime_smoke/$row
    archive_path "$output" "smoke_${row}"
    archive_path "${output}.egl_preflight.json" "smoke_${row}_preflight"
    wait_for_gpu || return 1
    export SIMVLA_FIXED_2X2_RUN=1
    log "runtime_smoke_start row=$row"
    bash "$ROOT/architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh" \
      --row "$row" \
      --output "$output" \
      --manifest "${MANIFESTS[seed01]}" \
      --manifest-sha256 "${MANIFEST_SHA[seed01]}" \
      --bundle-root "$BUNDLE" \
      --condition-checkpoint "$CONDITION_CHECKPOINT" \
      --source-lock "$SOURCE_LOCK" \
      --control-manifest "$CONTROL_MANIFEST" \
      --parity-gate "${PARITY_GATES[seed01]}" \
      --physical-gpu-id "$GPU" \
      --classification RB2_CONFIRMATORY_EGL \
      --inference-seed seed01 \
      --task-ids 0 \
      --episodes-per-task-limit 1 \
      2>&1 | tee -a "$LOG_ROOT/smoke_${row}.log"
    rc=${PIPESTATUS[0]}
    ((rc == 0)) || return "$rc"
  done
  printf 'REPLAY_CONTROL_RUNTIME_SMOKE_PASS\n' > "$RESULT/metadata/runtime_smoke.pass"
  log "runtime_smoke_pass rows=3"
}

aggregate_results() {
  local args=() seed row root
  for seed in seed01 seed02 seed03; do
    args+=(--manifest-sha256 "$seed=${MANIFEST_SHA[$seed]}")
    for row in full_nfe10 condition_kc2_ng3_coupled "$KC2_NAIVE" "$LEARNED_REPLAY" "$PURE_REPLAY"; do
      root=$(existing_root "$seed" "$row")
      args+=(--cell "$seed:$row=$root")
    done
  done
  archive_path "$RESULT/aggregate" aggregate_rebuild
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.replay_control_aggregate \
    --output "$RESULT/aggregate" "${args[@]}" \
    2>&1 | tee -a "$LOG_ROOT/aggregate.log"
  return "${PIPESTATUS[0]}"
}

main() {
  local cell seed row
  write_status REPLAY_CONTROL_RUNNING pending static_preflight
  static_preflight || { write_status REPLAY_CONTROL_FAILED 1 static_preflight; return 1; }
  prepare_provenance || { write_status REPLAY_CONTROL_FAILED 1 provenance; return 1; }
  if [[ "$MODE" == "--preflight" ]]; then
    write_status REPLAY_CONTROL_PREFLIGHT_PASS 0 preflight
    return 0
  fi
  runtime_smoke || { write_status REPLAY_CONTROL_FAILED 1 runtime_smoke; return 1; }
  if [[ "$MODE" == "--smoke" ]]; then
    write_status REPLAY_CONTROL_RUNTIME_SMOKE_PASS 0 runtime_smoke
    return 0
  fi
  for cell in "${NEW_CELLS[@]}"; do
    seed=${cell%%:*}
    row=${cell#*:}
    write_status REPLAY_CONTROL_RUNNING pending "$cell"
    run_cell "$seed" "$row" || {
      write_status REPLAY_CONTROL_FAILED 1 "$cell"
      return 1
    }
  done
  aggregate_results || { write_status REPLAY_CONTROL_FAILED 1 aggregate; return 1; }
  write_status REPLAY_CONTROL_THREE_SEED_COMPLETE 0 complete
  log "campaign_complete summary=$RESULT/aggregate/replay_control_three_seed_summary.json"
}

exec 9>"$LOCK"
if ! flock -n 9; then
  write_status REPLAY_CONTROL_BLOCKED 1 gpu_lock
  log "another replay-control launcher holds $LOCK"
  exit 1
fi
main
