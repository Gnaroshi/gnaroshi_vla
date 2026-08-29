#!/usr/bin/env bash
# Complete the unique seed02 LIBERO-Long SimVLA paper grid on rb2 GPU 0.

set -uo pipefail

MODE=${1:---all}
case "$MODE" in
  --all|--preflight|--dry-run) ;;
  *) echo "Usage: $0 [--all|--preflight|--dry-run]" >&2; exit 2 ;;
esac

ROOT=${SIMVLA_PAPER_GRID_ROOT:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_paper_grid_seed02}
PYTHON=${SIMVLA_PAPER_GRID_PYTHON:-/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
STORAGE=${SIMVLA_STORAGE_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage}
GPU=${SIMVLA_PAPER_GRID_GPU_ID:-0}
INPUTS=$STORAGE/artifacts/simvla/fixed_2x2_inputs_v1
BUNDLE=$INPUTS/generation_bundle
CACHE=$STORAGE/results/simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/03_exact_teacher_cache
CONDITION_CHECKPOINT=$INPUTS/condition/native_v0_step_150000.pt
MANIFEST=$STORAGE/results/simvla/latentloop/generation_loop_ng2_rb2_v1/online/step_030000_long500_egl_seed02_v1/episode_manifest.json
MANIFEST_SHA=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
PARITY_GATE=$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1/gates/fixed_2x2_parity.json
BASE_SOURCE_LOCK=$INPUTS/fixed_2x2_source_lock.json
LIBERO_ROOT=$STORAGE/datasets/LIBERO
LIBERO_CONFIG=$STORAGE/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config
CAMPAIGN=$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1
PROVENANCE=$CAMPAIGN/provenance
PLAN=$CAMPAIGN/metadata/execution_plan.json
FAILURES=$CAMPAIGN/metadata/failures.tsv
STATUS=$CAMPAIGN/pipeline.status
LOCK=$STORAGE/locks/simvla_paper_grid_gpu0.lock
CURRENT_STAGE=initializing

mkdir -p "$CAMPAIGN/metadata" "$CAMPAIGN/logs" "$STORAGE/locks"
export PYTHONPATH="$ROOT:$UPSTREAM:$LIBERO_ROOT:${PYTHONPATH:-}"
export SIMVLA_UPSTREAM_ROOT="$UPSTREAM"

known_root() {
  case "$1" in
    full_nfe10)
      echo "$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1/compatibility_from_generation_v1/full_nfe10" ;;
    generation_ng3)
      echo "$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1/compatibility_from_generation_v1/generation_ng3" ;;
    condition_kc2_ng10)
      echo "$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1/condition_kc2_ng10/merged" ;;
    condition_kc2_ng3)
      echo "$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1/condition_kc2_ng3/merged" ;;
    naive_nfe3)
      echo "$STORAGE/results/simvla/generation_control/naive_confirmatory_v1/seed02/naive_nfe3/merged" ;;
    condition_kc2_ng3_coupled)
      echo "$STORAGE/results/simvla/coupled_condition_generation/kc2_ng3_real_cj_projection10k_seed02_v1/online/condition_kc2_ng3_coupled/merged" ;;
    condition_kc3_ng3)
      echo "$STORAGE/results/simvla/coupled_condition_generation/kc3_ng3_real_cj_projection10k_seed02_v1/online/condition_kc3_ng3/merged" ;;
    condition_kc3_ng3_coupled)
      echo "$STORAGE/results/simvla/coupled_condition_generation/kc3_ng3_real_cj_projection10k_seed02_v1/online/condition_kc3_ng3_coupled/merged" ;;
    *) echo "$CAMPAIGN/rows/$1/merged" ;;
  esac
}

mapfile -t ALL_ROWS < <(
  "$PYTHON" - <<'PY'
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import PAPER_GRID_ROWS
print("\n".join(PAPER_GRID_ROWS))
PY
)
[[ ${#ALL_ROWS[@]} -eq 27 ]] || {
  echo "Failed to load the exact 27-row paper grid" >&2
  exit 1
}
KNOWN_COMPLETE=(
  full_nfe10 generation_ng3 condition_kc2_ng10 condition_kc2_ng3
  naive_nfe3 condition_kc2_ng3_coupled condition_kc3_ng3
  condition_kc3_ng3_coupled
)

row_root_args() {
  local row
  for row in "${ALL_ROWS[@]}"; do
    printf '%s\0%s\0' --row-root "$row=$(known_root "$row")"
  done
}

run_plan() {
  local args=()
  while IFS= read -r -d '' value; do args+=("$value"); done < <(row_root_args)
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_grid \
    plan --output "$PLAN" --manifest-sha256 "$MANIFEST_SHA" "${args[@]}"
}

validate_row() {
  local row=$1
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_grid \
    validate-row --row "$row" --root "$(known_root "$row")" \
    --manifest-sha256 "$MANIFEST_SHA" >/dev/null
}

archive_path() {
  local path=$1
  [[ -e "$path" ]] || return 0
  local stamp destination
  stamp=$(date +%Y%m%d_%H%M%S)
  destination="${path}.failed_${stamp}"
  mv "$path" "$destination"
  echo "ARCHIVED source=$path destination=$destination"
}

record_failure() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "$FAILURES"
}

static_preflight() {
  CURRENT_STAGE=static_preflight
  local command required
  for command in bash df flock git grep nvidia-smi tee; do
    command -v "$command" >/dev/null || {
      echo "Missing required command: $command" >&2
      return 1
    }
  done
  for required in \
    "$PYTHON" "$UPSTREAM" "$LIBERO_ROOT" "$LIBERO_CONFIG" "$MANIFEST" \
    "$CACHE/manifest.json" "$CONDITION_CHECKPOINT" "$BASE_SOURCE_LOCK" \
    "$PARITY_GATE" "$BUNDLE/checkpoint/generation_step_030000.pt" \
    "$BUNDLE/norm/libero_norm_official_32700d0.json" \
    "$BUNDLE/transfer_manifest.json"; do
    [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; return 1; }
  done
  [[ "$GPU" == 0 ]] || { echo "rb2 paper grid is locked to physical GPU 0" >&2; return 1; }
  [[ "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)" == \
    exp/simvla-paper-grid-seed02-rb2-20260828 ]] || {
      echo "Unexpected paper-grid branch" >&2; return 1;
    }
  [[ -z "$(git -C "$ROOT" status --porcelain)" ]] || {
    echo "Paper-grid worktree is dirty" >&2; return 1;
  }
  local observed_manifest
  observed_manifest=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$MANIFEST") \
    || return 1
  [[ "$observed_manifest" == "$MANIFEST_SHA" ]] || {
    echo "Manifest SHA mismatch: $observed_manifest" >&2; return 1;
  }
  "$PYTHON" - <<'PY' || return 1
import importlib.metadata as metadata
import platform
import numpy
import torch
assert platform.python_version() == "3.10.20", platform.python_version()
assert numpy.__version__ == "1.26.3", numpy.__version__
assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert metadata.version("mujoco") == "2.3.7", metadata.version("mujoco")
assert metadata.version("transformers") == "4.57.3", metadata.version("transformers")
print("RUNTIME_CONTRACT_PASS")
PY
  "$PYTHON" - "$BUNDLE/checkpoint/generation_step_030000.pt" \
    "$CONDITION_CHECKPOINT" <<'PY' || return 1
import sys
import torch

generation = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
condition = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
assert int(generation["optimizer_step"]) == 30_000
assert int(generation["training_config"]["n_g"]) == 2
assert tuple(generation["model_config"]["supported_n_g"]) == (3, 2)
assert int(condition["global_optimizer_step"]) == 150_000
print("CHECKPOINT_METADATA_CONTRACT_PASS")
PY
  "$PYTHON" -m compileall -q \
    "$ROOT/architectures/simvla/adapters/latentloop/efficient_multirate" \
    "$ROOT/architectures/simvla/wrappers" || return 1
  bash -n "$ROOT/architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh" \
    || return 1
  bash -n "$ROOT/architectures/simvla/wrappers/run_paper_grid_seed02_rb2.sh" \
    || return 1
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest -q \
    "$ROOT/tests/simvla_fixed_2x2/test_paper_grid.py" \
    "$ROOT/tests/simvla_fixed_2x2/test_coupled_condition_generation.py" \
    || return 1
  local row
  for row in "${KNOWN_COMPLETE[@]}"; do
    validate_row "$row" || {
      echo "Known seed02 artifact failed validation: $row" >&2
      return 1
    }
  done
  local free_bytes
  free_bytes=$(df -PB1 "$STORAGE" | awk 'NR==2 {print $4}') || return 1
  ((free_bytes >= 100 * 1024 * 1024 * 1024)) || {
    echo "Less than 100 GiB free under storage root" >&2; return 1;
  }
  run_plan >/dev/null || return 1
  echo "STATIC_PREFLIGHT_PASS plan=$PLAN"
}

gpu_busy() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
    2>/dev/null | grep -q '[0-9]'
}

wait_for_gpu() {
  CURRENT_STAGE=wait_for_gpu0
  local clear_polls=0
  while ((clear_polls < 2)); do
    if gpu_busy; then
      clear_polls=0
      echo "[$(date --iso-8601=seconds)] GPU0 busy; waiting 60s"
    else
      clear_polls=$((clear_polls + 1))
      echo "[$(date --iso-8601=seconds)] GPU0 clear poll=$clear_polls/2"
    fi
    ((clear_polls >= 2)) || sleep 60
  done
  echo "GPU0_STABLY_IDLE"
}

prepare_runtime_provenance() {
  CURRENT_STAGE=prepare_runtime_provenance
  if [[ -d "$PROVENANCE" ]]; then
    if "$PYTHON" - "$PROVENANCE/fixed_eval_source_lock.json" "$ROOT" <<'PY'
import json, subprocess, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
head = subprocess.check_output(["git", "-C", sys.argv[2], "rev-parse", "HEAD"], text=True).strip()
if payload.get("root_commit") != head:
    print(
        f"PROVENANCE_COMMIT_STALE observed={payload.get('root_commit')} expected={head}"
    )
    raise SystemExit(1)
print("PROVENANCE_REUSE_PASS")
PY
    then
      return 0
    fi
    archive_path "$PROVENANCE"
  fi
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock \
    --base-fixed-source-lock "$BASE_SOURCE_LOCK" \
    --base-control-manifest "$BUNDLE/transfer_manifest.json" \
    --output "$PROVENANCE"
}

verify_runtime_provenance() {
  CURRENT_STAGE=verify_runtime_provenance
  local output=$CAMPAIGN/gates/runtime_provenance.json
  mkdir -p "$CAMPAIGN/gates"
  "$PYTHON" - "$BUNDLE" "$CONDITION_CHECKPOINT" \
    "$PROVENANCE/fixed_eval_source_lock.json" \
    "$PROVENANCE/control_manifest.json" "$output" <<'PY'
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import (
    _verify_provenance,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    atomic_write_json,
)

args = SimpleNamespace(
    bundle_root=sys.argv[1],
    condition_checkpoint=sys.argv[2],
    fixed_2x2_source_lock=sys.argv[3],
    control_manifest=sys.argv[4],
    classification="RB2_CONFIRMATORY_EGL",
)
report = _verify_provenance(args)
atomic_write_json(Path(sys.argv[5]), report)
assert report["verdict"] == "FROZEN_PROVENANCE_PASS"
assert report["paper_runtime_match"] is True
print("RUNTIME_PROVENANCE_PASS")
PY
}

condition_code_parity() {
  CURRENT_STAGE=condition_code_parity
  local output=$CAMPAIGN/gates/condition_change_code_parity.json
  if [[ -f "$output" ]] && "$PYTHON" - "$output" <<'PY'
import json, sys
assert json.load(open(sys.argv[1], encoding="utf-8"))["verdict"] == "CONDITION_CHANGE_CODE_PARITY_PASS"
PY
  then
    echo "CONDITION_CHANGE_CODE_PARITY_REUSE"
    return 0
  fi
  archive_path "$output"
  mkdir -p "$CAMPAIGN/gates"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_parity \
    --output "$output" --cache "$CACHE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" --sequences 16
}

valid_projection_train() {
  local k_c=$1 n_g=$2 train=$3
  "$PYTHON" - "$train" "$k_c" "$n_g" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
config = json.load(open(root / "training_config.json", encoding="utf-8"))
summary = json.load(open(root / "run_summary.json", encoding="utf-8"))
checkpoint = root / "checkpoints" / "coupled_generation_step_010000.pt"
assert config["k_c"] == int(sys.argv[2])
assert config["n_g"] == int(sys.argv[3])
assert config["projection_audit"]["trainable_parameters"] == 16384
assert summary["verdict"] == "COUPLED_PROJECTION_TRAINING_COMPLETE"
assert summary["optimizer_step"] == 10000
assert checkpoint.is_file() and checkpoint.stat().st_size > 0
PY
}

run_projection_train() {
  local k_c=$1 n_g=$2 root=$3
  local smoke=$root/smoke/projection_20 train=$root/train/projection_10k
  local parent=$BUNDLE/checkpoint/generation_step_030000.pt
  local norm=$BUNDLE/norm/libero_norm_official_32700d0.json
  local attempt port rc
  mkdir -p "$root/logs"
  if ! valid_projection_train "$k_c" "$n_g" "$train" 2>/dev/null; then
    for attempt in 1 2; do
      archive_path "$smoke"
      archive_path "$train"
      port=$((29820 + k_c * 10 + n_g + attempt))
      CURRENT_STAGE="coupled_kc${k_c}_ng${n_g}_smoke_attempt${attempt}"
      SIMVLA_GPU_IDS="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc-per-node=1 --master-port "$port" \
        -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_generation_train \
        --output "$smoke" --cache "$CACHE" \
        --parent-generation-checkpoint "$parent" \
        --condition-checkpoint "$CONDITION_CHECKPOINT" --norm-stats "$norm" \
        --k-c "$k_c" --n-g "$n_g" --stop-step 20 --local-batch-size 2 \
        --warmup-steps 2 --save-interval 20 --log-interval 5 \
        2>&1 | tee "$root/logs/smoke_attempt${attempt}.log"
      rc=${PIPESTATUS[0]}
      ((rc == 0)) || continue
      CURRENT_STAGE="coupled_kc${k_c}_ng${n_g}_train10k_attempt${attempt}"
      SIMVLA_GPU_IDS="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
        --nproc-per-node=1 --master-port "$((port + 20))" \
        -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_generation_train \
        --output "$train" --cache "$CACHE" \
        --parent-generation-checkpoint "$parent" \
        --condition-checkpoint "$CONDITION_CHECKPOINT" --norm-stats "$norm" \
        --k-c "$k_c" --n-g "$n_g" --stop-step 10000 --local-batch-size 2 \
        --warmup-steps 500 --save-interval 5000 --log-interval 50 \
        --wandb-project gnaroshi-simvla-paper-grid \
        --wandb-name "simvla_seed02_kc${k_c}_ng${n_g}_coupled_projection10k" \
        2>&1 | tee "$root/logs/train10k_attempt${attempt}.log"
      rc=${PIPESTATUS[0]}
      if ((rc == 0)) && valid_projection_train "$k_c" "$n_g" "$train"; then
        return 0
      fi
    done
    return 1
  fi
  echo "COUPLED_TRAIN_REUSE k_c=$k_c n_g=$n_g"
}

run_projection_offline() {
  local k_c=$1 n_g=$2 root=$3
  local train=$root/train/projection_10k
  local offline=$root/offline/projection_10k_512
  local checkpoint=$train/checkpoints/coupled_generation_step_010000.pt
  local attempt rc
  if "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_grid \
    validate-coupled --k-c "$k_c" --n-g "$n_g" --train-root "$train" \
    --offline-root "$offline" >/dev/null 2>&1; then
    echo "COUPLED_OFFLINE_REUSE k_c=$k_c n_g=$n_g"
    return 0
  fi
  for attempt in 1 2; do
    archive_path "$offline"
    CURRENT_STAGE="coupled_kc${k_c}_ng${n_g}_offline512_attempt${attempt}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_generation_offline \
      --output "$offline" --cache "$CACHE" \
      --parent-generation-checkpoint "$BUNDLE/checkpoint/generation_step_030000.pt" \
      --coupled-generation-checkpoint "$checkpoint" \
      --condition-checkpoint "$CONDITION_CHECKPOINT" \
      --norm-stats "$BUNDLE/norm/libero_norm_official_32700d0.json" \
      --k-c "$k_c" --n-g "$n_g" --queries 512 \
      2>&1 | tee "$root/logs/offline512_attempt${attempt}.log"
    rc=${PIPESTATUS[0]}
    if ((rc == 0)) && "$PYTHON" \
      -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_grid \
      validate-coupled --k-c "$k_c" --n-g "$n_g" --train-root "$train" \
      --offline-root "$offline" >/dev/null; then
      return 0
    fi
  done
  return 1
}

prepare_coupled_artifact() {
  local k_c=$1 n_g=$2
  local root=$CAMPAIGN/coupled_artifacts/kc${k_c}_ng${n_g}
  if ! run_projection_train "$k_c" "$n_g" "$root"; then
    record_failure "coupled_artifact" "kc${k_c}_ng${n_g}_training"
    return 1
  fi
  if ! run_projection_offline "$k_c" "$n_g" "$root"; then
    record_failure "coupled_artifact" "kc${k_c}_ng${n_g}_offline"
    return 1
  fi
}

coupled_checkpoint_for_row() {
  local row=$1
  if [[ "$row" =~ ^condition_kc([23])_ng([235])_coupled$ ]]; then
    echo "$CAMPAIGN/coupled_artifacts/kc${BASH_REMATCH[1]}_ng${BASH_REMATCH[2]}/train/projection_10k/checkpoints/coupled_generation_step_010000.pt"
    return 0
  fi
  return 1
}

run_row_once() {
  local row=$1 output=$2 limit=$3
  local extra=()
  local checkpoint
  if checkpoint=$(coupled_checkpoint_for_row "$row"); then
    [[ -f "$checkpoint" ]] || return 1
    extra=(--coupled-generation-checkpoint "$checkpoint")
  fi
  SIMVLA_FIXED_2X2_RUN=1 \
  SIMVLA_FIXED_2X2_ROOT="$ROOT" \
  SIMVLA_FIXED_2X2_PYTHON="$PYTHON" \
  SIMVLA_UPSTREAM_ROOT="$UPSTREAM" \
  SIMVLA_LIBERO_ROOT="$LIBERO_ROOT" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG" \
  bash "$ROOT/architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh" \
    --row "$row" --output "$output" --manifest "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA" --bundle-root "$BUNDLE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" \
    --source-lock "$PROVENANCE/fixed_eval_source_lock.json" \
    --control-manifest "$PROVENANCE/control_manifest.json" \
    --parity-gate "$PARITY_GATE" --physical-gpu-id "$GPU" \
    --classification RB2_CONFIRMATORY_EGL --inference-seed seed02 \
    --task-ids 0,1,2,3,4,5,6,7,8,9 \
    --episodes-per-task-limit "$limit" "${extra[@]}"
}

valid_smoke() {
  local row=$1 output=$2
  "$PYTHON" - "$output/shard_rank0_tasks_0_9/shard_summary.json" "$row" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["row"] == sys.argv[2]
assert payload["episodes"] == 10
assert payload["verdict"].endswith("_SHARD_PASS")
assert payload["all_episode_counter_gates_pass"] is True
PY
}

recover_existing_row() {
  local row=$1 output=$2
  local shard=$output/shard_rank0_tasks_0_9
  local merged=$output/merged
  local extra=()
  local checkpoint
  [[ -d "$shard" ]] || return 1
  if checkpoint=$(coupled_checkpoint_for_row "$row"); then
    [[ -f "$checkpoint" ]] || return 1
    extra=(--generation-checkpoint "$checkpoint")
  fi
  mkdir -p "$output/logs"
  CURRENT_STAGE="row_recovery_${row}"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.row_postprocess_recovery \
    --row "$row" --shard "$shard" --merged "$merged" \
    --expected-manifest-sha256 "$MANIFEST_SHA" "${extra[@]}" \
    2>&1 | tee "$output/logs/restart_recovery.log"
  local rc=${PIPESTATUS[0]}
  ((rc == 0)) && validate_row "$row"
}

smoke_row() {
  local row=$1 output=$CAMPAIGN/smoke/$1
  local attempt rc
  if valid_smoke "$row" "$output" 2>/dev/null; then
    echo "ROW_SMOKE_REUSE row=$row"
    return 0
  fi
  for attempt in 1 2; do
    archive_path "$output"
    archive_path "${output}.egl_preflight.json"
    CURRENT_STAGE="row_smoke_${row}_attempt${attempt}"
    run_row_once "$row" "$output" 1 \
      2>&1 | tee "$CAMPAIGN/logs/smoke_${row}_attempt${attempt}.log"
    rc=${PIPESTATUS[0]}
    if ((rc == 0)) && valid_smoke "$row" "$output"; then
      return 0
    fi
  done
  record_failure "row_smoke" "$row"
  return 1
}

evaluate_row() {
  local row=$1 output=$CAMPAIGN/rows/$1
  local attempt rc
  if validate_row "$row" 2>/dev/null; then
    echo "ROW_ALREADY_COMPLETE row=$row root=$(known_root "$row")"
    return 0
  fi
  if recover_existing_row "$row" "$output" 2>/dev/null; then
    echo "ROW_RECOVERED_ON_RESTART row=$row root=$(known_root "$row")"
    return 0
  fi
  if ! smoke_row "$row"; then
    echo "ROW_SKIPPED_AFTER_SMOKE_FAILURE row=$row" >&2
    return 1
  fi
  for attempt in 1 2; do
    archive_path "$output"
    archive_path "${output}.egl_preflight.json"
    CURRENT_STAGE="row_full_${row}_attempt${attempt}"
    run_row_once "$row" "$output" 50 \
      2>&1 | tee "$CAMPAIGN/logs/full_${row}_attempt${attempt}.log"
    rc=${PIPESTATUS[0]}
    if ((rc == 0)) && validate_row "$row"; then
      echo "ROW_COMPLETE row=$row root=$(known_root "$row")"
      return 0
    fi
  done
  record_failure "row_full" "$row"
  return 1
}

aggregate_grid() {
  CURRENT_STAGE=aggregate_grid
  local args=()
  while IFS= read -r -d '' value; do args+=("$value"); done < <(row_root_args)
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_grid \
    aggregate --output "$CAMPAIGN/aggregate" --manifest-sha256 "$MANIFEST_SHA" \
    "${args[@]}"
}

run_all() {
  static_preflight || return 1
  if [[ "$MODE" == --preflight || "$MODE" == --dry-run ]]; then
    prepare_runtime_provenance || return 1
    verify_runtime_provenance || return 1
    "$PYTHON" - "$PLAN" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"complete={p['complete_row_count']} missing={p['missing_row_count']}")
print("missing_rows=" + ",".join(p["missing_rows"]))
PY
    return 0
  fi

  exec 9>"$LOCK"
  flock -n 9 || { echo "Another paper-grid launcher holds $LOCK" >&2; return 1; }
  prepare_runtime_provenance || return 1
  verify_runtime_provenance || return 1
  wait_for_gpu
  condition_code_parity || return 1

  local k_c n_g row
  for k_c in 2 3; do
    for n_g in 2 5; do
      prepare_coupled_artifact "$k_c" "$n_g" || true
    done
  done

  run_plan >/dev/null
  mapfile -t missing_rows < <(
    "$PYTHON" - "$PLAN" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8"))["missing_rows"]:
    print(row)
PY
  )
  for row in "${missing_rows[@]}"; do
    evaluate_row "$row" || true
    run_plan >/dev/null
  done

  if aggregate_grid; then
    printf 'verdict=PAPER_GRID_SEED02_COMPLETE\nexit_code=0\nresult=%s\n' \
      "$CAMPAIGN/aggregate/paper_grid_summary.json" > "$STATUS"
    echo "PAPER_GRID_SEED02_COMPLETE result=$CAMPAIGN/aggregate/paper_grid_summary.json"
    return 0
  fi
  run_plan >/dev/null
  printf 'verdict=PAPER_GRID_SEED02_INCOMPLETE\nexit_code=1\nplan=%s\nfailures=%s\n' \
    "$PLAN" "$FAILURES" > "$STATUS"
  echo "PAPER_GRID_SEED02_INCOMPLETE plan=$PLAN failures=$FAILURES" >&2
  return 1
}

export HF_HOME=${HF_HOME:-$STORAGE/cache/simvla/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=20260816
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-offline}

run_all
rc=$?
if ((rc != 0)) && [[ ! -f "$STATUS" ]]; then
  printf 'verdict=PAPER_GRID_PIPELINE_FAILED\nexit_code=%s\nstage=%s\nplan=%s\n' \
    "$rc" "$CURRENT_STAGE" "$PLAN" > "$STATUS"
fi
exit "$rc"
