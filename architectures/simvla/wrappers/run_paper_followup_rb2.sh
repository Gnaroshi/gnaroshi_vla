#!/usr/bin/env bash
# Resume the selected three-inference-seed SimVLA paper follow-up on rb2 GPU 0.

set -uo pipefail

MODE=${1:---all}
case "$MODE" in
  --all|--preflight|--dry-run) ;;
  *) echo "Usage: $0 [--all|--preflight|--dry-run]" >&2; exit 2 ;;
esac

ROOT=${SIMVLA_PAPER_FOLLOWUP_ROOT:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_paper_grid_seed02}
PYTHON=${SIMVLA_PAPER_FOLLOWUP_PYTHON:-/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
STORAGE=${SIMVLA_STORAGE_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage}
GPU=${SIMVLA_PAPER_FOLLOWUP_GPU_ID:-0}
INPUTS=$STORAGE/artifacts/simvla/fixed_2x2_inputs_v1
BUNDLE=$INPUTS/generation_bundle
CACHE=$STORAGE/results/simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/03_exact_teacher_cache
CONDITION_CHECKPOINT=$INPUTS/condition/native_v0_step_150000.pt
BASE_SOURCE_LOCK=$INPUTS/fixed_2x2_source_lock.json
LIBERO_ROOT=$STORAGE/datasets/LIBERO
LIBERO_CONFIG=$STORAGE/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config
GENERATION_EXP=$STORAGE/results/simvla/latentloop/generation_loop_ng2_rb2_v1
CAMPAIGN=$STORAGE/results/simvla/paper_followup/three_seed_long500_primary_v1
PROVENANCE=$CAMPAIGN/provenance
PLAN=$CAMPAIGN/metadata/execution_plan.json
FAILURES=$CAMPAIGN/metadata/failures.tsv
STATUS=$CAMPAIGN/pipeline.status
LOCK=$STORAGE/locks/simvla_paper_followup_gpu0.lock
NONLONG=$STORAGE/results/simvla/paper_nonlong_seed01_primary_v1/summary/selected_matrix_summary.json
MECHANICAL=$STORAGE/results/simvla/mechanical_controls/kc2_ng3_long500_seed02_v1/comparison/mechanical_control_summary.json
CURRENT_STAGE=initializing

declare -A MANIFESTS MANIFEST_SHA PARITY_GATES
MANIFESTS[seed01]=$GENERATION_EXP/online/step_010000_long500_egl_paired_v1/episode_manifest.json
MANIFESTS[seed02]=$GENERATION_EXP/online/step_030000_long500_egl_seed02_v1/episode_manifest.json
MANIFESTS[seed03]=$GENERATION_EXP/online/step_030000_long500_egl_seed03_v1/episode_manifest.json
MANIFEST_SHA[seed01]=d1d9bf5a0ff6b20c235eb92dae80189ed3ebdc9eb1591a51fd0d8d572521e74a
MANIFEST_SHA[seed02]=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
MANIFEST_SHA[seed03]=25c3741fd73034cff2d83640dccb675a9fc526c2dc4b406490209e53fd76c61d
for seed in seed01 seed02 seed03; do
  PARITY_GATES[$seed]=$STORAGE/results/simvla/action_equivalent_refresh/three_seed_long500_v1/gates/$seed/fixed_2x2_parity.json
done

ROWS=(
  full_nfe10
  naive_nfe3
  generation_ng3
  condition_kc2_ng10
  condition_kc2_ng3
  condition_kc2_ng2_coupled
  condition_kc2_ng3_coupled
  condition_kc2_ng5_coupled
)
REUSED_CELLS=(
  seed01:full_nfe10 seed01:generation_ng3
  seed02:full_nfe10 seed02:generation_ng3
  seed03:full_nfe10 seed03:generation_ng3
  seed02:naive_nfe3 seed03:naive_nfe3
  seed02:condition_kc2_ng10 seed02:condition_kc2_ng3
  seed02:condition_kc2_ng2_coupled
  seed02:condition_kc2_ng3_coupled
  seed02:condition_kc2_ng5_coupled
)
# Main paper candidates run first; naive seed01 is the last missing control.
EXPECTED_NEW_CELLS=(
  seed01:condition_kc2_ng3_coupled seed03:condition_kc2_ng3_coupled
  seed01:condition_kc2_ng5_coupled seed03:condition_kc2_ng5_coupled
  seed01:condition_kc2_ng2_coupled seed03:condition_kc2_ng2_coupled
  seed01:condition_kc2_ng3 seed03:condition_kc2_ng3
  seed01:condition_kc2_ng10 seed03:condition_kc2_ng10
  seed01:naive_nfe3
)

mkdir -p "$CAMPAIGN/metadata" "$CAMPAIGN/logs" "$STORAGE/locks"
export PYTHONPATH="$ROOT:$UPSTREAM:$LIBERO_ROOT:${PYTHONPATH:-}"
export SIMVLA_UPSTREAM_ROOT="$UPSTREAM"

legacy_source() {
  local seed=$1 row=$2
  case "$seed:$row" in
    seed01:full_nfe10)
      echo "$GENERATION_EXP/online/step_010000_long500_egl_paired_v1/baseline_k1" ;;
    seed01:generation_ng3)
      echo "$GENERATION_EXP/online/step_030000_long500_egl_paired_v1/generation_ng3" ;;
    seed02:full_nfe10)
      echo "$GENERATION_EXP/online/step_030000_long500_egl_seed02_v1/baseline_k1" ;;
    seed02:generation_ng3)
      echo "$GENERATION_EXP/online/step_030000_long500_egl_seed02_v1/generation_ng3" ;;
    seed03:full_nfe10)
      echo "$GENERATION_EXP/online/step_030000_long500_egl_seed03_v1/baseline_k1" ;;
    seed03:generation_ng3)
      echo "$GENERATION_EXP/online/step_030000_long500_egl_seed03_v1/generation_ng3" ;;
    *) return 1 ;;
  esac
}

known_root() {
  local seed=$1 row=$2
  case "$seed:$row" in
    seed01:full_nfe10|seed01:generation_ng3|seed02:full_nfe10|seed02:generation_ng3|seed03:full_nfe10|seed03:generation_ng3)
      echo "$CAMPAIGN/reused/$seed/$row" ;;
    seed02:naive_nfe3)
      echo "$STORAGE/results/simvla/generation_control/naive_confirmatory_v1/seed02/naive_nfe3/merged" ;;
    seed03:naive_nfe3)
      echo "$STORAGE/results/simvla/generation_control/naive_confirmatory_v1/seed03/naive_nfe3/merged" ;;
    seed02:condition_kc2_ng10)
      echo "$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1/condition_kc2_ng10/merged" ;;
    seed02:condition_kc2_ng3)
      echo "$STORAGE/results/simvla/fixed_2x2/kc2_ng3_seed02_v1/condition_kc2_ng3/merged" ;;
    seed02:condition_kc2_ng2_coupled)
      echo "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/rows/condition_kc2_ng2_coupled/merged" ;;
    seed02:condition_kc2_ng3_coupled)
      echo "$STORAGE/results/simvla/coupled_condition_generation/kc2_ng3_real_cj_projection10k_seed02_v1/online/condition_kc2_ng3_coupled/merged" ;;
    seed02:condition_kc2_ng5_coupled)
      echo "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/rows/condition_kc2_ng5_coupled/merged" ;;
    *) echo "$CAMPAIGN/rows/$seed/$row/merged" ;;
  esac
}

cell_root_args() {
  local seed row
  for seed in seed01 seed02 seed03; do
    for row in "${ROWS[@]}"; do
      printf '%s\0%s\0' --cell-root "$seed:$row=$(known_root "$seed" "$row")"
    done
  done
}

manifest_args() {
  local seed
  for seed in seed01 seed02 seed03; do
    printf '%s\0%s\0' --manifest-sha256 "$seed=${MANIFEST_SHA[$seed]}"
  done
}

paper_followup_args() {
  local value
  while IFS= read -r -d '' value; do printf '%s\0' "$value"; done < <(cell_root_args)
  while IFS= read -r -d '' value; do printf '%s\0' "$value"; done < <(manifest_args)
}

run_plan() {
  local args=() value
  while IFS= read -r -d '' value; do args+=("$value"); done < <(paper_followup_args)
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_followup \
    plan --output "$PLAN" "${args[@]}"
}

validate_cell() {
  local seed=$1 row=$2
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_followup \
    validate-cell --seed "$seed" --row "$row" --root "$(known_root "$seed" "$row")" \
    --manifest-sha256 "${MANIFEST_SHA[$seed]}" >/dev/null
}

archive_path() {
  local path=$1
  [[ -e "$path" ]] || return 0
  local destination="${path}.failed_$(date +%Y%m%d_%H%M%S)"
  mv "$path" "$destination"
  echo "ARCHIVED source=$path destination=$destination"
}

record_failure() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" >> "$FAILURES"
}

write_running_status() {
  printf 'verdict=PAPER_FOLLOWUP_RUNNING\nexit_code=pending\nstage=%s\nplan=%s\n' \
    "$1" "$PLAN" > "$STATUS"
}

handle_signal() {
  local rc=$1 signal=$2
  printf 'verdict=PAPER_FOLLOWUP_INTERRUPTED\nexit_code=%s\nsignal=%s\nstage=%s\nplan=%s\n' \
    "$rc" "$signal" "$CURRENT_STAGE" "$PLAN" > "$STATUS"
  echo "PAPER_FOLLOWUP_INTERRUPTED signal=$signal stage=$CURRENT_STAGE" >&2
  exit "$rc"
}

trap 'handle_signal 130 INT' INT
trap 'handle_signal 143 TERM' TERM

coupled_checkpoint_for_row() {
  case "$1" in
    condition_kc2_ng2_coupled)
      echo "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/coupled_artifacts/kc2_ng2/train/projection_10k/checkpoints/coupled_generation_step_010000.pt" ;;
    condition_kc2_ng3_coupled)
      echo "$STORAGE/results/simvla/coupled_condition_generation/kc2_ng3_real_cj_projection10k_seed02_v1/train/projection_10k/checkpoints/coupled_generation_step_010000.pt" ;;
    condition_kc2_ng5_coupled)
      echo "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/coupled_artifacts/kc2_ng5/train/projection_10k/checkpoints/coupled_generation_step_010000.pt" ;;
    *) return 1 ;;
  esac
}

validate_external_evidence() {
  "$PYTHON" - "$NONLONG" "$MECHANICAL" <<'PY'
import json, sys
expected = {
    sys.argv[1]: "PAPER_SELECTED_MATRIX_COMPLETE",
    sys.argv[2]: "MECHANICAL_CONTROL_COMPARISON_COMPLETE",
}
for path, verdict in expected.items():
    payload = json.load(open(path, encoding="utf-8"))
    assert payload.get("verdict") == verdict, (path, payload.get("verdict"), verdict)
print("EXTERNAL_EVIDENCE_PASS")
PY
}

validate_parity_gates() {
  "$PYTHON" - \
    "${PARITY_GATES[seed01]}" "${MANIFEST_SHA[seed01]}" \
    "${PARITY_GATES[seed02]}" "${MANIFEST_SHA[seed02]}" \
    "${PARITY_GATES[seed03]}" "${MANIFEST_SHA[seed03]}" <<'PY'
import json, sys
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import (
    FROZEN_CONDITION_CHECKPOINT_SHA256,
    FROZEN_CONDITION_SOURCE_SHA256,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_CHECKPOINT_SHA256,
    FROZEN_GENERATION_SOURCE_SHA256,
)
for index in range(1, len(sys.argv), 2):
    payload = json.load(open(sys.argv[index], encoding="utf-8"))
    expected_manifest = sys.argv[index + 1]
    checks = {
        "verdict": payload.get("verdict") == "FIXED_2X2_PARITY_PASS",
        "manifest": payload.get("manifest_sha256") == expected_manifest,
        "condition_source": payload.get("condition_source_combined_sha256") == FROZEN_CONDITION_SOURCE_SHA256,
        "generation_source": payload.get("generation_source_combined_sha256") == FROZEN_GENERATION_SOURCE_SHA256,
        "condition_checkpoint": payload.get("condition_checkpoint_sha256") == FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint": payload.get("generation_checkpoint_sha256") == FROZEN_GENERATION_CHECKPOINT_SHA256,
    }
    assert all(checks.values()), (sys.argv[index], checks)
print("THREE_SEED_PARITY_GATES_PASS")
PY
}

validate_coupled_artifacts() {
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_grid \
    validate-coupled --k-c 2 --n-g 2 \
    --train-root "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/coupled_artifacts/kc2_ng2/train/projection_10k" \
    --offline-root "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/coupled_artifacts/kc2_ng2/offline/projection_10k_512" >/dev/null || return 1
  "$PYTHON" - \
    "$STORAGE/results/simvla/coupled_condition_generation/kc2_ng3_real_cj_projection10k_seed02_v1" <<'PY' || return 1
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
train = root / "train" / "projection_10k"
checkpoint = train / "checkpoints" / "coupled_generation_step_010000.pt"
config = json.load(open(train / "training_config.json", encoding="utf-8"))
summary = json.load(open(train / "run_summary.json", encoding="utf-8"))
offline = json.load(open(root / "offline" / "projection_10k_512" / "offline_screen.json", encoding="utf-8"))
online = json.load(open(root / "online" / "condition_kc2_ng3_coupled" / "merged" / "row_summary.json", encoding="utf-8"))
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
checks = {
    "train_k_c": int(config.get("k_c", -1)) == 2,
    "train_n_g": int(config.get("n_g", -1)) == 3,
    "projection_parameters": int(config.get("projection_audit", {}).get("trainable_parameters", -1)) == 16_384,
    "train_verdict": summary.get("verdict") == "COUPLED_PROJECTION_TRAINING_COMPLETE",
    "optimizer_step": int(summary.get("optimizer_step", -1)) == 10_000,
    "offline_verdict": offline.get("verdict") == "COUPLED_OFFLINE_INTEGRITY_PASS",
    "offline_queries": int(offline.get("queries", -1)) == 512,
    "offline_checks": bool(offline.get("checks")) and all(offline["checks"].values()),
    "projection_state": offline.get("projection_only_state_audit", {}).get("verdict") == "PROJECTION_ONLY_STATE_PASS",
    "online_row": online.get("row") == "condition_kc2_ng3_coupled",
    "online_checkpoint_sha256": online.get("generation_checkpoint_sha256") == digest,
}
assert all(checks.values()), checks
print("LEGACY_KC2_NG3_COUPLED_ARTIFACT_PASS", digest)
PY
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_grid \
    validate-coupled --k-c 2 --n-g 5 \
    --train-root "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/coupled_artifacts/kc2_ng5/train/projection_10k" \
    --offline-root "$STORAGE/results/simvla/paper_grid/seed02_long500_egl_v1/coupled_artifacts/kc2_ng5/offline/projection_10k_512" >/dev/null
}

validate_frozen_coupled_checkpoints() {
  "$PYTHON" - \
    "$BUNDLE/checkpoint/generation_step_030000.pt" \
    "$CONDITION_CHECKPOINT" \
    "$BUNDLE/norm/libero_norm_official_32700d0.json" \
    "$BUNDLE/exact_cache_contract" \
    "condition_kc2_ng2_coupled=$(coupled_checkpoint_for_row condition_kc2_ng2_coupled)" \
    "condition_kc2_ng3_coupled=$(coupled_checkpoint_for_row condition_kc2_ng3_coupled)" \
    "condition_kc2_ng5_coupled=$(coupled_checkpoint_for_row condition_kc2_ng5_coupled)" <<'PY' || return 1
import sys

from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    audit_projection_only_state,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock import (
    verify_frozen_coupled_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)

parent_path, condition_path, norm_path, cache_path = sys.argv[1:5]
parent, _ = load_generation_checkpoint(parent_path, device="cpu")
for item in sys.argv[5:]:
    row, checkpoint_path = item.split("=", 1)
    updater, payload = load_generation_checkpoint(checkpoint_path, device="cpu")
    report = verify_frozen_coupled_checkpoint(
        checkpoint_path,
        payload,
        row=row,
        parent_generation_checkpoint=parent_path,
        condition_checkpoint=condition_path,
        norm_stats=norm_path,
        exact_cache=cache_path,
    )
    assert report["verdict"] == "FROZEN_COUPLED_CHECKPOINT_PASS", report
    state = audit_projection_only_state(parent, updater)
    assert state["verdict"] == "PROJECTION_ONLY_STATE_PASS", (row, state)
    print(
        "FROZEN_COUPLED_CHECKPOINT_PASS",
        row,
        report["observed"]["checkpoint_sha256"],
    )
PY
}

static_preflight() {
  CURRENT_STAGE=static_preflight
  local command required seed free_bytes
  for command in bash df flock git grep nvidia-smi tee; do
    command -v "$command" >/dev/null || {
      echo "Missing required command: $command" >&2
      return 1
    }
  done
  for required in \
    "$PYTHON" "$UPSTREAM" "$LIBERO_ROOT" "$LIBERO_CONFIG" \
    "$CACHE/manifest.json" "$CONDITION_CHECKPOINT" "$BASE_SOURCE_LOCK" \
    "$BUNDLE/checkpoint/generation_step_030000.pt" \
    "$BUNDLE/norm/libero_norm_official_32700d0.json" \
    "$BUNDLE/transfer_manifest.json" "$NONLONG" "$MECHANICAL"; do
    [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; return 1; }
  done
  for seed in seed01 seed02 seed03; do
    [[ -f "${MANIFESTS[$seed]}" ]] || { echo "Missing manifest: ${MANIFESTS[$seed]}" >&2; return 1; }
    [[ -f "${PARITY_GATES[$seed]}" ]] || { echo "Missing parity gate: ${PARITY_GATES[$seed]}" >&2; return 1; }
  done
  [[ "$GPU" == 0 ]] || { echo "rb2 follow-up is locked to physical GPU 0" >&2; return 1; }
  [[ "$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)" == \
    exp/simvla-paper-grid-seed02-rb2-20260828 ]] || {
      echo "Unexpected follow-up branch" >&2; return 1;
    }
  [[ -z "$(git -C "$ROOT" status --porcelain)" ]] || {
    echo "Follow-up worktree is dirty" >&2; return 1;
  }
  "$PYTHON" - \
    "${MANIFESTS[seed01]}" "${MANIFEST_SHA[seed01]}" \
    "${MANIFESTS[seed02]}" "${MANIFEST_SHA[seed02]}" \
    "${MANIFESTS[seed03]}" "${MANIFEST_SHA[seed03]}" <<'PY' || return 1
import json, sys
for index in range(1, len(sys.argv), 2):
    payload = json.load(open(sys.argv[index], encoding="utf-8"))
    assert payload["manifest_sha256"] == sys.argv[index + 1]
    assert payload["suite"] == "libero_10"
    assert int(payload["trials_per_task"]) == 50
print("THREE_MANIFEST_CONTRACT_PASS")
PY
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
  validate_parity_gates || return 1
  validate_coupled_artifacts || return 1
  validate_frozen_coupled_checkpoints || return 1
  validate_external_evidence || return 1
  "$PYTHON" -m compileall -q \
    "$ROOT/architectures/simvla/adapters/latentloop/efficient_multirate" \
    "$ROOT/architectures/simvla/wrappers" || return 1
  bash -n "$ROOT/architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh" || return 1
  bash -n "$ROOT/architectures/simvla/wrappers/run_paper_followup_rb2.sh" || return 1
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest -q \
    "$ROOT/tests/simvla_fixed_2x2/test_coupled_condition_generation.py" \
    "$ROOT/tests/simvla_fixed_2x2/test_paper_followup.py" \
    "$ROOT/tests/simvla_fixed_2x2/test_paper_grid.py" \
    "$ROOT/tests/simvla_fixed_2x2/test_legacy_generation_row_materialization.py" || return 1
  free_bytes=$(df -PB1 "$STORAGE" | awk 'NR==2 {print $4}') || return 1
  ((free_bytes >= 100 * 1024 * 1024 * 1024)) || {
    echo "Less than 100 GiB free under storage root" >&2; return 1;
  }
  echo "STATIC_PREFLIGHT_PASS"
}

prepare_runtime_provenance() {
  CURRENT_STAGE=prepare_runtime_provenance
  if [[ -d "$PROVENANCE" ]]; then
    if "$PYTHON" - "$PROVENANCE/fixed_eval_source_lock.json" "$ROOT" <<'PY'
import json, subprocess, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
head = subprocess.check_output(["git", "-C", sys.argv[2], "rev-parse", "HEAD"], text=True).strip()
if payload.get("root_commit") != head:
    print(f"PROVENANCE_STALE observed={payload.get('root_commit')} expected={head}")
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
import sys
from pathlib import Path
from types import SimpleNamespace
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import _verify_provenance
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import atomic_write_json
args = SimpleNamespace(
    bundle_root=sys.argv[1], condition_checkpoint=sys.argv[2],
    fixed_2x2_source_lock=sys.argv[3], control_manifest=sys.argv[4],
    classification="RB2_CONFIRMATORY_EGL",
)
report = _verify_provenance(args)
atomic_write_json(Path(sys.argv[5]), report)
assert report["verdict"] == "FROZEN_PROVENANCE_PASS"
assert report["paper_runtime_match"] is True
print("RUNTIME_PROVENANCE_PASS")
PY
}

prepare_compatibility_rows() {
  CURRENT_STAGE=prepare_compatibility_rows
  local seed row output source
  for seed in seed01 seed02 seed03; do
    for row in full_nfe10 generation_ng3; do
      output=$(known_root "$seed" "$row")
      if validate_cell "$seed" "$row" 2>/dev/null; then
        echo "COMPATIBILITY_REUSE seed=$seed row=$row"
        continue
      fi
      archive_path "$output"
      source=$(legacy_source "$seed" "$row") || return 1
      "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_aggregate \
        materialize-legacy-row --source "$source" --output "$output" \
        --row "$row" --inference-seed "$seed" \
        --classification RB2_CONFIRMATORY_EGL \
        --expected-manifest-sha256 "${MANIFEST_SHA[$seed]}" \
        --paper-runtime-match >/dev/null || return 1
      validate_cell "$seed" "$row" || return 1
      echo "COMPATIBILITY_READY seed=$seed row=$row"
    done
  done
}

validate_reused_cells() {
  CURRENT_STAGE=validate_reused_cells
  local cell seed row
  for cell in "${REUSED_CELLS[@]}"; do
    seed=${cell%%:*}
    row=${cell#*:}
    validate_cell "$seed" "$row" || {
      echo "Required reusable cell failed validation: $cell" >&2
      return 1
    }
  done
  echo "REUSED_CELL_CONTRACT_PASS count=${#REUSED_CELLS[@]}"
}

validate_plan_scope() {
  "$PYTHON" - "$PLAN" "${EXPECTED_NEW_CELLS[@]}" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
allowed = set(sys.argv[2:])
missing = set(plan["missing_cells"])
assert plan["target_cell_count"] == 24, plan
assert missing <= allowed, (sorted(missing), sorted(allowed))
assert plan["complete_cell_count"] + plan["missing_cell_count"] == 24
print(
    f"FOLLOWUP_PLAN_PASS complete={plan['complete_cell_count']} "
    f"missing={plan['missing_cell_count']} new_episodes={plan['new_episode_count']}"
)
print("missing_cells=" + ",".join(plan["missing_cells"]))
PY
}

gpu_busy() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -q '[0-9]'
}

wait_for_gpu() {
  CURRENT_STAGE=wait_for_gpu0
  local clear_polls=0
  while ((clear_polls < 2)); do
    if gpu_busy; then
      clear_polls=0
      echo "[$(date --iso-8601=seconds)] rb2 GPU0 busy; waiting 60s"
    else
      clear_polls=$((clear_polls + 1))
      echo "[$(date --iso-8601=seconds)] rb2 GPU0 clear poll=$clear_polls/2"
    fi
    ((clear_polls >= 2)) || sleep 60
  done
  echo "GPU0_STABLY_IDLE"
}

wait_for_gpu_available() {
  CURRENT_STAGE=wait_for_gpu0_available
  while gpu_busy; do
    echo "[$(date --iso-8601=seconds)] rb2 GPU0 became busy; waiting 60s"
    sleep 60
  done
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
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_parity \
    --output "$output" --cache "$CACHE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" --sequences 16
}

run_cell_once() {
  local seed=$1 row=$2 output=$3 limit=$4
  local extra=() checkpoint
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
    --row "$row" --output "$output" --manifest "${MANIFESTS[$seed]}" \
    --manifest-sha256 "${MANIFEST_SHA[$seed]}" --bundle-root "$BUNDLE" \
    --condition-checkpoint "$CONDITION_CHECKPOINT" \
    --source-lock "$PROVENANCE/fixed_eval_source_lock.json" \
    --control-manifest "$PROVENANCE/control_manifest.json" \
    --parity-gate "${PARITY_GATES[$seed]}" --physical-gpu-id "$GPU" \
    --classification RB2_CONFIRMATORY_EGL --inference-seed "$seed" \
    --task-ids 0,1,2,3,4,5,6,7,8,9 \
    --episodes-per-task-limit "$limit" "${extra[@]}"
}

valid_smoke() {
  local seed=$1 row=$2 output=$3
  "$PYTHON" - "$output/shard_rank0_tasks_0_9/shard_summary.json" "$seed" "$row" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["row"] == sys.argv[3]
assert payload["inference_seed"] == sys.argv[2]
assert payload["episodes"] == 10
assert payload["verdict"].endswith("_SHARD_PASS")
assert payload["all_episode_counter_gates_pass"] is True
PY
}

recover_existing_cell() {
  local seed=$1 row=$2 output=$3
  local shard=$output/shard_rank0_tasks_0_9 merged=$output/merged
  local extra=() checkpoint
  [[ -d "$shard" ]] || return 1
  if checkpoint=$(coupled_checkpoint_for_row "$row"); then
    [[ -f "$checkpoint" ]] || return 1
    extra=(--generation-checkpoint "$checkpoint")
  fi
  mkdir -p "$output/logs"
  CURRENT_STAGE="cell_recovery_${seed}_${row}"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.row_postprocess_recovery \
    --row "$row" --shard "$shard" --merged "$merged" \
    --expected-manifest-sha256 "${MANIFEST_SHA[$seed]}" "${extra[@]}" \
    2>&1 | tee "$output/logs/restart_recovery.log"
  local rc=${PIPESTATUS[0]}
  ((rc == 0)) && validate_cell "$seed" "$row"
}

smoke_cell() {
  local seed=$1 row=$2 output=$CAMPAIGN/smoke/$1/$2
  local attempt rc
  if valid_smoke "$seed" "$row" "$output" 2>/dev/null; then
    echo "CELL_SMOKE_REUSE seed=$seed row=$row"
    return 0
  fi
  for attempt in 1 2; do
    archive_path "$output"
    archive_path "${output}.egl_preflight.json"
    wait_for_gpu_available
    CURRENT_STAGE="cell_smoke_${seed}_${row}_attempt${attempt}"
    run_cell_once "$seed" "$row" "$output" 1 \
      2>&1 | tee "$CAMPAIGN/logs/smoke_${seed}_${row}_attempt${attempt}.log"
    rc=${PIPESTATUS[0]}
    if ((rc == 0)) && valid_smoke "$seed" "$row" "$output"; then
      return 0
    fi
  done
  record_failure "cell_smoke" "$seed:$row"
  return 1
}

evaluate_cell() {
  local seed=$1 row=$2 output=$CAMPAIGN/rows/$1/$2
  local attempt rc
  if validate_cell "$seed" "$row" 2>/dev/null; then
    echo "CELL_ALREADY_COMPLETE seed=$seed row=$row root=$(known_root "$seed" "$row")"
    return 0
  fi
  if recover_existing_cell "$seed" "$row" "$output" 2>/dev/null; then
    echo "CELL_RECOVERED_ON_RESTART seed=$seed row=$row"
    return 0
  fi
  if ! smoke_cell "$seed" "$row"; then
    echo "CELL_CONTINUING_AFTER_SMOKE_FAILURE seed=$seed row=$row" >&2
    return 1
  fi
  for attempt in 1 2; do
    archive_path "$output"
    archive_path "${output}.egl_preflight.json"
    wait_for_gpu_available
    CURRENT_STAGE="cell_full_${seed}_${row}_attempt${attempt}"
    run_cell_once "$seed" "$row" "$output" 50 \
      2>&1 | tee "$CAMPAIGN/logs/full_${seed}_${row}_attempt${attempt}.log"
    rc=${PIPESTATUS[0]}
    if ((rc == 0)) && validate_cell "$seed" "$row"; then
      echo "CELL_COMPLETE seed=$seed row=$row root=$(known_root "$seed" "$row")"
      return 0
    fi
  done
  record_failure "cell_full" "$seed:$row"
  echo "CELL_CONTINUING_AFTER_FULL_FAILURE seed=$seed row=$row" >&2
  return 1
}

aggregate_followup() {
  CURRENT_STAGE=aggregate_followup
  local args=() value
  while IFS= read -r -d '' value; do args+=("$value"); done < <(paper_followup_args)
  "$PYTHON" -m architectures.simvla.adapters.latentloop.efficient_multirate.paper_followup \
    aggregate --output "$CAMPAIGN/aggregate" "${args[@]}" \
    --external-artifact "nonlong_seed01=$NONLONG" \
    --external-artifact "mechanical_controls_seed02=$MECHANICAL"
}

run_all() {
  exec 9>"$LOCK"
  flock -n 9 || { echo "Another follow-up launcher holds $LOCK" >&2; return 1; }
  static_preflight || return 1
  prepare_runtime_provenance || return 1
  verify_runtime_provenance || return 1
  prepare_compatibility_rows || return 1
  validate_reused_cells || return 1
  run_plan >/dev/null || return 1
  validate_plan_scope || return 1
  if [[ "$MODE" == --preflight || "$MODE" == --dry-run ]]; then
    printf 'verdict=PAPER_FOLLOWUP_PREFLIGHT_PASS\nexit_code=0\nplan=%s\n' \
      "$PLAN" > "$STATUS"
    return 0
  fi

  write_running_status waiting_for_gpu0
  wait_for_gpu
  condition_code_parity || return 1
  local cell seed row
  for cell in "${EXPECTED_NEW_CELLS[@]}"; do
    seed=${cell%%:*}
    row=${cell#*:}
    if validate_cell "$seed" "$row" 2>/dev/null; then
      echo "CELL_SKIP_VALID seed=$seed row=$row"
      continue
    fi
    write_running_status "$seed:$row"
    if ! evaluate_cell "$seed" "$row"; then
      printf 'verdict=PAPER_FOLLOWUP_PIPELINE_FAILED\nexit_code=1\nstage=%s\ncell=%s:%s\nplan=%s\nfailures=%s\n' \
        "$CURRENT_STAGE" "$seed" "$row" "$PLAN" "$FAILURES" > "$STATUS"
      echo "PAPER_FOLLOWUP_ABORTED_AFTER_CELL_FAILURE seed=$seed row=$row" >&2
      return 1
    fi
    run_plan >/dev/null || return 1
  done

  if aggregate_followup; then
    printf 'verdict=PAPER_FOLLOWUP_COMPLETE\nexit_code=0\nresult=%s\n' \
      "$CAMPAIGN/aggregate/paper_followup_three_seed_summary.json" > "$STATUS"
    echo "PAPER_FOLLOWUP_COMPLETE result=$CAMPAIGN/aggregate/paper_followup_three_seed_summary.json"
    return 0
  fi
  run_plan >/dev/null || true
  printf 'verdict=PAPER_FOLLOWUP_INCOMPLETE\nexit_code=1\nplan=%s\nfailures=%s\n' \
    "$PLAN" "$FAILURES" > "$STATUS"
  echo "PAPER_FOLLOWUP_INCOMPLETE plan=$PLAN failures=$FAILURES" >&2
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
if ((rc != 0)) && ! grep -Eq \
  '^verdict=PAPER_FOLLOWUP_(INTERRUPTED|PIPELINE_FAILED)$' "$STATUS" 2>/dev/null; then
  printf 'verdict=PAPER_FOLLOWUP_PIPELINE_FAILED\nexit_code=%s\nstage=%s\nplan=%s\n' \
    "$rc" "$CURRENT_STAGE" "$PLAN" > "$STATUS"
fi
exit "$rc"
