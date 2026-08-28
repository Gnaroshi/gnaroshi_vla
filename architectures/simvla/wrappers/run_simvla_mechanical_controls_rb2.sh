#!/usr/bin/env bash
# Run paired LIBERO-Long mechanism controls for K_C=2, N_G=3 on rb2.

set -uo pipefail

MODE=${1:---all}
case "${MODE}" in
  --all|--preflight) ;;
  *) echo "usage: $0 [--all|--preflight]" >&2; exit 2 ;;
esac

if [[ "${SIMVLA_MECHANICAL_CONTROL_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_MECHANICAL_CONTROL_RUN=1" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STORAGE=${SIMVLA_STORAGE_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
PYTHON=${SIMVLA_PYTHON:-${STORAGE}/envs/simvla/libero_mujoco237/bin/python}
LIBERO_ROOT=${STORAGE}/datasets/LIBERO
LIBERO_CONFIG=${STORAGE}/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config
BUNDLE=${STORAGE}/artifacts/simvla/generation_eval_bundle_20260824_v1
CONDITION_CHECKPOINT=${STORAGE}/artifacts/simvla/fixed_2x2_inputs_v1/condition/native_v0_step_150000.pt
SOURCE_LOCK=${SIMVLA_MECHANICAL_SOURCE_LOCK:-${STORAGE}/artifacts/simvla/mechanical_controls_kc2_ng3_seed02_v1/source_lock.json}
MANIFEST=${STORAGE}/results/simvla/latentloop/generation_loop_ng2_rb2_v1/online/step_030000_long500_egl_seed02_v1/episode_manifest.json
PARITY_GATE=${STORAGE}/results/simvla/fixed_2x2/kc2_ng3_seed02_v1/gates/fixed_2x2_parity.json
RESULT=${SIMVLA_MECHANICAL_OUTPUT:-${STORAGE}/results/simvla/mechanical_controls/kc2_ng3_long500_seed02_v1}
STATUS=${RESULT}/pipeline.status
LOG_ROOT=${RESULT}/logs
FAILED_ROOT=${RESULT}/failed_attempts
GPU_ID=${SIMVLA_RB2_GPU_ID:-0}
MINIMUM_FREE_MIB=${SIMVLA_MINIMUM_FREE_MIB:-28000}
GPU_WAIT_SECONDS=${SIMVLA_GPU_WAIT_SECONDS:-120}
EXPECTED_COMMIT=${SIMVLA_MECHANICAL_COMMIT:?Set SIMVLA_MECHANICAL_COMMIT to the reviewed branch commit}

ROWS=(
  condition_kc2_ng3
  mechanical_hold_condition_kc2_ng3
  mechanical_native_chunk_replay_kc2_ng3
  mechanical_hold_action_kc2_ng3
  mechanical_no_observation_kc2_ng3
)

BASE_RESULT=${STORAGE}/results/simvla/fixed_2x2/kc2_ng3_seed02_v1
REFERENCE_FULL=${BASE_RESULT}/compatibility_from_generation_v1/full_nfe10
REFERENCE_GENERATION=${BASE_RESULT}/compatibility_from_generation_v1/generation_ng3
REFERENCE_CONDITION=${BASE_RESULT}/condition_kc2_ng10/merged
REFERENCE_NAIVE=${STORAGE}/results/simvla/generation_control/naive_confirmatory_v1/seed02/naive_nfe3/merged

mkdir -p "${LOG_ROOT}" "${FAILED_ROOT}"

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${LOG_ROOT}/pipeline.log"; }

on_signal() {
  printf 'MECHANICAL_CONTROL_INTERRUPTED signal=%s time=%s\n' "$1" "$(timestamp)" > "${STATUS}"
  log "interrupted signal=$1; completed rows remain reusable"
  exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

require_file() { [[ -f "$1" ]] || { log "PREFLIGHT_FAIL missing_file=$1"; return 1; }; }
require_dir() { [[ -d "$1" ]] || { log "PREFLIGHT_FAIL missing_directory=$1"; return 1; }; }

manifest_sha() {
  "${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "${MANIFEST}"
}

source_audit() {
  local observed lock_commit
  require_file "${PYTHON}" || return 1
  require_dir "${ROOT}" || return 1
  require_dir "${UPSTREAM}" || return 1
  require_dir "${LIBERO_ROOT}" || return 1
  require_dir "${LIBERO_CONFIG}" || return 1
  require_file "${CONDITION_CHECKPOINT}" || return 1
  require_file "${SOURCE_LOCK}" || return 1
  require_file "${MANIFEST}" || return 1
  require_file "${PARITY_GATE}" || return 1
  require_file "${BUNDLE}/transfer_manifest.json" || return 1
  observed=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null) || return 1
  [[ "${observed}" == "${EXPECTED_COMMIT}" ]] || {
    log "PREFLIGHT_FAIL commit=${observed} expected=${EXPECTED_COMMIT}"
    return 1
  }
  lock_commit=$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["root_commit"])' "${SOURCE_LOCK}") || return 1
  [[ "${lock_commit}" == "${EXPECTED_COMMIT}" ]] || {
    log "PREFLIGHT_FAIL source_lock_commit=${lock_commit} expected=${EXPECTED_COMMIT}"
    return 1
  }
  if ! git -C "${ROOT}" diff --quiet -- \
      architectures/simvla/adapters/latentloop/efficient_multirate \
      architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh \
      architectures/simvla/wrappers/run_simvla_mechanical_controls_rb2.sh; then
    log "PREFLIGHT_FAIL tracked_runtime_files_dirty"
    return 1
  fi
  if ! git -C "${ROOT}" diff --cached --quiet -- \
      architectures/simvla/adapters/latentloop/efficient_multirate \
      architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh \
      architectures/simvla/wrappers/run_simvla_mechanical_controls_rb2.sh; then
    log "PREFLIGHT_FAIL staged_runtime_files_dirty"
    return 1
  fi
  if ! LIBERO_CONFIG_PATH="${LIBERO_CONFIG}" \
    PYTHONPATH="${ROOT}:${UPSTREAM}:${LIBERO_ROOT}" "${PYTHON}" - <<'PY'
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import MECHANICAL_CONTROL_ROWS
from libero.libero import benchmark

assert len(MECHANICAL_CONTROL_ROWS) == 4
assert "libero_10" in benchmark.get_benchmark_dict()
print("MECHANICAL_CONTROL_IMPORT_PASS")
PY
  then
    log "PREFLIGHT_FAIL runtime_import"
    return 1
  fi
  log "source_artifact_import_audit_pass commit=${observed}"
}

set_runtime() {
  local assignment
  export SIMVLA_FIXED_2X2_ROOT=${ROOT}
  export SIMVLA_FIXED_2X2_PYTHON=${PYTHON}
  export SIMVLA_UPSTREAM_ROOT=${UPSTREAM}
  export SIMVLA_LIBERO_ROOT=${LIBERO_ROOT}
  export LIBERO_CONFIG_PATH=${LIBERO_CONFIG}
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
  unset GALLIUM_DRIVER LIBGL_ALWAYS_SOFTWARE LP_NUM_THREADS EGL_DEVICE_ID
  while IFS= read -r assignment; do
    [[ -n "${assignment}" ]] && export "${assignment}"
  done < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import json, shlex, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
for name, value in d["renderer"].items():
    print(f"{name}={shlex.quote(str(value))}")
PY
  )
}

wait_for_isolated_gpu() {
  local free_mib pids
  while true; do
    free_mib=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    pids=$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MINIMUM_FREE_MIB )) && [[ -z "${pids}" ]]; then
      return 0
    fi
    log "gpu_wait gpu=${GPU_ID} free_mib=${free_mib:-unknown} external_pids=${pids//$'\n'/,} retry_seconds=${GPU_WAIT_SECONDS}"
    sleep "${GPU_WAIT_SECONDS}"
  done
}

row_complete() {
  local row=$1 root=$2 expected
  expected=$(manifest_sha) || return 1
  "${PYTHON}" - "${row}" "${root}" "${expected}" <<'PY'
import csv, json, pathlib, sys
row, root, expected = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
summary = json.load(open(root / "merged" / "row_summary.json", encoding="utf-8"))
with open(root / "merged" / "episode_metrics.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
keys = {(int(item["task_id"]), int(item["trial_id"])) for item in rows}
assert summary["row"] == row
assert summary["manifest_sha256"] == expected
assert summary["verdict"].endswith("_ROW_PASS")
assert len(rows) == len(keys) == 500
assert keys == {(task, trial) for task in range(10) for trial in range(50)}
PY
}

quarantine() {
  local path=$1 label=$2
  [[ -e "${path}" ]] || return 0
  local destination=${FAILED_ROOT}/${label}_$(date +%Y%m%d_%H%M%S)_$$
  mv "${path}" "${destination}"
  log "quarantined label=${label} destination=${destination}"
}

runtime_smoke() {
  local row output rc expected
  expected=$(manifest_sha) || return 1
  for row in "${ROWS[@]}"; do
    output=${RESULT}/runtime_smoke/${row}
    quarantine "${output}" "runtime_smoke_${row}"
    quarantine "${output}.egl_preflight.json" "runtime_smoke_${row}_preflight"
    wait_for_isolated_gpu || return 1
    set_runtime || return 1
    export SIMVLA_FIXED_2X2_RUN=1
    log "runtime_smoke_start row=${row} episodes=1"
    bash "${ROOT}/architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh" \
      --row "${row}" \
      --output "${output}" \
      --manifest "${MANIFEST}" \
      --manifest-sha256 "${expected}" \
      --bundle-root "${BUNDLE}" \
      --condition-checkpoint "${CONDITION_CHECKPOINT}" \
      --source-lock "${SOURCE_LOCK}" \
      --control-manifest "${BUNDLE}/transfer_manifest.json" \
      --parity-gate "${PARITY_GATE}" \
      --physical-gpu-id "${GPU_ID}" \
      --classification RB2_CONFIRMATORY_EGL \
      --inference-seed seed02 \
      --task-ids 0 \
      --episodes-per-task-limit 1 \
      2>&1 | tee -a "${LOG_ROOT}/runtime_smoke_${row}.log"
    rc=${PIPESTATUS[0]}
    if ((rc != 0)); then
      log "runtime_smoke_failed row=${row} rc=${rc}"
      return "${rc}"
    fi
    log "runtime_smoke_complete row=${row}"
  done
  log "runtime_smoke_suite_complete rows=${#ROWS[@]}"
}

run_row() {
  local row=$1 cell=${RESULT}/rows/${row} expected attempt rc=1
  expected=$(manifest_sha) || return 1
  if row_complete "${row}" "${cell}"; then
    log "row_skip_complete row=${row}"
    return 0
  fi
  for attempt in 1 2; do
    quarantine "${cell}" "${row}_attempt${attempt}"
    quarantine "${cell}.egl_preflight.json" "${row}_preflight_attempt${attempt}"
    wait_for_isolated_gpu || return 1
    set_runtime || return 1
    export SIMVLA_FIXED_2X2_RUN=1
    log "row_start row=${row} attempt=${attempt} episodes=500"
    bash "${ROOT}/architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh" \
      --row "${row}" \
      --output "${cell}" \
      --manifest "${MANIFEST}" \
      --manifest-sha256 "${expected}" \
      --bundle-root "${BUNDLE}" \
      --condition-checkpoint "${CONDITION_CHECKPOINT}" \
      --source-lock "${SOURCE_LOCK}" \
      --control-manifest "${BUNDLE}/transfer_manifest.json" \
      --parity-gate "${PARITY_GATE}" \
      --physical-gpu-id "${GPU_ID}" \
      --classification RB2_CONFIRMATORY_EGL \
      --inference-seed seed02 \
      --task-ids 0,1,2,3,4,5,6,7,8,9 \
      --save-failure-videos \
      2>&1 | tee -a "${LOG_ROOT}/${row}.log"
    rc=${PIPESTATUS[0]}
    if ((rc == 0)) && row_complete "${row}" "${cell}"; then
      log "row_complete row=${row} attempt=${attempt}"
      return 0
    fi
    log "row_attempt_failed row=${row} attempt=${attempt} rc=${rc}"
  done
  return 1
}

aggregate_controls() {
  local expected output=${RESULT}/comparison
  expected=$(manifest_sha) || return 1
  quarantine "${output}" comparison_rebuild
  PYTHONPATH="${ROOT}:${UPSTREAM}:${LIBERO_ROOT}" "${PYTHON}" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.mechanical_control_aggregate \
    --output "${output}" \
    --expected-manifest-sha256 "${expected}" \
    --row "full_nfe10=${REFERENCE_FULL}" \
    --row "generation_ng3=${REFERENCE_GENERATION}" \
    --row "naive_nfe3=${REFERENCE_NAIVE}" \
    --row "condition_kc2_ng10=${REFERENCE_CONDITION}" \
    --row "condition_kc2_ng3=${RESULT}/rows/condition_kc2_ng3/merged" \
    --row "mechanical_hold_condition_kc2_ng3=${RESULT}/rows/mechanical_hold_condition_kc2_ng3/merged" \
    --row "mechanical_native_chunk_replay_kc2_ng3=${RESULT}/rows/mechanical_native_chunk_replay_kc2_ng3/merged" \
    --row "mechanical_hold_action_kc2_ng3=${RESULT}/rows/mechanical_hold_action_kc2_ng3/merged" \
    --row "mechanical_no_observation_kc2_ng3=${RESULT}/rows/mechanical_no_observation_kc2_ng3/merged" \
    2>&1 | tee -a "${LOG_ROOT}/aggregate.log"
  return "${PIPESTATUS[0]}"
}

main() {
  printf 'MECHANICAL_CONTROL_RUNNING mode=%s time=%s\n' "${MODE}" "$(timestamp)" > "${STATUS}"
  log "pipeline_start mode=${MODE} result=${RESULT} gpu=${GPU_ID}"
  source_audit || return 1
  if [[ "${MODE}" == "--preflight" ]]; then
    printf 'MECHANICAL_CONTROL_PREFLIGHT_PASS\n' > "${STATUS}"
    log "preflight_complete"
    return 0
  fi
  runtime_smoke || {
    printf 'MECHANICAL_CONTROL_FAILED stage=runtime_smoke\n' > "${STATUS}"
    return 1
  }
  local row
  for row in "${ROWS[@]}"; do
    run_row "${row}" || {
      printf 'MECHANICAL_CONTROL_FAILED row=%s\n' "${row}" > "${STATUS}"
      return 1
    }
  done
  aggregate_controls || {
    printf 'MECHANICAL_CONTROL_FAILED stage=aggregate\n' > "${STATUS}"
    return 1
  }
  printf 'MECHANICAL_CONTROL_COMPLETE\n' > "${STATUS}"
  log "pipeline_complete summary=${RESULT}/comparison/mechanical_control_summary.json"
}

main
