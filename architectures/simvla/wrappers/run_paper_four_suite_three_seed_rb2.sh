#!/usr/bin/env bash
# Run the frozen SimVLA paper matrix on rb2. The wrapper is resumable and
# preserves independent rows when another row fails.

set -uo pipefail

MODE=${1:---all}
case "${MODE}" in
  --all|--primary-only|--preflight) ;;
  *) echo "usage: $0 [--all|--primary-only|--preflight]" >&2; exit 2 ;;
esac

STORAGE=${SIMVLA_STORAGE_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DRIVER_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
GENERATION_ROOT=${SIMVLA_GENERATION_ROOT:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_generation_loop}
CONTROL_ROOT=${SIMVLA_CONTROL_ROOT:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_generation_control_egl}
FIXED_ROOT=${SIMVLA_FIXED_ROOT:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_action_refresh_3seed_rb2}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
PYTHON=${SIMVLA_PYTHON:-${STORAGE}/envs/simvla/libero_mujoco237/bin/python}
HELPER=${SIMVLA_PAPER_MATRIX_HELPER:-${DRIVER_ROOT}/tools/simvla/paper_suite_matrix.py}
LIBERO_ROOT=${STORAGE}/datasets/LIBERO
LIBERO_CONFIG=${STORAGE}/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config
BUNDLE=${STORAGE}/artifacts/simvla/generation_eval_bundle_20260824_v1
CONDITION_CHECKPOINT=${STORAGE}/artifacts/simvla/fixed_2x2_inputs_v1/condition/native_v0_step_150000.pt
FIXED_LOCK=${STORAGE}/results/simvla/action_equivalent_refresh/three_seed_long500_v1/provenance/fixed_eval_source_lock.json
RESULT=${SIMVLA_PAPER_MATRIX_OUTPUT:-${STORAGE}/results/simvla/paper_four_suite_three_seed_v1}
REGISTRY=${RESULT}/metadata/experiment_registry.json
AUDIT=${RESULT}/metadata/preflight_audit.json
SUMMARY=${RESULT}/summary/selected_matrix_summary.json
LOG_ROOT=${RESULT}/logs
STATUS=${RESULT}/launcher.status
FAILURES=${RESULT}/metadata/failed_cells.tsv
CHECKPOINT=YuankaiLuo/SimVLA-LIBERO
SMOLVLM=HuggingFaceTB/SmolVLM-500M-Instruct
GPU_ID=${SIMVLA_RB2_GPU_ID:-0}
MINIMUM_FREE_MIB=${SIMVLA_MINIMUM_FREE_MIB:-28000}
GPU_WAIT_SECONDS=${SIMVLA_GPU_WAIT_SECONDS:-120}

IFS=',' read -r -a SUITES <<< "${SIMVLA_PAPER_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
IFS=',' read -r -a SEEDS <<< "${SIMVLA_PAPER_SEEDS:-seed01,seed02,seed03}"
IFS=',' read -r -a MATRIX_ROWS <<< "${SIMVLA_PAPER_ROWS:-full_nfe10,generation_ng3,condition_kc2_ng3,condition_kc2_ng10,naive_nfe3}"
IFS=',' read -r -a PRIMARY_ROWS <<< "${SIMVLA_PAPER_PRIMARY_ROWS:-full_nfe10,generation_ng3,condition_kc2_ng3}"
IFS=',' read -r -a CONTROL_ROWS <<< "${SIMVLA_PAPER_CONTROL_ROWS:-condition_kc2_ng10,naive_nfe3}"

SUITES_CSV=$(IFS=,; printf '%s' "${SUITES[*]}")
SEEDS_CSV=$(IFS=,; printf '%s' "${SEEDS[*]}")
ROWS_CSV=$(IFS=,; printf '%s' "${MATRIX_ROWS[*]}")

mkdir -p "${RESULT}/metadata" "${RESULT}/summary" "${LOG_ROOT}" "${RESULT}/failed_attempts"
touch "${FAILURES}"

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${LOG_ROOT}/launcher.log"; }

on_signal() {
  local signal=$1
  printf 'PAPER_MATRIX_INTERRUPTED signal=%s time=%s\n' "${signal}" "$(timestamp)" > "${STATUS}"
  log "interrupted signal=${signal}; completed rows remain reusable"
  exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

require_file() {
  [[ -f "$1" ]] || { log "PREFLIGHT_FAIL missing_file=$1"; return 1; }
}

require_dir() {
  [[ -d "$1" ]] || { log "PREFLIGHT_FAIL missing_directory=$1"; return 1; }
}

long_manifest() {
  case "$1" in
    seed01) printf '%s\n' "${STORAGE}/results/simvla/latentloop/generation_loop_ng2_rb2_v1/online/step_010000_long500_egl_paired_v1/episode_manifest.json" ;;
    seed02) printf '%s\n' "${STORAGE}/results/simvla/latentloop/generation_loop_ng2_rb2_v1/online/step_030000_long500_egl_seed02_v1/episode_manifest.json" ;;
    seed03) printf '%s\n' "${STORAGE}/results/simvla/latentloop/generation_loop_ng2_rb2_v1/online/step_030000_long500_egl_seed03_v1/episode_manifest.json" ;;
    *) return 2 ;;
  esac
}

manifest_path() {
  printf '%s\n' "${RESULT}/manifests/$1/$2/episode_manifest.json"
}

prepare_manifests() {
  local suite seed base output
  for suite in "${SUITES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      base=$(long_manifest "${seed}") || return 1
      output=$(manifest_path "${suite}" "${seed}")
      require_file "${base}" || return 1
      "${PYTHON}" "${HELPER}" prepare-manifest \
        --base-manifest "${base}" \
        --output "${output}" \
        --suite "${suite}" \
        --seed "${seed}" \
        > "${LOG_ROOT}/manifest_${suite}_${seed}.log" 2>&1 || {
          log "PREFLIGHT_FAIL manifest suite=${suite} seed=${seed}"
          return 1
        }
    done
  done
  "${PYTHON}" "${HELPER}" build-registry \
    --result-root "${RESULT}" \
    --storage "${STORAGE}" \
    --suites "${SUITES_CSV}" \
    --seeds "${SEEDS_CSV}" \
    --rows "${ROWS_CSV}" \
    --output "${REGISTRY}" \
    > "${LOG_ROOT}/registry.log" 2>&1 || return 1
  local planned reused
  planned=$("${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["episodes_to_run"])' "${REGISTRY}")
  reused=$("${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["episodes_reused"])' "${REGISTRY}")
  log "manifest_registry_pass new_episodes=${planned} reused_episodes=${reused}"
}

source_audit() {
  require_file "${PYTHON}" || return 1
  require_file "${HELPER}" || return 1
  require_dir "${GENERATION_ROOT}" || return 1
  require_dir "${CONTROL_ROOT}" || return 1
  require_dir "${FIXED_ROOT}" || return 1
  require_dir "${UPSTREAM}" || return 1
  require_dir "${LIBERO_ROOT}" || return 1
  require_dir "${LIBERO_CONFIG}" || return 1
  require_file "${CONDITION_CHECKPOINT}" || return 1
  require_file "${FIXED_LOCK}" || return 1
  command -v nvidia-smi >/dev/null 2>&1 || { log "PREFLIGHT_FAIL nvidia-smi_missing"; return 1; }
  local driver_commit expected_driver_commit
  driver_commit=$(git -C "${DRIVER_ROOT}" rev-parse HEAD 2>/dev/null) || {
    log "PREFLIGHT_FAIL driver_git_commit_unavailable"
    return 1
  }
  expected_driver_commit=${SIMVLA_PAPER_DRIVER_COMMIT:-}
  if [[ -n "${expected_driver_commit}" && "${driver_commit}" != "${expected_driver_commit}" ]]; then
    log "PREFLIGHT_FAIL driver_commit=${driver_commit} expected=${expected_driver_commit}"
    return 1
  fi
  if ! git -C "${DRIVER_ROOT}" diff --quiet -- \
      architectures/simvla/wrappers/run_paper_four_suite_three_seed_rb2.sh \
      tools/simvla/paper_suite_matrix.py; then
    log "PREFLIGHT_FAIL driver_files_have_uncommitted_changes"
    return 1
  fi
  if ! git -C "${DRIVER_ROOT}" diff --cached --quiet -- \
      architectures/simvla/wrappers/run_paper_four_suite_three_seed_rb2.sh \
      tools/simvla/paper_suite_matrix.py; then
    log "PREFLIGHT_FAIL driver_files_have_staged_changes"
    return 1
  fi
  "${PYTHON}" "${HELPER}" audit \
    --generation-root "${GENERATION_ROOT}" \
    --control-root "${CONTROL_ROOT}" \
    --fixed-root "${FIXED_ROOT}" \
    --upstream "${UPSTREAM}" \
    --storage "${STORAGE}" \
    --bundle "${BUNDLE}" \
    --condition-checkpoint "${CONDITION_CHECKPOINT}" \
    --fixed-lock "${FIXED_LOCK}" \
    --helper "${HELPER}" \
    --launcher "$0" \
    --minimum-free-gib 100 \
    --output "${AUDIT}" \
    > "${LOG_ROOT}/source_audit.log" 2>&1 || {
      log "PREFLIGHT_FAIL source_audit=${AUDIT}"
      return 1
    }
  log "source_artifact_runtime_audit_pass"
}

set_runtime() {
  local manifest=$1 assignment
  export CUDA_VISIBLE_DEVICES=${GPU_ID}
  export MUJOCO_EGL_DEVICE_ID=${GPU_ID}
  export SIMVLA_UPSTREAM_ROOT=${UPSTREAM}
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
  done < <("${PYTHON}" "${HELPER}" manifest-env --manifest "${manifest}" --shell)
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

manifest_sha() {
  "${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest_sha256"])' "$1"
}

validate_egl() {
  local path=$1 suite=$2
  "${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["verdict"]=="EGL_PREFLIGHT_PASS"; assert d["physical_gpu_id"]==int(sys.argv[2]); assert d["libero_task"]["suite"]==sys.argv[3]; assert d["libero_reset_pass"] and d["libero_step_pass"]' "${path}" "${GPU_ID}" "${suite}" >/dev/null 2>&1
}

quarantine_path() {
  local path=$1 label=$2
  [[ -e "${path}" ]] || return 0
  local destination=${RESULT}/failed_attempts/${label}_$(date +%Y%m%d_%H%M%S)_$$
  mkdir -p "$(dirname "${destination}")"
  mv "${path}" "${destination}"
  log "quarantined label=${label} destination=${destination}"
}

ensure_egl() {
  local suite=$1 seed=$2 manifest preflight log_file
  manifest=$(manifest_path "${suite}" "${seed}")
  preflight=${RESULT}/gates/${suite}/${seed}/egl_preflight.json
  log_file=${LOG_ROOT}/egl_${suite}_${seed}.log
  set_runtime "${manifest}" || return 1
  if validate_egl "${preflight}" "${suite}"; then
    return 0
  fi
  quarantine_path "${preflight}" "egl_${suite}_${seed}"
  mkdir -p "$(dirname "${preflight}")"
  wait_for_isolated_gpu || return 1
  (cd "${FIXED_ROOT}" && \
    PYTHONPATH="${FIXED_ROOT}:${UPSTREAM}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" "${FIXED_ROOT}/tools/simvla/simvla_egl_preflight.py" \
      --output "${preflight}" \
      --gpu-id "${GPU_ID}" \
      --suite "${suite}" \
      --task-id 0 \
      --environment-seed 7 \
      --resolution 256) \
      > "${log_file}" 2>&1 || return 1
  validate_egl "${preflight}" "${suite}"
}

validate_parity() {
  local path=$1 manifest=$2 expected
  expected=$(manifest_sha "${manifest}") || return 1
  "${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["verdict"]=="FIXED_2X2_PARITY_PASS"; assert d["manifest_sha256"]==sys.argv[2]; assert all(d["checks"].values())' "${path}" "${expected}" >/dev/null 2>&1
}

ensure_parity() {
  local suite=$1 seed=$2 manifest preflight parity expected log_file
  manifest=$(manifest_path "${suite}" "${seed}")
  preflight=${RESULT}/gates/${suite}/${seed}/egl_preflight.json
  parity=${RESULT}/gates/${suite}/${seed}/fixed_2x2_parity.json
  log_file=${LOG_ROOT}/parity_${suite}_${seed}.log
  ensure_egl "${suite}" "${seed}" || return 1
  set_runtime "${manifest}" || return 1
  if validate_parity "${parity}" "${manifest}"; then
    return 0
  fi
  quarantine_path "${parity}" "parity_${suite}_${seed}"
  expected=$(manifest_sha "${manifest}") || return 1
  wait_for_isolated_gpu || return 1
  (cd "${FIXED_ROOT}" && \
    PYTHONPATH="${FIXED_ROOT}:${UPSTREAM}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_parity \
      --output "${parity}" \
      --manifest "${manifest}" \
      --expected-manifest-sha256 "${expected}" \
      --bundle-root "${BUNDLE}" \
      --condition-checkpoint "${CONDITION_CHECKPOINT}" \
      --fixed-2x2-source-lock "${FIXED_LOCK}" \
      --egl-preflight "${preflight}" \
      --physical-gpu-id "${GPU_ID}" \
      --classification RB2_CONFIRMATORY_EGL \
      --checkpoint "${CHECKPOINT}" \
      --smolvlm-model "${SMOLVLM}") \
      > "${log_file}" 2>&1 || return 1
  validate_parity "${parity}" "${manifest}"
}

registry_field() {
  local suite=$1 seed=$2 row=$3 field=$4
  "${PYTHON}" "${HELPER}" lookup \
    --registry "${REGISTRY}" --suite "${suite}" --seed "${seed}" --row "${row}" --field "${field}"
}

row_complete() {
  local root=$1 manifest=$2 row=$3 report=$4
  "${PYTHON}" "${HELPER}" validate-row \
    --root "${root}" --manifest "${manifest}" --row "${row}" --output "${report}" \
    >/dev/null 2>&1
}

run_generation_control_row() {
  local suite=$1 seed=$2 row=$3 cell=$4 manifest=$5
  local expected preflight shard log_file attempt rc validation
  expected=$(manifest_sha "${manifest}") || return 1
  preflight=${RESULT}/gates/${suite}/${seed}/egl_preflight.json
  log_file=${LOG_ROOT}/row_${suite}_${seed}_${row}.log
  validation=${RESULT}/metadata/row_validation/${suite}/${seed}/${row}.json
  if row_complete "${cell}" "${manifest}" "${row}" "${validation}"; then
    log "row_skip_complete suite=${suite} seed=${seed} row=${row}"
    return 0
  fi
  for attempt in 1 2; do
    quarantine_path "${cell}" "${suite}_${seed}_${row}_attempt${attempt}"
    mkdir -p "${cell}/logs"
    shard=${cell}/shard_rank0_tasks_0_9
    ensure_egl "${suite}" "${seed}" || rc=1
    if [[ ${rc:-0} -eq 0 ]]; then
      set_runtime "${manifest}" || rc=1
      wait_for_isolated_gpu || rc=1
    fi
    if [[ ${rc:-0} -eq 0 ]]; then
      log "row_start suite=${suite} seed=${seed} row=${row} attempt=${attempt} episodes=500"
      nvidia-smi -q -i "${GPU_ID}" > "${cell}/logs/nvidia_smi_before.txt" 2>&1 || true
      (cd "${CONTROL_ROOT}" && \
        PYTHONPATH="${CONTROL_ROOT}:${UPSTREAM}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval \
          --row "${row}" \
          --output "${shard}" \
          --manifest "${manifest}" \
          --expected-manifest-sha256 "${expected}" \
          --bundle-root "${BUNDLE}" \
          --egl-preflight "${preflight}" \
          --physical-gpu-id "${GPU_ID}" \
          --task-ids 0,1,2,3,4,5,6,7,8,9 \
          --classification RB2_CONFIRMATORY_EGL \
          --inference-seed "${seed}" \
          --checkpoint "${CHECKPOINT}" \
          --smolvlm-model "${SMOLVLM}" \
          --tqdm-mininterval 1.0 \
          --save-video --video-failures-only --video-stride 2 --video-max-per-task 1) \
          2>&1 | tee -a "${log_file}"
      rc=${PIPESTATUS[0]}
      nvidia-smi -q -i "${GPU_ID}" > "${cell}/logs/nvidia_smi_after.txt" 2>&1 || true
    fi
    if [[ ${rc:-1} -eq 0 ]] && row_complete "${cell}" "${manifest}" "${row}" "${validation}"; then
      log "row_complete suite=${suite} seed=${seed} row=${row} attempt=${attempt}"
      return 0
    fi
    log "row_attempt_failed suite=${suite} seed=${seed} row=${row} attempt=${attempt} rc=${rc:-1}"
    rc=0
  done
  return 1
}

run_fixed_row() {
  local suite=$1 seed=$2 row=$3 cell=$4 manifest=$5
  local expected preflight parity shard log_file attempt rc validation
  expected=$(manifest_sha "${manifest}") || return 1
  preflight=${RESULT}/gates/${suite}/${seed}/egl_preflight.json
  parity=${RESULT}/gates/${suite}/${seed}/fixed_2x2_parity.json
  log_file=${LOG_ROOT}/row_${suite}_${seed}_${row}.log
  validation=${RESULT}/metadata/row_validation/${suite}/${seed}/${row}.json
  if row_complete "${cell}" "${manifest}" "${row}" "${validation}"; then
    log "row_skip_complete suite=${suite} seed=${seed} row=${row}"
    return 0
  fi
  ensure_parity "${suite}" "${seed}" || return 1
  for attempt in 1 2; do
    quarantine_path "${cell}" "${suite}_${seed}_${row}_attempt${attempt}"
    mkdir -p "${cell}/logs"
    shard=${cell}/shard_rank0_tasks_0_9
    set_runtime "${manifest}" || rc=1
    wait_for_isolated_gpu || rc=1
    if [[ ${rc:-0} -eq 0 ]]; then
      log "row_start suite=${suite} seed=${seed} row=${row} attempt=${attempt} episodes=500"
      nvidia-smi -q -i "${GPU_ID}" > "${cell}/logs/nvidia_smi_before.txt" 2>&1 || true
      (cd "${FIXED_ROOT}" && \
        PYTHONPATH="${FIXED_ROOT}:${UPSTREAM}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval \
          --row "${row}" \
          --output "${shard}" \
          --manifest "${manifest}" \
          --expected-manifest-sha256 "${expected}" \
          --bundle-root "${BUNDLE}" \
          --condition-checkpoint "${CONDITION_CHECKPOINT}" \
          --fixed-2x2-source-lock "${FIXED_LOCK}" \
          --control-manifest "${BUNDLE}/transfer_manifest.json" \
          --fixed-2x2-parity-gate "${parity}" \
          --egl-preflight "${preflight}" \
          --physical-gpu-id "${GPU_ID}" \
          --task-ids 0,1,2,3,4,5,6,7,8,9 \
          --classification RB2_CONFIRMATORY_EGL \
          --inference-seed "${seed}" \
          --checkpoint "${CHECKPOINT}" \
          --smolvlm-model "${SMOLVLM}" \
          --tqdm-mininterval 1.0 \
          --save-video --video-failures-only --video-stride 2 --video-max-per-task 1) \
          2>&1 | tee -a "${log_file}"
      rc=${PIPESTATUS[0]}
      nvidia-smi -q -i "${GPU_ID}" > "${cell}/logs/nvidia_smi_after.txt" 2>&1 || true
    fi
    if [[ ${rc:-1} -eq 0 ]] && row_complete "${cell}" "${manifest}" "${row}" "${validation}"; then
      log "row_complete suite=${suite} seed=${seed} row=${row} attempt=${attempt}"
      return 0
    fi
    log "row_attempt_failed suite=${suite} seed=${seed} row=${row} attempt=${attempt} rc=${rc:-1}"
    rc=0
  done
  return 1
}

run_cell() {
  local suite=$1 seed=$2 row=$3 cell manifest reused validation
  cell=$(registry_field "${suite}" "${seed}" "${row}" path) || return 1
  manifest=$(registry_field "${suite}" "${seed}" "${row}" manifest) || return 1
  reused=$(registry_field "${suite}" "${seed}" "${row}" reused) || return 1
  validation=${RESULT}/metadata/row_validation/${suite}/${seed}/${row}.json
  if [[ "${reused}" == "true" ]]; then
    if row_complete "${cell}" "${manifest}" "${row}" "${validation}"; then
      log "row_reuse_pass suite=${suite} seed=${seed} row=${row} path=${cell}"
      return 0
    fi
    log "row_reuse_invalid suite=${suite} seed=${seed} row=${row}"
    return 1
  fi
  case "${row}" in
    full_nfe10|generation_ng3|naive_nfe3)
      run_generation_control_row "${suite}" "${seed}" "${row}" "${cell}" "${manifest}"
      ;;
    condition_kc2_ng3|condition_kc2_ng10)
      run_fixed_row "${suite}" "${seed}" "${row}" "${cell}" "${manifest}"
      ;;
    *) return 2 ;;
  esac
}

record_failure() {
  local suite=$1 seed=$2 row=$3 phase=$4
  printf '%s\t%s\t%s\t%s\t%s\n' "$(timestamp)" "${phase}" "${suite}" "${seed}" "${row}" >> "${FAILURES}"
  log "cell_failed_but_pipeline_continues phase=${phase} suite=${suite} seed=${seed} row=${row}"
}

run_phase() {
  local phase=$1
  shift
  local rows=("$@") suite seed row
  log "phase_start name=${phase} rows=${rows[*]}"
  for suite in "${SUITES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for row in "${rows[@]}"; do
        run_cell "${suite}" "${seed}" "${row}" || record_failure "${suite}" "${seed}" "${row}" "${phase}"
      done
    done
  done
  "${PYTHON}" "${HELPER}" aggregate \
    --registry "${REGISTRY}" \
    --output "${RESULT}/summary/${phase}_partial_summary.json" \
    --allow-partial \
    > "${LOG_ROOT}/aggregate_${phase}.log" 2>&1 || true
  log "phase_end name=${phase}"
}

preflight_suite_smokes() {
  local suite
  for suite in "${SUITES[@]}"; do
    ensure_parity "${suite}" seed01 || {
      log "PREFLIGHT_FAIL suite_parity=${suite}/seed01"
      return 1
    }
    log "suite_preflight_pass suite=${suite} seed=seed01"
  done
}

main() {
  : > "${STATUS}"
  log "launcher_start mode=${MODE} result=${RESULT} gpu=${GPU_ID}"
  source_audit || return 1
  prepare_manifests || return 1
  preflight_suite_smokes || return 1
  if [[ "${MODE}" == "--preflight" ]]; then
    printf 'PAPER_MATRIX_PREFLIGHT_PASS\n' > "${STATUS}"
    log "preflight_complete"
    return 0
  fi

  run_phase primary "${PRIMARY_ROWS[@]}"
  if [[ "${MODE}" == "--primary-only" ]]; then
    if "${PYTHON}" "${HELPER}" aggregate \
        --registry "${REGISTRY}" --output "${SUMMARY}" \
        > "${LOG_ROOT}/aggregate_final.log" 2>&1; then
      printf 'PAPER_SELECTED_MATRIX_COMPLETE\n' > "${STATUS}"
      log "primary_only_complete summary=${SUMMARY}"
      return 0
    fi
    printf 'PAPER_SELECTED_MATRIX_INCOMPLETE\n' > "${STATUS}"
    log "primary_only_incomplete summary=${SUMMARY} failures=${FAILURES}"
    return 1
  fi

  if (( ${#CONTROL_ROWS[@]} > 0 )); then
    run_phase controls "${CONTROL_ROWS[@]}"
  fi

  if "${PYTHON}" "${HELPER}" aggregate \
      --registry "${REGISTRY}" --output "${SUMMARY}" \
      > "${LOG_ROOT}/aggregate_final.log" 2>&1; then
    printf 'PAPER_FOUR_SUITE_THREE_SEED_COMPLETE\n' > "${STATUS}"
    log "pipeline_complete summary=${SUMMARY}"
    return 0
  fi
  printf 'PAPER_FOUR_SUITE_THREE_SEED_INCOMPLETE\n' > "${STATUS}"
  log "pipeline_incomplete summary=${SUMMARY} failures=${FAILURES}"
  return 1
}

main
