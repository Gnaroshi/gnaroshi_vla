#!/usr/bin/env bash
# Resumable rb2 paper evaluation: matched periodic baseline and selective refresh.

set -uo pipefail

if [[ ${SIMVLA_ACTION_REFRESH_3SEED_RUN:-0} != 1 ]]; then
  echo "Refusing launch: export SIMVLA_ACTION_REFRESH_3SEED_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_ACTION_REFRESH_ROOT:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_action_refresh_3seed_rb2}
UPSTREAM=/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream
STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
PYTHON=${STORAGE}/envs/simvla/libero_mujoco237/bin/python
LIBERO_ROOT=${STORAGE}/datasets/LIBERO
LIBERO_CONFIG=${STORAGE}/results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config
EXPECTED_COMMIT=${SIMVLA_ACTION_REFRESH_EXPECTED_COMMIT:-}
MAX_ATTEMPTS=${SIMVLA_ACTION_REFRESH_MAX_ATTEMPTS:-3}
MODE=${1:---all}

OUTPUT=${SIMVLA_ACTION_REFRESH_3SEED_OUTPUT:-${STORAGE}/results/simvla/action_equivalent_refresh/three_seed_long500_v1}
OFFLINE=${STORAGE}/results/simvla/action_equivalent_refresh/primary_v1
RISK=${OFFLINE}/risk_head_2k/action_fidelity_head.pt
INPUTS=${STORAGE}/artifacts/simvla/fixed_2x2_inputs_v1
FIXED_BUNDLE=${INPUTS}/generation_bundle
CONDITION=${INPUTS}/condition/native_v0_step_150000.pt
BASE_FIXED_SOURCE=${INPUTS}/fixed_2x2_source_lock.json
BASE_CONTROL_MANIFEST=${FIXED_BUNDLE}/transfer_manifest.json
GENERATION_BUNDLE=${STORAGE}/artifacts/simvla/generation_eval_bundle_20260824_v1
GENERATION=${GENERATION_BUNDLE}/checkpoint/generation_step_030000.pt
NORM=${GENERATION_BUNDLE}/norm/libero_norm_official_32700d0.json
EXP=${STORAGE}/results/simvla/latentloop/generation_loop_ng2_rb2_v1

declare -A MANIFESTS MANIFEST_SHAS BASELINES GENERATIONS
MANIFESTS[seed01]=${GENERATION_BUNDLE}/manifests/seed01_libero_long500_egl_manifest.json
MANIFESTS[seed02]=${EXP}/online/step_030000_long500_egl_seed02_v1/episode_manifest.json
MANIFESTS[seed03]=${EXP}/online/step_030000_long500_egl_seed03_v1/episode_manifest.json
MANIFEST_SHAS[seed01]=d1d9bf5a0ff6b20c235eb92dae80189ed3ebdc9eb1591a51fd0d8d572521e74a
MANIFEST_SHAS[seed02]=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
MANIFEST_SHAS[seed03]=25c3741fd73034cff2d83640dccb675a9fc526c2dc4b406490209e53fd76c61d
BASELINES[seed01]=${EXP}/online/step_010000_long500_egl_paired_v1/baseline_k1
BASELINES[seed02]=${EXP}/online/step_030000_long500_egl_seed02_v1/baseline_k1
BASELINES[seed03]=${EXP}/online/step_030000_long500_egl_seed03_v1/baseline_k1
GENERATIONS[seed01]=${EXP}/online/step_030000_long500_egl_paired_v1/generation_ng3
GENERATIONS[seed02]=${EXP}/online/step_030000_long500_egl_seed02_v1/generation_ng3
GENERATIONS[seed03]=${EXP}/online/step_030000_long500_egl_seed03_v1/generation_ng3

LOGS=${OUTPUT}/logs
GATES=${OUTPUT}/gates
FAILED=${OUTPUT}/failed_attempts
STATUS=${OUTPUT}/pipeline.status
PROVENANCE=${OUTPUT}/provenance
FIXED_SOURCE=${PROVENANCE}/fixed_eval_source_lock.json
CONTROL_MANIFEST=${PROVENANCE}/control_manifest.json
mkdir -p "${LOGS}" "${GATES}" "${FAILED}"

export SIMVLA_UPSTREAM_ROOT=${UPSTREAM}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG}
export PYTHONPATH="${ROOT}:${UPSTREAM}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${STORAGE}/cache/simvla/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER}
export MPLCONFIGDIR=/tmp/matplotlib_${USER}
unset GALLIUM_DRIVER
unset LIBGL_ALWAYS_SOFTWARE

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
record() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${LOGS}/pipeline.log"; }
fail() {
  local message=$1 rc=${2:-1}
  printf 'ACTION_EQUIVALENT_REFRESH_3SEED_FAILED rc=%s reason=%s\n' "${rc}" "${message}" | tee "${STATUS}" >&2
  return "${rc}"
}

[[ ${MODE} == --all || ${MODE} == --preflight ]] || { fail "usage: $0 [--preflight|--all]" 2; exit 2; }
[[ -n ${EXPECTED_COMMIT} ]] || { fail "SIMVLA_ACTION_REFRESH_EXPECTED_COMMIT is required" 2; exit 2; }
[[ ${MAX_ATTEMPTS} =~ ^[1-9][0-9]*$ ]] || { fail "SIMVLA_ACTION_REFRESH_MAX_ATTEMPTS must be a positive integer" 2; exit 2; }
for path in "${PYTHON}" "${ROOT}" "${UPSTREAM}" "${LIBERO_ROOT}" \
  "${OFFLINE}/pipeline.status" "${RISK}" "${CONDITION}" "${BASE_FIXED_SOURCE}" \
  "${BASE_CONTROL_MANIFEST}" \
  "${GENERATION}" "${NORM}"; do
  [[ -e ${path} ]] || { fail "missing required path: ${path}" 2; exit 2; }
done
observed_commit=$(git -C "${ROOT}" rev-parse HEAD) || exit 2
[[ ${observed_commit} == "${EXPECTED_COMMIT}" ]] || { fail "commit ${observed_commit} != ${EXPECTED_COMMIT}" 2; exit 2; }
[[ -z $(git -C "${ROOT}" status --porcelain --untracked-files=no) ]] || { fail "tracked worktree changes are present" 2; exit 2; }

prepare_periodic_provenance() {
  if [[ -s ${FIXED_SOURCE} && -s ${CONTROL_MANIFEST} ]] && \
    COMMIT=${EXPECTED_COMMIT} FIXED_SOURCE=${FIXED_SOURCE} "${PYTHON}" - <<'PY' >/dev/null 2>&1
import json, os
d=json.load(open(os.environ['FIXED_SOURCE'], encoding='utf-8'))
assert d['root_commit']==os.environ['COMMIT']
assert d['file_sha256']
PY
  then
    record "periodic_provenance=resume_skip commit=${EXPECTED_COMMIT}"
    return 0
  fi
  if [[ -e ${PROVENANCE} ]]; then
    mv "${PROVENANCE}" "${FAILED}/provenance_$(date +%s)"
  fi
  cd "${ROOT}" || return 1
  CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m \
    architectures.simvla.adapters.latentloop.efficient_multirate.coupled_source_lock \
    --base-fixed-source-lock "${BASE_FIXED_SOURCE}" \
    --base-control-manifest "${BASE_CONTROL_MANIFEST}" \
    --output "${PROVENANCE}" \
    >"${LOGS}/prepare_periodic_provenance.log" 2>&1 || return 1
  COMMIT=${EXPECTED_COMMIT} FIXED_SOURCE=${FIXED_SOURCE} CONTROL_MANIFEST=${CONTROL_MANIFEST} \
    "${PYTHON}" - <<'PY'
import json, os
fixed=json.load(open(os.environ['FIXED_SOURCE'], encoding='utf-8'))
control=json.load(open(os.environ['CONTROL_MANIFEST'], encoding='utf-8'))
assert fixed['root_commit']==os.environ['COMMIT']
assert fixed['file_sha256'] and control['control_file_sha256']
print('PERIODIC_PROVENANCE_PASS', fixed['root_commit'])
PY
}

wait_for_gpu() {
  while true; do
    local used
    used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || return 1
    [[ ${used} =~ ^[0-9]+$ ]] || return 1
    if (( used <= 2048 )); then
      record "gpu=0 state=free_or_teleop_only memory_used_mib=${used}"
      return 0
    fi
    record "gpu=0 state=waiting memory_used_mib=${used} threshold_mib=2048 next_check=60s"
    sleep 60
  done
}

set_seed_runtime() {
  local seed=$1 manifest=${MANIFESTS[$1]}
  mapfile -t renderer < <("${PYTHON}" - "${manifest}" <<'PY'
import json, sys
r=json.load(open(sys.argv[1],encoding='utf-8'))['renderer']
for key in ('CUBLAS_WORKSPACE_CONFIG','CUDA_DEVICE_MAX_CONNECTIONS','PYTHONHASHSEED','SIMVLA_RENDER_AXIS'):
    print(r[key])
PY
  )
  [[ ${#renderer[@]} -eq 4 ]] || return 1
  export CUBLAS_WORKSPACE_CONFIG=${renderer[0]}
  export CUDA_DEVICE_MAX_CONNECTIONS=${renderer[1]}
  export PYTHONHASHSEED=${renderer[2]}
  export SIMVLA_RENDER_AXIS=${renderer[3]}
  export CUDA_VISIBLE_DEVICES=0
  record "seed=${seed} renderer_axis=${SIMVLA_RENDER_AXIS} pythonhashseed=${PYTHONHASHSEED}"
}

validate_input_axis() {
  local seed manifest sha
  for seed in seed01 seed02 seed03; do
    manifest=${MANIFESTS[$seed]}
    sha=${MANIFEST_SHAS[$seed]}
    [[ -s ${manifest} ]] || return 1
    [[ -s ${BASELINES[$seed]}/row_summary.json ]] || return 1
    [[ -s ${GENERATIONS[$seed]}/row_summary.json ]] || return 1
    SEED=${seed} MANIFEST=${manifest} SHA=${sha} BASELINE=${BASELINES[$seed]} GENERATION_ROOT=${GENERATIONS[$seed]} "${PYTHON}" - <<'PY' || return 1
import csv, hashlib, json, os
from pathlib import Path
m=json.load(open(os.environ['MANIFEST'],encoding='utf-8'))
claimed=m.pop('manifest_sha256')
actual=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert claimed==actual==os.environ['SHA']
assert m['suite']=='libero_10' and len(m['episodes'])==500 and m['trials_per_task']==50
assert m['selected_physical_gpu_ids']==[0]
assert m['renderer']['MUJOCO_GL']=='egl' and m['renderer']['PYOPENGL_PLATFORM']=='egl'
for root, row, step in ((Path(os.environ['BASELINE']),'baseline_k1',None),(Path(os.environ['GENERATION_ROOT']),'generation_ng3',30000)):
    summary=json.load(open(root/'row_summary.json',encoding='utf-8'))
    assert summary['row']==row and summary['episodes']==500 and summary['manifest_sha256']==claimed
    if step is not None: assert summary['optimizer_step']==step
    paths=[root/'episode_metrics.csv'] if (root/'episode_metrics.csv').is_file() else sorted(root.glob('shard_rank*_tasks_*/episode_metrics.csv'))
    assert paths and sum(sum(1 for _ in csv.DictReader(open(path,newline='',encoding='utf-8'))) for path in paths)==500
print('INPUT_AXIS_PASS',os.environ['SEED'],claimed)
PY
  done
}

preflight_valid() {
  local path=$1 seed=$2
  [[ -s ${path} ]] || return 1
  PATH_TO_CHECK=${path} MANIFEST=${MANIFESTS[$seed]} "${PYTHON}" - <<'PY' >/dev/null 2>&1
import json, os
d=json.load(open(os.environ['PATH_TO_CHECK'],encoding='utf-8'))
m=json.load(open(os.environ['MANIFEST'],encoding='utf-8'))
assert d['verdict']=='EGL_PREFLIGHT_PASS' and d['physical_gpu_id']==0
for key,value in m['renderer'].items(): assert d['environment'].get(key)==value
assert d['libero_reset_pass'] and d['libero_step_pass']
PY
}

ensure_preflight() {
  local seed=$1 root=${GATES}/$1 path=${GATES}/$1/egl_preflight.json
  mkdir -p "${root}"
  if preflight_valid "${path}" "${seed}"; then
    record "seed=${seed} egl_preflight=resume_skip"
    return 0
  fi
  [[ ! -e ${path} ]] || mv "${path}" "${FAILED}/${seed}_egl_preflight_$(date +%s).json"
  set_seed_runtime "${seed}" || return 1
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${ROOT}/tools/simvla/simvla_egl_preflight.py" \
    --output "${path}" --gpu-id 0 --suite libero_10 --environment-seed 7 --resolution 256 \
    >"${LOGS}/${seed}_egl_preflight.log" 2>&1 || return 1
  preflight_valid "${path}" "${seed}"
}

parity_valid() {
  local path=$1 seed=$2
  [[ -s ${path} ]] || return 1
  PATH_TO_CHECK=${path} SHA=${MANIFEST_SHAS[$seed]} "${PYTHON}" - <<'PY' >/dev/null 2>&1
import json, os
d=json.load(open(os.environ['PATH_TO_CHECK'],encoding='utf-8'))
assert d['verdict']=='FIXED_2X2_PARITY_PASS'
assert d['manifest_sha256']==os.environ['SHA']
assert all(d['checks'].values())
PY
}

ensure_parity() {
  local seed=$1 path=${GATES}/$1/fixed_2x2_parity.json
  if parity_valid "${path}" "${seed}"; then
    record "seed=${seed} fixed_2x2_parity=resume_skip"
    return 0
  fi
  [[ ! -e ${path} ]] || mv "${path}" "${FAILED}/${seed}_fixed_2x2_parity_$(date +%s).json"
  set_seed_runtime "${seed}" || return 1
  cd "${ROOT}" || return 1
  PYTHONPATH="${ROOT}:${UPSTREAM}:${LIBERO_ROOT}" CUDA_VISIBLE_DEVICES=0 "${PYTHON}" \
    -m architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_parity \
    --output "${path}" --manifest "${MANIFESTS[$seed]}" \
    --expected-manifest-sha256 "${MANIFEST_SHAS[$seed]}" \
    --bundle-root "${FIXED_BUNDLE}" --condition-checkpoint "${CONDITION}" \
    --fixed-2x2-source-lock "${FIXED_SOURCE}" \
    --egl-preflight "${GATES}/${seed}/egl_preflight.json" --physical-gpu-id 0 \
    --classification RB2_CONFIRMATORY_EGL \
    >"${LOGS}/${seed}_fixed_2x2_parity.log" 2>&1 || return 1
  parity_valid "${path}" "${seed}"
}

periodic_root() { printf '%s/%s/periodic_kc3_ng3' "${OUTPUT}" "$1"; }
action_root() { printf '%s/%s/action_equivalent_refresh_ng3' "${OUTPUT}" "$1"; }
merged_root() { printf '%s/%s/merged' "${OUTPUT}" "$1"; }

periodic_complete() {
  local seed=$1 summary
  summary=$(periodic_root "${seed}")/merged/row_summary.json
  [[ -s ${summary} ]] || return 1
  SUMMARY=${summary} SHA=${MANIFEST_SHAS[$seed]} "${PYTHON}" - <<'PY' >/dev/null 2>&1
import json, os
d=json.load(open(os.environ['SUMMARY'],encoding='utf-8'))
assert d['row']=='condition_kc3_ng3' and d['episodes']==500
assert d['manifest_sha256']==os.environ['SHA']
PY
}

run_periodic_once() {
  local seed=$1 root archive
  root=$(periodic_root "${seed}")
  if [[ -e ${root} || -e ${root}.egl_preflight.json ]]; then
    archive=${FAILED}/${seed}_periodic_kc3_ng3_$(date +%s)
    mkdir -p "${archive}"
    [[ ! -e ${root} ]] || mv "${root}" "${archive}/output"
    [[ ! -e ${root}.egl_preflight.json ]] || mv "${root}.egl_preflight.json" "${archive}/egl_preflight.json"
  fi
  set_seed_runtime "${seed}" || return 1
  cd "${ROOT}" || return 1
  export SIMVLA_FIXED_2X2_RUN=1
  export SIMVLA_FIXED_2X2_ROOT=${ROOT}
  export SIMVLA_FIXED_2X2_PYTHON=${PYTHON}
  export SIMVLA_UPSTREAM_ROOT=${UPSTREAM}
  PYTHONPATH="${ROOT}:${UPSTREAM}:${LIBERO_ROOT}" \
    bash architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh \
      --row condition_kc3_ng3 --output "${root}" \
      --manifest "${MANIFESTS[$seed]}" --manifest-sha256 "${MANIFEST_SHAS[$seed]}" \
      --bundle-root "${FIXED_BUNDLE}" --condition-checkpoint "${CONDITION}" \
      --source-lock "${FIXED_SOURCE}" --parity-gate "${GATES}/${seed}/fixed_2x2_parity.json" \
      --control-manifest "${CONTROL_MANIFEST}" \
      --physical-gpu-id 0 --classification RB2_CONFIRMATORY_EGL \
      --inference-seed "${seed}" --save-failure-videos \
      >"${LOGS}/${seed}_periodic_kc3_ng3.log" 2>&1
}

action_complete() {
  local seed=$1 root summary
  root=$(action_root "${seed}")
  summary=${root}/shard_summary.json
  [[ -s ${summary} ]] || return 1
  SUMMARY=${summary} SHA=${MANIFEST_SHAS[$seed]} COMMIT=${EXPECTED_COMMIT} "${PYTHON}" - <<'PY' >/dev/null 2>&1
import json, os
d=json.load(open(os.environ['SUMMARY'],encoding='utf-8'))
s=json.load(open(os.path.dirname(os.environ['SUMMARY'])+'/source_lock.json',encoding='utf-8'))
assert d['verdict']=='ACTION_EQUIVALENT_REFRESH_ONLINE_SHARD_COMPLETE'
assert d['episodes']==500 and d['manifest_sha256']==os.environ['SHA']
assert d['all_counter_gates_pass'] and s['root_commit']==os.environ['COMMIT']
PY
}

run_action_once() {
  local seed=$1 root
  root=$(action_root "${seed}")
  mkdir -p "${root}"
  set_seed_runtime "${seed}" || return 1
  cd "${ROOT}" || return 1
  "${PYTHON}" -m architectures.simvla.adapters.latentloop.action_equivalent_refresh.online_evaluator \
    --output "${root}" --manifest "${MANIFESTS[$seed]}" \
    --expected-manifest-sha256 "${MANIFEST_SHAS[$seed]}" --offline-root "${OFFLINE}" \
    --risk-checkpoint "${RISK}" --condition-checkpoint "${CONDITION}" \
    --generation-checkpoint "${GENERATION}" --norm-stats "${NORM}" \
    --expected-root-commit "${EXPECTED_COMMIT}" \
    --egl-preflight "${GATES}/${seed}/egl_preflight.json" --physical-gpu-id 0 \
    --task-ids 0,1,2,3,4,5,6,7,8,9 --classification RB2_HOST_LOCAL_EGL_LONG500 \
    --tqdm-mininterval 1.0 --save-video --video-stride 2 --video-fps 10 --video-max-per-task 2 \
    >"${LOGS}/${seed}_action_equivalent_refresh.log" 2>&1
}

run_until_complete() {
  local label=$1 check=$2 runner=$3 seed=$4 attempt=0 rc
  while ! "${check}" "${seed}"; do
    attempt=$((attempt + 1))
    wait_for_gpu || { record "label=${label} seed=${seed} gpu_query_failed retry=60s"; sleep 60; continue; }
    record "label=${label} seed=${seed} attempt=${attempt} state=start_or_resume"
    "${runner}" "${seed}"
    rc=$?
    if "${check}" "${seed}"; then
      record "label=${label} seed=${seed} attempt=${attempt} state=complete child_rc=${rc}"
      return 0
    fi
    if (( rc == 2 )); then
      record "label=${label} seed=${seed} attempt=${attempt} state=deterministic_configuration_failure child_rc=2"
      return 2
    fi
    if (( attempt >= MAX_ATTEMPTS )); then
      record "label=${label} seed=${seed} attempt=${attempt} state=retry_budget_exhausted child_rc=${rc}"
      return 1
    fi
    record "label=${label} seed=${seed} attempt=${attempt} state=incomplete child_rc=${rc} retry=60s"
    sleep 60
  done
  record "label=${label} seed=${seed} state=resume_skip"
}

aggregate_seed() {
  local seed=$1 root
  root=$(merged_root "${seed}")
  cd "${ROOT}" || return 1
  CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m \
    architectures.simvla.adapters.latentloop.action_equivalent_refresh.online_aggregate \
    --output "${root}" --manifest "${MANIFESTS[$seed]}" \
    --expected-manifest-sha256 "${MANIFEST_SHAS[$seed]}" \
    --classification RB2_HOST_LOCAL_EGL_LONG500 \
    --shards "$(action_root "${seed}")" \
    --full-control "${BASELINES[$seed]}" --generation-control "${GENERATIONS[$seed]}" \
    --periodic-kc3-control "$(periodic_root "${seed}")/merged" \
    >"${LOGS}/${seed}_aggregate.log" 2>&1
}

validate_input_axis || { fail "3-seed input-axis validation failed" 2; exit 2; }
prepare_periodic_provenance || { fail "periodic evaluator provenance preparation failed" 2; exit 2; }
wait_for_gpu || { fail "cannot query rb2 GPU0" 2; exit 2; }
for seed in seed01 seed02 seed03; do
  set_seed_runtime "${seed}" || { fail "renderer contract failed for ${seed}" 2; exit 2; }
  ensure_preflight "${seed}" || { fail "EGL preflight failed for ${seed}" 2; exit 2; }
  ensure_parity "${seed}" || { fail "fixed-2x2 parity failed for ${seed}" 2; exit 2; }
done
if [[ ${MODE} == --preflight ]]; then
  printf 'ACTION_EQUIVALENT_REFRESH_3SEED_PREFLIGHT_PASS\n' | tee "${STATUS}"
  exit 0
fi
printf 'ACTION_EQUIVALENT_REFRESH_3SEED_RUNNING\n' | tee "${STATUS}"

# Distinct seeds first. Seed02 runs last because the same seed is already in
# progress on sd1, while rb2 measurements remain host-local for latency.
for seed in seed01 seed03 seed02; do
  run_until_complete periodic_kc3_ng3 periodic_complete run_periodic_once "${seed}" || {
    fail "periodic_kc3_ng3 failed for ${seed}"
    exit 1
  }
  run_until_complete action_equivalent_refresh_ng3 action_complete run_action_once "${seed}" || {
    fail "action_equivalent_refresh_ng3 failed for ${seed}"
    exit 1
  }
  aggregate_seed "${seed}" || { fail "per-seed aggregation failed for ${seed}"; exit 1; }
  record "seed=${seed} paired_aggregation=complete"
done

FINAL=${OUTPUT}/three_seed_summary
CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m \
  architectures.simvla.adapters.latentloop.action_equivalent_refresh.three_seed_aggregate \
  --output "${FINAL}" \
  --seed01 "$(merged_root seed01)/online_comparison_summary.json" \
  --seed02 "$(merged_root seed02)/online_comparison_summary.json" \
  --seed03 "$(merged_root seed03)/online_comparison_summary.json" \
  >"${LOGS}/three_seed_aggregate.log" 2>&1 || { fail "three-seed aggregation failed"; exit 1; }

grep -q 'ACTION_EQUIVALENT_REFRESH_THREE_INFERENCE_SEED_COMPLETE' \
  "${FINAL}/three_inference_seed_summary.json" || { fail "final completion verdict missing"; exit 1; }
printf 'ACTION_EQUIVALENT_REFRESH_THREE_INFERENCE_SEED_COMPLETE\n' | tee "${STATUS}"
"${PYTHON}" - "${FINAL}/three_inference_seed_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
for name,row in d['aggregate'].items():
    print(f"RESULT row={name} success={row['successes']}/{row['episodes']} sr={100*row['success_rate']:.2f}% latency_action_ms={row['latency_per_executed_action_ms_seed_mean']:.3f}")
print('SUMMARY',sys.argv[1])
PY
