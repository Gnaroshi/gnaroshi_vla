#!/usr/bin/env bash
# Resumable four-GPU Long-500 evaluation for selective condition refresh.

set -uo pipefail

if [[ ${SIMVLA_ACTION_REFRESH_ONLINE_RUN:-0} != 1 ]]; then
  echo "Refusing launch: export SIMVLA_ACTION_REFRESH_ONLINE_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_ACTION_REFRESH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PYTHON=${SIMVLA_ACTION_REFRESH_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
EXPECTED_COMMIT=${SIMVLA_ACTION_REFRESH_EXPECTED_COMMIT:-}
MODE=${1:---all}

STORAGE=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla
OUTPUT=${SIMVLA_ACTION_REFRESH_ONLINE_OUTPUT:-${STORAGE}/results/simvla/action_equivalent_refresh/online_seed02_long500_v1}
OFFLINE=${SIMVLA_ACTION_REFRESH_OFFLINE_ROOT:-${STORAGE}/results/simvla/action_equivalent_refresh/primary_v1}
MANIFEST=${SIMVLA_ACTION_REFRESH_MANIFEST:-${STORAGE}/artifacts/simvla/fixed_2x2_sd1_v1/manifests/seed02_libero_long500_egl_manifest.json}
MANIFEST_SHA=9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48
RISK=${OFFLINE}/risk_head_2k/action_fidelity_head.pt
CONDITION=${STORAGE}/results/simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/checkpoints/native_v0_step_150000.pt
BUNDLE=${STORAGE}/artifacts/simvla/generation_eval_bundle_20260824_v1
GENERATION=${BUNDLE}/checkpoint/generation_step_030000.pt
NORM=${BUNDLE}/norm/libero_norm_official_32700d0.json

FULL_CONTROL=${STORAGE}/results/simvla/fixed_2x2/sd1_seed01_seed02_v1/seed02/full_nfe10/merged
GENERATION_CONTROL=${STORAGE}/results/simvla/fixed_2x2/sd1_seed01_seed02_v1/seed02/generation_ng3/merged
PERIODIC_KC3_CONTROL=${STORAGE}/results/simvla/kc_efficiency_frontier/kc3_kc4_ng10_ng3_sd1_seed02_v1/rows/condition_kc3_ng3/merged
PERIODIC_KC4_CONTROL=${STORAGE}/results/simvla/kc_efficiency_frontier/kc3_kc4_ng10_ng3_sd1_seed02_v1/rows/condition_kc4_ng3/merged

GPU_IDS=(4 5 6 7)
TASK_PARTITIONS=("0,1,4" "2,5,7" "3,9" "6,8")
EXPECTED_EPISODES=(150 150 100 100)
LOGS=${OUTPUT}/logs
STATUS=${OUTPUT}/pipeline.status
MERGED=${OUTPUT}/merged
FAILED=${OUTPUT}/failed_attempts
mkdir -p "${LOGS}" "${FAILED}"
cd "${ROOT}" || exit 2

export PYTHONPATH="${ROOT}:${UPSTREAM}:${UPSTREAM}/evaluation/libero/LIBERO${PYTHONPATH:+:${PYTHONPATH}}"
export SIMVLA_UPSTREAM_ROOT=${UPSTREAM}
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONHASHSEED=20260816
export SIMVLA_RENDER_AXIS=rb2_egl_long500_seed02_v1
unset GALLIUM_DRIVER
unset LIBGL_ALWAYS_SOFTWARE

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
record() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${LOGS}/pipeline.log"; }
die() {
  local message=$1 rc=${2:-1}
  printf 'ACTION_EQUIVALENT_REFRESH_ONLINE_FAILED rc=%s reason=%s\n' "${rc}" "${message}" | tee "${STATUS}" >&2
  printf 'Inspect: %s\n' "${LOGS}" >&2
  exit "${rc}"
}

if [[ ${MODE} != --all && ${MODE} != --preflight ]]; then
  die "usage: $0 [--preflight|--all]" 2
fi
[[ -n ${EXPECTED_COMMIT} ]] || die "SIMVLA_ACTION_REFRESH_EXPECTED_COMMIT is required" 2
for path in "${PYTHON}" "${UPSTREAM}" "${MANIFEST}" "${RISK}" "${CONDITION}" "${GENERATION}" "${NORM}"; do
  [[ -e ${path} ]] || die "missing required path: ${path}" 2
done
for root in "${FULL_CONTROL}" "${GENERATION_CONTROL}" "${PERIODIC_KC3_CONTROL}" "${PERIODIC_KC4_CONTROL}"; do
  [[ -s ${root}/row_summary.json && -s ${root}/episode_metrics.csv ]] || die "missing control: ${root}" 2
done

observed_commit=$(git -C "${ROOT}" rev-parse HEAD) || die "cannot read worktree commit"
[[ ${observed_commit} == "${EXPECTED_COMMIT}" ]] || die "worktree commit ${observed_commit} != ${EXPECTED_COMMIT}"
tracked_status=$(git -C "${ROOT}" status --porcelain --untracked-files=no) || die "cannot inspect worktree"
[[ -z ${tracked_status} ]] || die "tracked worktree changes are present"

wait_for_gpus() {
  while true; do
    local busy=() gpu used
    for gpu in "${GPU_IDS[@]}"; do
      used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || die "nvidia-smi failed for GPU ${gpu}"
      [[ ${used} =~ ^[0-9]+$ ]] || die "invalid memory value for GPU ${gpu}: ${used}"
      (( used <= 512 )) || busy+=("${gpu}:${used}MiB")
    done
    if (( ${#busy[@]} == 0 )); then
      record "gpu_pool=4,5,6,7 state=free"
      return
    fi
    record "gpu_pool state=waiting busy=${busy[*]} next_check=60s"
    sleep 60
  done
}

run_cpu_preflight() {
  local marker=${OUTPUT}/cpu_preflight.json
  if [[ -s ${marker} ]] && "${PYTHON}" - "${marker}" "${EXPECTED_COMMIT}" <<'PY' >/dev/null 2>&1
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
assert d['verdict']=='ACTION_EQUIVALENT_REFRESH_CPU_PREFLIGHT_PASS'
assert d['commit']==sys.argv[2]
PY
  then
    record "cpu_preflight state=resume_skip"
    return
  fi
  record "cpu_preflight state=start"
  CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m pytest -q \
    tests/latentloop/test_action_equivalent_refresh.py \
    >"${LOGS}/cpu_preflight.log" 2>&1 || {
      tail -100 "${LOGS}/cpu_preflight.log" >&2
      die "CPU unit tests failed"
    }
  "${PYTHON}" - "${marker}" "${EXPECTED_COMMIT}" <<'PY'
import json, os, sys
path, commit=sys.argv[1:]
tmp=f'{path}.tmp-{os.getpid()}'
with open(tmp,'w',encoding='utf-8') as f:
    json.dump({'verdict':'ACTION_EQUIVALENT_REFRESH_CPU_PREFLIGHT_PASS','commit':commit},f,indent=2,sort_keys=True)
    f.write('\n')
os.replace(tmp,path)
PY
  record "cpu_preflight state=complete"
}

preflight_valid() {
  local path=$1 gpu=$2
  [[ -s ${path} ]] || return 1
  "${PYTHON}" - "${path}" "${gpu}" <<'PY' >/dev/null 2>&1
import json, sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['verdict']=='EGL_PREFLIGHT_PASS'
assert int(d['physical_gpu_id'])==int(sys.argv[2])
assert d['environment']['MUJOCO_GL']=='egl'
assert d['environment']['PYOPENGL_PLATFORM']=='egl'
assert d['libero_reset_pass'] and d['libero_step_pass']
PY
}

run_egl_preflights() {
  local pids=() labels=() index gpu path task
  for index in 0 1 2 3; do
    gpu=${GPU_IDS[$index]}
    path=${OUTPUT}/egl_preflight_gpu${gpu}.json
    task=${TASK_PARTITIONS[$index]%%,*}
    if preflight_valid "${path}" "${gpu}"; then
      record "egl_preflight gpu=${gpu} state=resume_skip"
      continue
    fi
    if [[ -e ${path} ]]; then
      mv "${path}" "${FAILED}/egl_preflight_gpu${gpu}_$(date +%s).json"
    fi
    record "egl_preflight gpu=${gpu} state=start"
    CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} \
      "${PYTHON}" tools/simvla/simvla_egl_preflight.py \
        --output "${path}" --gpu-id "${gpu}" --suite libero_10 \
        --task-id "${task}" --environment-seed 7 --resolution 256 \
        >"${LOGS}/egl_preflight_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    labels+=("${gpu}")
  done
  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      record "egl_preflight gpu=${labels[$index]} state=complete"
    else
      failed=1
      record "egl_preflight gpu=${labels[$index]} state=failed"
    fi
  done
  (( failed == 0 )) || {
    tail -100 "${LOGS}"/egl_preflight_gpu*.log >&2
    die "one or more EGL preflights failed"
  }
  for gpu in "${GPU_IDS[@]}"; do
    preflight_valid "${OUTPUT}/egl_preflight_gpu${gpu}.json" "${gpu}" || die "invalid EGL preflight for GPU ${gpu}"
  done
}

shard_output() { printf '%s/shards/gpu%s_tasks_%s' "${OUTPUT}" "$1" "${2//,/_}"; }

shard_complete() {
  local output=$1 expected=$2
  [[ -s ${output}/shard_summary.json ]] || return 1
  "${PYTHON}" - "${output}/shard_summary.json" "${expected}" "${MANIFEST_SHA}" "${EXPECTED_COMMIT}" <<'PY' >/dev/null 2>&1
import json, sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
assert p['verdict']=='ACTION_EQUIVALENT_REFRESH_ONLINE_SHARD_COMPLETE'
assert int(p['episodes'])==int(sys.argv[2])
assert p['manifest_sha256']==sys.argv[3]
s=json.load(open(sys.argv[1].rsplit('/',1)[0]+'/source_lock.json',encoding='utf-8'))
assert s['root_commit']==sys.argv[4]
PY
}

run_shard() {
  local index=$1 gpu=${GPU_IDS[$1]} tasks=${TASK_PARTITIONS[$1]}
  local output log
  output=$(shard_output "${gpu}" "${tasks}")
  log=${LOGS}/evaluate_gpu${gpu}.log
  if shard_complete "${output}" "${EXPECTED_EPISODES[$index]}"; then
    record "online_shard gpu=${gpu} tasks=${tasks} state=resume_skip"
    return 0
  fi
  mkdir -p "${output}"
  record "online_shard gpu=${gpu} tasks=${tasks} state=start_or_resume"
  CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} \
    "${PYTHON}" -m \
      architectures.simvla.adapters.latentloop.action_equivalent_refresh.online_evaluator \
      --output "${output}" \
      --manifest "${MANIFEST}" \
      --expected-manifest-sha256 "${MANIFEST_SHA}" \
      --offline-root "${OFFLINE}" \
      --risk-checkpoint "${RISK}" \
      --condition-checkpoint "${CONDITION}" \
      --generation-checkpoint "${GENERATION}" \
      --norm-stats "${NORM}" \
      --expected-root-commit "${EXPECTED_COMMIT}" \
      --egl-preflight "${OUTPUT}/egl_preflight_gpu${gpu}.json" \
      --physical-gpu-id "${gpu}" \
      --task-ids "${tasks}" \
      --classification SD1_HOST_LOCAL_EGL_LONG500 \
      --tqdm-mininterval 1.0 \
      --save-video --video-stride 2 --video-fps 10 --video-max-per-task 2 \
      >"${log}" 2>&1
}

progress_snapshot() {
  "${PYTHON}" - "${OUTPUT}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])/'shards'
paths=list(root.glob('*/episodes/*.json'))
success=0
for path in paths:
    try: success += int(json.load(open(path,encoding='utf-8'))['metrics']['success'])
    except Exception: pass
n=len(paths)
print(f'completed={n}/500 success={success}/{n} sr={100.0*success/max(1,n):.1f}%')
PY
}

finalize_shard() {
  local index=$1 gpu=${GPU_IDS[$1]} tasks=${TASK_PARTITIONS[$1]}
  local output
  output=$(shard_output "${gpu}" "${tasks}")
  CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_EGL_DEVICE_ID=${gpu} \
    "${PYTHON}" -m \
      architectures.simvla.adapters.latentloop.action_equivalent_refresh.online_evaluator \
      --output "${output}" --manifest "${MANIFEST}" \
      --expected-manifest-sha256 "${MANIFEST_SHA}" \
      --offline-root "${OFFLINE}" --risk-checkpoint "${RISK}" \
      --condition-checkpoint "${CONDITION}" --generation-checkpoint "${GENERATION}" \
      --norm-stats "${NORM}" --expected-root-commit "${EXPECTED_COMMIT}" \
      --egl-preflight "${OUTPUT}/egl_preflight_gpu${gpu}.json" \
      --physical-gpu-id "${gpu}" --task-ids "${tasks}" \
      --classification SD1_HOST_LOCAL_EGL_LONG500 --finalize-only \
      >"${LOGS}/finalize_gpu${gpu}.log" 2>&1
}

run_cpu_preflight
wait_for_gpus
run_egl_preflights
if [[ ${MODE} == --preflight ]]; then
  printf 'ACTION_EQUIVALENT_REFRESH_ONLINE_PREFLIGHT_PASS\n' | tee "${STATUS}"
  exit 0
fi

pids=() indices=()
for index in 0 1 2 3; do
  if shard_complete "$(shard_output "${GPU_IDS[$index]}" "${TASK_PARTITIONS[$index]}")" "${EXPECTED_EPISODES[$index]}"; then
    record "online_shard gpu=${GPU_IDS[$index]} state=already_complete"
    continue
  fi
  run_shard "${index}" &
  pids+=("$!")
  indices+=("${index}")
done

while (( ${#pids[@]} > 0 )); do
  alive=0
  for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && alive=1
  done
  record "online_progress $(progress_snapshot)"
  (( alive == 1 )) || break
  sleep 60
done

process_failed=0
for position in "${!pids[@]}"; do
  if wait "${pids[$position]}"; then
    record "online_shard gpu=${GPU_IDS[${indices[$position]}]} state=process_complete"
  else
    process_failed=1
    record "online_shard gpu=${GPU_IDS[${indices[$position]}]} state=process_failed"
  fi
done

# Always rebuild summaries from atomic episode files. This recovers a completed
# rollout whose process failed only during final CSV/JSON postprocessing.
failed=0
for index in 0 1 2 3; do
  finalize_shard "${index}" || failed=1
  shard_complete "$(shard_output "${GPU_IDS[$index]}" "${TASK_PARTITIONS[$index]}")" "${EXPECTED_EPISODES[$index]}" || failed=1
done
if (( process_failed == 1 && failed == 0 )); then
  record "online_shards state=recovered_from_atomic_episode_files"
fi
(( failed == 0 )) || {
  for log in "${LOGS}"/evaluate_gpu*.log "${LOGS}"/finalize_gpu*.log; do
    [[ -f ${log} ]] && { echo "===== ${log} =====" >&2; tail -80 "${log}" >&2; }
  done
  die "one or more shards are incomplete after postprocess recovery"
}

shards=()
for index in 0 1 2 3; do
  shards+=("$(shard_output "${GPU_IDS[$index]}" "${TASK_PARTITIONS[$index]}")")
done
record "aggregate state=start"
CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m \
  architectures.simvla.adapters.latentloop.action_equivalent_refresh.online_aggregate \
  --output "${MERGED}" --manifest "${MANIFEST}" \
  --expected-manifest-sha256 "${MANIFEST_SHA}" \
  --shards "${shards[@]}" \
  --full-control "${FULL_CONTROL}" \
  --generation-control "${GENERATION_CONTROL}" \
  --periodic-kc3-control "${PERIODIC_KC3_CONTROL}" \
  --periodic-kc4-control "${PERIODIC_KC4_CONTROL}" \
  >"${LOGS}/aggregate.log" 2>&1 || {
    tail -120 "${LOGS}/aggregate.log" >&2
    die "Long-500 aggregation failed"
  }
grep -q 'ACTION_EQUIVALENT_REFRESH_ONLINE_COMPLETE' "${MERGED}/online_comparison_summary.json" || die "completion verdict missing"
record "aggregate state=complete"
printf 'ACTION_EQUIVALENT_REFRESH_ONLINE_COMPLETE\n' | tee "${STATUS}"
"${PYTHON}" - "${MERGED}/online_comparison_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
c=d['candidate']
print(f"RESULT success={c['successes']}/{c['episodes']} ({100*c['success_rate']:.2f}%) exact_fraction={c['observed_exact_fraction']:.4f} effective_KC={c['effective_k_c']:.3f} latency_action_ms={c['latency_per_executed_action_ms']:.3f}")
print(f"SUMMARY {sys.argv[1]}")
PY
