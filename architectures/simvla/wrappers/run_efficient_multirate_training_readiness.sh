#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=/home/mingyujung/private/gnaroshi_vla
PY=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
SHARED=${SIMVLA_SHARED_RESULTS_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop}
PARENT=${SHARED}/correct_native_v0_seed20260815_v1
EXP=${SIMVLA_EFFICIENT_CACHE_RESULT_ROOT:-${SHARED}/simvla_efficient_coupled_multirate_latentloop_sigfix_v1}
WRAP=${ROOT}/architectures/simvla/wrappers/simvla_efficient_multirate.sh

COMPACT=${PARENT}/00_training_cache_libero10_r5
NORM=${ROOT}/architectures/simvla/adapters/latentloop/assets/libero_norm_official_32700d0.json
CHECKPOINT=YuankaiLuo/SimVLA-LIBERO
SOURCE_LOCK=${EXP}/00_source/source_lock.json
CACHE_PILOT=${EXP}/02_cache_pilot/exact_teacher_cache_pilot.json
EXACT_CACHE=${EXP}/03_exact_teacher_cache
CACHE_GATE=${EXACT_CACHE}/cache_generation_complete.json
CACHE_VALIDATION=${EXP}/03_cache_validation/cache_validation.json

PARENT_TRAIN_CONFIG=${PARENT}/08_train_150k/training_config.json
PARENT_MODE_AB=${PARENT}/05_mode_ab/decision/mode_ab_decision.json
APPROVED_WEIGHTS=${PARENT}/06_loss_calibration/approved_loss_weights.json
MONITOR_REFERENCE=${PARENT}/06_loss_calibration/loss_scale_calibration.json

MODE_AB=${EXP}/04_mode_ab/mode_ab_decision.json
BATCH_GATE=${EXP}/05_batch/selected_batch_contract.json
MODE_B_BENCH=${EXP}/05_batch/mode_b_benchmark_1200
MODE_D_BENCH=${EXP}/06_mode_d/mode_d_benchmark_1200
MODE_D_GATE=${EXP}/06_mode_d/mode_d_decision.json
WALLCLOCK_GATE=${EXP}/07_wallclock/wallclock_gate.json

if [[ ${SIMVLA_EFFICIENT_TRAINING_READINESS_RUN:-0} != 1 ]]; then
  echo "Set SIMVLA_EFFICIENT_TRAINING_READINESS_RUN=1 to enable this bounded pipeline." >&2
  exit 2
fi
if [[ -z ${SIMVLA_GPU_IDS:-} ]]; then
  echo "Set SIMVLA_GPU_IDS to exactly two idle physical GPUs, for example 6,7." >&2
  exit 2
fi

if [[ -n ${TMUX:-} ]]; then
  tmux set-option -p remain-on-exit on 2>/dev/null || true
fi

mkdir -p "${EXP}/logs"
LOG=${EXP}/logs/training_readiness.log
exec > >(tee -a "${LOG}") 2>&1

on_error() {
  local rc=$?
  echo "TRAINING_READINESS_FAILED rc=${rc} line=${BASH_LINENO[0]} command=${BASH_COMMAND}" >&2
  echo "Inspect: ${LOG}" >&2
  exit "${rc}"
}
trap on_error ERR

export SIMVLA_EFFICIENT_MULTIRATE_RUN=1
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for path in "${SOURCE_LOCK}" "${CACHE_PILOT}" "${CACHE_GATE}" "${CACHE_VALIDATION}"; do
  [[ -s ${path} ]] || { echo "Missing prerequisite: ${path}" >&2; exit 2; }
done
for path in "${MODE_AB}" "${BATCH_GATE}" "${MODE_B_BENCH}" "${MODE_D_BENCH}" "${MODE_D_GATE}" "${WALLCLOCK_GATE}"; do
  [[ ! -e ${path} ]] || { echo "Refusing existing stage output: ${path}" >&2; exit 2; }
done

"${PY}" -c 'import json,sys; source=json.load(open(sys.argv[1]))["combined_sha256"]; complete=json.load(open(sys.argv[2])); valid=json.load(open(sys.argv[3])); assert complete["verdict"]=="EXACT_TEACHER_CACHE_COMPLETE",complete; assert valid["verdict"]=="EXACT_TEACHER_CACHE_VALID",valid; assert complete["source_combined_sha256"]==valid["source_combined_sha256"]==source; print("EXACT_CACHE_PREREQUISITES_PASS",source)' "${SOURCE_LOCK}" "${CACHE_GATE}" "${CACHE_VALIDATION}"

echo "[1/5] Issue source-identical Mode-B gate"
bash "${WRAP}" mode-ab \
  --output "${MODE_AB}" \
  --source-lock "${SOURCE_LOCK}" \
  --pilot-gate "${CACHE_PILOT}" \
  --cache-gate "${CACHE_GATE}" \
  --parent-mode-ab-gate "${PARENT_MODE_AB}"

echo "[2/5] Issue effective-unique-global-batch-one gate"
bash "${WRAP}" batch \
  --output "${BATCH_GATE}" \
  --source-lock "${SOURCE_LOCK}"
"${PY}" -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); assert a["verdict"]=="MODE_B_APPROVED",a; assert b["verdict"]=="BATCH_CONFIGURATION_SELECTED" and b["effective_unique_global_batch"]==1,b; print(a["verdict"],b["verdict"])' "${MODE_AB}" "${BATCH_GATE}"

echo "[3/5] Measure optimized Mode-B for 1,200 steps"
bash "${WRAP}" benchmark \
  --output "${MODE_B_BENCH}" \
  --cache "${EXACT_CACHE}" \
  --source-lock "${SOURCE_LOCK}" \
  --cache-gate "${CACHE_GATE}" \
  --objective-gate "${MODE_AB}" \
  --batch-gate "${BATCH_GATE}" \
  --parent-training-config "${PARENT_TRAIN_CONFIG}" \
  --approved-weights "${APPROVED_WEIGHTS}" \
  --monitor-reference "${MONITOR_REFERENCE}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM}" \
  --objective-mode B \
  --benchmark \
  --max-steps 1200 \
  --peak-lr 1e-4 \
  --weight-decay 0 \
  --num-workers 2 \
  --prefetch-factor 2 \
  --log-interval 100 \
  --validation-interval 1200 \
  --validation-batches 16 \
  --save-interval 10000 \
  --profile-warmup-steps 100 \
  --profile-log-interval 1000

MODE_B_HOURS=$("${PY}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["measured_steps"]>=1000,p; print(p["projected_150k_hours_from_run"])' "${MODE_B_BENCH}/run_summary.json")
echo "mode_b_projected_150k_hours=${MODE_B_HOURS}"

NEEDS_MODE_D=$("${PY}" -c 'import sys; print(int(float(sys.argv[1])>12.0))' "${MODE_B_HOURS}")
if [[ ${NEEDS_MODE_D} == 0 ]]; then
  echo "[4/5] Mode B meets the 12-hour target; Mode D is not required"
  bash "${WRAP}" mode-d-not-required \
    --output "${MODE_D_GATE}" \
    --source-lock "${SOURCE_LOCK}" \
    --mode-b-summary "${MODE_B_BENCH}/run_summary.json" \
    --amortized-overhead-seconds 60
  SELECTED_MODE=B
  SELECTED_BENCH=${MODE_B_BENCH}
  SELECTED_OBJECTIVE_GATE=${MODE_AB}
else
  echo "[4/5] Mode B exceeds 12 hours; run matched Mode-D benchmark"
  bash "${WRAP}" benchmark \
    --output "${MODE_D_BENCH}" \
    --cache "${EXACT_CACHE}" \
    --source-lock "${SOURCE_LOCK}" \
    --cache-gate "${CACHE_GATE}" \
    --objective-gate "${MODE_AB}" \
    --batch-gate "${BATCH_GATE}" \
    --parent-training-config "${PARENT_TRAIN_CONFIG}" \
    --approved-weights "${APPROVED_WEIGHTS}" \
    --monitor-reference "${MONITOR_REFERENCE}" \
    --checkpoint "${CHECKPOINT}" \
    --norm-stats "${NORM}" \
    --objective-mode D \
    --benchmark \
    --max-steps 1200 \
    --peak-lr 1e-4 \
    --weight-decay 0 \
    --num-workers 2 \
    --prefetch-factor 2 \
    --log-interval 100 \
    --validation-interval 1200 \
    --validation-batches 16 \
    --save-interval 10000 \
    --profile-warmup-steps 100 \
    --profile-log-interval 1000
  bash "${WRAP}" mode-d \
    --output "${MODE_D_GATE}" \
    --source-lock "${SOURCE_LOCK}" \
    --mode-b-summary "${MODE_B_BENCH}/run_summary.json" \
    --mode-d-summary "${MODE_D_BENCH}/run_summary.json"
  MODE_D_VERDICT=$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "${MODE_D_GATE}")
  if [[ ${MODE_D_VERDICT} == MODE_D_APPROVED ]]; then
    SELECTED_MODE=D
    SELECTED_BENCH=${MODE_D_BENCH}
    SELECTED_OBJECTIVE_GATE=${MODE_D_GATE}
  else
    SELECTED_MODE=B
    SELECTED_BENCH=${MODE_B_BENCH}
    SELECTED_OBJECTIVE_GATE=${MODE_AB}
  fi
fi

echo "[5/5] Issue measured wall-clock gate for selected Mode ${SELECTED_MODE}"
bash "${WRAP}" wallclock \
  --output "${WALLCLOCK_GATE}" \
  --source-lock "${SOURCE_LOCK}" \
  --throughput-summary "${SELECTED_BENCH}/run_summary.json" \
  --objective-gate "${SELECTED_OBJECTIVE_GATE}" \
  --amortized-overhead-seconds 60

WALLCLOCK_VERDICT=$("${PY}" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["verdict"])' "${WALLCLOCK_GATE}")
echo "selected_mode=${SELECTED_MODE}"
echo "wallclock_verdict=${WALLCLOCK_VERDICT}"
echo "SIMVLA_EFFICIENT_TRAINING_READINESS_COMPLETE"
echo "result_root=${EXP}"
