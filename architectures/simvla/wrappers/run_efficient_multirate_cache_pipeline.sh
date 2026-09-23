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
CHECKPOINT_REV=93dc4d90b0596c652ad2840ad743c62b9c4473fb
PARENT_LOCK=${PARENT}/08_train_150k/source_lock.json
PARENT_TRAIN_CONFIG=${PARENT}/08_train_150k/training_config.json

SOURCE_LOCK=${EXP}/00_source/source_lock.json
PROJECTION=${EXP}/01_cache_projection/projection.json
CACHE_PILOT=${EXP}/02_cache_pilot/exact_teacher_cache_pilot.json
EXACT_CACHE=${EXP}/03_exact_teacher_cache
CACHE_VALIDATION=${EXP}/03_cache_validation/cache_validation.json

if [[ ${SIMVLA_EFFICIENT_CACHE_PIPELINE_RUN:-0} != 1 ]]; then
  echo "Set SIMVLA_EFFICIENT_CACHE_PIPELINE_RUN=1 to approve this pipeline." >&2
  exit 2
fi
if [[ -z ${SIMVLA_GPU_IDS:-} ]]; then
  echo "Set SIMVLA_GPU_IDS to exactly two idle physical GPUs, for example 6,7." >&2
  exit 2
fi
if [[ -e ${EXP} ]]; then
  echo "Refusing existing result root: ${EXP}" >&2
  exit 2
fi

if [[ -n ${TMUX:-} ]]; then
  tmux set-option -p remain-on-exit on 2>/dev/null || true
fi

mkdir -p "${EXP}/logs"
LOG=${EXP}/logs/cache_pipeline.log
exec > >(tee -a "${LOG}") 2>&1

on_error() {
  local rc=$?
  echo "CACHE_PIPELINE_FAILED rc=${rc} line=${BASH_LINENO[0]} command=${BASH_COMMAND}" >&2
  echo "Inspect: ${LOG}" >&2
  exit "${rc}"
}
trap on_error ERR
trap 'unset SIMVLA_EXACT_CACHE_GENERATION_APPROVED' EXIT

export SIMVLA_EFFICIENT_MULTIRATE_RUN=1
export HF_HOME=${HF_HOME:-${ROOT}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCE_FILES=(
  architectures/simvla/adapters/latentloop/efficient_multirate/__init__.py
  architectures/simvla/adapters/latentloop/efficient_multirate/contracts.py
  architectures/simvla/adapters/latentloop/efficient_multirate/decisions.py
  architectures/simvla/adapters/latentloop/efficient_multirate/efficient_delta.py
  architectures/simvla/adapters/latentloop/efficient_multirate/efficient_v0_long_eval.py
  architectures/simvla/adapters/latentloop/efficient_multirate/efficient_v0_offline.py
  architectures/simvla/adapters/latentloop/efficient_multirate/efficient_v0_train.py
  architectures/simvla/adapters/latentloop/efficient_multirate/exact_teacher_cache.py
  architectures/simvla/adapters/latentloop/efficient_multirate/generation_audit.py
  architectures/simvla/adapters/latentloop/efficient_multirate/generation_hidden.py
  architectures/simvla/adapters/latentloop/efficient_multirate/generation_objective.py
  architectures/simvla/adapters/latentloop/efficient_multirate/gpu_contract.py
  architectures/simvla/adapters/latentloop/efficient_multirate/lineage_bridge.py
  architectures/simvla/adapters/latentloop/efficient_multirate/v0_objectives.py
  architectures/simvla/wrappers/simvla_efficient_multirate.sh
  methods/latentloop/modules/simvla_generation_loop.py
)
SOURCE_ARGS=()
for file in "${SOURCE_FILES[@]}"; do
  SOURCE_ARGS+=(--source-file "${file}")
done

echo "[1/5] Create source lock"
bash "${WRAP}" source-lock \
  --output "${SOURCE_LOCK}" \
  --repository "${ROOT}" \
  --parent-source-lock "${PARENT_LOCK}" \
  --parent-training-config "${PARENT_TRAIN_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-revision "${CHECKPOINT_REV}" \
  --norm-stats "${NORM}" \
  --compact-cache "${COMPACT}" \
  "${SOURCE_ARGS[@]}"

echo "[2/5] Project exact FP32 cache storage"
bash "${WRAP}" cache-project \
  --output "${PROJECTION}" \
  --compact-cache "${COMPACT}" \
  --source-lock "${SOURCE_LOCK}" \
  --shared-storage-path "${SHARED}" \
  --shard-queries 1024
"${PY}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["verdict"]=="EXACT_FP32_STORAGE_GATE_PASS",p; print(p["verdict"])' "${PROJECTION}"

echo "[3/5] Re-run bounded identity pilot under the fixed source lock"
bash "${WRAP}" cache-pilot \
  --output "${EXP}/02_cache_pilot" \
  --compact-cache "${COMPACT}" \
  --source-lock "${SOURCE_LOCK}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM}" \
  --pilot-windows 2 \
  --decode-batch-size 4 \
  --action-noise-seed-base 20260822
"${PY}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["verdict"]=="EXACT_TEACHER_CACHE_PASS",p; assert p["condition_max_abs_difference"]==0.0,p; assert p["action_max_abs_difference"]==0.0,p; print(p["verdict"])' "${CACHE_PILOT}"

echo "[4/5] Generate approved two-rank production exact cache"
export SIMVLA_EXACT_CACHE_GENERATION_APPROVED=1
bash "${WRAP}" cache-generate \
  --output "${EXACT_CACHE}" \
  --compact-cache "${COMPACT}" \
  --source-lock "${SOURCE_LOCK}" \
  --pilot-gate "${CACHE_PILOT}" \
  --checkpoint "${CHECKPOINT}" \
  --norm-stats "${NORM}" \
  --shard-queries 1024 \
  --decode-batch-size 4 \
  --action-noise-seed-base 20260822
unset SIMVLA_EXACT_CACHE_GENERATION_APPROVED

echo "[5/5] Verify every production shard checksum"
bash "${WRAP}" cache-validate \
  --output "${CACHE_VALIDATION}" \
  --cache "${EXACT_CACHE}" \
  --verify-checksums
"${PY}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["verdict"]=="EXACT_TEACHER_CACHE_VALID",p; print(p["verdict"],"queries=",p["queries"],"shards=",p["shards"])' "${CACHE_VALIDATION}"

echo "SIMVLA_EFFICIENT_EXACT_CACHE_PIPELINE_PASS"
echo "result_root=${EXP}"
