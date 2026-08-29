#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODE=${1:---all}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}
PYTHON=${SIMVLA_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
RESULTS_ROOT=${SIMVLA_ACTION_REFRESH_RESULTS:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/action_equivalent_refresh/primary_v1}
GPU_IDS_RAW=${SIMVLA_ACTION_REFRESH_GPUS:-"4 5 6 7"}
read -r -a GPU_IDS <<<"${GPU_IDS_RAW//,/ }"

CACHE=${SIMVLA_ACTION_REFRESH_CACHE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/03_exact_teacher_cache}
CONDITION=${SIMVLA_ACTION_REFRESH_CONDITION:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/checkpoints/native_v0_step_150000.pt}
GENERATION=${SIMVLA_ACTION_REFRESH_GENERATION:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/artifacts/simvla/generation_eval_bundle_20260824_v1/checkpoint/generation_step_030000.pt}
NORM=${SIMVLA_ACTION_REFRESH_NORM:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream/norm_stats/libero_norm.json}

COMPACT=${RESULTS_ROOT}/compact
LOGS=${RESULTS_ROOT}/logs
TRAIN=${RESULTS_ROOT}/risk_head_2k
STATUS=${RESULTS_ROOT}/pipeline.status
mkdir -p "${COMPACT}/shards" "${LOGS}" "${RESULTS_ROOT}/failed_attempts"

export PYTHONPATH="${ROOT}:${UPSTREAM}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=20260827
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_MAX_CONNECTIONS=1

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
record() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${LOGS}/pipeline.log"; }

fail() {
  local rc=$?
  printf 'ACTION_EQUIVALENT_REFRESH_FAILED rc=%s line=%s\n' "${rc}" "${BASH_LINENO[0]}" | tee "${STATUS}"
  printf 'Inspect logs: %s\n' "${LOGS}"
  exit "${rc}"
}
trap fail ERR

if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "Exactly four sd1 GPUs are required; received: ${GPU_IDS[*]}" >&2
  exit 2
fi
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! ${gpu} =~ ^[4-7]$ ]]; then
    echo "sd1 contract permits only GPUs 4,5,6,7; received ${gpu}" >&2
    exit 2
  fi
done

if [[ ${MODE} != --all && ${MODE} != --preflight ]]; then
  echo "Usage: $0 [--preflight|--all]" >&2
  exit 2
fi

for path in "${PYTHON}" "${CACHE}/manifest.json" "${CONDITION}" "${GENERATION}" "${NORM}"; do
  test -e "${path}"
done
test -d "${UPSTREAM}"

for gpu in "${GPU_IDS[@]}"; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if (( used > 512 )); then
    echo "GPU ${gpu} is not free (${used} MiB used); refusing to collide with another process." >&2
    exit 3
  fi
done

{
  echo "started_at=$(timestamp)"
  echo "root=${ROOT}"
  echo "branch=$(git -C "${ROOT}" branch --show-current)"
  echo "commit=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "gpus=${GPU_IDS[*]}"
  echo "cache=${CACHE}"
  echo "condition=${CONDITION}"
  echo "generation=${GENERATION}"
  echo "norm=${NORM}"
} >"${RESULTS_ROOT}/run_contract.env"

"${PYTHON}" -m architectures.simvla.adapters.latentloop.action_equivalent_refresh.dataset_builder --help >/dev/null
"${PYTHON}" -m architectures.simvla.adapters.latentloop.action_equivalent_refresh.training --help >/dev/null
"${PYTHON}" -m architectures.simvla.adapters.latentloop.action_equivalent_refresh.offline_evaluator --help >/dev/null
if [[ ${MODE} == --preflight ]]; then
  printf 'ACTION_EQUIVALENT_REFRESH_PREFLIGHT_PASS\n' | tee "${STATUS}"
  printf 'GPUs: %s\nResults: %s\n' "${GPU_IDS[*]}" "${RESULTS_ROOT}"
  exit 0
fi

is_complete() {
  local summary=$1 verdict=$2
  test -s "${summary}" && grep -q "\"verdict\": \"${verdict}\"" "${summary}"
}

isolate_incomplete() {
  local path=$1 summary=$2
  if [[ -e ${path} ]] && ! is_complete "${summary}" ACTION_FIDELITY_COMPACT_SHARD_COMPLETE; then
    local failed="${RESULTS_ROOT}/failed_attempts/$(date +%Y%m%d_%H%M%S)_$(basename "${path}")"
    mv "${path}" "${failed}"
    [[ -e ${summary} ]] && mv "${summary}" "${failed}.summary.json"
  fi
}

extract_split() {
  local split=$1
  local split_root="${COMPACT}/shards/${split}"
  mkdir -p "${split_root}"
  local pids=() labels=()
  for shard in 0 1 2 3; do
    local gpu=${GPU_IDS[$shard]}
    local output="${split_root}/shard$(printf '%02d' "${shard}").pt"
    local summary="${output%.pt}.summary.json"
    local log="${LOGS}/extract_${split}_shard$(printf '%02d' "${shard}").log"
    if [[ -s ${output} ]] && is_complete "${summary}" ACTION_FIDELITY_COMPACT_SHARD_COMPLETE; then
      record "split=${split} shard=${shard} state=resume_skip"
      continue
    fi
    isolate_incomplete "${output}" "${summary}"
    record "split=${split} shard=${shard} gpu=${gpu} state=start"
    CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON}" -m \
      architectures.simvla.adapters.latentloop.action_equivalent_refresh.dataset_builder \
      extract \
      --cache "${CACHE}" \
      --split "${split}" \
      --output "${output}" \
      --norm-stats "${NORM}" \
      --condition-checkpoint "${CONDITION}" \
      --generation-checkpoint "${GENERATION}" \
      --split-seed 20260822 \
      --seed 20260827 \
      --shard-index "${shard}" \
      --num-shards 4 \
      --batch-size 1 \
      --num-workers 0 \
      --device cuda >"${log}" 2>&1 &
    pids+=("$!")
    labels+=("${split}/shard${shard}")
  done
  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      record "job=${labels[$index]} state=complete"
    else
      record "job=${labels[$index]} state=failed"
      failed=1
    fi
  done
  if (( failed )); then
    for log in "${LOGS}"/extract_"${split}"_shard*.log; do
      echo "===== ${log} =====" >&2
      tail -80 "${log}" >&2 || true
    done
    return 1
  fi
  for shard in 0 1 2 3; do
    local summary="${split_root}/shard$(printf '%02d' "${shard}").summary.json"
    is_complete "${summary}" ACTION_FIDELITY_COMPACT_SHARD_COMPLETE
  done
}

merge_split() {
  local split=$1
  local output="${COMPACT}/${split}.pt"
  local summary="${COMPACT}/${split}.summary.json"
  if [[ -s ${output} ]] && is_complete "${summary}" ACTION_FIDELITY_COMPACT_MERGE_COMPLETE; then
    record "split=${split} merge=resume_skip"
    return
  fi
  if [[ -e ${output} ]]; then
    mv "${output}" "${RESULTS_ROOT}/failed_attempts/$(date +%Y%m%d_%H%M%S)_${split}.pt"
  fi
  record "split=${split} merge=start"
  CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m \
    architectures.simvla.adapters.latentloop.action_equivalent_refresh.dataset_builder \
    merge \
    --split "${split}" \
    --inputs "${COMPACT}/shards/${split}"/shard*.pt \
    --output "${output}" >"${LOGS}/merge_${split}.log" 2>&1
  is_complete "${summary}" ACTION_FIDELITY_COMPACT_MERGE_COMPLETE
  record "split=${split} merge=complete"
}

for split in train checkpoint_validation final_offline; do
  extract_split "${split}"
  merge_split "${split}"
done

if [[ ! -s ${TRAIN}/action_fidelity_head.pt ]] || [[ ! -s ${TRAIN}/training_summary.json ]] || ! grep -q 'ACTION_FIDELITY_HEAD_TRAINING_COMPLETE' "${TRAIN}/training_summary.json"; then
  record "risk_head_2k gpu=${GPU_IDS[0]} state=start"
  mkdir -p "${TRAIN}"
  CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} "${PYTHON}" -m \
    architectures.simvla.adapters.latentloop.action_equivalent_refresh.training \
    --train-data "${COMPACT}/train.pt" \
    --validation-data "${COMPACT}/checkpoint_validation.pt" \
    --output "${TRAIN}" \
    --device cuda \
    --max-steps 2000 \
    --batch-size 256 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --seed 20260827 \
    --target-exact-fraction 0.3333333333333333 \
    >"${LOGS}/risk_head_2k.log" 2>&1
  grep -q 'ACTION_FIDELITY_HEAD_TRAINING_COMPLETE' "${TRAIN}/training_summary.json"
  record "risk_head_2k state=complete"
else
  record "risk_head_2k state=resume_skip"
fi

OFFLINE=${RESULTS_ROOT}/final_offline_comparison.json
if [[ ! -s ${OFFLINE} ]] || ! grep -q 'OFFLINE_ACTION_FIDELITY_COMPARISON_COMPLETE' "${OFFLINE}"; then
  record "final_offline_comparison state=start"
  CUDA_VISIBLE_DEVICES='' "${PYTHON}" -m \
    architectures.simvla.adapters.latentloop.action_equivalent_refresh.offline_evaluator \
    --checkpoint "${TRAIN}/action_fidelity_head.pt" \
    --validation-data "${COMPACT}/checkpoint_validation.pt" \
    --final-data "${COMPACT}/final_offline.pt" \
    --output "${OFFLINE}" \
    --device cpu >"${LOGS}/final_offline_comparison.log" 2>&1
  grep -q 'OFFLINE_ACTION_FIDELITY_COMPARISON_COMPLETE' "${OFFLINE}"
  record "final_offline_comparison state=complete"
else
  record "final_offline_comparison state=resume_skip"
fi

printf 'ACTION_EQUIVALENT_REFRESH_OFFLINE_COMPLETE\n' | tee "${STATUS}"
printf 'Results: %s\n' "${RESULTS_ROOT}"
printf 'No LIBERO online evaluation was launched.\n'
