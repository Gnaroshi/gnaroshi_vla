#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:---all}"
case "${MODE}" in
  --preflight|--smoke|--all) ;;
  *) echo "usage: $0 [--preflight|--smoke|--all]" >&2; exit 2 ;;
esac

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
STORAGE="${SIMVLA_STORAGE_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage}"
UPSTREAM="${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}"
PYTHON="${SIMVLA_PYTHON:-${STORAGE}/envs/simvla/libero_mujoco237/bin/python}"
GPU="${SIMVLA_ANALYSIS_GPU:-0}"
MIN_FREE_MIB="${SIMVLA_ANALYSIS_MIN_FREE_MIB:-28000}"
WAIT_SECONDS="${SIMVLA_ANALYSIS_WAIT_SECONDS:-120}"
RESULT_ROOT="${SIMVLA_ANALYSIS_RESULT_ROOT:-${STORAGE}/results/simvla/analysis/reviewer_mechanism_v1}"
CACHE="${SIMVLA_EXACT_CACHE:-${STORAGE}/results/simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/03_exact_teacher_cache}"
CONDITION_CKPT="${SIMVLA_CONDITION_CHECKPOINT:-${STORAGE}/artifacts/simvla/fixed_2x2_inputs_v1/condition/native_v0_step_150000.pt}"
PARENT_GENERATION_CKPT="${SIMVLA_PARENT_GENERATION_CHECKPOINT:-${STORAGE}/artifacts/simvla/generation_eval_bundle_20260824_v1/checkpoint/generation_step_030000.pt}"
COUPLED_GENERATION_CKPT="${SIMVLA_COUPLED_GENERATION_CHECKPOINT:-${STORAGE}/results/simvla/coupled_condition_generation/kc2_ng3_real_cj_projection10k_seed02_v1/train/projection_10k/checkpoints/coupled_generation_step_010000.pt}"
NORM_STATS="${SIMVLA_NORM_STATS:-${UPSTREAM}/norm_stats/libero_norm.json}"

mkdir -p "${RESULT_ROOT}/logs"
STATUS="${RESULT_ROOT}/pipeline.status"
LOG="${RESULT_ROOT}/logs/launcher.log"
exec > >(tee -a "${LOG}") 2>&1

on_error() {
  local rc=$?
  printf 'verdict=LATENT_FIDELITY_ANALYSIS_FAILED\nexit_code=%s\nstage=%s\n' \
    "${rc}" "${STAGE:-unknown}" > "${STATUS}"
  echo "LATENT_FIDELITY_ANALYSIS_FAILED stage=${STAGE:-unknown} rc=${rc}"
  exit "${rc}"
}
trap on_error ERR

preflight() {
  STAGE=preflight
  for path in \
    "${PYTHON}" \
    "${UPSTREAM}/models/modeling_smolvlm_vla.py" \
    "${CACHE}/manifest.json" \
    "${CONDITION_CKPT}" \
    "${PARENT_GENERATION_CKPT}" \
    "${COUPLED_GENERATION_CKPT}" \
    "${NORM_STATS}"; do
    test -e "${path}" || { echo "missing required path: ${path}" >&2; return 1; }
  done
  test -x "${PYTHON}"
  test "$(hostname)" = "jbr-TRX50" || {
    echo "this launcher is locked to rb2 host jbr-TRX50" >&2
    return 1
  }
  PYTHONPATH="${ROOT}:${UPSTREAM}" "${PYTHON}" -c \
    'import torch; from architectures.simvla.adapters.latentloop.efficient_multirate.latent_fidelity_analysis import build_parser; assert torch.cuda.is_available(); print("IMPORT_PASS", torch.__version__)'
  PYTHONPATH="${ROOT}:${UPSTREAM}" "${PYTHON}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("complete") is True; assert int(p.get("window_count",0)) == 6525; print("CACHE_MANIFEST_PASS", p["query_count"], p["window_count"])' \
    "${CACHE}/manifest.json"
  echo "PREFLIGHT_PASS"
}

wait_for_gpu() {
  STAGE=wait_for_gpu
  while true; do
    local free
    free="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free}" =~ ^[0-9]+$ ]] && (( free >= MIN_FREE_MIB )); then
      echo "GPU_READY gpu=${GPU} free_mib=${free}"
      return 0
    fi
    echo "GPU_WAIT gpu=${GPU} free_mib=${free:-unknown} required_mib=${MIN_FREE_MIB} sleep=${WAIT_SECONDS}s"
    sleep "${WAIT_SECONDS}"
  done
}

run_analysis() {
  local label=$1
  local condition_windows=$2
  local action_queries=$3
  local generation_queries=$4
  local schedule_queries=$5
  local latency_warmup=$6
  local latency_repeats=$7
  local final="${RESULT_ROOT}/${label}"
  local summary="${final}/latent_fidelity_analysis.json"
  if test -f "${summary}" && grep -q 'LATENT_FIDELITY_ANALYSIS_COMPLETE' "${summary}"; then
    echo "SKIP_COMPLETE label=${label} output=${final}"
    return 0
  fi
  test ! -e "${final}" || {
    echo "refusing incomplete/existing output: ${final}" >&2
    return 1
  }
  local staging="${RESULT_ROOT}/.${label}.incomplete.$$.${RANDOM}"
  STAGE="${label}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH="${ROOT}:${UPSTREAM}" \
  SIMVLA_UPSTREAM_ROOT="${UPSTREAM}" \
  HF_HOME="${HF_HOME:-/home/mingyujung/private/gnaroshi_vla/.cache/huggingface}" \
  TOKENIZERS_PARALLELISM=false \
  PYTHONHASHSEED=20260831 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" -m architectures.simvla.adapters.latentloop.efficient_multirate.latent_fidelity_analysis \
    --output "${staging}" \
    --cache "${CACHE}" \
    --condition-checkpoint "${CONDITION_CKPT}" \
    --parent-generation-checkpoint "${PARENT_GENERATION_CKPT}" \
    --coupled-generation-checkpoint "${COUPLED_GENERATION_CKPT}" \
    --norm-stats "${NORM_STATS}" \
    --condition-windows "${condition_windows}" \
    --action-queries "${action_queries}" \
    --generation-queries "${generation_queries}" \
    --schedule-queries "${schedule_queries}" \
    --latency-warmup "${latency_warmup}" \
    --latency-repeats "${latency_repeats}" \
    --batch-size 4 \
    --seed 20260831 \
    --physical-gpu "${GPU}"
  test -f "${staging}/latent_fidelity_analysis.json"
  test -f "${staging}/generation_method_fidelity_rows.csv"
  test -f "${staging}/gate_ablation_rows.csv"
  test -f "${staging}/generation_schedule_rows.csv"
  test -f "${staging}/latency_samples.csv"
  grep -q 'LATENT_FIDELITY_ANALYSIS_COMPLETE' "${staging}/latent_fidelity_analysis.json"
  mv "${staging}" "${final}"
  echo "ANALYSIS_COMPLETE label=${label} output=${final}"
}

preflight
if [[ "${MODE}" == "--preflight" ]]; then
  printf 'verdict=LATENT_FIDELITY_PREFLIGHT_PASS\nexit_code=0\nstage=preflight\n' > "${STATUS}"
  exit 0
fi

wait_for_gpu
run_analysis smoke 8 8 4 4 1 3
if [[ "${MODE}" == "--smoke" ]]; then
  printf 'verdict=LATENT_FIDELITY_SMOKE_COMPLETE\nexit_code=0\nstage=smoke\n' > "${STATUS}"
  exit 0
fi

run_analysis full 0 512 512 128 10 50
printf 'verdict=LATENT_FIDELITY_ANALYSIS_COMPLETE\nexit_code=0\nstage=complete\nresult=%s\n' \
  "${RESULT_ROOT}/full" > "${STATUS}"
echo "LATENT_FIDELITY_ANALYSIS_COMPLETE output=${RESULT_ROOT}/full"
