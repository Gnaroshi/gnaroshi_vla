#!/usr/bin/env bash
# Run the bounded recursive-versus-teacher-forced condition-drift diagnostic.

set -uo pipefail

if [[ "${SIMVLA_CONDITION_DRIFT_P1_RUN:-0}" != "1" ]]; then
  echo "Refusing launch: export SIMVLA_CONDITION_DRIFT_P1_RUN=1" >&2
  exit 2
fi

ROOT=${SIMVLA_CONDITION_DRIFT_P1_ROOT:?Set SIMVLA_CONDITION_DRIFT_P1_ROOT}
PYTHON=${SIMVLA_CONDITION_DRIFT_P1_PYTHON:?Set SIMVLA_CONDITION_DRIFT_P1_PYTHON}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:?Set SIMVLA_UPSTREAM_ROOT}
STORAGE=${SIMVLA_CONDITION_DRIFT_P1_STORAGE:?Set SIMVLA_CONDITION_DRIFT_P1_STORAGE}
OUTPUT_ROOT=${SIMVLA_CONDITION_DRIFT_P1_OUTPUT:?Set SIMVLA_CONDITION_DRIFT_P1_OUTPUT}
GPU_IDS=${SIMVLA_GPU_IDS:-4,5,6,7}
EXPECTED_BRANCH=exp/simvla-condition-drift-p1-20260825
MODE=${1:---all}

BASE=$STORAGE/results/simvla/latentloop/correct_native_v0_seed20260815_v1
CACHE=$BASE/00_training_cache_libero10_r5
V0_CHECKPOINT=$BASE/08_train_150k/checkpoints/native_v0_step_150000.pt
NORM_STATS=$ROOT/architectures/simvla/adapters/latentloop/assets/libero_norm_official_32700d0.json
PARITY_GATE=$BASE/02_k1_parity/k1_parity.json
PARAMETER_GATE=$BASE/04_parameter_audit/simvla_v0_parameter_audit.json
TRAINING_GATE=$BASE/08_train_150k/run_summary.json
MODULE=architectures.simvla.adapters.latentloop.efficient_multirate.condition_drift_p1
CURRENT_OUTPUT=""

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if ((${#GPUS[@]} < 1 || ${#GPUS[@]} > 4)); then
  echo "SIMVLA_GPU_IDS must contain one to four distinct physical GPU IDs" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
  echo "SIMVLA_GPU_IDS contains duplicate GPU IDs" >&2
  exit 2
fi

gpu_pids() {
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | sed '/^[[:space:]]*$/d'
}

wait_for_gpus() {
  local gpu
  local -a busy=()
  while true; do
    busy=()
    for gpu in "${GPUS[@]}"; do
      [[ -z "$(gpu_pids "$gpu")" ]] || busy+=("$gpu")
    done
    if ((${#busy[@]} == 0)); then
      echo "[$(date --iso-8601=seconds)] GPUs ${GPUS[*]} are idle"
      return 0
    fi
    if [[ "${SIMVLA_WAIT_FOR_GPUS:-0}" != "1" ]]; then
      echo "Selected GPUs are busy: ${busy[*]}" >&2
      return 1
    fi
    echo "[$(date --iso-8601=seconds)] waiting=60s busy_gpus=${busy[*]}"
    sleep 60
  done
}

recover_if_complete() {
  local output=$1
  [[ -d "$output" && ! -f "$output/condition_drift_p1_summary.json" ]] || return 0
  echo "POSTPROCESS_RECOVERY output=$output"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -m "$MODULE" aggregate --output "$output"
}

on_exit() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)) && [[ -n "$CURRENT_OUTPUT" ]]; then
    recover_if_complete "$CURRENT_OUTPUT" || true
  fi
  exit "$rc"
}
trap on_exit EXIT

preflight() {
  test -x "$PYTHON"
  test -d "$UPSTREAM/evaluation/libero/LIBERO"
  test -f "$CACHE/manifest.json"
  test -f "$V0_CHECKPOINT"
  test -f "$NORM_STATS"
  test -f "$PARITY_GATE"
  test -f "$PARAMETER_GATE"
  test -f "$TRAINING_GATE"
  test "$(git -C "$ROOT" branch --show-current)" = "$EXPECTED_BRANCH"
  test -z "$(git -C "$ROOT" status --short)"
  if [[ ! -e "$ROOT/architectures/simvla/upstream" ]]; then
    ln -s "$UPSTREAM" "$ROOT/architectures/simvla/upstream"
  fi
  test "$(readlink -f "$ROOT/architectures/simvla/upstream")" = "$(readlink -f "$UPSTREAM")"
  "$PYTHON" -m pytest -q -p no:cacheprovider \
    tests/simvla_fixed_2x2/test_condition_drift_p1.py
}

run_one() {
  local name=$1 max_sequences=$2 port=$3 output=$OUTPUT_ROOT/$name rc
  CURRENT_OUTPUT=$output
  if [[ -f "$output/condition_drift_p1_summary.json" ]]; then
    echo "ALREADY_COMPLETE output=$output"
    return 0
  fi
  if [[ -e "$output" ]]; then
    recover_if_complete "$output" || {
      echo "Refusing unresolved partial output: $output" >&2
      return 1
    }
    [[ -f "$output/condition_drift_p1_summary.json" ]] && return 0
  fi
  wait_for_gpus
  mkdir -p "$OUTPUT_ROOT/logs"
  set +e
  "$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${#GPUS[@]}" --master-port="$port" \
    -m "$MODULE" evaluate \
    --output "$output" \
    --cache "$CACHE" \
    --v0-checkpoint "$V0_CHECKPOINT" \
    --checkpoint YuankaiLuo/SimVLA-LIBERO \
    --norm-stats "$NORM_STATS" \
    --smolvlm-model HuggingFaceTB/SmolVLM-500M-Instruct \
    --parity-gate "$PARITY_GATE" \
    --parameter-gate "$PARAMETER_GATE" \
    --training-gate "$TRAINING_GATE" \
    --heldout-fraction 0.2 \
    --split-seed 20260822 \
    --seed 20260815 \
    --max-sequences "$max_sequences" \
    --flush-interval 25 \
    --log-interval 25 \
    2>&1 | tee "$OUTPUT_ROOT/logs/$name.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf 'exit_code=%s\n' "$rc" > "$OUTPUT_ROOT/logs/$name.status"
  if ((rc != 0)); then
    recover_if_complete "$output" || true
    return "$rc"
  fi
  test -f "$output/condition_drift_p1_summary.json"
}

run_all() {
  set -euo pipefail
  cd "$ROOT"
  export PYTHONPATH="$ROOT:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HOME=${HF_HOME:-/home/mingyujung/private/gnaroshi_vla/.cache/huggingface}
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export PYTHONHASHSEED=20260815
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NVIDIA_TF32_OVERRIDE=0
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"

  echo "[1/3] P1 source, test, and artifact preflight"
  preflight
  case "$MODE" in
    --smoke)
      echo "[2/3] 16-sequence smoke"
      run_one smoke_16 16 29673
      ;;
    --full)
      echo "[3/3] full heldout diagnostic"
      run_one full_heldout 0 29674
      ;;
    --recover)
      recover_if_complete "$OUTPUT_ROOT/smoke_16"
      recover_if_complete "$OUTPUT_ROOT/full_heldout"
      ;;
    --all)
      echo "[2/3] 16-sequence smoke"
      run_one smoke_16 16 29673
      echo "[3/3] full heldout diagnostic"
      run_one full_heldout 0 29674
      ;;
    *)
      echo "usage: $0 [--smoke|--full|--recover|--all]" >&2
      return 2
      ;;
  esac
  echo "CONDITION_DRIFT_P1_COMPLETE output=$OUTPUT_ROOT"
}

run_all
