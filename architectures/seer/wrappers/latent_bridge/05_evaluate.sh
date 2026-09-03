#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
preset="${BRIDGE_PRESET:-full}"
bridge="${LATENT_BRIDGE_RESULT_ROOT}/r1_${preset}/training/best.pt"
[[ -s "${bridge}" ]] || latent_bridge_fail "R1 best checkpoint is absent: ${bridge}"
stage="${LATENT_BRIDGE_RESULT_ROOT}/evaluation_${preset}"
if ! latent_bridge_prepare_stage "${stage}"; then exit 0; fi
read -r -a seeds <<< "${EVAL_SEEDS:-42 43 44}"
read -r -a periods <<< "${REFRESH_PERIODS:-2 3 4}"
comparisons=()
base_port="${MASTER_PORT_BASE:-18100}"
port="$((base_port + 100))"

for seed in "${seeds[@]}"; do
    seed_root="${stage}/seed${seed}"
    mkdir -p "${seed_root}"
    manifest="${seed_root}/exact_manifest.json"
    python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/make_eval_manifest.py" \
        --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" --checkpoint "${LATENT_BRIDGE_PUBLIC33}" \
        --output "${manifest}" --seed "${seed}" --num-tasks 10 --episodes-per-task 20 \
        --renderer osmesa

    baseline="${seed_root}/f1_frozen_seer"
    mkdir -p "${baseline}"
    latent_bridge_run_eval \
        architectures.seer.adapters.latent_bridge.baseline_entry \
        "${baseline}" "public33_f1_seed${seed}" "${seed}" 20 10 "${port}"
    port=$((port + 10))
    row_args=(--row "f1=${baseline}")
    for period in "${periods[@]}"; do
        row="${seed_root}/f${period}_latent_bridge"
        mkdir -p "${row}"
        export SEER_LATENT_BRIDGE_CHECKPOINT="${bridge}"
        export SEER_LATENT_BRIDGE_REFRESH_PERIOD="${period}"
        export SEER_LATENT_BRIDGE_PRECISION=bf16
        export SEER_LATENT_BRIDGE_COMPILE=1
        unset SEER_LATENT_BRIDGE_DAGGER_OUTPUT || true
        latent_bridge_run_eval \
            architectures.seer.adapters.latent_bridge.evaluation_entry \
            "${row}" "public33_bridge_f${period}_seed${seed}" "${seed}" 20 10 "${port}"
        port=$((port + 10))
        row_args+=(--row "f${period}=${row}")
    done
    python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/aggregate_evaluations.py" \
        --manifest "${manifest}" "${row_args[@]}" --baseline f1 \
        --output-dir "${seed_root}/comparison"
    comparisons+=(--comparison "${seed_root}/comparison/comparison.json")
done

python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/aggregate_seeds.py" \
    "${comparisons[@]}" --output-dir "${stage}/three_seed_summary"
python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/latency_benchmark.py" \
    --checkpoint "${LATENT_BRIDGE_PUBLIC33}" --vit-checkpoint "${LATENT_BRIDGE_VIT}" \
    --dataset-root "${LATENT_BRIDGE_DATASET_ROOT}" --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" \
    --bridge-checkpoint "${bridge}" --output "${stage}/component_latency_rtx3090.json"
printf 'SEER_LATENT_BRIDGE_EVALUATION_COMPLETE\n' > "${stage}/COMPLETE"
