#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
latent_bridge_require_runtime
latent_bridge_efficient_verify_source_lock

preset="${BRIDGE_PRESET:-full}"
bridge="${LATENT_BRIDGE_RESULT_ROOT}/r1_${preset}/training/best.pt"
[[ -s "${bridge}" ]] || latent_bridge_fail "efficient R1 best checkpoint is absent: ${bridge}"
stage="${LATENT_BRIDGE_RESULT_ROOT}/evaluation_${preset}"
if ! latent_bridge_prepare_stage "${stage}" resume; then exit 0; fi
read -r -a seeds <<< "${EVAL_SEEDS:-42}"
read -r -a periods <<< "${REFRESH_PERIODS:-2 3 4}"
episodes_per_task="${EVAL_EPISODES_PER_TASK:-50}"
num_tasks="${EVAL_NUM_TASKS:-10}"
comparisons=()
port="$(( ${MASTER_PORT_BASE:-18200} + 100 ))"

for seed in "${seeds[@]}"; do
    seed_root="${stage}/seed${seed}"
    mkdir -p "${seed_root}"
    manifest="${seed_root}/exact_manifest.json"
    python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/make_eval_manifest.py" \
        --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" --checkpoint "${LATENT_BRIDGE_PUBLIC33}" \
        --output "${manifest}" --seed "${seed}" --num-tasks "${num_tasks}" \
        --episodes-per-task "${episodes_per_task}" \
        --renderer "${LATENT_BRIDGE_RENDERER}" --reuse-if-matching

    baseline="${seed_root}/f1_frozen_seer"
    if latent_bridge_prepare_eval_row \
        "${baseline}" "${seed}" "${episodes_per_task}" "${num_tasks}"; then
        latent_bridge_run_eval \
            architectures.seer.adapters.latent_bridge.baseline_entry \
            "${baseline}" "public33_compute_matched_f1_seed${seed}" "${seed}" \
            "${episodes_per_task}" "${num_tasks}" "${port}"
    fi
    port=$((port + 10))
    row_args=(--row "f1=${baseline}")
    for period in "${periods[@]}"; do
        row="${seed_root}/f${period}_latent_bridge"
        if latent_bridge_prepare_eval_row \
            "${row}" "${seed}" "${episodes_per_task}" "${num_tasks}"; then
            export SEER_LATENT_BRIDGE_CHECKPOINT="${bridge}"
            export SEER_LATENT_BRIDGE_REFRESH_PERIOD="${period}"
            export SEER_LATENT_BRIDGE_PRECISION=bf16
            export SEER_LATENT_BRIDGE_COMPILE=1
            unset SEER_LATENT_BRIDGE_DAGGER_OUTPUT || true
            latent_bridge_run_eval \
                architectures.seer.adapters.latent_bridge.evaluation_entry \
                "${row}" "public33_compute_matched_f${period}_seed${seed}" "${seed}" \
                "${episodes_per_task}" "${num_tasks}" "${port}"
        fi
        port=$((port + 10))
        row_args+=(--row "f${period}=${row}")
    done
    comparison_root="${seed_root}/comparison"
    if [[ ! -s "${comparison_root}/COMPLETE" ]] || ! latent_bridge_json_has_status \
        "${comparison_root}/comparison.json" \
        "SEER_LATENT_BRIDGE_EVALUATION_AGGREGATION_PASS"; then
        [[ ! -e "${comparison_root}" ]] || latent_bridge_archive_partial "${comparison_root}"
        python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/aggregate_evaluations.py" \
            --manifest "${manifest}" "${row_args[@]}" --baseline f1 \
            --output-dir "${comparison_root}"
    fi
    comparisons+=(--comparison "${seed_root}/comparison/comparison.json")
done

seed_summary="${stage}/execution_seed_summary"
if [[ ! -s "${seed_summary}/COMPLETE" ]] || ! latent_bridge_json_has_status \
    "${seed_summary}/execution_seed_summary.json" \
    "SEER_LATENT_BRIDGE_EXECUTION_SEED_AGGREGATION_COMPLETE"; then
    [[ ! -e "${seed_summary}" ]] || latent_bridge_archive_partial "${seed_summary}"
    python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/aggregate_seeds.py" \
        "${comparisons[@]}" --output-dir "${seed_summary}"
fi

latency_output="${stage}/component_latency_rtx3090.json"
if ! latent_bridge_json_has_status "${latency_output}" "PASS"; then
    [[ ! -e "${latency_output}" ]] || latent_bridge_archive_partial "${latency_output}"
    python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/latency_benchmark.py" \
        --checkpoint "${LATENT_BRIDGE_PUBLIC33}" --vit-checkpoint "${LATENT_BRIDGE_VIT}" \
        --dataset-root "${LATENT_BRIDGE_DATASET_ROOT}" --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" \
        --bridge-checkpoint "${bridge}" --output "${latency_output}"
fi
printf 'SEER_LATENT_BRIDGE_EFFICIENT_EVALUATION_COMPLETE\n' > "${stage}/COMPLETE"
