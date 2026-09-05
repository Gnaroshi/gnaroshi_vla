#!/usr/bin/env bash

set -euo pipefail

wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${wrapper_dir}/../latent_bridge/common.sh"
latent_bridge_require_runtime

bridge="${LATENT_BRIDGE_R1_CHECKPOINT:-${LATENT_BRIDGE_RESULT_ROOT}/r1_${BRIDGE_PRESET:-full}/training/best.pt}"
[[ -s "${bridge}" ]] || latent_bridge_fail "R1 bridge checkpoint is absent: ${bridge}"

stage="${LATENT_BRIDGE_RESULT_ROOT}/paper_evaluation_${BRIDGE_PRESET:-full}"
if ! latent_bridge_prepare_stage "${stage}" resume; then
    exit 0
fi

export EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}"
export REFRESH_PERIODS="${REFRESH_PERIODS:-3 4}"
export EVAL_EPISODES_PER_TASK="${EVAL_EPISODES_PER_TASK:-50}"
export EVAL_NUM_TASKS="${EVAL_NUM_TASKS:-10}"
read -r -a seeds <<< "${EVAL_SEEDS}"
read -r -a periods <<< "${REFRESH_PERIODS}"
episodes_per_task="${EVAL_EPISODES_PER_TASK}"
num_tasks="${EVAL_NUM_TASKS}"
run_baseline="${RUN_BASELINE:-1}"
run_component_latency="${RUN_COMPONENT_LATENCY:-0}"
[[ "${run_baseline}" == "1" ]] || \
    latent_bridge_fail "paper aggregation currently requires RUN_BASELINE=1"

python - "${bridge}" "${stage}/evaluation_contract.json" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

from architectures.seer.adapters.latent_bridge.checkpoint import (
    load_bridge_checkpoint,
    validate_bridge_runtime_provenance,
)
from architectures.seer.adapters.latent_bridge.provenance import sha256_file

checkpoint, output = Path(sys.argv[1]), Path(sys.argv[2])
_, payload = load_bridge_checkpoint(checkpoint, map_location="cpu")
provenance = validate_bridge_runtime_provenance(payload)
record = {
    "status": "SEER_LATENT_BRIDGE_PAPER_EVALUATION_CONTRACT_PASS",
    "suite": os.environ["LATENT_BRIDGE_SUITE"],
    "base_checkpoint": os.environ["LATENT_BRIDGE_BASE_CHECKPOINT"],
    "base_checkpoint_sha256": os.environ["LATENT_BRIDGE_BASE_CHECKPOINT_SHA256"],
    "bridge_checkpoint": str(checkpoint),
    "bridge_checkpoint_sha256": sha256_file(checkpoint),
    "bridge_provenance": provenance,
    "renderer": os.environ["LATENT_BRIDGE_RENDERER"],
    "eval_seeds": [int(x) for x in os.environ.get("EVAL_SEEDS", "42 43 44").split()],
    "refresh_periods": [int(x) for x in os.environ.get("REFRESH_PERIODS", "3 4").split()],
    "episodes_per_task": int(os.environ.get("EVAL_EPISODES_PER_TASK", "50")),
    "num_tasks": int(os.environ.get("EVAL_NUM_TASKS", "10")),
    "policy_contract": {
        "control_hz": 20,
        "image_size": 128,
        "rgb_pad": 10,
        "gripper_pad": 4,
        "traj_cons": True,
        "action_pred_steps": 3,
        "temporal_ensembling": True,
        "max_policy_steps": 600,
    },
    "source_commit": subprocess.check_output(
        ["git", "-C", os.environ["LATENT_BRIDGE_REPO_ROOT"], "rev-parse", "HEAD"],
        text=True,
    ).strip(),
}
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(record, indent=2))
PY

comparisons=()
port="$(( ${MASTER_PORT_BASE:-18800} + 100 ))"
for seed in "${seeds[@]}"; do
    seed_root="${stage}/seed${seed}"
    mkdir -p "${seed_root}"
    manifest="${seed_root}/exact_manifest.json"
    python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/make_eval_manifest.py" \
        --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" \
        --checkpoint "${LATENT_BRIDGE_BASE_CHECKPOINT}" \
        --suite "${LATENT_BRIDGE_SUITE}" --output "${manifest}" \
        --seed "${seed}" --num-tasks "${num_tasks}" \
        --episodes-per-task "${episodes_per_task}" \
        --renderer "${LATENT_BRIDGE_RENDERER}" --reuse-if-matching

    baseline="${seed_root}/f1_frozen_seer"
    if latent_bridge_prepare_eval_row \
        "${baseline}" "${seed}" "${episodes_per_task}" "${num_tasks}"; then
        latent_bridge_run_eval \
            architectures.seer.adapters.latent_bridge.baseline_entry \
            "${baseline}" "${LATENT_BRIDGE_RUN_LABEL}_f1_seed${seed}" "${seed}" \
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
                "${row}" "${LATENT_BRIDGE_RUN_LABEL}_f${period}_seed${seed}" "${seed}" \
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
    comparisons+=(--comparison "${comparison_root}/comparison.json")
done

seed_summary="${stage}/execution_seed_summary"
if [[ ! -s "${seed_summary}/COMPLETE" ]] || ! latent_bridge_json_has_status \
    "${seed_summary}/execution_seed_summary.json" \
    "SEER_LATENT_BRIDGE_EXECUTION_SEED_AGGREGATION_COMPLETE"; then
    [[ ! -e "${seed_summary}" ]] || latent_bridge_archive_partial "${seed_summary}"
    python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/aggregate_seeds.py" \
        "${comparisons[@]}" --output-dir "${seed_summary}"
fi

if [[ "${run_component_latency}" == "1" ]]; then
    latency_output="${stage}/component_latency_rtx3090.json"
    if ! latent_bridge_json_has_status "${latency_output}" "PASS"; then
        [[ ! -e "${latency_output}" ]] || latent_bridge_archive_partial "${latency_output}"
        python "${LATENT_BRIDGE_REPO_ROOT}/tools/seer_latent_bridge/latency_benchmark.py" \
            --checkpoint "${LATENT_BRIDGE_BASE_CHECKPOINT}" \
            --checkpoint-sha256 "${LATENT_BRIDGE_BASE_CHECKPOINT_SHA256}" \
            --vit-checkpoint "${LATENT_BRIDGE_VIT}" \
            --dataset-root "${LATENT_BRIDGE_DATASET_ROOT}" \
            --dataset-name "${LATENT_BRIDGE_DATASET_NAME}" \
            --dataset-info "${LATENT_BRIDGE_DATASET_INFO}" \
            --libero-path "${LATENT_BRIDGE_LIBERO_PATH}" \
            --bridge-checkpoint "${bridge}" --output "${latency_output}"
    fi
fi

printf 'SEER_LATENT_BRIDGE_PAPER_EVALUATION_COMPLETE\n' > "${stage}/COMPLETE"
echo "[DONE] ${LATENT_BRIDGE_SUITE}: ${seed_summary}/execution_seed_summary.json"
