#!/usr/bin/env bash
# Edit here, or override with these environment variables. No robot connection.
max_steps="${SIMVLA_DOLL_TRAIN_STEPS:-5000}"
local_batch="${SIMVLA_DOLL_MICROBATCH:-1}"
effective_batch="${SIMVLA_DOLL_BATCH:-32}"
action_lr="${SIMVLA_DOLL_ACTION_LR:-1e-4}"
vlm_lr_scale="${SIMVLA_DOLL_VLM_LR_SCALE:-0.1}"
warmup_steps="${SIMVLA_DOLL_WARMUP:-200}"
schedule="${SIMVLA_DOLL_SCHEDULE:-constant}"
validation_interval="${SIMVLA_DOLL_VALIDATION_INTERVAL:-500}"
validation_stride="${SIMVLA_DOLL_VALIDATION_STRIDE:-5}"

set -uo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
storage="${SIMVLA_REAL_ASSET_ROOT:-${HOME}/private/gnaroshi_vla_storage}"
python="${SIMVLA_REAL_PYTHON:-${storage}/envs/simvla/libero_mujoco237/bin/python}"
dataset="${SIMVLA_REAL_DATASET:-${storage}/datasets/simvla_real/stackcupanddoll_hdf5_v3}"
output="${SIMVLA_DOLL_JOINT_OUTPUT:-${storage}/results/simvla/real_world/doll_joint_v1}"
gpu_ids="${SIMVLA_REAL_GPU_IDS:-0}"
mode="${1:---preflight}"

main() {
    set -e
    cd "${repo}"
    [[ "${mode}" == --preflight || "${mode}" == --all || "${mode}" == --wait ]] || {
        echo "Usage: bash $0 --preflight|--all|--wait" >&2; return 2;
    }
    [[ -x "${python}" ]] || { echo "Python missing: ${python}" >&2; return 2; }
    export PYTHONPATH="${repo}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    unset PYTHONHOME
    export HF_HOME="${SIMVLA_REAL_HF_HOME:-${storage}/cache/simvla/huggingface}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" CUDA_VISIBLE_DEVICES="${gpu_ids}"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    base="${SIMVLA_REAL_BASE:-${HF_HOME}/hub/models--YuankaiLuo--SimVLA-LIBERO/snapshots/93dc4d90b0596c652ad2840ad743c62b9c4473fb}"
    processor="${SIMVLA_REAL_PROCESSOR:-${HF_HOME}/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47}"
    IFS=',' read -r -a devices <<< "${gpu_ids}"
    [[ "${local_batch}" =~ ^[1-9][0-9]*$ && "${effective_batch}" =~ ^[1-9][0-9]*$ ]] || return 2
    for id in "${devices[@]}"; do
        [[ "${id}" =~ ^[0-9]+$ ]] || return 2
        if [[ "$(hostname -s)" == jbrserver1 || "$(hostname -s)" == sd1 ]]; then
            [[ "${id}" =~ ^[4-7]$ ]] || { echo "sd1: GPU4,5,6,7 only" >&2; return 2; }
        fi
    done
    count="${#devices[@]}"
    (( effective_batch % (local_batch * count) == 0 )) || {
        echo "Effective batch must be divisible by microbatch x GPU count" >&2; return 2;
    }
    accumulation=$((effective_batch / (local_batch * count)))
    args=(--dataset "${dataset}" --checkpoint "${base}" --processor "${processor}"
          --output "${output}" --max-steps "${max_steps}" --local-batch-size "${local_batch}"
          --accumulation "${accumulation}" --learning-rate "${action_lr}" --vlm-lr-scale "${vlm_lr_scale}"
          --warmup-steps "${warmup_steps}" --schedule "${schedule}"
          --validation-interval "${validation_interval}" --validation-stride "${validation_stride}")
    "${python}" -m architectures.simvla.adapters.real_world_training.train_joint_baseline "${args[@]}" --preflight
    [[ "${mode}" != --preflight ]] || return 0
    [[ "${SIMVLA_REAL_TRAIN_RUN:-0}" == 1 ]] || { echo "Set SIMVLA_REAL_TRAIN_RUN=1" >&2; return 2; }
    mkdir -p "${output}/logs"
    exec 9>"${output}/.train.lock"
    flock -n 9 || { echo "Doll joint training is already active" >&2; return 2; }
    if [[ "${mode}" == --wait ]]; then
        [[ "${gpu_ids}" == 0 && "$(hostname -s)" == jbr-TRX50 ]] || {
            echo "--wait is the rb2 GPU0 queue; use --all for explicit sd1 execution" >&2; return 2;
        }
        parents="${SIMVLA_REAL_WAIT_PIDS:-$(pgrep -f '[r]un_latent_bridge_large_nonlong.py all' | paste -sd, - || true)}"
        "${python}" -m tools.simvla.wait_real_training --parent-pids "${parents:-none}" --dataset-manifest "${dataset}/manifest.json" --gpu-id 0
    fi
    for id in "${devices[@]}"; do
        pids=$(nvidia-smi --id="${id}" --query-compute-apps=pid --format=csv,noheader)
        [[ -z "${pids}" ]] || { echo "GPU ${id} busy: ${pids}. Use --wait on rb2." >&2; return 2; }
    done
    [[ -f "${output}/resume.pt" ]] && args+=(--resume)
    if [[ ! -f "${output}/verification/verification.json" ]]; then
        "${python}" -m tools.simvla.verify_doll_joint --dataset "${dataset}" --checkpoint "${base}" --processor "${processor}" --output "${output}/verification" --device cuda
    fi
    export WANDB_MODE="${WANDB_MODE:-online}"
    args+=(--wandb-project "${SIMVLA_DOLL_WANDB_PROJECT:-gnaroshi-simvla-real}")
    echo "Joint VLM+head: steps=${max_steps} effective_batch=${effective_batch} action_lr=${action_lr} vlm_multiplier=${vlm_lr_scale}"
    "${python}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${count}" -m architectures.simvla.adapters.real_world_training.train_joint_baseline "${args[@]}" 2>&1 | tee -a "${output}/logs/train.log"
    "${python}" -m tools.simvla.prepare_joint_baseline --run "${output}" --dataset "${dataset}" --checkpoint "${base}" --processor "${processor}" --output "${output}/deployment_manifest.json"
}

# Strict errors stay inside the child shell, not the user's tmux shell.
set +e
( main )
status=$?
if [[ "${mode}" != --preflight && -d "${output}/logs" ]]; then printf '%s\n' "${status}" > "${output}/logs/launcher.exit_code"; fi
if (( status != 0 )); then
    echo "DOLL_JOINT_FAILED rc=${status}; inspect the traceback above. tmux pane remains open."
else
    echo "DOLL_JOINT_COMPLETE mode=${mode}; no robot was connected."
fi
exit 0
