#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  train_real_stackcupanddoll.sh --preflight
  SIMVLA_REAL_TRAIN_RUN=1 train_real_stackcupanddoll.sh --all

Required environment:
  SIMVLA_REAL_RAW_DATA   Local stackcupanddoll directory containing 40 episode dirs

Optional environment:
  SIMVLA_REAL_STORAGE    Output root (default: shared/NVMe when present)
  SIMVLA_REAL_GPU_IDS    One or more physical GPU IDs (default: 4,5,6,7)
  SIMVLA_REAL_LOCAL_BATCH_SIZE
                        Per-process baseline microbatch (default: 4)
  SIMVLA_REAL_EFFECTIVE_BATCH_SIZE
                        Baseline global effective batch (default: 64)
  SIMVLA_REAL_PYTHON     Python from the simvla_libero environment
  SIMVLA_REAL_BASE       Local YuankaiLuo/SimVLA-LIBERO snapshot
  SIMVLA_REAL_PROCESSOR  Local SmolVLM-500M-Instruct snapshot

This wrapper never initializes robot hardware and never starts deployment.
EOF
}

mode="${1:---preflight}"
if [[ "${mode}" != "--preflight" && "${mode}" != "--all" ]]; then
    usage >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
python_bin="${SIMVLA_REAL_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}"
gpu_ids="${SIMVLA_REAL_GPU_IDS:-4,5,6,7}"
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
world_size=${#gpu_array[@]}
if (( world_size < 1 )); then
    echo "[ERROR] SIMVLA_REAL_GPU_IDS must contain at least one ID" >&2
    exit 2
fi
for gpu_id in "${gpu_array[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
        echo "[ERROR] Invalid GPU ID: ${gpu_id}" >&2
        exit 2
    }
done
local_batch_size="${SIMVLA_REAL_LOCAL_BATCH_SIZE:-4}"
effective_batch_size="${SIMVLA_REAL_EFFECTIVE_BATCH_SIZE:-64}"
[[ "${local_batch_size}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] SIMVLA_REAL_LOCAL_BATCH_SIZE must be a positive integer" >&2
    exit 2
}
[[ "${effective_batch_size}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] SIMVLA_REAL_EFFECTIVE_BATCH_SIZE must be a positive integer" >&2
    exit 2
}
microbatches_per_step=$((local_batch_size * world_size))
if (( effective_batch_size % microbatches_per_step != 0 )); then
    echo "[ERROR] Effective batch ${effective_batch_size} is not divisible by local batch ${local_batch_size} x world size ${world_size}" >&2
    exit 2
fi
gradient_accumulation_steps=$((effective_batch_size / microbatches_per_step))

default_storage="/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/real_world/stackcupanddoll"
if [[ ! -d "$(dirname "${default_storage}")" ]]; then
    default_storage="${repo_root}/results/simvla/real_world/stackcupanddoll"
fi
storage="${SIMVLA_REAL_STORAGE:-${default_storage}}"
raw_data="${SIMVLA_REAL_RAW_DATA:-}"

base="${SIMVLA_REAL_BASE:-}"
processor="${SIMVLA_REAL_PROCESSOR:-}"
if [[ -z "${base}" ]]; then
    base=$(find "${repo_root}/.cache/huggingface/hub/models--YuankaiLuo--SimVLA-LIBERO/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)
fi
if [[ -z "${processor}" ]]; then
    processor=$(find "${repo_root}/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)
fi

dataset_root="${storage}/dataset"
condition_cache="${storage}/condition_cache_fp32"
baseline_root="${storage}/baseline"
condition_root="${storage}/ours/condition_kc2"
generation_root="${storage}/ours/generation_ng3"
bundle_root="${storage}/deployment"
norm_stats="${dataset_root}/real_norm.json"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}
[[ -x "${python_bin}" ]] || fail "Python not found: ${python_bin}"
[[ -n "${raw_data}" && -d "${raw_data}" ]] || fail "Set SIMVLA_REAL_RAW_DATA to the 40-episode local dataset"
[[ -f "${base}/model.safetensors" ]] || fail "Official SimVLA snapshot is incomplete: ${base}"
[[ -f "${processor}/model.safetensors" ]] || fail "SmolVLM processor/model snapshot is incomplete: ${processor}"
[[ -f "${repo_root}/architectures/simvla/third_party/simvla_upstream_32700d0/models/modeling_smolvlm_vla.py" ]] || fail "Vendored SimVLA source is absent"
"${python_bin}" -c 'import h5py, numpy, scipy, torch, torchvision, transformers' || fail "Python environment is incomplete"

cat <<EOF
[SimVLA real training]
repo=${repo_root}
raw_data=${raw_data}
storage=${storage}
official_checkpoint=${base}
processor=${processor}
gpus=${gpu_ids}
world_size=${world_size}
local_batch_size=${local_batch_size}
gradient_accumulation_steps=${gradient_accumulation_steps}
effective_global_batch=${effective_batch_size}
protocol=full official initialization; frozen VLM; fresh H=10; execute R=5
baseline=K_C=1,N_G=10
ours=K_C=2,N_G=3
robot_hardware_initialized=false
EOF

if [[ "${mode}" == "--preflight" ]]; then
    echo "REAL_TRAINING_PREFLIGHT_PASS"
    exit 0
fi
[[ "${SIMVLA_REAL_TRAIN_RUN:-0}" == "1" ]] || fail "Set SIMVLA_REAL_TRAIN_RUN=1 for --all"

mkdir -p "${storage}/logs" "${bundle_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "${dataset_root}/manifest.json" ]]; then
    convert_args=()
    [[ -d "${dataset_root}" ]] && convert_args+=(--resume)
    "${python_bin}" -m architectures.simvla.adapters.real_world_training.convert_dataset \
        --source "${raw_data}" \
        --output "${dataset_root}" \
        "${convert_args[@]}" \
        2>&1 | tee "${storage}/logs/01_convert.log"
fi
"${python_bin}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["verdict"]=="REAL_DATASET_CONTRACT_PASS"' "${dataset_root}/manifest.json"

if [[ ! -f "${condition_cache}/manifest.json" ]]; then
    cache_args=()
    [[ -d "$(dirname "${condition_cache}")/.${condition_cache##*/}.building" ]] && cache_args+=(--resume)
    CUDA_VISIBLE_DEVICES="${gpu_ids}" "${python_bin}" -m torch.distributed.run \
        --standalone --nproc_per_node="${world_size}" \
        -m architectures.simvla.adapters.real_world_training.condition_cache \
        --dataset-manifest "${dataset_root}/manifest.json" \
        --checkpoint "${base}" \
        --processor "${processor}" \
        --norm-stats "${norm_stats}" \
        --output "${condition_cache}" \
        --batch-size 4 --num-workers 2 \
        "${cache_args[@]}" \
        2>&1 | tee "${storage}/logs/02_condition_cache.log"
fi

baseline_args=()
if [[ -f "${baseline_root}/resume.pt" && ! -f "${baseline_root}/run_summary.json" ]]; then
    baseline_args+=(--resume)
fi
if [[ ! -f "${baseline_root}/run_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_ids}" "${python_bin}" -m torch.distributed.run \
        --standalone --nproc_per_node="${world_size}" \
        -m architectures.simvla.adapters.real_world_training.train_baseline \
        --condition-cache "${condition_cache}" \
        --checkpoint "${base}" \
        --processor "${processor}" \
        --norm-stats "${norm_stats}" \
        --output "${baseline_root}" \
        --max-steps 3000 --local-batch-size "${local_batch_size}" \
        --gradient-accumulation-steps "${gradient_accumulation_steps}" \
        --save-interval 1000 --keep-checkpoints 2 \
        "${baseline_args[@]}" \
        2>&1 | tee "${storage}/logs/03_baseline_train.log"
fi
baseline_checkpoint=$(<"${baseline_root}/latest_checkpoint.txt")
[[ -f "${baseline_checkpoint}" ]] || fail "baseline checkpoint is absent"

condition_args=()
generation_args=()
[[ -f "${condition_root}/resume.pt" && ! -f "${condition_root}/run_summary.json" ]] && condition_args+=(--resume)
[[ -f "${generation_root}/resume.pt" && ! -f "${generation_root}/run_summary.json" ]] && generation_args+=(--resume)

run_condition_updater() {
    CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "${python_bin}" \
        -m architectures.simvla.adapters.real_world_training.train_updater condition \
        --condition-cache "${condition_cache}" --checkpoint "${base}" \
        --processor "${processor}" --norm-stats "${norm_stats}" \
        --baseline-action-checkpoint "${baseline_checkpoint}" \
        --output "${condition_root}" --max-steps 10000 \
        "${condition_args[@]}" \
        >"${storage}/logs/04_condition_train.log" 2>&1
}

generation_gpu="${gpu_array[1]:-${gpu_array[0]}}"
run_generation_updater() {
    CUDA_VISIBLE_DEVICES="${generation_gpu}" "${python_bin}" \
        -m architectures.simvla.adapters.real_world_training.train_updater generation \
        --condition-cache "${condition_cache}" --checkpoint "${base}" \
        --processor "${processor}" --norm-stats "${norm_stats}" \
        --baseline-action-checkpoint "${baseline_checkpoint}" \
        --output "${generation_root}" --max-steps 10000 \
        "${generation_args[@]}" \
        >"${storage}/logs/05_generation_train.log" 2>&1
}

if (( world_size >= 2 )); then
    pids=()
    names=()
    if [[ ! -f "${condition_root}/run_summary.json" ]]; then
        run_condition_updater &
        pids+=("$!")
        names+=("condition")
    fi
    if [[ ! -f "${generation_root}/run_summary.json" ]]; then
        run_generation_updater &
        pids+=("$!")
        names+=("generation")
    fi
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            fail "${names[$index]} updater failed; inspect ${storage}/logs"
        fi
    done
else
    if [[ ! -f "${condition_root}/run_summary.json" ]]; then
        run_condition_updater || fail "condition updater failed; inspect ${storage}/logs"
    fi
    if [[ ! -f "${generation_root}/run_summary.json" ]]; then
        run_generation_updater || fail "generation updater failed; inspect ${storage}/logs"
    fi
fi

condition_checkpoint=$(<"${condition_root}/latest_checkpoint.txt")
generation_checkpoint=$(<"${generation_root}/latest_checkpoint.txt")
"${python_bin}" -m architectures.simvla.adapters.real_world_training.build_deployment_manifest \
    --template "${repo_root}/artifacts/simvla/real_world/deployment_manifest.example.json" \
    --checkpoint "${base}" --processor "${processor}" --norm-stats "${norm_stats}" \
    --baseline-action-checkpoint "${baseline_checkpoint}" \
    --condition-updater "${condition_checkpoint}" \
    --generation-updater "${generation_checkpoint}" \
    --deployment-id "stackcupanddoll_simvla_real_v1" \
    --instruction "Pick up the white cup and place it on top of the upside-down pink cup, then pick up the blue penguin plush toy and put it in the white cup" \
    --output "${bundle_root}/deployment_manifest.local.json" \
    2>&1 | tee "${storage}/logs/06_manifest.log"

echo "REAL_TRAINING_PIPELINE_COMPLETE"
echo "manifest=${bundle_root}/deployment_manifest.local.json"
echo "No robot command was issued. Run artifact/read-only preflights before requesting live authorization."
