#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  train_real_stackcupanddoll.sh --preflight
  SIMVLA_REAL_TRAIN_RUN=1 train_real_stackcupanddoll.sh --all

Required environment:
  SIMVLA_REAL_RAW_DATA   Local stackcupanddoll directory containing 40 episode dirs,
                        unless SIMVLA_REAL_DATASET names a converted dataset

Optional environment:
  SIMVLA_REAL_STORAGE    Output root (default: shared/NVMe when present)
  SIMVLA_REAL_DATASET    Existing converted dataset containing manifest.json
  SIMVLA_REAL_LEGACY_CONDITION_CACHE
                        Optional v1 cache reused only after strict label-only
                        migration checks; no frozen VLM recomputation
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
# Isolate imports from the caller's worktree for preflight and every train stage.
cd -- "${repo_root}"
python_bin="${SIMVLA_REAL_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}"
gpu_ids="${SIMVLA_REAL_GPU_IDS:-4,5,6,7}"
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
world_size=${#gpu_array[@]}
if (( world_size < 1 )); then
    echo "[ERROR] SIMVLA_REAL_GPU_IDS must contain at least one ID" >&2
    exit 2
fi
declare -A selected_gpu_ids=()
machine_name=$(hostname -s)
for gpu_id in "${gpu_array[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || {
        echo "[ERROR] Invalid GPU ID: ${gpu_id}" >&2
        exit 2
    }
    if [[ -n "${selected_gpu_ids[${gpu_id}]:-}" ]]; then
        echo "[ERROR] Duplicate GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    selected_gpu_ids[${gpu_id}]=1
    if [[ "${machine_name}" == "jbrserver1" || "${machine_name}" == "sd1" ]]; then
        [[ "${gpu_id}" =~ ^[4-7]$ ]] || {
            echo "[ERROR] sd1 permits only physical GPU IDs 4,5,6,7" >&2
            exit 2
        }
    fi
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

default_storage="/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/real_world/stackcupanddoll_v2_corrected"
if [[ ! -d "$(dirname "${default_storage}")" ]]; then
    default_storage="${repo_root}/results/simvla/real_world/stackcupanddoll"
fi
storage="${SIMVLA_REAL_STORAGE:-${default_storage}}"
raw_data="${SIMVLA_REAL_RAW_DATA:-}"

base="${SIMVLA_REAL_BASE:-}"
processor="${SIMVLA_REAL_PROCESSOR:-}"
hf_home="${HF_HOME:-${repo_root}/.cache/huggingface}"
resolve_hf_main_snapshot() {
    local model_cache="$1"
    local reference="${hf_home}/hub/${model_cache}/refs/main"
    [[ -f "${reference}" ]] || return 1
    local revision
    revision=$(<"${reference}")
    local snapshot="${hf_home}/hub/${model_cache}/snapshots/${revision}"
    [[ -d "${snapshot}" ]] || return 1
    printf '%s\n' "${snapshot}"
}
if [[ -z "${base}" ]]; then
    base=$(resolve_hf_main_snapshot "models--YuankaiLuo--SimVLA-LIBERO" || true)
fi
if [[ -z "${processor}" ]]; then
    processor=$(resolve_hf_main_snapshot "models--HuggingFaceTB--SmolVLM-500M-Instruct" || true)
fi

dataset_root="${SIMVLA_REAL_DATASET:-${storage}/dataset_v3}"
dataset_is_external=0
[[ -n "${SIMVLA_REAL_DATASET:-}" ]] && dataset_is_external=1
condition_cache="${storage}/condition_cache_fp32_v2"
condition_cache_building="$(dirname "${condition_cache}")/.${condition_cache##*/}.building"
legacy_condition_cache="${SIMVLA_REAL_LEGACY_CONDITION_CACHE:-}"
condition_cache_attestation="${storage}/provenance/condition_cache_attestation.json"
baseline_root="${storage}/baseline"
condition_root="${storage}/ours/condition_kc2"
generation_root="${storage}/ours/generation_ng3"
coupled_root="${storage}/ours/coupled_kc2_ng3"
bundle_root="${storage}/deployment_v4"
portable_bundle_root="${storage}/deployment_bundle_v4"
norm_stats="${dataset_root}/real_norm.json"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}
export PYTHONPATH="${repo_root}"

dataset_manifest_valid() {
    "${python_bin}" -c \
        'import sys; from architectures.simvla.adapters.real_world_training.artifact_validation import validate_real_dataset_manifest; validate_real_dataset_manifest(sys.argv[1], verify_episode_checksums=False)' \
        "$1" >/dev/null 2>&1
}

[[ -x "${python_bin}" ]] || fail "Python not found: ${python_bin}"
if [[ -f "${dataset_root}/manifest.json" ]]; then
    if ! dataset_manifest_valid "${dataset_root}/manifest.json"; then
        if [[ "${mode}" == "--preflight" || ${dataset_is_external} -eq 1 ]]; then
            fail "Existing dataset failed the corrected command_t/schema-v3 contract: ${dataset_root}"
        fi
        [[ -n "${raw_data}" && -d "${raw_data}" ]] || fail \
            "Owned dataset is invalid and cannot be regenerated without SIMVLA_REAL_RAW_DATA"
        echo "[WARN] Owned dataset will be quarantined and regenerated: ${dataset_root}"
    fi
else
    [[ -n "${raw_data}" && -d "${raw_data}" ]] || fail "Set SIMVLA_REAL_RAW_DATA, or point SIMVLA_REAL_DATASET to a converted dataset"
fi
[[ -f "${base}/model.safetensors" ]] || fail "Official SimVLA snapshot is incomplete: ${base}"
[[ -f "${processor}/model.safetensors" ]] || fail "SmolVLM processor/model snapshot is incomplete: ${processor}"
[[ -f "${repo_root}/architectures/simvla/third_party/simvla_upstream_32700d0/models/modeling_smolvlm_vla.py" ]] || fail "Vendored SimVLA source is absent"
"${python_bin}" -c 'import h5py, numpy, scipy, torch, torchvision, transformers' || fail "Python environment is incomplete"
"${python_bin}" -c \
    'import json,sys; from architectures.simvla.adapters.real_world_training.model_io import official_base_identity; print(json.dumps(official_base_identity(sys.argv[1], sys.argv[2]).to_dict(), sort_keys=True))' \
    "${base}" "${processor}" || fail "Pinned YuankaiLuo/SimVLA-LIBERO identity check failed"

cat <<EOF
[SimVLA real training]
repo=${repo_root}
raw_data=${raw_data}
dataset=${dataset_root}
storage=${storage}
legacy_condition_cache=${legacy_condition_cache:-none}
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
coupling=Condition Updater delta code; projection-only 16,384 parameters
robot_hardware_initialized=false
EOF

if [[ "${mode}" == "--preflight" ]]; then
    echo "REAL_TRAINING_PREFLIGHT_PASS"
    exit 0
fi
[[ "${SIMVLA_REAL_TRAIN_RUN:-0}" == "1" ]] || fail "Set SIMVLA_REAL_TRAIN_RUN=1 for --all"

mkdir -p "${storage}/logs" "${storage}/provenance" "${bundle_root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

validate_chain() {
    "${python_bin}" -m architectures.simvla.adapters.real_world_training.artifact_validation \
        --condition-cache "${condition_cache}" \
        --checkpoint "${base}" \
        --processor "${processor}" \
        --norm-stats "${norm_stats}" \
        --condition-cache-attestation "${condition_cache_attestation}" \
        "$@"
}

quarantine_path() {
    local run_root="$1"
    local label="$2"
    [[ -e "${run_root}" ]] || return 0
    if [[ ! -d "${run_root}" || -n "$(find "${run_root}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
        local quarantine_root="${storage}/quarantine"
        local stamp
        stamp=$(date +%Y%m%d_%H%M%S)
        mkdir -p "${quarantine_root}"
        local destination="${quarantine_root}/${label}_${stamp}"
        mv "${run_root}" "${destination}"
        echo "[WARN] Preserved invalid or incomplete ${label} artifact at ${destination}"
    fi
}

completed_checkpoint() {
    local run_root="$1"
    local expected_verdict="$2"
    local expected_steps="$3"
    "${python_bin}" -c \
        'import json,pathlib,sys; r=pathlib.Path(sys.argv[1]); p=json.load(open(r/"run_summary.json")); assert p.get("verdict")==sys.argv[2]; assert int(p.get("optimizer_steps",-1))==int(sys.argv[3]); c=pathlib.Path((r/"latest_checkpoint.txt").read_text().strip()).expanduser(); assert c.is_file(); print(c.resolve())' \
        "${run_root}" "${expected_verdict}" "${expected_steps}" 2>/dev/null
}

if [[ -f "${dataset_root}/manifest.json" ]] && ! dataset_manifest_valid "${dataset_root}/manifest.json"; then
    (( dataset_is_external == 0 )) || fail "Refusing to move an external invalid dataset"
    quarantine_path "${dataset_root}" dataset_v3
fi

if [[ ! -f "${dataset_root}/manifest.json" ]]; then
    convert_args=()
    [[ -d "${dataset_root}" ]] && convert_args+=(--resume)
    "${python_bin}" -m architectures.simvla.adapters.real_world_training.convert_dataset \
        --source "${raw_data}" \
        --output "${dataset_root}" \
        "${convert_args[@]}" \
        2>&1 | tee "${storage}/logs/01_convert.log"
fi
"${python_bin}" -c \
    'import sys; from architectures.simvla.adapters.real_world_training.artifact_validation import validate_real_dataset_manifest; validate_real_dataset_manifest(sys.argv[1], verify_episode_checksums=True); print("REAL_DATASET_FULL_INTEGRITY_PASS")' \
    "${dataset_root}/manifest.json" 2>&1 | tee "${storage}/logs/01b_dataset_integrity.log"
[[ -f "${norm_stats}" ]] || fail "converted dataset norm stats are absent: ${norm_stats}"

if [[ -d "${condition_cache}" ]]; then
    if ! "${python_bin}" -c \
        'import sys; from architectures.simvla.adapters.real_world_training.condition_cache import validate_real_condition_cache; validate_real_condition_cache(sys.argv[1], verify_array_checksums=False)' \
        "${condition_cache}" >/dev/null 2>&1; then
        quarantine_path "${condition_cache}" condition_cache_fp32_v2
    fi
fi
if [[ -d "${condition_cache_building}" ]]; then
    if ! "${python_bin}" -c \
        'import sys; from architectures.simvla.adapters.real_world_training.condition_cache import validate_condition_cache_building; validate_condition_cache_building(sys.argv[1], dataset_manifest=sys.argv[2], checkpoint=sys.argv[3], processor=sys.argv[4], norm_stats=sys.argv[5])' \
        "${condition_cache_building}" "${dataset_root}/manifest.json" "${base}" "${processor}" "${norm_stats}" \
        >/dev/null 2>&1; then
        quarantine_path "${condition_cache_building}" condition_cache_building
    fi
fi
if [[ ! -f "${condition_cache}/manifest.json" ]]; then
    if [[ -n "${legacy_condition_cache}" ]]; then
        [[ -f "${legacy_condition_cache}/manifest.json" ]] || fail "Legacy Condition cache manifest is absent: ${legacy_condition_cache}"
        "${python_bin}" -m architectures.simvla.adapters.real_world_training.migrate_condition_cache \
            --legacy-condition-cache "${legacy_condition_cache}" \
            --corrected-dataset-manifest "${dataset_root}/manifest.json" \
            --output "${condition_cache}" \
            2>&1 | tee "${storage}/logs/02_condition_cache_migration.log"
    else
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
fi
CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "${python_bin}" \
    -m architectures.simvla.adapters.real_world_training.verify_condition_cache \
    --condition-cache "${condition_cache}" \
    --checkpoint "${base}" \
    --processor "${processor}" \
    --norm-stats "${norm_stats}" \
    --output "${condition_cache_attestation}" \
    --batch-size 4 --device cuda \
    2>&1 | tee "${storage}/logs/02a_condition_cache_attestation.log"
validate_chain \
    2>&1 | tee "${storage}/logs/02b_condition_cache_validate.log"

baseline_checkpoint=""
if [[ -d "${baseline_root}" ]]; then
    if candidate=$(completed_checkpoint "${baseline_root}" REAL_BASELINE_FINETUNE_COMPLETE 3000) \
        && validate_chain --baseline-action-checkpoint "${candidate}" >/dev/null 2>&1; then
        baseline_checkpoint="${candidate}"
        echo "[REUSE] validated baseline ${baseline_checkpoint}"
    else
        quarantine_path "${baseline_root}" baseline
    fi
fi
if [[ -z "${baseline_checkpoint}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_ids}" "${python_bin}" -m torch.distributed.run \
        --standalone --nproc_per_node="${world_size}" \
        -m architectures.simvla.adapters.real_world_training.train_baseline \
        --condition-cache "${condition_cache}" \
        --condition-cache-attestation "${condition_cache_attestation}" \
        --checkpoint "${base}" \
        --processor "${processor}" \
        --norm-stats "${norm_stats}" \
        --output "${baseline_root}" \
        --max-steps 3000 --local-batch-size "${local_batch_size}" \
        --gradient-accumulation-steps "${gradient_accumulation_steps}" \
        --save-interval 1000 --keep-checkpoints 2 \
        2>&1 | tee "${storage}/logs/03_baseline_train.log"
fi
baseline_checkpoint=$(completed_checkpoint "${baseline_root}" REAL_BASELINE_FINETUNE_COMPLETE 3000) \
    || fail "baseline completion contract is absent"
[[ -f "${baseline_checkpoint}" ]] || fail "baseline checkpoint is absent"
validate_chain --baseline-action-checkpoint "${baseline_checkpoint}" \
    2>&1 | tee "${storage}/logs/03b_baseline_validate.log"

condition_checkpoint=""
if [[ -d "${condition_root}" ]]; then
    if candidate=$(completed_checkpoint "${condition_root}" REAL_CONDITION_UPDATER_COMPLETE 10000) \
        && validate_chain --baseline-action-checkpoint "${baseline_checkpoint}" \
            --condition-updater "${candidate}" >/dev/null 2>&1; then
        condition_checkpoint="${candidate}"
        echo "[REUSE] validated Condition Updater ${condition_checkpoint}"
    else
        quarantine_path "${condition_root}" condition_kc2
    fi
fi
generation_checkpoint=""
if [[ -d "${generation_root}" ]]; then
    if candidate=$(completed_checkpoint "${generation_root}" REAL_GENERATION_UPDATER_COMPLETE 10000) \
        && validate_chain --baseline-action-checkpoint "${baseline_checkpoint}" \
            --generation-updater "${candidate}" >/dev/null 2>&1; then
        generation_checkpoint="${candidate}"
        echo "[REUSE] validated Generation Updater ${generation_checkpoint}"
    else
        quarantine_path "${generation_root}" generation_ng3
    fi
fi

run_condition_updater() {
    CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "${python_bin}" \
        -m architectures.simvla.adapters.real_world_training.train_updater condition \
        --condition-cache "${condition_cache}" --checkpoint "${base}" \
        --condition-cache-attestation "${condition_cache_attestation}" \
        --processor "${processor}" --norm-stats "${norm_stats}" \
        --baseline-action-checkpoint "${baseline_checkpoint}" \
        --output "${condition_root}" --max-steps 10000
}

generation_gpu="${gpu_array[1]:-${gpu_array[0]}}"
run_generation_updater() {
    CUDA_VISIBLE_DEVICES="${generation_gpu}" "${python_bin}" \
        -m architectures.simvla.adapters.real_world_training.train_updater generation \
        --condition-cache "${condition_cache}" --checkpoint "${base}" \
        --condition-cache-attestation "${condition_cache_attestation}" \
        --processor "${processor}" --norm-stats "${norm_stats}" \
        --baseline-action-checkpoint "${baseline_checkpoint}" \
        --output "${generation_root}" --max-steps 10000
}

if (( world_size >= 2 )); then
    pids=()
    names=()
    if [[ -z "${condition_checkpoint}" ]]; then
        run_condition_updater >"${storage}/logs/04_condition_train.log" 2>&1 &
        pids+=("$!")
        names+=("condition")
    fi
    if [[ -z "${generation_checkpoint}" ]]; then
        run_generation_updater >"${storage}/logs/05_generation_train.log" 2>&1 &
        pids+=("$!")
        names+=("generation")
    fi
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            fail "${names[$index]} updater failed; inspect ${storage}/logs"
        fi
    done
else
    if [[ -z "${condition_checkpoint}" ]]; then
        run_condition_updater 2>&1 | tee "${storage}/logs/04_condition_train.log" \
            || fail "condition updater failed; inspect ${storage}/logs"
    fi
    if [[ -z "${generation_checkpoint}" ]]; then
        run_generation_updater 2>&1 | tee "${storage}/logs/05_generation_train.log" \
            || fail "generation updater failed; inspect ${storage}/logs"
    fi
fi

condition_checkpoint=$(completed_checkpoint "${condition_root}" REAL_CONDITION_UPDATER_COMPLETE 10000) \
    || fail "Condition Updater completion contract is absent"
generation_checkpoint=$(completed_checkpoint "${generation_root}" REAL_GENERATION_UPDATER_COMPLETE 10000) \
    || fail "Generation Updater completion contract is absent"
validate_chain \
    --baseline-action-checkpoint "${baseline_checkpoint}" \
    --condition-updater "${condition_checkpoint}" \
    --generation-updater "${generation_checkpoint}" \
    2>&1 | tee "${storage}/logs/05b_updaters_validate.log"
coupled_checkpoint=""
if [[ -d "${coupled_root}" ]]; then
    if candidate=$(completed_checkpoint "${coupled_root}" REAL_COUPLED_GENERATION_COMPLETE 10000) \
        && validate_chain \
            --baseline-action-checkpoint "${baseline_checkpoint}" \
            --condition-updater "${condition_checkpoint}" \
            --generation-updater "${generation_checkpoint}" \
            --coupled-generation-updater "${candidate}" >/dev/null 2>&1; then
        coupled_checkpoint="${candidate}"
        echo "[REUSE] validated coupled Generation Updater ${coupled_checkpoint}"
    else
        quarantine_path "${coupled_root}" coupled_kc2_ng3
    fi
fi
if [[ -z "${coupled_checkpoint}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_array[0]}" "${python_bin}" \
        -m architectures.simvla.adapters.real_world_training.train_coupled_generation \
        --condition-cache "${condition_cache}" \
        --condition-cache-attestation "${condition_cache_attestation}" \
        --checkpoint "${base}" \
        --processor "${processor}" \
        --norm-stats "${norm_stats}" \
        --baseline-action-checkpoint "${baseline_checkpoint}" \
        --condition-updater-checkpoint "${condition_checkpoint}" \
        --parent-generation-checkpoint "${generation_checkpoint}" \
        --output "${coupled_root}" \
        --max-steps 10000 --batch-size 2 \
        --learning-rate 1e-4 --warmup-steps 500 \
        --minimum-lr-ratio 0.1 --save-interval 5000 \
        2>&1 | tee "${storage}/logs/06_coupled_train.log"
fi
coupled_checkpoint=$(completed_checkpoint "${coupled_root}" REAL_COUPLED_GENERATION_COMPLETE 10000) \
    || fail "coupled Generation completion contract is absent"
[[ -f "${coupled_checkpoint}" ]] || fail "coupled Generation checkpoint is absent"
validate_chain \
    --baseline-action-checkpoint "${baseline_checkpoint}" \
    --condition-updater "${condition_checkpoint}" \
    --generation-updater "${generation_checkpoint}" \
    --coupled-generation-updater "${coupled_checkpoint}" \
    2>&1 | tee "${storage}/logs/06b_coupled_validate.log"
"${python_bin}" -m architectures.simvla.adapters.real_world_training.build_deployment_manifest \
    --template "${repo_root}/artifacts/simvla/real_world/deployment_manifest.example.json" \
    --checkpoint "${base}" --processor "${processor}" --norm-stats "${norm_stats}" \
    --dataset-manifest "${dataset_root}/manifest.json" \
    --condition-cache-manifest "${condition_cache}/manifest.json" \
    --condition-cache-attestation "${condition_cache_attestation}" \
    --baseline-action-checkpoint "${baseline_checkpoint}" \
    --condition-updater "${condition_checkpoint}" \
    --generation-updater "${generation_checkpoint}" \
    --coupled-generation-updater "${coupled_checkpoint}" \
    --deployment-id "stackcupanddoll_simvla_real_v2_corrected" \
    --instruction "Pick up the white cup and place it on top of the upside-down pink cup, then pick up the blue penguin plush toy and put it in the white cup" \
    --output "${bundle_root}/deployment_manifest.local.json" \
    2>&1 | tee "${storage}/logs/07_manifest.log"

if [[ -d "${portable_bundle_root}" ]]; then
    if ! "${python_bin}" -c 'import sys; from architectures.simvla.adapters.latentloop_real_deploy.contracts import load_deployment_contract; load_deployment_contract(sys.argv[1]); print("EXISTING_PORTABLE_BUNDLE_PASS")' "${portable_bundle_root}/deployment_manifest.json"; then
        stamp=$(date +%Y%m%d_%H%M%S)
        mkdir -p "${storage}/quarantine"
        destination="${storage}/quarantine/deployment_bundle_v4_${stamp}"
        mv "${portable_bundle_root}" "${destination}"
        echo "[WARN] Preserved stale deployment bundle at ${destination}"
    fi
fi
if [[ ! -d "${portable_bundle_root}" ]]; then
    "${python_bin}" -m architectures.simvla.adapters.real_world_training.build_deployment_bundle \
        --manifest "${bundle_root}/deployment_manifest.local.json" \
        --output "${portable_bundle_root}" \
        2>&1 | tee "${storage}/logs/08_deployment_bundle.log"
fi
"${python_bin}" -c 'import sys; from architectures.simvla.adapters.latentloop_real_deploy.contracts import load_deployment_contract; c=load_deployment_contract(sys.argv[1]); assert not c.live_authorized; print("PORTABLE_DEPLOYMENT_BUNDLE_PASS")' \
    "${portable_bundle_root}/deployment_manifest.json"

echo "REAL_TRAINING_PIPELINE_COMPLETE"
echo "manifest=${bundle_root}/deployment_manifest.local.json"
echo "portable_bundle=${portable_bundle_root}"
echo "No robot command was issued. Populate and review hardware fields before read-only or live use."
