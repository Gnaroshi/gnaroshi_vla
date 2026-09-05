#!/usr/bin/env bash
# Host-specific paths only; the shared real training wrapper owns the protocol.
set -euo pipefail
mode="${1:---preflight}"
[[ $# -le 1 ]] || exit 2
case "${mode}" in --preflight|--all) ;; *) echo "Usage: $0 [--preflight|--all]" >&2; exit 2 ;; esac
[[ "$(hostname -s)" == "jbr-TRX50" ]] || { echo "Run this training on rb2" >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
storage=/home/mingyujung/private/gnaroshi_vla_storage
export SIMVLA_REAL_PYTHON="${storage}/envs/simvla/libero_mujoco237/bin/python"
export SIMVLA_REAL_DATASET="${storage}/datasets/simvla_real/stackcupanddoll_hdf5_v3"
export SIMVLA_REAL_STORAGE="${storage}/results/simvla/real_world/stackcupanddoll_v2_corrected"
export SIMVLA_REAL_LEGACY_CONDITION_CACHE="${storage}/results/simvla/real_world/stackcupanddoll_v1/condition_cache_fp32"
export HF_HOME="${storage}/cache/simvla/huggingface"
export SIMVLA_REAL_BASE="${HF_HOME}/hub/models--YuankaiLuo--SimVLA-LIBERO/snapshots/93dc4d90b0596c652ad2840ad743c62b9c4473fb"
export SIMVLA_REAL_PROCESSOR="${HF_HOME}/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47"
export SIMVLA_REAL_GPU_IDS=0
export SIMVLA_REAL_LOCAL_BATCH_SIZE=4
export SIMVLA_REAL_EFFECTIVE_BATCH_SIZE=64
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${SIMVLA_REAL_PYTHON}"):${PATH}"
unset SIMVLA_REAL_RAW_DATA
if [[ "${mode}" == "--all" ]]; then
    [[ "${SIMVLA_REAL_TRAIN_RUN:-0}" == 1 ]] || { echo "Set SIMVLA_REAL_TRAIN_RUN=1" >&2; exit 2; }
    command -v flock >/dev/null
    mkdir -p "${SIMVLA_REAL_STORAGE}/logs"
    exec 9>"${SIMVLA_REAL_STORAGE}/.pipeline.lock"
    flock -n 9 || { echo "Corrected Doll pipeline is already active" >&2; exit 2; }
fi
bash "${repo}/architectures/simvla/wrappers/train_real_stackcupanddoll.sh" --preflight
if [[ "${mode}" == "--all" ]]; then
    set +e
    bash "${repo}/architectures/simvla/wrappers/train_real_stackcupanddoll.sh" --all \
        2>&1 | tee -a "${SIMVLA_REAL_STORAGE}/logs/pipeline.log"
    codes=("${PIPESTATUS[@]}")
    set -e
    code="${codes[0]}"
    (( code != 0 || codes[1] == 0 )) || code="${codes[1]}"
    printf '%s\n' "${code}" > "${SIMVLA_REAL_STORAGE}/logs/pipeline.exit_code"
    exit "${code}"
fi
