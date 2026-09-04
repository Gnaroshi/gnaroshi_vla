#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
python_bin="${SIMVLA_REAL_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}"
source_root="${SIMVLA_REAL_RAW_DATA:?Set SIMVLA_REAL_RAW_DATA to the 40-episode source directory}"
output_root="${SIMVLA_REAL_DATASET_OUTPUT:?Set SIMVLA_REAL_DATASET_OUTPUT to the final dataset directory}"
remote_destination="${SIMVLA_REAL_DATASET_REMOTE:-}"
building_root="${output_root}.building"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ -x "${python_bin}" ]] || fail "Python not found: ${python_bin}"
[[ -d "${source_root}" ]] || fail "Raw dataset not found: ${source_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

validate_manifest() {
    "${python_bin}" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); assert p["verdict"]=="REAL_DATASET_CONTRACT_PASS", p["verdict"]; print("DATASET_PASS", p["dataset_identity_sha256"])' \
        "$1"
}

if [[ -f "${output_root}/manifest.json" ]] && validate_manifest "${output_root}/manifest.json"; then
    echo "[data] reusing validated dataset: ${output_root}"
else
    [[ ! -e "${output_root}" ]] || fail "Output exists but is not valid; inspect or move it first: ${output_root}"
    [[ ! -e "${building_root}" ]] || fail "Staging output already exists; inspect or move it first: ${building_root}"
    mkdir -p "$(dirname -- "${output_root}")"
    echo "[data] converting 40 synchronized trajectories"
    if ! "${python_bin}" -m architectures.simvla.adapters.real_world_training.convert_dataset \
        --source "${source_root}" \
        --output "${building_root}"; then
        fail "Conversion failed; no transfer was attempted. Inspect ${building_root}/manifest.json"
    fi
    validate_manifest "${building_root}/manifest.json" \
        || fail "Manifest validation failed; no transfer was attempted"
    mv "${building_root}" "${output_root}"
fi

if [[ -n "${remote_destination}" ]]; then
    [[ "${remote_destination}" == *:* ]] \
        || fail "SIMVLA_REAL_DATASET_REMOTE must have host:/absolute/path form"
    remote_host="${remote_destination%%:*}"
    remote_path="${remote_destination#*:}"
    [[ "${remote_path}" == /* ]] || fail "Remote dataset path must be absolute"
    echo "[data] transferring validated dataset to ${remote_destination}"
    ssh "${remote_host}" mkdir -p "${remote_path}"
    rsync -aH --partial --info=progress2 "${output_root}/" "${remote_host}:${remote_path}/"
    ssh "${remote_host}" test -f "${remote_path}/manifest.json"
fi

echo "REAL_DATASET_PREPARATION_COMPLETE"
echo "dataset=${output_root}"
