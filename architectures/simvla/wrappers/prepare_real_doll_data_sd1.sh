#!/usr/bin/env bash
# CPU-only data correction on sd1, then verified compact-data transfer to rb2.
set -euo pipefail
mode="${1:---preflight}"
[[ $# -le 1 ]] || exit 2
case "${mode}" in --preflight|--all) ;; *) echo "Usage: $0 [--preflight|--all]" >&2; exit 2 ;; esac
[[ "$(hostname -s)" == "jbrserver1" ]] || { echo "Run this preparation on sd1" >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
py=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
raw=/home/mingyujung/shared/ssd1/mingyujung/FlowFLA_RWDatasets/teleop_datasets/stackcupanddoll
data=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/datasets/simvla_real/stackcupanddoll_hdf5_v3
remote=/home/mingyujung/private/gnaroshi_vla_storage/datasets/simvla_real/stackcupanddoll_hdf5_v3
remote_stage="${remote}.incoming"
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
[[ -x "${py}" && -d "${raw}" ]] || { echo "Missing source data or Python" >&2; exit 2; }
command -v rsync >/dev/null
command -v flock >/dev/null
"${py}" -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); episodes=[x for x in p.iterdir() if x.is_dir()]; assert len(episodes)==40, len(episodes); print("RAW_DOLL_EPISODES=40")' "${raw}"
if [[ "${mode}" == "--preflight" ]]; then
    echo "DATA_PATH_PREFLIGHT_PASS; conversion and data validation have not run"
    exit 0
fi
mkdir -p "$(dirname "${data}")"
exec 9>"${data}.prepare.lock"
flock -n 9 || { echo "Doll data preparation is already active" >&2; exit 2; }
if [[ ! -f "${data}/manifest.json" ]]; then
    "${py}" -m architectures.simvla.adapters.real_world_training.convert_dataset \
        --source "${raw}" --output "${data}" --resume
fi
"${py}" -c 'import sys; from architectures.simvla.adapters.real_world_training.artifact_validation import validate_real_dataset_manifest; p=validate_real_dataset_manifest(sys.argv[1], verify_episode_checksums=True); print("CORRECTED_DOLL_DATASET_PASS",p["dataset_identity_sha256"])' "${data}/manifest.json"
ssh_args=(-o BatchMode=yes -o ConnectTimeout=10)
if ssh "${ssh_args[@]}" rb2 "test -e '${remote}'"; then
    # An existing completed destination is reused only if every file is identical.
    changes=$(rsync -anrc --delete --out-format='%i %n' -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' "${data}/" "rb2:${remote}/")
    [[ -z "${changes}" ]] || { printf '[ERROR] Existing destination differs; not overwritten:\n%s\n' "${changes}" >&2; exit 1; }
    echo "CORRECTED_DOLL_DATASET_ALREADY_TRANSFERRED"
else
    ssh "${ssh_args[@]}" rb2 "test ! -L '${remote_stage}' && mkdir -p '${remote_stage}'"
    rsync -a --checksum --partial --info=progress2 \
        -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' "${data}/" "rb2:${remote_stage}/"
    ssh "${ssh_args[@]}" rb2 "test ! -e '${remote}' && test -f '${remote_stage}/manifest.json' && mv -T -n '${remote_stage}' '${remote}' && test ! -e '${remote_stage}'"
    echo "CORRECTED_DOLL_DATASET_TRANSFERRED"
fi
echo "NEXT: on rb2 run SIMVLA_REAL_TRAIN_RUN=1 bash architectures/simvla/wrappers/train_real_doll_rb2.sh --all"
