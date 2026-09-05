#!/usr/bin/env bash
# This script never activates, upgrades, or removes another project's env.
set -euo pipefail
mode="${1:---check}"
[[ $# -le 1 ]] || { echo "Usage: $0 [--check|--install]" >&2; exit 2; }
case "${mode}" in --check|--install) ;; *) echo "Usage: $0 [--check|--install]" >&2; exit 2 ;; esac
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
prefix="${SIMVLA_REAL_ENV_PREFIX:-${HOME}/gnaroshi_vla_runtime/envs/simvla_real}"
requirements="${repo}/architectures/simvla/env/requirements.real_deploy.txt"
[[ "${prefix}" == /* && "${prefix}" != / && "${prefix}" != "${HOME}" ]] || {
    echo "[ERROR] An absolute dedicated environment prefix is required" >&2; exit 2;
}
if [[ "${mode}" == "--install" ]]; then
    [[ "${SIMVLA_REAL_ENV_INSTALL:-0}" == 1 ]] || {
        echo "[ERROR] Set SIMVLA_REAL_ENV_INSTALL=1 for the isolated install" >&2; exit 2;
    }
    conda_bin="${SIMVLA_REAL_CONDA:-${HOME}/miniconda3/bin/conda}"
    [[ -x "${conda_bin}" ]] || { echo "[ERROR] Set SIMVLA_REAL_CONDA to conda" >&2; exit 2; }
    command -v sha256sum >/dev/null
    command -v flock >/dev/null
    signature="python310-torch260-cu124-$(sha256sum "${requirements}" | cut -d ' ' -f1)"
    state="${prefix}.simvla-setup"
    [[ ! -L "${prefix}" && ! -L "${state}" ]] || { echo "[ERROR] Symlink prefixes are not accepted" >&2; exit 2; }
    if [[ -e "${prefix}" && ! -f "${state}/spec" ]]; then
        echo "[ERROR] Refusing to modify an existing unowned environment: ${prefix}" >&2
        exit 2
    fi
    mkdir -p "${state}"
    exec 9>"${state}/lock"
    flock -n 9 || { echo "[ERROR] Another environment install is active" >&2; exit 2; }
    if [[ -f "${state}/spec" ]]; then
        [[ "$(<"${state}/spec")" == "${signature}" ]] || {
            echo "[ERROR] Environment specification changed; choose a NEW prefix" >&2; exit 2;
        }
    else
        printf '%s\n' "${signature}" > "${state}/spec"
    fi
    export CONDA_PKGS_DIRS="${HOME}/gnaroshi_vla_runtime/downloads/conda"
    export PIP_CACHE_DIR="${HOME}/gnaroshi_vla_runtime/downloads/pip"
    if [[ ! -f "${state}/installed" ]]; then
        if [[ ! -x "${prefix}/bin/python" ]]; then
            "${conda_bin}" create --yes --override-channels -c conda-forge \
                --prefix "${prefix}" python=3.10 pip tk ffmpeg
        fi
        "${prefix}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu124 \
            torch==2.6.0 torchvision==0.21.0
        "${prefix}/bin/python" -m pip install -r "${requirements}"
        "${prefix}/bin/python" -m pip check
        "${prefix}/bin/python" -m pip freeze > "${state}/pip_freeze.txt"
        "${conda_bin}" list --prefix "${prefix}" --explicit > "${state}/conda_explicit.txt"
        printf '%s\n' "${signature}" > "${state}/installed"
    fi
fi
[[ -x "${prefix}/bin/python" ]] || { echo "[ERROR] Environment not installed: ${prefix}" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=''
export PATH="${prefix}/bin:${PATH}"
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
"${prefix}/bin/python" -c 'import json; from architectures.simvla.adapters.latentloop_real_deploy.environment import inspect_environment; result=inspect_environment(require_cuda=False, require_gui=False); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["verdict"] == "REAL_ENVIRONMENT_PASS" else 1)'
echo "ENVIRONMENT_READY_FOR_GPU_AND_GUI_CHECKS python=${prefix}/bin/python"
echo 'No robot connection, camera stream, training, or CUDA job was started.'
