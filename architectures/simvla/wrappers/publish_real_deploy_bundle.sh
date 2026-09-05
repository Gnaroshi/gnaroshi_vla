#!/usr/bin/env bash
# Validate before transfer, publish atomically, never overwrite deployed assets.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
bundle=""
send=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle) [[ $# -ge 2 ]] || exit 2; bundle="$2"; shift 2 ;;
        --send) send=1; shift ;;
        -h|--help)
            echo "Usage: $0 --bundle DIR [--send]"
            echo "Default: verify only. --send requires SIMVLA_REAL_TRANSFER_RUN=1."
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -d "${bundle}" ]] || { echo "[ERROR] --bundle must name the completed deployment_bundle_v4" >&2; exit 2; }
python_bin="${SIMVLA_REAL_PYTHON:-python}"
python_bin=$(command -v "${python_bin}")
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" -c '
import json, sys
from pathlib import Path
from architectures.simvla.adapters.latentloop_real_deploy.contracts import load_deployment_contract
p = Path(sys.argv[1])
inventory = json.loads((p / "bundle_inventory.json").read_text())
if inventory.get("verdict") != "REAL_SIMVLA_DEPLOYMENT_BUNDLE_PASS":
    raise RuntimeError("Bundle was not completed successfully")
contract = load_deployment_contract(p / "deployment_manifest.json", verify_artifacts=True)
if contract.live_authorized:
    raise RuntimeError("Transfer only an unapproved portable bundle, not a live site manifest")
print("BUNDLE_VALIDATED", contract.deployment_id)
' "${bundle}"
if (( ! send )); then
    echo "TRANSFER_NOT_REQUESTED"
    exit 0
fi
[[ "${SIMVLA_REAL_TRANSFER_RUN:-0}" == 1 ]] || { echo "[ERROR] Set SIMVLA_REAL_TRANSFER_RUN=1" >&2; exit 2; }
destination="${SIMVLA_REAL_REMOTE:-jbr@210.107.197.121}"
port="${SIMVLA_REAL_SSH_PORT:-9000}"
remote_root="${SIMVLA_REAL_REMOTE_BUNDLE:-/home/jbr/gnaroshi_vla_runtime/artifacts/stackcupanddoll}"
[[ "${destination}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*@[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || {
    echo "[ERROR] Destination must be user@host" >&2; exit 2;
}
[[ "${port}" =~ ^[0-9]+$ && ${#port} -le 5 ]] || exit 2
(( 10#${port} > 0 && 10#${port} <= 65535 )) || exit 2
[[ "${remote_root}" =~ ^/[A-Za-z0-9_./-]+$ && "${remote_root}" != / && "${remote_root}" != */ ]] || exit 2
[[ "/${remote_root}/" != *"/../"* && "/${remote_root}/" != *"/./"* ]] || exit 2
command -v rsync >/dev/null
command -v sha256sum >/dev/null
digest=$(sha256sum "${bundle}/deployment_manifest.json")
digest="${digest%% *}"
staging="${remote_root}.incoming-${digest:0:16}"
ssh_options=(-o BatchMode=yes -o ConnectTimeout=10 -p "${port}")
ssh "${ssh_options[@]}" "${destination}" \
    "test ! -e '${remote_root}' && mkdir -p '${staging}'"
rsync -a --checksum --partial --safe-links --info=progress2 \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10 -p ${port}" \
    "${bundle}/" "${destination}:${staging}/"
ssh "${ssh_options[@]}" "${destination}" \
    "test ! -e '${remote_root}' && test -f '${staging}/bundle_inventory.json' && mv -T -n '${staging}' '${remote_root}' && test ! -e '${staging}'"
echo "BUNDLE_TRANSFERRED destination=${destination}:${remote_root}"
echo "Run deploy_latentloop_real.sh prepare on the receiver; transfer is NOT model or hardware approval."
echo "Sender Git commit: $(git -C "${repo}" rev-parse HEAD)"
