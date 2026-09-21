#!/usr/bin/env bash
# Separate assets/logs from the previous Doll model. Live GUI is operator-run only.
main() (
    set -euo pipefail
    root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
    export SIMVLA_REAL_PYTHON=${SIMVLA_REAL_PYTHON:-${HOME}/gnaroshi_vla_runtime/envs/simvla_real/bin/python}
    export SIMVLA_REAL_LOG_ROOT=${SIMVLA_REAL_LOG_ROOT:-${HOME}/gnaroshi_vla_runtime/results/simvla/doll_joint_v1}
    manifest=${SIMVLA_DOLL_MANIFEST:-${HOME}/gnaroshi_vla_runtime/artifacts/doll_joint_v1/deployment_manifest.site.json}
    wrapper=${root}/architectures/simvla/wrappers
    mode=${1:---live}
    case "$mode" in
        --preflight|--profile|--check|--live) if (($#)); then shift; fi ;;
        *) echo 'Use --preflight (no robot connection), --profile (receive only), --check, or --live'; return 2 ;;
    esac
    case "$mode" in
        --preflight)
            bash "$wrapper/deploy_latentloop_real.sh" artifact-preflight --manifest "$manifest" --method baseline "$@"
            ;;
        --profile)
            bash "$wrapper/deploy_latentloop_real.sh" read-only-profile --manifest "$manifest" --method baseline --steps 15 "$@"
            ;;
        --check)
            bash "$wrapper/deploy_doll_baseline.sh" --manifest "$manifest" --check "$@"
            ;;
        --live)
            [[ -n ${DISPLAY:-} && -t 0 ]] || { echo 'Run from the monitor-connected inference-computer terminal.'; return 2; }
            # Fresh inputs are checked for the NEW checkpoint; never reuse old-model evidence.
            bash "$wrapper/deploy_latentloop_real.sh" read-only-profile --manifest "$manifest" --method baseline --steps 15
            bash "$wrapper/deploy_doll_baseline.sh" --manifest "$manifest" "$@"
            ;;
    esac
)
set +e
main "$@"
rc=$?
if [[ $rc == 0 ]]; then
    echo 'DOLL_JOINT_COMMAND_COMPLETE'
else
    echo "DOLL_JOINT_COMMAND_FAILED rc=$rc; inspect the doll_joint_v1 log directory."
    # Do not close the caller's tmux pane under errexit.
    if [[ ${SIMVLA_STRICT_EXIT:-0} == 1 ]]; then exit "$rc"; fi
fi
exit 0
