#!/usr/bin/env bash
# Both Seer and SimVLA live in one repository; model identities stay separate.
main() (
    set -euo pipefail
    root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
    runtime=${SIMVLA_REAL_RUNTIME_ROOT:-${root}/runtime}
    export SIMVLA_REAL_RUNTIME_ROOT=$runtime
    export SIMVLA_REAL_PYTHON=${SIMVLA_REAL_PYTHON:-${runtime}/envs/simvla_real/bin/python}
    export SIMVLA_REAL_LOG_ROOT=${SIMVLA_REAL_LOG_ROOT:-${runtime}/results/simvla/doll_joint_v1}
    manifest=${SIMVLA_DOLL_MANIFEST:-${runtime}/artifacts/doll_joint_v1/deployment_manifest.site.json}
    mkdir -p "$SIMVLA_REAL_LOG_ROOT"
    launcher_log="$SIMVLA_REAL_LOG_ROOT/launcher.log"
    exec > >(tee -a "$launcher_log") 2>&1
    trap 'rc=$?; printf "%s\n" "$rc" > "$SIMVLA_REAL_LOG_ROOT/launcher.exit_code"' EXIT
    printf '\n[Doll joint] %s\nrepository=%s\nmanifest=%s\n' "$(date -Is)" "$root" "$manifest"
    [[ -x "$SIMVLA_REAL_PYTHON" ]] || { echo "Python not executable: $SIMVLA_REAL_PYTHON"; return 2; }
    [[ -f "$manifest" ]] || { echo "Manifest missing: $manifest"; return 2; }
    wrapper=${root}/architectures/simvla/wrappers
    mode=${1:---live}
    case "$mode" in
        --preflight|--profile|--check|--display-check|--live) if (($#)); then shift; fi ;;
        --max-steps|--control-hz|--camera-fps|--num-rollouts|--warmup-steps) mode=--live ;;
        -h|--help)
            echo 'Usage: deploy_doll_joint_baseline.sh [--live|--preflight|--profile|--check|--display-check] [options]'
            echo 'Live options: --max-steps N --control-hz HZ --camera-fps FPS --num-rollouts N --warmup-steps N'
            echo 'Defaults are editable at the top of deploy_doll_baseline.sh.'
            return 0 ;;
        *) echo 'Unknown mode. Use --help for supported modes/options.'; return 2 ;;
    esac
    configure_display() {
        local display_exports
        display_exports=$(PYTHONPATH="$root" "$SIMVLA_REAL_PYTHON" -m tools.simvla.launch_doll_baseline --desktop-environment) || return
        eval "$display_exports"
        echo "GUI_CONNECTION_PASS DISPLAY=$DISPLAY"
    }
    case "$mode" in
        --display-check)
            configure_display
            ;;
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
            [[ -t 0 ]] || {
                echo "Cannot open live GUI: DISPLAY=${DISPLAY:-unset}, interactive_stdin=$([[ -t 0 ]] && echo yes || echo no)."
                echo 'Run this command in a terminal on the inference-computer desktop; --preflight needs no desktop.'
                return 2
            }
            if [[ -z ${DISPLAY:-} ]]; then
                configure_display
            fi
            # Fresh inputs are checked for the NEW checkpoint; never reuse old-model evidence.
            echo '[1/2] Current-camera/state check (receive only; no robot commands)'
            bash "$wrapper/deploy_latentloop_real.sh" read-only-profile --manifest "$manifest" --method baseline --steps 15
            echo '[2/2] Open baseline GUI'
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
