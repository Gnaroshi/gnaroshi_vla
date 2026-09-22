#!/usr/bin/env bash
# SimVLA 공통 배포 실행기. 모델·상태·action 설정은 선택된 manifest를 사용합니다.

# 평소에는 이 블록만 수정하세요. CLI 인자가 이 설정보다 우선합니다.
DEPLOY_PRESET="doll_joint_baseline"
# DEPLOY_PRESET="doll_joint_ours"       # 미준비: 새 baseline용 updater가 필요합니다.
# 다른 task는 --list에서 확인하세요. SimVLA 배포 자산이 없으면 실행하지 않습니다.
MAX_STEPS=5000
CONTROL_HZ=60
CAMERA_FPS=60
NUM_ROLLOUTS=15
WARMUP_STEPS=3

main() (
    set -euo pipefail
    root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
    cd "$root"
    mode=--live
    preset="${SIMVLA_DEPLOY_PRESET:-$DEPLOY_PRESET}"
    max_steps="${SIMVLA_DOLL_MAX_STEPS:-$MAX_STEPS}"
    control_hz="${SIMVLA_DOLL_CONTROL_HZ:-$CONTROL_HZ}"
    camera_fps="${SIMVLA_DOLL_CAMERA_FPS:-$CAMERA_FPS}"
    num_rollouts="${SIMVLA_DOLL_NUM_ROLLOUTS:-$NUM_ROLLOUTS}"
    warmup_steps="${SIMVLA_DOLL_WARMUP_STEPS:-$WARMUP_STEPS}"
    check_options=()
    while (($#)); do
        case "$1" in
            --live|--preflight|--profile|--check|--inspect|--display-check) mode=$1; shift ;;
            --refresh-checks) check_options+=(--refresh-checks); shift ;;
            --preset|--max-steps|--control-hz|--camera-fps|--num-rollouts|--warmup-steps)
                (($# >= 2)) || { echo "값이 필요합니다: $1"; return 2; }
                case "$1" in
                    --preset) preset=$2 ;;
                    --max-steps) max_steps=$2 ;;
                    --control-hz) control_hz=$2 ;;
                    --camera-fps) camera_fps=$2 ;;
                    --num-rollouts) num_rollouts=$2 ;;
                    --warmup-steps) warmup_steps=$2 ;;
                esac
                shift 2 ;;
            --list)
                printf '%s\n' \
                    'doll_joint_baseline: 최신 Doll baseline' \
                    'doll_joint_ours: 최신 baseline용 Ours 미준비; 실행 불가' \
                    'basketball / stack_cups / cabinet / fruit: SimVLA 배포 자산 미준비; 실행 불가'
                return 0 ;;
            -h|--help)
                echo '사용: deploy_ll.sh [--preset NAME] [--live|--inspect|--check|--preflight|--profile|--display-check]'
                echo '설정: --max-steps N --control-hz HZ --camera-fps FPS --num-rollouts N --warmup-steps N'
                echo '동일 설정의 점검은 재사용합니다. 센서 배치를 바꿨으면 --refresh-checks로 다시 점검하세요.'
                echo '기본값은 이 파일 상단에서 수정합니다. --inspect는 모델/로봇/카메라를 실행하지 않습니다.'
                return 0 ;;
            *) echo "알 수 없는 인자: $1 (--help 참조)"; return 2 ;;
        esac
    done
    case "$preset" in
        doll_joint_baseline|doll_joint_ours)
            task=doll; deployment=doll_joint_v1; artifact_dir=doll_joint_v1; log_dir=doll_joint_v1
            if [[ $preset == doll_joint_baseline ]]; then method=baseline; else method=condition_loop; fi ;;
        basketball|stack_cups|cabinet|fruit)
            echo "$preset: SimVLA checkpoint, 정규화, task별 현장 설정이 아직 설치되지 않았습니다. 다른 모델로 대체하지 않습니다."
            return 2 ;;
        *) echo "알 수 없는 preset: $preset (--list 참조)"; return 2 ;;
    esac
    unset PYTHONHOME
    export PYTHONNOUSERSITE=1 PYTHONPATH="$root"
    export SIMVLA_REAL_RUNTIME_ROOT="${SIMVLA_REAL_RUNTIME_ROOT:-${root}/runtime}"
    export SIMVLA_REAL_PYTHON="${SIMVLA_REAL_PYTHON:-${SIMVLA_REAL_RUNTIME_ROOT}/envs/simvla_real/bin/python}"
    export SIMVLA_REAL_LOG_ROOT="${SIMVLA_REAL_LOG_ROOT:-${SIMVLA_REAL_RUNTIME_ROOT}/results/simvla/${log_dir}}"
    export PATH="$(dirname -- "$SIMVLA_REAL_PYTHON"):$PATH"
    manifest="${SIMVLA_DOLL_MANIFEST:-${SIMVLA_REAL_RUNTIME_ROOT}/artifacts/${artifact_dir}/deployment_manifest.site.json}"
    mkdir -p "$SIMVLA_REAL_LOG_ROOT"
    exec > >(tee -a "$SIMVLA_REAL_LOG_ROOT/launcher.log") 2>&1
    trap 'rc=$?; printf "%s\n" "$rc" > "$SIMVLA_REAL_LOG_ROOT/launcher.exit_code"' EXIT
    printf '\n[SimVLA] %s preset=%s task=%s method=%s mode=%s\nmanifest=%s\n' "$(date -Is)" "$preset" "$task" "$method" "$mode" "$manifest"
    [[ -x $SIMVLA_REAL_PYTHON ]] || { echo "Python을 찾을 수 없습니다: $SIMVLA_REAL_PYTHON"; return 2; }
    options=(--manifest "$manifest" --method "$method" --site-profile seer_doll
        --expected-deployment-id "$deployment" --expected-task-id stackcupanddoll
        --max-steps "$max_steps" --control-hz "$control_hz" --camera-fps "$camera_fps"
        --num-rollouts "$num_rollouts" --warmup-steps "$warmup_steps")
    configure_display() {
        local exports
        exports=$("$SIMVLA_REAL_PYTHON" -m tools.simvla.launch_doll_baseline --desktop-environment) || return
        eval "$exports"
        echo "GUI_CONNECTION_PASS DISPLAY=$DISPLAY"
    }
    if [[ $mode == --display-check ]]; then configure_display; return; fi
    # Reject the wrong task/teacher/method before opening sensors or allocating a GPU.
    "$SIMVLA_REAL_PYTHON" -m tools.simvla.launch_doll_baseline "${options[@]}" --inspect
    case "$mode" in
        --inspect) return 0 ;;
        --check) "$SIMVLA_REAL_PYTHON" -m tools.simvla.launch_doll_baseline "${options[@]}" --check ;;
        --preflight)
            bash "$root/architectures/simvla/wrappers/deploy_latentloop_real.sh" artifact-preflight --manifest "$manifest" --method "$method" ;;
        --profile)
            bash "$root/architectures/simvla/wrappers/deploy_latentloop_real.sh" read-only-profile --manifest "$manifest" --method "$method" --steps 15 ;;
        --live)
            [[ -t 0 ]] || {
                echo "Cannot open live GUI: DISPLAY=${DISPLAY:-unset}, interactive_stdin=no."
                echo '추론 컴퓨터의 대화형 터미널에서 실행하세요.'
                return 2
            }
            if [[ -z ${DISPLAY:-} ]]; then configure_display; fi
            echo '완료된 점검 확인 후 GUI 열기 (모델/센서 점검은 설정 변경 시에만 갱신)'
            "$SIMVLA_REAL_PYTHON" -m tools.simvla.launch_doll_baseline "${options[@]}" "${check_options[@]}"
            ;;
    esac
)

set +e
main "$@"
rc=$?
if [[ $rc == 0 ]]; then
    echo SIMVLA_DEPLOY_COMMAND_COMPLETE
else
    echo "SIMVLA_DEPLOY_COMMAND_FAILED rc=$rc"
    if [[ ${SIMVLA_STRICT_EXIT:-0} == 1 ]]; then exit "$rc"; fi
fi
exit 0
