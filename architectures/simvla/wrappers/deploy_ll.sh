#!/usr/bin/env bash
# SimVLA 공통 배포 실행기. 모델·상태·action 설정은 선택된 manifest를 사용합니다.

# 평소에는 이 블록만 수정하세요. CLI 인자가 이 설정보다 우선합니다.
deploy_presets=(
    "doll_baseline"
    # "doll_ours"                    # 미준비: 새 baseline용 updater가 필요합니다.
)
# 다른 task는 --list에서 확인하세요. SimVLA 배포 자산이 없으면 실행하지 않습니다.
execution_mode="live"               # live | read_only_profile
preflight_only=0                     # 1: 모의 입력으로 모델만 점검, 로봇/카메라 접근 없음
cuda_device=0                       # inference computer의 GPU
gui_font_backend="system"           # system: 시스템 Tk 글꼴 | conda: 환경 기본 Tk
profile_steps=15                     # read_only_profile 전용; 로봇 명령 없음
MAX_STEPS=5000
CONTROL_HZ=60
CAMERA_FPS=60
NUM_ROLLOUTS=15
WARMUP_STEPS=3
# 수집 episode 0511_172010의 첫 관절각. TCP 좌표가 아닙니다. 마지막 0은 열린 그리퍼.
DOLL_HOME_POSE='[3.0502887,-1.6030570,1.8191951,-1.8019783,-1.5417574,-1.6144441,0.0]'
DOLL_HOME_SOURCE='stackcupanddoll/0511_172010/2026-05-11T17:20:10.585482.pkl:joint_positions; efficient_Seer-main/pi05_relative_deploy_vtp.py:TASK_SPECS[3]; gripper open'

main() (
    set -euo pipefail
    root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
    cd "$root"
    (( ${#deploy_presets[@]} == 1 )) || { echo 'deploy_presets에서 하나만 선택하세요.'; return 2; }
    case "$execution_mode" in
        live) mode=--live ;;
        read_only_profile) mode=--profile ;;
        *) echo 'execution_mode는 live 또는 read_only_profile이어야 합니다.'; return 2 ;;
    esac
    case "$preflight_only" in
        0) ;; 1) mode=--preflight ;; *) echo 'preflight_only는 0 또는 1이어야 합니다.'; return 2 ;;
    esac
    preset="${deploy_presets[0]}"
    max_steps="$MAX_STEPS"
    control_hz="$CONTROL_HZ"
    camera_fps="$CAMERA_FPS"
    num_rollouts="$NUM_ROLLOUTS"
    warmup_steps="$WARMUP_STEPS"
    print_config_only=0
    check_options=()
    while (($#)); do
        case "$1" in
            --live|--preflight|--profile|--check|--inspect|--display-check) mode=$1; shift ;;
            --print-config) print_config_only=1; shift ;;
            --refresh-checks) check_options+=(--refresh-checks); shift ;;
            --preset|--max-steps|--control-hz|--control-freq|--camera-fps|--num-rollouts|--warmup-steps|--profile-steps|--gui-font-backend|--cuda-device)
                (($# >= 2)) || { echo "값이 필요합니다: $1"; return 2; }
                case "$1" in
                    --preset) preset=$2 ;;
                    --max-steps) max_steps=$2 ;;
                    --control-hz|--control-freq) control_hz=$2 ;;
                    --camera-fps) camera_fps=$2 ;;
                    --num-rollouts) num_rollouts=$2 ;;
                    --warmup-steps) warmup_steps=$2 ;;
                    --profile-steps) profile_steps=$2 ;;
                    --gui-font-backend) gui_font_backend=$2 ;;
                    --cuda-device) cuda_device=$2 ;;
                esac
                shift 2 ;;
            --list)
                printf '%s\n' \
                    'doll_baseline: 최신 Doll baseline (doll_joint_baseline과 동일)' \
                    'doll_ours: 최신 baseline용 Ours 미준비; 실행 불가' \
                    'doll_joint_baseline: 최신 Doll baseline' \
                    'doll_joint_ours: 최신 baseline용 Ours 미준비; 실행 불가' \
                    'basketball / stack_cups / cabinet / fruit: SimVLA 배포 자산 미준비; 실행 불가'
                return 0 ;;
            -h|--help)
                echo '사용: deploy_ll.sh [--preset NAME] [--live|--print-config|--inspect|--check|--preflight|--profile|--display-check]'
                echo '설정: --max-steps N --control-hz HZ --camera-fps FPS --num-rollouts N --warmup-steps N'
                echo '추가: --control-freq HZ (= --control-hz), --profile-steps N, --gui-font-backend system|conda, --cuda-device N'
                echo '동일 설정의 점검은 재사용합니다. 센서 배치를 바꿨으면 --refresh-checks로 다시 점검하세요.'
                echo '기본값은 이 파일 상단에서 수정합니다. --print-config는 Python 없이 설정만 출력합니다.'
                echo '--inspect는 checkpoint 경로/방법을 검사하며 모델/로봇/카메라를 실행하지 않습니다.'
                return 0 ;;
            *) echo "알 수 없는 인자: $1 (--help 참조)"; return 2 ;;
        esac
    done
    case "$preset" in
        doll_baseline|doll_ours|doll_joint_baseline|doll_joint_ours)
            task=doll; deployment=doll_joint_v1; artifact_dir=doll_joint_v1; log_dir=doll_joint_v1
            if [[ $preset == *_baseline ]]; then method=baseline; else method=condition_loop; fi ;;
        basketball|stack_cups|cabinet|fruit)
            echo "$preset: SimVLA checkpoint, 정규화, task별 현장 설정이 아직 설치되지 않았습니다. 다른 모델로 대체하지 않습니다."
            return 2 ;;
        *) echo "알 수 없는 preset: $preset (--list 참조)"; return 2 ;;
    esac
    [[ $gui_font_backend == system || $gui_font_backend == conda ]] || { echo 'gui_font_backend는 system 또는 conda이어야 합니다.'; return 2; }
    [[ $cuda_device =~ ^[0-9]+$ ]] || { echo 'cuda_device는 하나의 물리 GPU 번호이어야 합니다.'; return 2; }
    [[ $profile_steps =~ ^[1-9][0-9]*$ ]] || { echo 'profile_steps는 양의 정수이어야 합니다.'; return 2; }
    unset PYTHONHOME
    export PYTHONNOUSERSITE=1 PYTHONPATH="$root"
    export SIMVLA_REAL_RUNTIME_ROOT="${SIMVLA_REAL_RUNTIME_ROOT:-${root}/runtime}"
    export SIMVLA_REAL_PYTHON="${SIMVLA_REAL_PYTHON:-${SIMVLA_REAL_RUNTIME_ROOT}/envs/simvla_real/bin/python}"
    export SIMVLA_REAL_LOG_ROOT="${SIMVLA_REAL_LOG_ROOT:-${SIMVLA_REAL_RUNTIME_ROOT}/results/simvla/${log_dir}}"
    export SIMVLA_REAL_CUDA_DEVICE="$cuda_device"
    export PATH="$(dirname -- "$SIMVLA_REAL_PYTHON"):$PATH"
    manifest="${SIMVLA_DOLL_MANIFEST:-${SIMVLA_REAL_RUNTIME_ROOT}/artifacts/${artifact_dir}/deployment_manifest.site.json}"
    print_config() {
        printf '%s\n' "preset=$preset" "task=$task" "method=$method" "mode=$mode" \
            "manifest=$manifest" "target_control_hz=$control_hz" "camera_fps=$camera_fps" \
            "max_steps=$max_steps" "num_rollouts=$num_rollouts" "warmup_steps=$warmup_steps" \
            "profile_steps=$profile_steps" "cuda_device=$cuda_device" "gui_font_backend=$gui_font_backend" \
            "home_pose=$DOLL_HOME_POSE" "home_pose_source=$DOLL_HOME_SOURCE" \
            '모델/action 설정은 SimVLA manifest 사용. 목표 Hz와 실제 달성 Hz는 다릅니다.'
    }
    if (( print_config_only )); then
        print_config
        echo '설정 출력만 수행했습니다. checkpoint/실행 가능 여부는 --inspect로 확인하세요.'
        return 0
    fi
    mkdir -p "$SIMVLA_REAL_LOG_ROOT"
    exec > >(tee -a "$SIMVLA_REAL_LOG_ROOT/launcher.log") 2>&1
    trap 'rc=$?; printf "%s\n" "$rc" > "$SIMVLA_REAL_LOG_ROOT/launcher.exit_code"' EXIT
    printf '\n[SimVLA] %s preset=%s task=%s method=%s mode=%s\nmanifest=%s\n' "$(date -Is)" "$preset" "$task" "$method" "$mode" "$manifest"
    print_config
    [[ -x $SIMVLA_REAL_PYTHON ]] || { echo "Python을 찾을 수 없습니다: $SIMVLA_REAL_PYTHON"; return 2; }
    options=(--manifest "$manifest" --method "$method" --site-profile seer_doll
        --expected-deployment-id "$deployment" --expected-task-id stackcupanddoll
        --max-steps "$max_steps" --control-hz "$control_hz" --camera-fps "$camera_fps"
        --num-rollouts "$num_rollouts" --warmup-steps "$warmup_steps"
        --home-pose-json "$DOLL_HOME_POSE" --home-pose-source "$DOLL_HOME_SOURCE"
        --gui-font-backend "$gui_font_backend")
    configure_display() {
        local exports
        exports=$("$SIMVLA_REAL_PYTHON" -m tools.simvla.launch_doll_baseline --desktop-environment --gui-font-backend "$gui_font_backend") || return
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
            bash "$root/architectures/simvla/wrappers/deploy_latentloop_real.sh" read-only-profile --manifest "$manifest" --method "$method" \
                --steps "$profile_steps" --profile-target-hz "$control_hz" --profile-camera-fps "$camera_fps" ;;
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
