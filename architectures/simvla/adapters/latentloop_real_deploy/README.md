# SimVLA 실물 배포

일반 배포는 공통 실행기 하나를 사용합니다.

```bash
bash architectures/simvla/wrappers/deploy_ll.sh
```

이 파일 상단에서 `DEPLOY_PRESET`, `MAX_STEPS`, `CONTROL_HZ`, `CAMERA_FPS`,
`NUM_ROLLOUTS`, `WARMUP_STEPS`를 수정합니다. 기본값은 최신 Doll joint baseline,
목표 60 Hz, 최대 5000 steps입니다. CLI 인자는 파일의 기본값보다 우선합니다.

## 모델과 task 선택

- `doll_joint_baseline`: Doll 데이터로 VLM과 action transformer를 공동 학습한 baseline.
- `doll_joint_ours`: 이 baseline에 대응하는 updater가 설치되어야 실행할 수 있습니다.
- `basketball`, `stack_cups`, `cabinet`, `fruit`: 현재 SimVLA 배포 자산 미설치로 실행하지 않습니다.

task instruction, home pose, 카메라 역할, action/state 변환과 정규화는 선택된
SimVLA manifest에서 읽습니다. 다른 방법론의 task 번호, home pose나 그리퍼
임계값을 복사하지 않습니다. H=10, R=5 및 flow grid를 변경하지 않습니다.
이전 Doll baseline과 그에 맞춘 updater의 preset·실행기는 제거했습니다.
새 baseline에 이전 updater를 연결해서는 안 됩니다.

## 비구동 검사

```bash
# 파일 경로·task·방법 선택 확인. 모델 로딩/센서 연결 없음.
bash architectures/simvla/wrappers/deploy_ll.sh --inspect
bash architectures/simvla/wrappers/deploy_ll.sh --list

# 숨김 Tk 창으로 데스크톱 연결만 확인.
bash architectures/simvla/wrappers/deploy_ll.sh --display-check
```

`--inspect` 통과는 모델 추론이나 실물 task 성공을 뜻하지 않습니다.
`--preflight`는 모의 입력으로 모델을 실행합니다. `--profile`은 실제 카메라와
로봇 상태를 수신하지만 로봇 명령은 보내지 않습니다. 다른 배포가 같은 센서를
사용 중이면 센서 점검을 동시에 실행하지 마십시오.

실제 배포 명령은 대화형 터미널에서 실행합니다. DISPLAY가 없으면 같은 사용자의
로컬 데스크톱을 찾아 연결을 검사합니다. 여러 화면이 검색되면 임의 선택하지 않습니다.
모델·코드·환경·현장 설정이 같으면 완료된 비구동 점검을 재사용하므로,
매번 profile용 모델을 로드했다가 GUI용 모델을 다시 로드하지 않습니다.
checkpoint 해시 검사와 실제 실행 시 센서 연결, rollout warmup, Stop 기능은 유지합니다.
점검이 없거나 설정이 바뀌면 비구동 점검을 다시 수행하며 실패하면 GUI로 넘어가지 않습니다.
물리적으로 카메라를 옮기는 등의 변화는 파일만으로 감지할 수 없으므로 다음 명령으로 갱신합니다.

```bash
bash architectures/simvla/wrappers/deploy_ll.sh --refresh-checks
```

검증 정보는 자산 폴더의 `deployment_checks.json`에 저장합니다. 이것은 checkpoint나
성공률 결과가 아니라 같은 설정의 점검 재사용을 위한 작은 기록입니다.
GUI Start 시 home 이동 → 3회 warmup → policy 초기화 → rollout 순서는 그대로입니다.
기본 Doll home 관절각(rad)은 `[3.14, -1.57, 1.57, -1.57, -1.57, -1.57]`,
그리퍼는 열린 상태(0)로 Seer Doll과 같습니다. 물체 위치는 사용자가 재배치해야 합니다.

## 저장 위치

- 환경: `<repo>/runtime/envs/simvla_real`
- 현재 배포 자산: `<repo>/runtime/artifacts/doll_joint_v1`
- 실행 로그: `<repo>/runtime/results/simvla/doll_joint_v1`
- 실제 rollout 결과: 선택된 manifest의 `runtime.results_directory`

`deploy_doll_joint_baseline.sh`는 현재 baseline 명령을 위한 호환 연결입니다.
내부 실행 엔진 `deploy_latentloop_real.sh`, 공용 학습 코드와 보존 upstream snapshot은
task별 설정 파일이 아닙니다. Seer 실행기와 모델은 이 정리 대상이 아닙니다.

환경 검사는 `setup_real_deploy_env.sh --check`를 사용합니다. 기존 Seer 환경을
수정하지 않습니다. 향후 새 자산을 전송할 때는 `SIMVLA_REAL_REMOTE_BUNDLE`로
새 목적지를 명시해야 하며 기존 배포 자산을 덮어쓰지 않습니다.

배포 코드와 manifest는 checkpoint·정규화·학습 계보를 검증합니다. 실패한 검사를
우회하려고 식별값을 바꾸지 마십시오. 코드/자산 검증은 실물 task 성공 보장이
아니며, 성공/실패는 실제 rollout 결과로 기록합니다.
