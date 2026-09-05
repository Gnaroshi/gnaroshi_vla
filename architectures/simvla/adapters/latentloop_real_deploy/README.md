# SimVLA LatentLoop real-world deployment

This package adapts SimVLA to the proven UR5e/Robotiq/RealSense deployment
surface without editing either original source tree. It uses tracked snapshots
under `architectures/simvla/third_party/` and keeps all SimVLA-specific code in
this directory.

The selected LatentLoop policy preserves a fresh ten-action chunk every five
executed actions. It refreshes the full action condition every two policy
queries (`K_C=2`) and evaluates the full action transformer three times per
ten-step flow trajectory (`N_G=3`). The baseline uses `K_C=1,N_G=10` under the
same observation, action, and H=10/R=5 execution contract.

All deployment methods use the same resize-with-pad-224 then bicubic-384 image
transform as the real condition cache and action-head training path.

The real baseline starts from every tensor in the released SimVLA-LIBERO
checkpoint, freezes its VLM, and fine-tunes the existing action transformer;
there is no scratch or reinitialized-head ablation. LatentLoop updaters must be
trained from that exact real baseline. The deployment loader verifies the real
baseline, condition updater, parent generation updater, and projection-only
coupled generation checkpoint, together with data/cache/normalization lineage,
before constructing either policy. The `latentloop` method requires coupling;
an uncoupled checkpoint cannot be silently substituted.

This package proves source/artifact consistency and enforces reviewed command
limits. It does not prove real-task success. The frozen-VLM, 3,000-step real
action-head adaptation is a controlled low-data transfer protocol for the 40
available demonstrations, not a published SimVLA real-world recipe. A bounded
baseline canary must pass before LatentLoop can be authorized.

## Staged commands

실물 task 성공은 아래 검사로 보장되지 않습니다. 학습과 로봇 실행은 분리되며,
`prepare`는 로봇 연결 없이 환경과 세 가지 모델 구성을 순서대로 검사합니다.
기존 Seer 환경을 수정하지 말고, inference computer에 전용 환경을 설치합니다.

```bash
SIMVLA_REAL_ENV_INSTALL=1 \
bash architectures/simvla/wrappers/setup_real_deploy_env.sh --install

export SIMVLA_REAL_PYTHON="$HOME/gnaroshi_vla_runtime/envs/simvla_real/bin/python"
export SIMVLA_REAL_CUDA_DEVICE=0
export SIMVLA_REAL_LOG_ROOT="$HOME/gnaroshi_vla_runtime/results/simvla/real_deploy"

bash architectures/simvla/wrappers/deploy_latentloop_real.sh prepare \
  --manifest "$HOME/gnaroshi_vla_runtime/artifacts/stackcupanddoll/deployment_manifest.json" \
  --require-gui
```

`--require-gui`는 inference computer의 실제 데스크톱 세션에서 실행해야 합니다.
SSH/tmux에서 DISPLAY가 없는 경우 이를 빼고 모델 검사까지만 할 수 있습니다.
GUI 실행 시에는 display 검사가 다시 필수로 수행됩니다. DISPLAY 값을 추측해서
설정하지 마십시오. 승인된 환경에서도 GUI가 열리지 않으면 로봇 연결을 만들지 않습니다.

환경 설치는 Python 3.10, PyTorch 2.6.0/cu124와 명시된 패키지를 사용합니다.
전체 전이 의존성이 고정된 환경 파일은 아니며, 실제 설치 후 `pip check`,
`pip_freeze.txt`, `conda_explicit.txt`와 import 검사를 기록합니다.
기존 환경이나 다른 설치 명세가 있는 경로는 덮어쓰지 않습니다.

학습 wrapper가 만든 `deployment_bundle_v4/` 전체를 옮겨야 합니다.
norm stats, processor, 원본 모델, 네 개의 학습 checkpoint와 데이터 계보가 포함됩니다.
dataset episode와 condition cache 배열은 배포에 필요하지 않습니다.
수신측 Git 코드와 manifest의 runtime SHA가 달라지면 검사에서 중단합니다.
기존 checkpoint의 SHA를 수동으로 수정해서 우회하지 마십시오.

`MODELS_READY_FOR_SITE_REVIEW`는 모델 로딩/일정 검사 통과일 뿐입니다.
11개 모의 control step에서 3회 query를 검사합니다:
baseline은 VLM 3회/transformer 30회, condition-only는 2회/30회,
전체 Ours는 2회/9회입니다. 센서 프레임, 로봇 명령 또는 성공 판정은 발생하지 않습니다.

다음은 현장에서 별도로 확인해야 합니다.

- 외부/손목 카메라 serial, RGB 방향과 역할이 학습 데이터와 일치하는지
- robot IP, task별 home pose, workspace, 이동 및 tracking-error 제한
- gripper 초기화 시 자동 보정 동작과 소프트웨어 stop 부재, 물리 E-stop
- 세 가지 method의 read-only profile, 최신 프레임/TCP 값 및 실제 query 지연
- baseline의 감독하 짧은 동작 검사와 이후 동일 초기 조건의 task 비교

이 검토 전 `safety_review`를 일괄 true로 바꾸지 마십시오. 최초 baseline 검사에는
`baseline_bounded_canary_passed`가 필요하지 않지만 Ours 실행에는 필요합니다.
실물 성공/실패는 운영자가 GUI에서 표시하고 영상 및 실행 기록과 함께 보관합니다.

```bash
bash architectures/simvla/wrappers/deploy_latentloop_real.sh source-preflight

bash architectures/simvla/wrappers/deploy_latentloop_real.sh \
  artifact-preflight \
  --manifest artifacts/simvla/real_world/deployment_manifest.local.json \
  --method latentloop

bash architectures/simvla/wrappers/deploy_latentloop_real.sh \
  read-only-profile \
  --manifest artifacts/simvla/real_world/deployment_manifest.local.json \
  --method latentloop \
  --steps 300
```

Live mode intentionally requires three independent approvals:

1. `safety_review` fields in the verified manifest;
2. `SIMVLA_REAL_LIVE_RUN=1`;
3. `SIMVLA_REAL_DEPLOYMENT_ID` equal to the manifest deployment ID.

The wrapper defaults to `source-preflight`; it never defaults to live mode.

The operator-facing Stop/Retry actions raise the abort latch immediately and
stop the UR arm at the current RTDE command boundary. A command already inside
the RTDE call may finish before the software stop takes effect. The rollout is
discarded without an automatic home move. The copied Robotiq API has no stop
primitive, so the software can only stop issuing new gripper commands; the
physical emergency-stop path remains mandatory.
