# SimVLA real-world baseline and LatentLoop training

## 현재 Doll 학습 경로

`architectures/simvla/wrappers/train_doll_joint.sh`를 사용한다. 아래의
frozen-VLM/head-only 설명은 이전 실험 기록이며 새 baseline 학습 권장안이 아니다.

- 공식 SimVLA-LIBERO의 VLM과 action transformer 전체를 그대로 초기화하고 둘 다 학습한다.
- 원본 `SmolVLMVLA.forward`의 flow-matching loss를 호출한다. 새로운 보조 loss는 추가하지 않는다.
- 이미 변환한 v3 Doll 데이터, 32/8 시연 분할, RGB·state·action 변환은 유지한다.
- 캐시를 사용하지 않는다. VLM이 바뀌므로 이전의 frozen-condition 캐시는 새 학습에 사용할 수 없다.
- 기본값: 5,000 step, effective batch 32, action LR 1e-4, VLM LR 1e-5,
  warmup 200 이후 고정 LR. 메모리 절약은 gradient checkpointing과 accumulation으로 한다.
- 이 step·batch·warmup은 Doll용 시작 설정이지, 원 논문이 검증한 Doll 설정이 아니다.
  약 15,629개 training window를 기준으로 약 10.2회 노출이다.
- 500 step마다 held-out 8개 시연 각각의 처음부터 마지막까지 R=5 간격과 마지막
  유효 window를 평가한다. `validation_stride=1`이면 모든 중첩 window를 평가한다.
- 고정된 시연·frame별 noise로 10-step 생성한 action의 처음 5개 및 전체 10개 오차,
  translation(mm), rotation(deg), 연속 gripper 오차를 CSV와 시연별 JSON으로 저장한다.
- 선택 기준은 시연별 first5 action L1의 평균이다. best와 latest full checkpoint를
  보존하고, optimizer·RNG·data cursor는 `resume.pt`에 저장한다.
- `deployment_manifest.json`은 새 baseline만 허용한다. 이전 baseline용 Ours는 호환되지
  않으며, 새 baseline을 teacher로 다시 학습해야 한다. 자동 로봇 구동은 없다.

rb2에서:

```bash
SIMVLA_REAL_TRAIN_RUN=1 bash architectures/simvla/wrappers/train_doll_joint.sh --wait
```

`--wait`는 실행 중인 Latent Bridge launcher 및 GPU0의 사용이 끝나기를 기다린다.
`--all`은 GPU가 비어 있을 때 즉시 시작한다. 재실행하면 저장된 optimizer부터 재개한다.
설정은 `.sh` 맨 위에서 수정한다. 다른 설정의 실험은 새 `SIMVLA_DOLL_JOINT_OUTPUT`을
사용한다. 오류 코드는 `logs/launcher.exit_code`, traceback은 콘솔 및 학습 로그에 남는다.

배포 설정은 `deploy_doll_baseline.sh`의 `max_steps`, `control_hz`, `camera_fps`,
`num_rollouts`, `warmup_steps`에서 바꾼다. `deploy_doll_ours.sh`도 같은 설정을 사용한다.
Success/Failure는 저장 후 home으로 복귀한다. Stop/Retry/Exit는 home 이동도 취소한다.
새 `robot_commands_rollout_*.jsonl`은 실제 TCP와 전송 목표를 함께 기록한다.
데이터의 15 Hz와 GUI의 목표 60 Hz, 실제 달성 command Hz는 서로 구분해서 기록한다.

코드/오프라인 검증 통과는 로봇 성공률이 아니다. 이 문서는 성공을 보장한다고 주장하지 않는다.

## 이전 Head-only 실험 기록

This package prepares a controlled comparison on the 40 `stackcupanddoll`
teleoperation trajectories. Live robot control is implemented separately in
`architectures/simvla/adapters/latentloop_real_deploy` and remains disabled
until its artifact, hardware, timing, baseline-canary, and operator approvals
all pass.

## Initialization contract

There is no scratch-head or reinitialization ablation in this protocol. Every
baseline parameter is loaded from the complete released
`YuankaiLuo/SimVLA-LIBERO` checkpoint. Loading fails when Hugging Face reports
any missing, unexpected, or shape-mismatched tensor. The released VLM is then
frozen and the already initialized action transformer is fine-tuned on the real
demonstrations. The compact real checkpoint stores only that transformed head
and records the SHA-256 of its complete official parent.

## Data contract

- 40 trajectories are split by episode into 32 train and 8 validation episodes.
- The source and deployment control rate is 15 Hz. Source RGB, pose, and
  control records remain in their native synchronized order; the converter
  does not snap an already-15 Hz capture stream onto a second time grid.
- A transition whose capture interval differs from one nominal period by more
  than half a period is excluded. Every H=10 window crossing such a gap is
  omitted from training and counted in the dataset manifest.
- Exterior and wrist RGB are kept in that order. Cache creation, training, and
  deployment call the same JPEG95/subsampling0, resize-with-pad-224 then
  bicubic-384 transform and ImageNet normalization. The processor is used for
  text tokenization, not as an alternative image-normalization path.
- State is `[TCP xyz, TCP rotation-vector, +finger opening, -finger opening]`.
- `control[:6]` is an absolute joint command and is deliberately ignored.
- Each action is reconstructed from consecutive TCP poses as
  `inv(T_current) @ T_next`, then represented as local xyz / 0.02 m, XYZ Euler
  / 0.05 rad. The gripper target is the synchronized current-frame command
  `1 - 2 * command_t`, so it remains continuous with `+1=open,-1=close`.
- Pose-label clipping is rejected by default. The converter writes the measured
  extrema before stopping so that a scale change can be reviewed explicitly.

## Efficient training path

The VLM is frozen, so its exact FP32 action conditions are computed once and
memory-mapped. Baseline fine-tuning then trains only the existing action
transformer from this cache. The selected defaults are 3,000 optimizer steps,
local batch 4, and effective global batch 64. Gradient accumulation is derived
from the requested GPU count so that four GPUs use accumulation 4 and one GPU
uses accumulation 16. These are an engineering protocol for the available 40
demonstrations, not an official SimVLA paper setting.

After the baseline is fixed, the Condition Updater (`K_C=2`) and Generation
Updater (`N_G=3`, full evaluations at solver indices 0, 4, and 8) train in
parallel when at least two GPUs are available and sequentially on one GPU. Both
checkpoints must name the exact real baseline SHA-256 as their teacher. Loss
magnitudes are measured deterministically before training and normalized to
equal initial contribution. This balances numerical scales; it does not prove
that the chosen loss balance is optimal for physical task success.

A subsequent 10,000-step coupling stage freezes both trained updaters and
trains only the Generation Updater's existing 128-by-128 condition-code
projection (16,384 parameters). Its code is the same delta-encoder output used
by the Condition Updater. The objective is local-oracle hidden-state MSE under
the predicted condition. It is not end-to-end joint training and cannot by
itself establish that the approximate condition is correct. Deployment uses
this coupled checkpoint, not the uncoupled generation checkpoint.

Legacy data and checkpoints with next-frame gripper labels are not deployment
inputs. The wrapper requires dataset v3, cache v2, and v2 real checkpoint
formats. Optional cache migration reuses only frozen conditions after checking
image, proprioception, instruction, and record identities. A separate exact
condition check covers one query from each of all 40 episodes before reuse.

Training resume is deliberately not part of the scientific wrapper contract.
An interrupted optimizer run does not preserve the exact distributed sampler
and per-rank random-number state. The wrapper therefore moves an incomplete
run into `quarantine/` and restarts that stage from step zero. Completed stages
with a validated `run_summary.json` are reused.

## Comparison contract

Baseline and LatentLoop share all of the following:

- official parent checkpoint and real action-transformer overlay;
- normalization and image/state preprocessing;
- fresh ten-action output on every policy query;
- execution of the first five actions before the next query (`H=10,R=5`);
- deterministic per-query initial flow noise;
- camera order, task instruction, workspace, and safety gates.

Only the internal compute schedule changes: baseline uses `K_C=1,N_G=10`, and
LatentLoop uses `K_C=2,N_G=3`. Real trials should alternate method order within
paired scene resets and record success, policy latency, deadline misses, and
all module-call counters. The deployment manifest starts with live execution
disabled even after training completes.

## Commands

```bash
export SIMVLA_REAL_RAW_DATA=/path/to/stackcupanddoll
bash architectures/simvla/wrappers/train_real_stackcupanddoll.sh --preflight

SIMVLA_REAL_TRAIN_RUN=1 \
bash architectures/simvla/wrappers/train_real_stackcupanddoll.sh --all
```

For a machine that receives an already audited compact dataset, set
`SIMVLA_REAL_DATASET=/path/to/converted_dataset`. The wrapper verifies its
manifest and normalization file and skips raw-data conversion; the original
58 GB teleoperation directory does not need to be duplicated on that machine.

Before live use, run both artifact preflights and the read-only hardware profile
from `architectures/simvla/wrappers/deploy_latentloop_real.sh`. Live approval is
intentionally not part of the training wrapper.
