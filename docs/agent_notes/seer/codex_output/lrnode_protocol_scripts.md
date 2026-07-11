# LR-NODE Protocol Scripts

작성일: 2026-06-16 KST

## 1. 목적

이제 LR-NODE 학습 스크립트는 `scratch/distill/joint` 여부로 고정해서 구분한다.

목표 claim별로 프로토콜을 나눈다.

| Claim | Protocol | Script |
|---|---|---|
| scratch baseline을 만든다 | Scratch Seer baseline | `scratch.sh` |
| 같은 scratch 조건에서 LR-NODE detached distillation도 학습한다 | Scratch + LR-NODE teacher-student detached | `scratch_node.sh` |
| 기존 Seer baseline을 고정하고 LR-NODE만 distill한다 | Frozen-baseline LR-NODE distill/adapter | `distill_node.sh` |
| scratch Seer 학습 중 LR-NODE loss를 Seer/action head에도 결합한다 | Scratch + LR-NODE coupled joint | `scratch_node_joint.sh` |

새로 실행하는 학습/평가 결과는 모두 아래 root로 분리한다.

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616
```

기존 pre-protocol LR-NODE 결과는 삭제하지 않고 아래 archive로 이동했다.

```text
$SEER_WORKSPACE_ROOT/archived_experiment_results_20260616/pre_protocol_lrnode
```

## 2. Scratch Baseline

이 프로토콜은 LR-NODE를 사용하지 않는 plain Seer scratch baseline이다.

학습:

```bash
bash scripts/LIBERO_LONG/Seer/scratch.sh
```

이 checkpoint는 `scratch_node.sh`, `scratch_node_joint.sh`, `distill_node.sh` 비교의 기준 baseline으로 사용한다.

저장 위치:

```text
runs_lrnode_protocol_20260616/train/scratch/
runs_lrnode_protocol_20260616/train/_latest/scratch.env
```

## 3. Scratch + LR-NODE Teacher-Student Detached

이 프로토콜은 pretrained Seer checkpoint를 load하지 않는다. Seer base loss와 LR-NODE distillation loss를 같은 run에서 계산하지만, LR-NODE loss는 detach/freeze 설정 때문에 Seer backbone/action head를 직접 업데이트하지 않는다.

핵심 target:

```text
--lrnode_teacher_target_mode shifted_context
--lrnode_context_selected_step -1
```

핵심 gradient:

```text
base loss -> Seer/action head
LR-NODE loss -> LR-NODE modules only
```

학습:

```bash
bash scripts/LIBERO_LONG/Seer/scratch_node.sh
```

저장 위치:

```text
runs_lrnode_protocol_20260616/train/scratch_node/
runs_lrnode_protocol_20260616/train/_latest/scratch_node.env
```

## 4. Frozen-Baseline LR-NODE Distill/Adapter

이 프로토콜은 baseline ckpt를 로드하고, 기존 Seer/action head를 모두 freeze한다. LR-NODE 모듈만 distill한다.

핵심 target:

```text
--lrnode_teacher_target_mode shifted_context
--lrnode_context_selected_step -1
```

Trainable:

```text
lrnode_delta_encoder
lrnode_dynamics
```

Frozen:

```text
transformer_backbone
perceiver_resampler
text/state/image projectors
action_decoder
arm_action_decoder
gripper_action_decoder
image decoder
vision encoder
CLIP
```

학습:

```bash
bash scripts/LIBERO_LONG/Seer/distill_node.sh
```

평가:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
```

이 protocol에서 K=1 full-forward 결과는 baseline과 같거나 매우 가까워야 한다. 이 sanity가 깨지면 "LR-NODE만 붙였다"는 claim을 하면 안 된다.

저장 위치:

```text
runs_lrnode_protocol_20260616/train/distill_node/
runs_lrnode_protocol_20260616/train/_latest/distill_node.env
runs_lrnode_protocol_20260616/eval/lrnode_distill_compare_*/
```

## 5. Scratch + LR-NODE Coupled Joint

이 프로토콜은 Seer baseline loss와 LR-NODE auxiliary loss를 같이 학습하고, LR-NODE loss가 Seer/action head에도 영향을 줄 수 있게 둔다.

학습 baseline control:

```bash
bash scripts/LIBERO_LONG/Seer/scratch.sh
```

학습 ours:

```bash
bash scripts/LIBERO_LONG/Seer/scratch_node_joint.sh
```

평가:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_joint_compare.sh
```

이 protocol에서는 `Ours full K=1`이 baseline과 달라질 수 있다. 따라서 "baseline에 LR-NODE만 붙였다"가 아니라 "LR-NODE auxiliary loss를 포함한 scratch-trained Seer variant"라고 해석해야 한다.

저장 위치:

```text
runs_lrnode_protocol_20260616/train/scratch_node_joint/
runs_lrnode_protocol_20260616/train/_latest/scratch_node_joint.env
runs_lrnode_protocol_20260616/eval/lrnode_scratch_joint_compare_*/
```

## 6. Logging

모든 새 eval wrapper는 기본적으로 아래를 켠다.

```text
SAVE_VIDEO=1
SAVE_VIDEO_SUCC=1
SAVE_VIDEO_FAIL=1
SAVE_VIDEO_ALL_RANKS=1
LRNODE_EVAL_STEP_LOG=1
LRNODE_EVAL_SHADOW_FULL_FORWARD=0
```

`LRNODE_EVAL_SHADOW_FULL_FORWARD=1`은 skip step마다 extra full forward를 수행하므로 policy latency 측정을 오염시킨다. Efficiency 실험에서는 기본값 0을 유지한다.

저장되는 주요 artifact:

```text
analysis/eval_summary.json
analysis/eval_latency_profile.json
analysis/eval_episode_metrics.csv
analysis/eval_step_logs/*.csv
analysis/model_trainable_params.json
analysis/freeze_status_snapshot.json
analysis/lrnode_flags_snapshot.json
eval_videos/success/*.mp4
eval_videos/fail/*.mp4
```

## 6. Override 예시

Adapter ckpt id를 바꿔 평가:

```bash
OURS_CKPT_ID=20 bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
```

K sweep만 줄여 빠르게 평가:

```bash
LRNODE_QUERY_INTERVALS_STR="2 3" bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
```

Scratch joint 평가에서 baseline ckpt를 명시:

```bash
BASELINE_CKPT=/path/to/plain_scratch_baseline.pth \
OURS_CKPT=/path/to/lrnode_scratch_joint.pth \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_joint_compare.sh
```
