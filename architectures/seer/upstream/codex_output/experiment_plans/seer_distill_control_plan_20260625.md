# Seer-Only Distillation Control Plan

## 목적

현재 LR-NODE distill 결과에서 QRED20 K>1 성능이 오른 원인이 다음 중 무엇인지 분리한다.

1. LR-NODE latent update가 실제로 skipped step에서 더 좋은 action을 만든 효과
2. 같은 teacher checkpoint와 같은 dataset을 사용한 추가 distillation 자체가 성능을 올린 효과

이를 위해 LR-NODE를 완전히 끈 Seer-only teacher distillation control을 추가했다.

## 기존 LR-NODE Distill과 차이

기존 `distill_node.sh`:

- baseline Seer checkpoint를 로드한다.
- Seer backbone/action head는 freeze한다.
- LR-NODE module만 학습한다.
- K=1 full forward는 baseline teacher와 동일해야 한다.
- K>1에서만 LR-NODE update path가 사용된다.

새 `distill_seer.sh`:

- baseline Seer checkpoint를 frozen teacher로 사용한다.
- LR-NODE는 완전히 비활성화한다.
- student Seer 자체를 teacher action/latent에 맞추도록 학습한다.
- 평가는 K=1 full Seer만 가능하다.
- 목적은 "teacher KD 자체가 K=1 성능을 올리는가"를 확인하는 것이다.

## 새 코드

- `utils/arguments_utils.py`
  - `--seer_distill_teacher_ckpt`
  - `--seer_distill_action_weight`
  - `--seer_distill_latent_weight`
  - `--seer_distill_teacher_eval_mode`

- `train.py`
  - frozen raw teacher Seer를 별도 생성한다.
  - teacher checkpoint의 `module.` prefix를 제거해서 raw model에 로드한다.
  - teacher는 DDP로 감싸지 않는다.

- `utils/train_utils.py`
  - `seer_distill_teacher_model`이 주어지면 teacher forward를 `torch.no_grad()`로 수행한다.
  - student action과 teacher action 사이 L1 KD loss를 추가한다.
  - 선택적으로 action latent MSE KD loss를 추가한다.
  - LR-NODE distill과 Seer-only distill은 동시에 켤 수 없게 막았다.

- `scripts/LIBERO_LONG/Seer/distill_seer.sh`
  - Seer-only teacher distillation 학습 스크립트

- `scripts/LIBERO_LONG/Seer/eval_seer_distill.sh`
  - Seer-only distill checkpoint K=1 평가 스크립트

## 권장 실행 순서

먼저 현재 돌고 있는 QRED/HZUP eval과 겹치지 않게 한다.

순수 negative control:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
SEER_DISTILL_STUDENT_INIT=teacher \
SEER_DISTILL_USE_BASE_LOSS=0 \
SEER_DISTILL_ACTION_WEIGHT=1.0 \
SEER_DISTILL_LATENT_WEIGHT=0.0 \
bash scripts/LIBERO_LONG/Seer/distill_seer.sh
```

평가:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
CKPT_IDS="31 32 33 34 35 36 37 38 39" \
bash scripts/LIBERO_LONG/Seer/eval_seer_distill.sh
```

## 해석

만약 Seer-only distill K=1 성능이 baseline ckpt33보다 유의미하게 오른다면:

- LR-NODE QRED 상승의 일부는 "teacher/self-distillation regularization" 또는 추가 학습 효과일 수 있다.
- 이 경우 QRED K>1 상승을 LR-NODE만의 효과라고 주장하면 안 된다.

만약 Seer-only distill K=1 성능이 baseline과 같거나 낮다면:

- "같은 teacher와 같은 dataset으로 distill해서 성능이 오른 것"이라는 설명은 약해진다.
- QRED K>1 상승은 LR-NODE skip path, sparse full refresh, temporal dynamics, gripper timing 변화 등 skipped-step policy 변화에서 찾아야 한다.

## 주의

`distill_seer.sh`는 query reduction 실험이 아니다. LR-NODE가 꺼져 있으므로 K=2,3,4를 정의할 수 없다. 이 control은 오직 K=1 full-Seer 성능 변화 여부를 보기 위한 것이다.
