# LR-NODE QRED20 SR Gain Diagnosis - 2026-06-25

## Question

QRED20에서 LR-NODE distill adapter `ckpt39`가 K를 키울수록 성공률이 오히려 상승했다.

- Baseline full K=1: 83.0%
- Ours full K=1: 83.0%
- Ours skip K=2: 84.0%
- Ours skip K=3: 85.5%
- Ours skip K=4: 91.0%

이 문서는 K=4 상승이 어디서 발생했는지, 현재 데이터만으로 말할 수 있는 원인과 아직 검증해야 하는 원인을 분리한다.

## 1. 먼저 버그 가능성

`ours full K=1`이 baseline full K=1과 완전히 동일한 83.0%이고, task별 SR도 모두 동일하다.

따라서 아래 가능성은 현재 결과에서는 배제한다.

- adapter ckpt만 로드되어 Seer backbone/action head가 random인 문제
- baseline과 ours full-forward가 다른 checkpoint를 사용한 문제
- K=1에서 LR-NODE가 몰래 action에 개입한 문제

QRED20의 SR 상승은 `lrnode_eval_skip_full_forward=1`에서 실제로 full Seer query를 줄였을 때 생긴 policy 변화다.

## 2. K=4 상승이 어디서 생겼나

K=4는 baseline 대비 +8.0pp, 즉 200 episode 중 net +16 episode다.

Episode flip 기준:

- Baseline fail -> K4 success: 25 episodes
- Baseline success -> K4 fail: 9 episodes
- Net: +16 episodes
- Paired exact binomial test on discordant episodes: p ~= 0.009

즉 단순히 1-2개 episode 우연으로 보기는 어렵다.

## 3. Task별 변화

K=4에서 좋아진 task:

| Task | Baseline | K4 | Delta |
|---|---:|---:|---:|
| task0 | 80% | 90% | +10pp |
| task1 | 90% | 95% | +5pp |
| task5 | 90% | 100% | +10pp |
| task6 | 75% | 90% | +15pp |
| task7 | 95% | 100% | +5pp |
| task8 | 55% | 80% | +25pp |
| task9 | 70% | 80% | +10pp |

K4에서 유지된 task:

- task2: 100% -> 100%
- task3: 100% -> 100%
- task4: 75% -> 75%

가장 큰 기여는 task8이다.

- `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`
- Baseline 55% -> K4 80%
- episode flip: 9개 개선, 4개 악화, net +5/20

그 다음 큰 기여는 task6, task0, task5, task9다.

## 4. K4가 어떤 행동 변화를 만들었나

K4는 단순 smoothing이 아니다. 오히려 action dynamics가 더 커졌다.

| Metric | Baseline | K4 |
|---|---:|---:|
| action_delta_l2_mean | 0.066 | 0.088 |
| action_delta_l2_p95 | 0.117 | 0.229 |
| action_jerk_l2_mean | 0.065 | 0.109 |
| action_jerk_l2_p95 | 0.087 | 0.564 |
| arm_action_jerk | 0.020 | 0.030 |
| trans_action_jerk | 0.019 | 0.028 |
| rot_action_jerk | 0.006 | 0.008 |
| gripper_switch_rate | 0.011 | 0.020 |

주의할 점:

- `action_jerk_l2_p95`는 gripper의 binary jump 영향을 크게 받는다.
- K4의 arm/trans jerk도 증가했지만, p95가 0.56까지 커지는 주된 원인 중 하나는 gripper switch 증가다.
- 따라서 K4는 “더 부드러워져서 성공률이 올랐다”가 아니다.

더 정확한 해석:

> K4는 full Seer replanning 빈도를 줄이고, LR-NODE latent update가 action-relevant latent를 더 지속적으로 밀어주면서 일부 timeout/failure episode를 성공으로 바꾼다. 대신 gripper timing과 action jerk가 더 공격적으로 변한다.

## 5. 왜 성능이 올랐을 가능성이 큰가

현재 데이터로 가장 그럴듯한 원인은 다음이다.

### 5.1 Full Seer every-step replanning의 보수성/흔들림 완화

Baseline은 매 control step마다 full Seer를 다시 호출한다. 이때 observation, history, action ensembling의 작은 변화가 매번 새 action latent를 만든다.

K4에서는 full Seer를 5Hz 수준으로만 refresh하고, 중간 3 step은 LR-NODE가 이전 latent를 visual/proprio delta로 업데이트한다.

이 구조는 사실상 다음 효과를 낸다.

- full VLA query는 저주파 keyframe policy 역할
- LR-NODE는 중간 step의 reactive latent propagation 역할
- 매 step full replan의 작은 흔들림/보수성을 줄이고, 이전 intention을 더 오래 유지

task8, task6처럼 baseline이 timeout을 자주 내는 task에서 이 효과가 성공률 상승으로 나타난 것으로 보인다.

### 5.2 LR-NODE가 단순 hold보다 강한 latent perturbation을 만든다

K4의 평균 gate는 약 0.073, update norm은 약 0.615다. 즉 LR-NODE는 거의 0에 가까운 작은 correction만 하는 것이 아니라, action latent를 실제로 움직인다.

이 update가 일부 task에서는 baseline full-forward보다 더 유리한 trajectory를 만든다.

### 5.3 성공률 상승은 대부분 timeout rescue다

Baseline fail -> K4 success episode 25개는 baseline에서 대부분 600 step timeout이었다. K4에서는 이들이 149~512 step 사이에 성공한 경우가 많다.

즉 K4는 실패하던 episode를 단순히 더 오래 버티게 한 것이 아니라, 실제로 task completion까지 도달하게 했다.

## 6. 아직 확정할 수 없는 부분

현재 결과만으로는 아래를 “정확한 원인”으로 확정할 수 없다.

- LR-NODE latent update 자체가 좋은가?
- 아니면 full Seer query를 줄인 temporal persistence만 좋은가?
- action repeat/latent hold만 해도 비슷하게 좋아지는가?
- K4의 gripper timing 변화가 성공률 상승의 핵심인가?

이를 분리하려면 추가 ablation이 필요하다.

필요한 ablation:

1. K4 shadow full-forward
   - action은 LR-NODE로 실행
   - skipped step마다 full Seer action/latent를 기록
   - `shadow_action_l1`, `shadow_latent_mse`, `pred_vs_hold_improvement` 확인
2. latent hold / action hold baseline
   - full Seer는 K step마다만 호출
   - 중간 step은 LR-NODE 없이 cached latent/action을 hold
   - LR-NODE update가 정말 필요한지 확인
3. repeat seed
   - 같은 K4를 다른 seed 또는 repeated eval로 확인
   - 현재 paired flip p ~= 0.009라 유의미하지만, robotics eval은 반복 확인이 필요하다.

## 7. K를 더 키워야 하나?

예. 단, 목적은 “더 높은 SR 찾기”가 아니라 “붕괴 지점 찾기”다.

현재 K4는 이미 full query reduction 74.9%이고 SR 91.0%다. 하지만 jerk와 gripper switch가 크게 증가했다. 따라서 다음 실험은 K를 더 키워서 성능이 어디서 무너지는지 확인해야 한다.

추천 sweep:

- K=5
- K=6
- K=8

해석 기준:

- K5/K6에서 SR이 90% 근처로 유지되면 LR-NODE의 query reduction 주장이 더 강해진다.
- K8에서도 유지되면 매우 강한 결과지만, action jerk와 videos는 반드시 같이 봐야 한다.
- K5/K6/K8에서 SR이 급락하면 K4가 이 checkpoint의 sweet spot이다.

실행은 HZUP20Q가 끝난 뒤 해야 한다. 같은 GPU 4,5,6,7에서 동시에 돌리면 latency metric이 오염된다.

권장 명령:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
RUN_BASELINE=0 \
RUN_OURS_FULL=0 \
LRNODE_QUERY_INTERVALS_STR="5 6 8" \
EXPERIMENT_TAG=distill_qred20_klarge_20260625 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20.sh
```

그 다음 필요하면 더 극단:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
RUN_BASELINE=0 \
RUN_OURS_FULL=0 \
LRNODE_QUERY_INTERVALS_STR="10 12" \
EXPERIMENT_TAG=distill_qred20_kextreme_20260625 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20.sh
```
