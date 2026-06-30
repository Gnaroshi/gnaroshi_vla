# LR-NODE QRED20 SR Gain - Evidence Only Analysis

## 결론부터

QRED20의 K4 성능 상승은 checkpoint load/parity 오류가 아니다. 하지만 기존 로그만으로 “정확한 원인”을 완전히 확정할 수는 없다.

현재 로그로 확정 가능한 근거는 다음이다.

1. K=1 parity는 완전히 성립한다.
2. K4는 동일 200 episode에서 baseline failure 25개를 success로 바꿨고, baseline success 9개를 failure로 바꿨다.
3. net gain은 +16 episode, 즉 +8.0pp다.
4. 개선은 특정 task, 특히 task8/task6/task0/task5/task9의 timeout failure rescue에서 발생했다.
5. K4는 더 smooth해서 좋아진 것이 아니다. action jerk와 gripper switch는 증가했다.

따라서 현재 증거로 말할 수 있는 정확한 표현은 다음이다.

> K4 LR-NODE skip policy는 매-step full Seer policy와 다른 closed-loop trajectory를 만들고, 특히 baseline이 timeout하던 일부 episode를 성공 trajectory로 바꿨다. 성능 향상은 실제 episode flip으로 확인되지만, 이 변화가 LR-NODE latent update 자체 때문인지, full-forward 빈도 감소 때문인지, action/latent hold 효과인지, gripper timing 변화 때문인지는 아직 추가 ablation 없이는 확정할 수 없다.

## 1. Parity 확인

Baseline full K=1과 adapter-composed ours full K=1:

| Run | SR | Task별 결과 |
|---|---:|---|
| baseline full K=1 | 83.0% | reference |
| ours full K=1 | 83.0% | baseline과 전 task 동일 |

이는 다음을 의미한다.

- adapter-only ckpt가 단독으로 잘못 로드된 문제가 아니다.
- frozen baseline ckpt33 + adapter ckpt39 overlay가 정상이다.
- K=1 full-forward에서 LR-NODE skip path가 action에 개입하지 않는다.

## 2. K4의 정확한 episode-level 변화

Baseline과 K4는 같은 200개 episode를 평가했다.

| 변화 유형 | Episode 수 |
|---|---:|
| baseline fail -> K4 success | 25 |
| baseline success -> K4 fail | 9 |
| both fail | 9 |
| both success | 157 |

Net gain:

```text
25 - 9 = +16 episodes = +8.0pp
```

discordant episode 34개에서 K4가 이긴 횟수는 25회다. exact binomial 기준 p ~= 0.009로, 단순 1-2 episode noise로 보기 어렵다.

## 3. Task별 개선 위치

K4 task별 변화:

| Task | Baseline | K4 | Delta |
|---|---:|---:|---:|
| task0 | 80% | 90% | +10pp |
| task1 | 90% | 95% | +5pp |
| task2 | 100% | 100% | 0 |
| task3 | 100% | 100% | 0 |
| task4 | 75% | 75% | 0 |
| task5 | 90% | 100% | +10pp |
| task6 | 75% | 90% | +15pp |
| task7 | 95% | 100% | +5pp |
| task8 | 55% | 80% | +25pp |
| task9 | 70% | 80% | +10pp |

가장 큰 기여는 task8이다.

### task8

`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`

- Baseline fail ids: `[0, 1, 2, 4, 8, 10, 16, 17, 18]`
- K4 fail ids: `[3, 7, 15, 19]`
- baseline fail -> K4 success: 9
- baseline success -> K4 fail: 4
- net: +5/20

즉 task8에서는 baseline이 실패한 9개 episode를 K4가 모두 성공으로 바꿨고, 대신 다른 4개 episode를 새로 실패했다.

### task6

`LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`

- Baseline fail ids: `[9, 10, 14, 16, 17]`
- K4 fail ids: `[4, 14]`
- baseline fail -> K4 success: 4
- baseline success -> K4 fail: 1
- net: +3/20

### task0

`LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`

- Baseline fail ids: `[4, 5, 14, 17]`
- K4 fail ids: `[14, 18]`
- baseline fail -> K4 success: 3
- baseline success -> K4 fail: 1
- net: +2/20

## 4. 개선 episode는 baseline timeout을 성공으로 바꾼 것

Baseline failure는 모두 `num_steps=600` timeout이다.

K4가 rescue한 25개 episode의 K4 step 통계:

- mean steps: 330.1
- median steps: 333.0
- p95 steps: 471.8

즉 K4는 baseline timeout episode를 단순히 timeout까지 버틴 것이 아니라, 평균 330 step 근처에서 실제 success로 종료시켰다.

예시:

| Episode | Baseline | K4 |
|---|---:|---:|
| task8 exp0 | 600 fail | 386 success |
| task8 exp2 | 600 fail | 378 success |
| task8 exp10 | 600 fail | 384 success |
| task6 exp10 | 600 fail | 222 success |
| task5 exp15 | 600 fail | 149 success |

이것은 성능 향상이 평가 artifact가 아니라 실제 success termination 차이라는 직접 근거다.

## 5. K4는 smooth해져서 좋아진 것이 아니다

Summary metric:

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

따라서 다음 설명은 현재 데이터와 맞지 않는다.

> K4가 action을 smooth하게 만들어서 성공률이 올랐다.

반대로 현재 데이터가 말하는 것은:

> K4는 action dynamics와 gripper switch를 증가시켰고, 그럼에도 일부 timeout failure를 성공으로 바꿨다.

## 6. K4 update가 실제로 action path를 바꿨다는 근거

K4에서:

- full-forward ratio: 약 0.25
- skip ratio: 약 0.75
- effective full query Hz: 약 5.02Hz
- effective LR-NODE update Hz: 약 14.98Hz
- LR-NODE avg gate: 약 0.073
- LR-NODE avg update norm: 약 0.615

즉 K4는 “거의 full Seer와 같은 행동”이 아니다. 전체 policy step의 약 75%가 LR-NODE update이며, latent update norm도 0이 아니다.

하지만 이것은 “LR-NODE update 때문에 성능이 올랐다”의 충분조건은 아니다. 같은 K4 schedule에서 latent/action hold만 해도 좋아지는지 아직 모른다.

## 7. 기존 로그만으로는 아직 확정 불가능한 인과 원인

아래 원인은 현재 로그로 확정할 수 없다.

1. LR-NODE latent update 자체가 baseline보다 좋은가?
2. full Seer query를 줄여 temporal persistence가 생긴 것만으로도 좋아지는가?
3. action ensembling과 K-step refresh가 상호작용한 결과인가?
4. gripper timing 변화가 성공률 상승의 핵심인가?
5. K4가 task8의 특정 subgoal 순서를 더 잘 만든 것인가?

따라서 이 중 하나를 “정확한 원인”이라고 쓰면 근거 없는 주장이다.

## 8. 정확한 원인 규명을 위한 다음 ablation

K를 더 키우는 실험은 trend/붕괴점 확인에는 필요하지만, 원인 규명에는 충분하지 않다.

원인 규명을 위해 먼저 필요한 실험은 다음이다.

### A. K4 shadow full-forward

목적:

- 실행 action은 K4 LR-NODE 그대로 둔다.
- skipped step마다 full Seer도 shadow로 돌려서, LR-NODE action/latent가 full Seer와 얼마나 달라지는지 기록한다.

명령:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
RUN_BASELINE=0 \
RUN_OURS_FULL=0 \
LRNODE_QUERY_INTERVALS_STR="4" \
LRNODE_EVAL_SHADOW_FULL_FORWARD=1 \
EXPERIMENT_TAG=distill_qred20_K4_shadow_20260625 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20.sh
```

확인할 metric:

- `shadow_latent_mse`
- `shadow_latent_cos`
- `shadow_action_l1`
- `shadow_action_hold_l1`
- `pred_vs_hold_improvement`
- `by_cache_age`

이 실험으로 알 수 있는 것:

- LR-NODE가 full Seer action을 잘 근사하는지
- LR-NODE가 단순 latent/action hold보다 full Seer에 가까운지
- 성공 episode와 실패 episode에서 drift가 다른지

주의:

- shadow full-forward는 latency를 오염시킨다.
- SR/action은 그대로 해석 가능하지만 latency claim에는 쓰면 안 된다.

### B. K4 hold ablation

목적:

- 같은 K4 full-refresh schedule을 사용한다.
- 중간 step에서 LR-NODE update를 하지 않고 cached latent 또는 cached action을 hold한다.

필요한 이유:

- 이 실험이 없으면 “LR-NODE update가 좋아서”인지 “full query를 줄인 persistence가 좋아서”인지 분리할 수 없다.

해석:

- hold도 K4처럼 SR이 오르면 LR-NODE 자체보다 sparse full-query / temporal persistence 효과가 크다.
- hold는 낮고 LR-NODE만 높으면 LR-NODE update가 핵심 근거가 된다.

### C. no-ensembling ablation

목적:

- `--eval_libero_ensembling`을 끄고 K4 효과가 유지되는지 확인한다.

필요한 이유:

- 현재 Seer eval은 multi-step action ensembling을 사용한다.
- K-step refresh와 LR-NODE action sequence가 ensembling buffer에 들어가는 방식이 성능 향상에 영향을 줄 수 있다.

### D. K-large sweep

목적:

- 원인 규명이 아니라 붕괴점 확인.

명령:

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

이 실험은 HZUP20Q가 끝난 뒤 실행해야 한다.

## 9. 지금 당장 결론

현재 증거로 확정 가능한 결론:

> K4의 SR 상승은 실제 같은 episode set에서 timeout failure를 success로 바꾼 결과다. 특히 task8/task6/task0/task5/task9에서 개선이 집중된다. K4는 full Seer query를 75% 줄이면서 policy trajectory를 바꿨고, action jerk/gripper switch를 증가시키는 방향으로 행동을 더 공격적으로 만든다.

현재 증거로 아직 확정 불가능한 결론:

> 성능 향상의 원인이 LR-NODE latent update 자체다.

이 문장을 쓰려면 K4 hold ablation과 shadow full-forward evidence가 필요하다.
