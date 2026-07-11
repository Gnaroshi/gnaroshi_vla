# Baseline eval.sh 결과 분석

작성일: 2026-06-20 KST

분석 대상은 현재 repository에서 `scratch.sh`로 학습한 plain Seer baseline이다. LR-NODE는 사용하지 않았다.

## 1. 분석 대상

Baseline 학습 실행:

```text
sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040
```

Checkpoint root 경로:

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040
```

최신 eval result root:

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/eval/baseline_sweep_sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040_20260618_132356
```

동일 SR 재현 확인용 이전 eval root:

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/eval/baseline_sweep_sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040_20260617_095300
```

Eval 설정:

- benchmark: `libero_10`
- ckpt: 30-39
- episodes: ckpt당 200 episodes, task당 20 episodes
- control/action Hz: 20 Hz
- LR-NODE: disabled
- query interval: K=1
- full Seer forward calls: every environment step
- video: enabled, all ranks, success/fail 모두 저장

`eval_summary.json` 기준으로 모든 ckpt에서:

- `lrnode.enabled=false`
- `lrnode.lrnode_update_calls=0`
- `query_reduction.full_query_reduction_ratio=0.0`
- `.mp4` video count = 200 / ckpt

즉 이 결과는 LR-NODE가 끼지 않은 plain full-forward Seer baseline이다. 로그 prefix가 `[LR-NODE eval]`로 찍히는 부분은 공통 logging 함수 이름 때문에 생긴 표기이며, JSON상 LR-NODE는 꺼져 있다.

## 2. Checkpoint별 결과

최신 eval root `20260618_132356` 기준.

| ckpt | SR (%) | success / 200 | env steps | full calls | full ms | policy ms | env ms | videos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 78.0 | 156 | 69,616 | 69,616 | 68.71 | 88.51 | 275.11 | 200 |
| 31 | 79.5 | 159 | 69,581 | 69,581 | 68.78 | 88.69 | 271.34 | 200 |
| 32 | 77.5 | 155 | 69,942 | 69,942 | 68.53 | 88.51 | 272.46 | 200 |
| 33 | 83.0 | 166 | 66,564 | 66,564 | 69.13 | 89.34 | 281.20 | 200 |
| 34 | 79.0 | 158 | 68,443 | 68,443 | 69.18 | 89.50 | 282.20 | 200 |
| 35 | 77.0 | 154 | 69,725 | 69,725 | 69.13 | 89.45 | 282.05 | 200 |
| 36 | 81.0 | 162 | 67,696 | 67,696 | 69.31 | 89.59 | 281.55 | 200 |
| 37 | 80.0 | 160 | 67,635 | 67,635 | 69.47 | 89.97 | 289.56 | 200 |
| 38 | 81.0 | 162 | 69,029 | 69,029 | 67.44 | 83.35 | 290.10 | 200 |
| 39 | 83.0 | 166 | 67,797 | 67,797 | 68.58 | 88.36 | 268.33 | 200 |

용어:

- `SR`: success rate. `success episodes / total episodes`.
- `env steps`: evaluation 전체에서 실제 environment를 step한 횟수.
- `full calls`: Seer full forward 호출 횟수. baseline에서는 `full calls = env steps`.
- `full ms`: full Seer forward 1회 평균 latency.
- `policy ms`: preprocessing + policy inference를 포함한 policy step 평균 latency.
- `env ms`: environment step 평균 latency.
- `videos`: 저장된 `.mp4` episode video 수.

## 3. Best checkpoint

최고 SR은 ckpt 33과 ckpt 39가 모두 83.0%로 동률이다.

Primary baseline으로는 ckpt 33을 권장한다.

근거:

- ckpt 33: 83.0%, 166/200 success, 66,564 env steps
- ckpt 39: 83.0%, 166/200 success, 67,797 env steps
- 같은 SR에서 ckpt 33이 더 적은 environment step으로 episode를 끝냈다.
- latency는 별도 eval run의 runtime noise 영향을 받는다. 실제로 이전 sweep에서는 ckpt 33과 39의 latency 순위가 달라진다. 따라서 tie-breaker로 latency보다 SR과 env steps를 우선한다.

Primary checkpoint 경로:

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/33.pth
```

Tie backup checkpoint 경로:

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/39.pth
```

## 4. Task별 결과

Task ID 매핑:

| task | name |
|---:|---|
| 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` |
| 1 | `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` |
| 2 | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` |
| 3 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` |
| 4 | `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` |
| 5 | `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` |
| 6 | `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` |
| 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` |
| 8 | `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` |
| 9 | `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` |

Task별 SR (%):

| ckpt | avg | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 78.0 | 65 | 90 | 90 | 80 | 75 | 95 | 80 | 90 | 40 | 75 |
| 31 | 79.5 | 85 | 95 | 95 | 95 | 85 | 85 | 70 | 85 | 35 | 65 |
| 32 | 77.5 | 85 | 90 | 100 | 90 | 55 | 90 | 80 | 95 | 45 | 45 |
| 33 | 83.0 | 80 | 90 | 100 | 100 | 75 | 90 | 75 | 95 | 55 | 70 |
| 34 | 79.0 | 75 | 95 | 95 | 80 | 60 | 85 | 75 | 95 | 50 | 80 |
| 35 | 77.0 | 95 | 90 | 100 | 75 | 65 | 85 | 75 | 95 | 30 | 60 |
| 36 | 81.0 | 75 | 95 | 100 | 85 | 70 | 90 | 75 | 95 | 50 | 75 |
| 37 | 80.0 | 80 | 70 | 100 | 90 | 60 | 90 | 90 | 100 | 50 | 70 |
| 38 | 81.0 | 80 | 100 | 100 | 80 | 50 | 90 | 90 | 95 | 45 | 80 |
| 39 | 83.0 | 80 | 100 | 100 | 85 | 65 | 90 | 80 | 100 | 55 | 75 |

관찰:

- 가장 어려운 task는 task 8이다. 전체 ckpt에서 30-55% 범위이고, best도 55%다.
- task 4와 task 9도 ckpt에 따라 변동이 크다.
- task 2는 거의 포화 상태다. ckpt 32 이후 대부분 100%다.
- task 7도 강한 편이다. ckpt 37/39에서 100%다.
- ckpt 33은 task 3에서 100%를 찍고, ckpt 39는 task 1/7에서 더 강하다. 평균은 같다.

## 5. Smoothness / action 변화량

최신 eval root 기준.

| ckpt | delta mean | delta p95 | jerk mean | jerk p95 |
|---:|---:|---:|---:|---:|
| 30 | 0.064038 | 0.106529 | 0.062287 | 0.064359 |
| 31 | 0.062725 | 0.105169 | 0.060516 | 0.053394 |
| 32 | 0.063922 | 0.104985 | 0.062452 | 0.105195 |
| 33 | 0.066217 | 0.117215 | 0.064881 | 0.087384 |
| 34 | 0.064602 | 0.108396 | 0.062225 | 0.055544 |
| 35 | 0.064705 | 0.107446 | 0.063182 | 0.082671 |
| 36 | 0.064496 | 0.107635 | 0.062479 | 0.070015 |
| 37 | 0.064396 | 0.108257 | 0.062077 | 0.073840 |
| 38 | 0.063955 | 0.107900 | 0.061798 | 0.073923 |
| 39 | 0.065279 | 0.108326 | 0.063594 | 0.089346 |

정의:

- action delta는 같은 episode 안에서 인접 action의 L2 변화량이다.
  - \( \Delta a_t = a_t - a_{t-1} \)
  - `delta mean` = \( \mathbb{E}_t[\lVert \Delta a_t \rVert_2] \)
  - `delta p95` = \( \mathrm{Percentile}_{95}(\lVert \Delta a_t \rVert_2) \)
- action jerk는 action 변화량의 변화, 즉 2차 차분의 L2 값이다.
  - \( j_t = (a_t - a_{t-1}) - (a_{t-1} - a_{t-2}) = a_t - 2a_{t-1} + a_{t-2} \)
  - `jerk mean` = \( \mathbb{E}_t[\lVert j_t \rVert_2] \)
  - `jerk p95` = \( \mathrm{Percentile}_{95}(\lVert j_t \rVert_2) \)

Baseline끼리 비교하면 smoothness 차이는 크지 않다. LR-NODE K-sweep과 비교할 때는 SR뿐 아니라 `delta p95`, `jerk p95`, `gripper_switch_rate`까지 같이 봐야 한다. 특히 skip update가 action을 불안정하게 만들면 평균 SR이 비슷해도 jerk tail이 커질 수 있다.

## 6. Reproducibility check

두 baseline eval sweep의 ckpt별 SR은 완전히 동일했다.

| ckpt | 20260617_095300 SR | 20260618_132356 SR |
|---:|---:|---:|
| 30 | 78.0 | 78.0 |
| 31 | 79.5 | 79.5 |
| 32 | 77.5 | 77.5 |
| 33 | 83.0 | 83.0 |
| 34 | 79.0 | 79.0 |
| 35 | 77.0 | 77.0 |
| 36 | 81.0 | 81.0 |
| 37 | 80.0 | 80.0 |
| 38 | 81.0 | 81.0 |
| 39 | 83.0 | 83.0 |

SR은 재현됐다. latency는 두 sweep 사이에서 약간 다르므로, latency claim은 같은 실행 환경에서 baseline과 ours를 같은 script로 연속 평가한 결과를 우선해야 한다.

## 7. 다음 실험 기준

LR-NODE distill 실험을 baseline best ckpt에서 시작하려면:

```bash
BASELINE_CKPT_ID=33 bash scripts/LIBERO_LONG/Seer/distill_node.sh
```

ckpt 39도 동률 backup으로 확인하려면:

```bash
BASELINE_CKPT_ID=39 bash scripts/LIBERO_LONG/Seer/distill_node.sh
```

LR-NODE eval 비교 기준:

- baseline reference: ckpt 33 primary, ckpt 39 backup
- baseline query 감소율: 0%
- baseline full-query Hz: 20 Hz
- baseline full calls: env step마다 1회
- comparison target: LR-NODE K=2/3/4/5/6/8에서 SR 유지율, full-query reduction, full/policy/env latency, video, smoothness를 모두 기록

중요한 해석:

- 이 baseline 결과는 현재 repository의 새 protocol에서 생성된 유효한 기준점이다.
- pre-protocol old ours 결과와 직접 논문 claim을 만들면 안 된다.
- 새 LR-NODE 결과는 이 baseline run 및 동일한 eval logging 기준으로 다시 비교해야 한다.
