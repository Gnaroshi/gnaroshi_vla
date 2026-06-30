# LR-NODE Results Analysis 2026-06-26

## 현재 완료 상태

완료된 결과:

- baseline scratch K=1 sweep: ckpt30-39, 20Hz
- scratch-node K=1 sweep: ckpt30-39, 20Hz
- scratch-node ckpt36 QRED20/HZUP20Q
- distill-node ckpt39 load parity
- distill-node ckpt39 QRED20
- distill-node ckpt39 HZUP20Q 일부: 20Hz, 40Hz, 60Hz row 완료

아직 완료/실행 확인이 필요한 결과:

- distill-node ckpt31-38 QRED20 sweep: 아직 결과 파일 없음
- Seer-only distill control: 코드/스크립트는 준비됐지만 결과 없음
- distill-node HZUP20Q 80Hz row: 현재 eval이 계속 실행 중

## Baseline Scratch

baseline scratch의 20Hz K=1 최고 SR은 ckpt33/ckpt39의 83.0%다.

| ckpt | SR |
|---:|---:|
| 30 | 78.0 |
| 31 | 79.5 |
| 32 | 77.5 |
| 33 | 83.0 |
| 34 | 79.0 |
| 35 | 77.0 |
| 36 | 81.0 |
| 37 | 80.0 |
| 38 | 81.0 |
| 39 | 83.0 |

동일 baseline sweep을 두 번 수행했지만 SR/step/smoothness 값은 동일했고 latency만 실행 시점에 따라 달랐다. 따라서 baseline SR 자체는 재현됐다.

## Scratch-Node From-Scratch 결과

scratch-node의 K=1 최고 SR은 ckpt36의 82.5%다.

| ckpt | K=1 SR |
|---:|---:|
| 30 | 80.5 |
| 31 | 77.0 |
| 32 | 79.5 |
| 33 | 76.0 |
| 34 | 75.5 |
| 35 | 78.5 |
| 36 | 82.5 |
| 37 | 78.5 |
| 38 | 82.0 |
| 39 | 79.5 |

이 결과는 baseline과 별도 scratch 학습이므로, 논문용 핵심 인과 주장에는 조심해야 한다. 다만 LR-NODE skip infrastructure와 QRED/HZUP 동작성은 확인된다.

scratch-node ckpt36 QRED20:

| setting | SR | full query reduction | policy ms | LR-NODE ms | jerk p95 |
|---|---:|---:|---:|---:|---:|
| K1 | 82.5 | 0.0 | 70.388 | 0.000 | 0.079720 |
| K2 | 79.0 | 49.9 | 47.900 | 7.205 | 0.216848 |
| K3 | 82.5 | 66.6 | 36.843 | 6.965 | 0.259359 |
| K4 | 86.5 | 74.9 | 33.639 | 7.237 | 0.307808 |

scratch-node ckpt36 HZUP20Q:

| setting | SR | full query Hz | LR-NODE Hz | policy ms | jerk p95 |
|---|---:|---:|---:|---:|---:|
| 20Hz K1 | 82.5 | 20.0 | 0.0 | 82.818 | 0.079720 |
| 40Hz K1 | 83.0 | 40.0 | 0.0 | 75.570 | 0.037257 |
| 40Hz K2 | 84.5 | 20.0 | 20.0 | 45.061 | 0.166171 |
| 60Hz K1 | 73.5 | 60.0 | 0.0 | 76.322 | 0.030780 |
| 60Hz K3 | 78.5 | 20.0 | 40.0 | 36.392 | 0.310004 |
| 80Hz K1 | 77.5 | 80.0 | 0.0 | 75.659 | 0.026549 |
| 80Hz K4 | 78.0 | 20.0 | 60.0 | 32.855 | 0.616898 |

Scratch-node HZUP은 higher control Hz에서 full Seer query Hz를 20Hz로 유지하면서 action Hz를 올리는 실험으로는 의미가 있다. 하지만 모델이 baseline scratch와 동일하지 않으므로, 최종 주장에는 distill-node protocol이 더 중요하다.

## Distill-Node ckpt39 QRED20

이 결과가 현재 가장 중요한 결과다. 이유는 K=1에서 baseline full과 ours full이 정확히 같은 SR/step/smoothness를 보였기 때문이다.

| setting | SR | full query reduction | full query Hz | LR-NODE Hz | policy ms | full ms | LR-NODE ms | jerk p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline K1 | 83.0 | 0.0 | 20.00 | 0.00 | 73.159 | 62.870 | 0.000 | 0.087384 |
| ours K1 | 83.0 | 0.0 | 20.00 | 0.00 | 73.452 | 63.005 | 0.000 | 0.087384 |
| ours K2 | 84.0 | 49.9 | 10.01 | 9.99 | 44.985 | 62.468 | 6.690 | 0.183275 |
| ours K3 | 85.5 | 66.6 | 6.68 | 13.32 | 38.133 | 66.403 | 7.236 | 0.229699 |
| ours K4 | 91.0 | 74.9 | 5.02 | 14.98 | 32.281 | 65.299 | 6.928 | 0.563964 |

K=4는 baseline 대비 +8.0pp다. episode-paired flip 기준:

| setting | fail to success | success to fail | net |
|---|---:|---:|---:|
| K2 | 16 | 14 | +2 |
| K3 | 18 | 13 | +5 |
| K4 | 25 | 9 | +16 |

K4의 task별 net gain은 task8 +5, task6 +3, task0 +2, task5 +2, task9 +2가 크다.

중요한 부정적 근거:

- K가 커질수록 action jerk가 크게 증가한다.
- K4 jerk p95는 baseline 0.087384에서 0.563964로 증가했다.
- 따라서 현재 성능 상승은 "smooth한 action 때문"이라고 주장하면 안 된다.
- 현재 로그가 보여주는 것은 "LR-NODE skip path가 행동 궤적을 바꾸고, 그 결과 timeout/failure episode 일부가 success로 바뀌었다"까지다.
- "정확히 왜 K4에서 SR이 올랐는지"는 shadow full-forward/action comparison과 hold/no-delta ablation 없이는 확정할 수 없다.

## Distill-Node ckpt39 HZUP20Q

현재 완료된 row:

| setting | SR | full query Hz | LR-NODE Hz | reduction | policy ms | jerk p95 |
|---|---:|---:|---:|---:|---:|---:|
| 20Hz baseline K1 | 83.0 | 20.00 | 0.00 | 0.0 | 73.242 | 0.087384 |
| 20Hz ours K1 | 83.0 | 20.00 | 0.00 | 0.0 | 73.632 | 0.087384 |
| 40Hz baseline K1 | 83.5 | 40.00 | 0.00 | 0.0 | 71.003 | 0.035067 |
| 40Hz ours K1 | 83.5 | 40.00 | 0.00 | 0.0 | 75.968 | 0.035067 |
| 40Hz ours K2 | 83.5 | 20.01 | 19.99 | 50.0 | 43.735 | 0.150604 |
| 60Hz baseline K1 | 83.5 | 60.00 | 0.00 | 0.0 | 69.509 | 0.049304 |
| 60Hz ours K1 | 83.5 | 60.00 | 0.00 | 0.0 | 75.916 | 0.049304 |
| 60Hz ours K3 | 81.0 | 20.02 | 39.98 | 66.6 | 37.728 | 0.312946 |

해석:

- 20/40/60Hz에서 ours K1은 baseline K1과 SR이 같다. adapter load parity는 유지된다.
- 40Hz K2는 full Seer query를 약 20Hz로 줄였는데 SR 83.5%를 유지했다.
- 60Hz K3는 full Seer query를 약 20Hz로 줄였지만 SR이 83.5%에서 81.0%로 -2.5pp 하락했다.
- HZUP에서 현재 가장 좋은 근거는 40Hz K2다. "control/action Hz 40Hz, full Seer query Hz 20Hz, SR 유지, policy latency 감소"가 명확하다.
- 60Hz K3는 성능 보존이 아직 부족하다.

paired flip:

| setting | baseline SR | ours SR | fail to success | success to fail | net |
|---|---:|---:|---:|---:|---:|
| 40Hz K2 | 83.5 | 83.5 | 18 | 18 | 0 |
| 60Hz K3 | 83.5 | 81.0 | 13 | 18 | -5 |

## 현재 결론

현재 가장 강한 positive result는 distill-node ckpt39 QRED20 K4다.

- K=1 parity가 성립한다.
- K4에서 full Seer call을 약 75% 줄인다.
- policy step latency가 73.159 ms에서 32.281 ms로 감소한다.
- SR은 83.0%에서 91.0%로 증가한다.

하지만 아직 causal claim은 제한해야 한다.

- ckpt39만 본 결과이므로 ckpt31-38 sweep이 필요하다.
- Seer-only distill control 결과가 없다.
- hold/no-delta/shadow ablation이 없어서, 성능 상승이 NODE dynamics 때문인지, sparse refresh/temporal hold/gripper timing 변화 때문인지 확정되지 않았다.
- jerk가 크게 증가하므로 smoothness 기반 설명은 현재 데이터와 맞지 않는다.

## 다음 실험 우선순위

1. 현재 돌고 있는 distill HZUP20Q를 끝낸다.
2. distill ckpt31-38 QRED20 sweep을 실행해서 ckpt39가 특이한지 확인한다.
3. Seer-only distill control을 실행해서 teacher KD 자체의 K=1 성능 향상 여부를 확인한다.
4. K4에 대해 shadow full-forward를 켜서 LR-NODE action과 full Seer action의 차이를 age별로 측정한다.
5. hold/no-delta ablation을 추가해 LR-NODE dynamics 자체의 기여를 분리한다.
