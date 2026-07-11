# LR-NODE Current Experiment Results

작성일: 2026-06-16 KST

Archive note: 이 문서의 pre-protocol LR-NODE checkpoint/eval 결과는 아래 위치로 이동했다. 본문 표의 일부 경로는 이동 전 원래 경로를 기록한 것이다.

```text
$SEER_WORKSPACE_ROOT/archived_experiment_results_20260616/pre_protocol_lrnode
```

Protocol note: 아래 수치는 2026-06-17 `shifted_context` teacher target 수정 이전 결과다. 이 결과는 LR-NODE skip/eval/logging/video infrastructure가 동작한다는 구현 확인용으로만 사용하고, 새 방법론 성능 주장은 `runs_lrnode_protocol_20260616/` 아래 새 학습/평가 결과로 다시 작성해야 한다.

## 1. 실험 설정

비교 대상:

| 항목 | 값 |
|---|---|
| Baseline | Seer original, ckpt 37 |
| Baseline path | `$SEER_BASELINE_ROOT/checkpoints/sd1_libero_10_100pc_original_settings/37.pth` |
| Ours | `lrnode_student_v2_lw05_aw01_g4`, ckpt 35 |
| Ours path | `$SEER_WORKSPACE_ROOT/scratch_checkpoints_lrnode/sd1_scratch_libero_10_converted_seer_lrnode_student_v2_lw05_aw01_g4/35.pth` |
| Benchmark | LIBERO-10, 10 tasks x 20 episodes |
| Action rate | 20 Hz |
| Video | success/fail 모두 저장, 각 run 200개 |
| Result root | `scratch_eval_lrnode/lrnode_compare_lrnode_student_v2_lw05_aw01_g4_ckpt35_vs_seer_original_ckpt37_20260615_001039` |

## 2. 전체 결과

| Run | SR | Delta SR | Preservation | Full-query red. | Full Hz | Policy mean | Policy red. | Full call | LR call | Mean ep. jerk p95 | Videos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline K=1 | 86.0% | +0.0pp | 100.0% | 0.0% | 20.00 | 79.2 ms | 0.0% | 67.9 ms | 0.00 ms | 0.0707 | 172/28 |
| Ours full K=1 | 83.0% | -3.0pp | 96.5% | 0.0% | 20.00 | 79.2 ms | 0.0% | 67.9 ms | 0.00 ms | 0.0859 | 166/34 |
| Ours K=2 | 85.5% | -0.5pp | 99.4% | 49.9% | 10.00 | 49.7 ms | 37.3% | 68.6 ms | 7.67 ms | 0.1294 | 171/29 |
| Ours K=3 | 88.0% | +2.0pp | 102.3% | 66.6% | 6.67 | 39.4 ms | 50.3% | 68.4 ms | 7.52 ms | 0.1843 | 176/24 |
| Ours K=4 | 86.5% | +0.5pp | 100.6% | 74.9% | 5.00 | 34.7 ms | 56.3% | 69.2 ms | 7.62 ms | 0.4108 | 173/27 |
| Ours K=5 | 84.0% | -2.0pp | 97.7% | 79.9% | 4.00 | 31.3 ms | 60.5% | 68.8 ms | 7.51 ms | 0.3923 | 168/32 |
| Ours K=6 | 81.0% | -5.0pp | 94.2% | 83.2% | 3.33 | 29.4 ms | 63.0% | 69.0 ms | 7.53 ms | 0.4529 | 162/38 |
| Ours K=8 | 83.0% | -3.0pp | 96.5% | 87.4% | 2.50 | 26.9 ms | 66.1% | 69.4 ms | 7.56 ms | 0.4599 | 166/34 |

`Videos`는 `success/fail` 개수다.

## 3. 핵심 해석

가장 안전한 대표 결과는 K=2다.

- Full Seer query를 49.9% 줄인다.
- Policy inference latency를 37.3% 줄인다.
- SR은 85.5%로 baseline 86.0% 대비 -0.5pp다.
- Success preservation은 99.4%다.
- Smoothness는 나빠진다. mean episode jerk p95가 baseline 대비 1.83배다.

가장 좋은 frontier 결과는 K=3다.

- Full Seer query를 66.6% 줄인다.
- Policy inference latency를 50.3% 줄인다.
- SR은 88.0%로 이 run에서는 baseline보다 +2.0pp 높다.
- 단, mean episode jerk p95가 baseline 대비 2.61배다.

K=4는 성공률만 보면 좋지만 action quality 리스크가 크다.

- SR 86.5%, baseline 대비 +0.5pp다.
- Full-query reduction은 74.9%, policy latency reduction은 56.3%다.
- 하지만 mean episode jerk p95가 baseline 대비 5.81배까지 증가한다.

K=5, K=6, K=8은 main setting으로 쓰기 어렵다.

- K=5: SR 84.0%, -2.0pp
- K=6: SR 81.0%, -5.0pp
- K=8: SR 83.0%, -3.0pp
- 세 설정 모두 jerk가 baseline 대비 약 5.5배에서 6.5배 수준이다.
- 따라서 aggressive efficiency ablation 또는 upper-bound setting으로만 보여주는 것이 맞다.

## 4. Task별 패턴

| Task | Baseline | Ours full | K=2 | K=3 | K=4 | K=5 | K=6 | K=8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90% | 95% | 90% | 90% | 90% | 95% | 90% | 100% |
| 1 | 75% | 100% | 95% | 100% | 100% | 90% | 100% | 95% |
| 2 | 95% | 100% | 85% | 100% | 100% | 100% | 95% | 90% |
| 3 | 80% | 95% | 100% | 100% | 100% | 100% | 100% | 95% |
| 4 | 75% | 85% | 80% | 90% | 85% | 85% | 80% | 70% |
| 5 | 100% | 85% | 85% | 100% | 100% | 100% | 95% | 95% |
| 6 | 90% | 70% | 80% | 70% | 70% | 65% | 65% | 70% |
| 7 | 100% | 95% | 95% | 100% | 100% | 100% | 100% | 100% |
| 8 | 75% | 45% | 75% | 60% | 60% | 45% | 30% | 55% |
| 9 | 80% | 60% | 70% | 70% | 60% | 60% | 55% | 60% |

주요 관찰:

- Ours full K=1부터 task 6/8/9가 baseline보다 약하다. 즉 LR-NODE skip 때문만이 아니라 student checkpoint 자체가 일부 task에서 약하다.
- K=2는 task 8을 ours full 45%에서 75%로 회복한다.
- K=3은 task 1/2/3/5/7에서 100%를 기록하며 전체 SR 88.0%를 만든다.
- K=6은 task 8이 30%까지 떨어져 전체 SR을 크게 깎는다.
- K=8은 task 0/1/3/5/7은 강하지만 task 4/6/8/9가 낮다.

## 5. 발표/논문용 문장

방어적인 main claim:

```text
LR-NODE는 K=2에서 full Seer query를 약 50%, policy inference latency를 약 37% 줄이면서
LIBERO-10 success rate를 거의 보존한다.
```

Frontier claim:

```text
K=3에서는 full Seer query를 약 66.6%, policy inference latency를 약 50% 줄이고,
현재 run에서는 baseline보다 높은 success rate를 보인다.
```

주의 문장:

```text
K가 커질수록 action smoothness가 악화되며, K=4 이상은 success rate만으로는 안전한 설정이라고 보기 어렵다.
현재 LIBERO eval wall-clock은 env.step simulation/rendering이 지배하므로 overall runtime reduction으로 주장하면 안 된다.
```
