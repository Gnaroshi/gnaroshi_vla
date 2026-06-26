# LR-NODE QRED20 / HZUP20Q 결과 분석 - 2026-06-22

## 상태

- QRED20 best-checkpoint 평가는 완료되었다.
- HZUP20Q는 아직 실행 중이다. 이 문서 작성 시점에 완료된 행은 `20:1`, `40:2`, `60:3`이다.
- Scratch-node checkpoint: `ckpt36`.
- GPU 배치는 방법론/결과 claim의 일부가 아니다. 현재 이 repository의 표준은 `CUDA_VISIBLE_DEVICES=4,5,6,7`이다.
- 일부 이전 experiment tag에는 `gpu0123`이 들어 있다. 이는 naming artifact로 보고, 현재 실행 표준으로 해석하지 않는다.
- metric 정의, 수식, 코드 수준 timing 세부사항은 `codex_output/methodology/lrnode_eval_metrics_definition_20260622.md`에 정리되어 있다.

## QRED20: 20 Hz query reduction

QRED20은 LIBERO `control_freq=20`을 유지하면서 `K`를 키워 비싼 full-Seer call을 줄이는 실험이다.

| Hz | K | SR | K=1 대비 보존율 | Full Seer Hz | LR-NODE Hz | Full-query 감소율 | Policy ms | Budget ms | LR-NODE ms | Jerk p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 82.5 | 100.0 | 20.00 | 0.00 | 0.0 | 70.388 | 50.000 | 0.000 | 0.079720 |
| 20 | 2 | 79.0 | 95.758 | 10.00 | 10.00 | 49.939 | 47.900 | 50.000 | 7.205 | 0.216848 |
| 20 | 3 | 82.5 | 100.0 | 6.67 | 13.33 | 66.579 | 36.843 | 50.000 | 6.965 | 0.259359 |
| 20 | 4 | 86.5 | 104.848 | 5.00 | 15.00 | 74.891 | 33.639 | 50.000 | 7.237 | 0.307808 |

### 해석

QRED20은 핵심 query-reduction claim을 강하게 뒷받침한다.

- `K=2`는 K=1 SR의 95.8%를 유지하면서 full-Seer call을 약 50% 줄인다.
- `K=3`은 full-Seer call을 약 66.6% 줄이면서 K=1 SR과 정확히 같은 성공률을 낸다.
- `K=4`는 full-Seer call을 약 74.9% 줄이고, 이 run에서는 K=1 SR을 넘는다.
- Full-Seer K=1은 명목상 20 Hz policy budget에 들어오지 못한다. (`70.388 ms > 50 ms`)
- LR-NODE skip mode는 20 Hz policy budget 안에 들어온다.
  - K=2: `47.900 ms`
  - K=3: `36.843 ms`
  - K=4: `33.639 ms`

주의할 점은 smoothness다.

- Jerk p95는 K가 커질수록 단조 증가한다. `0.0797 -> 0.2168 -> 0.2594 -> 0.3078`
- 따라서 K=4는 평균 SR과 query reduction은 가장 좋지만, K=3이 더 보수적이고 깔끔한 operating point다. K=3은 SR 82.5%를 유지하면서 K=4보다 낮은 jerk를 보인다.

## QRED20 task별 메모

K=1 대비 가장 크게 떨어진 task:

- Task 6은 K=2와 K=3에서 `90 -> 75 -> 65`로 하락하지만, K=4에서 회복된다.
- Task 8은 K=2에서 `60 -> 40`으로 하락한 뒤 K=4에서 `65`로 회복된다.
- Task 9는 K=2/K=3에서 `70 -> 55 -> 65`로 하락한 뒤 K=4에서 회복된다.

가장 크게 개선된 task:

- Task 2는 K=3/K=4에서 `85`에서 `100`으로 개선된다.
- Task 5는 K=2에서 `80`에서 `95`로, K=3에서 `90`으로 개선된다.
- Task 4는 K=3에서 `95`에서 `100`으로 개선된다.

따라서 K=4의 평균 SR 개선은 모든 task에서 균일하게 좋아진 결과가 아니다. 일부 task의 이득이 다른 task의 손실을 상쇄한 평균값으로 해석해야 한다.

## HZUP20Q: 부분 결과

HZUP20Q는 실제 LIBERO `control_freq`를 높이면서, expensive full-Seer query rate를 약 20 Hz로 유지하는 실험이다.

완료된 행:

| Hz | K | SR | Full Seer Hz | LR-NODE Hz | Full-query 감소율 | Policy ms | Budget ms | LR-NODE ms | Jerk p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 82.5 | 20.00 | 0.00 | 0.0 | 82.818 | 50.000 | 0.000 | 0.079720 |
| 40 | 2 | 84.5 | 20.00 | 20.00 | 49.961 | 45.061 | 25.000 | 6.594 | 0.166171 |
| 60 | 3 | 78.5 | 20.00 | 40.00 | 66.639 | 36.392 | 16.667 | 6.815 | 0.310004 |

### 부분 해석

40 Hz 행은 task success 관점에서 긍정적이다.

- SR이 82.5%에서 84.5%로 증가했다.
- Full-Seer query rate는 20 Hz로 유지된다.
- LR-NODE가 skipped 20 Hz action update를 담당한다.

하지만 평균 policy-step wall time 기준으로는 아직 real-time 40 Hz policy budget을 만족하지 못한다.

- 40 Hz budget은 25 ms다.
- 평균 policy step은 45.061 ms다.
- budget은 `1000 / control_hz`로 계산하므로 `1000 / 40 = 25 ms`다.
- budget 초과 비율은 `45.061 / 25.0 = 1.802`다.
- LR-NODE-only skipped step은 `6.594 ms`로 싸지만, 평균 policy step에는 두 step마다 한 번씩 들어가는 full-Seer call도 포함된다.

따라서 현재 올바른 claim은 다음이다.

- HZUP20Q `40:2`는 expensive full-Seer query rate를 20 Hz로 유지하면서 성공률을 보존하거나 개선한다.
- 하지만 기록된 평균 policy-step latency만으로 real-time 40 Hz 실행을 주장하기에는 아직 부족하다.
- HZUP20Q `60:3`도 expensive full-Seer query rate를 20 Hz로 유지하지만, SR이 78.5%로 하락하고 jerk p95가 0.310004까지 증가하므로 현재는 `40:2`보다 약한 조건이다.

## 현재 권장안

논문/발표 표에 넣을 QRED20 주 행:

- Main QRED20 row: `K=3`
  - SR: 82.5%
  - 보존율: 100.0%
  - Full-query 감소율: 66.6%
  - Policy step: 36.8 ms, 20 Hz budget 이내
- Aggressive QRED20 row: `K=4`
  - SR: 86.5%
  - 보존율: 104.8%
  - Full-query 감소율: 74.9%
  - Policy step: 33.6 ms, 20 Hz budget 이내
  - 주의: jerk p95가 가장 높다.

HZUP20Q에 대해서는 다음처럼 정리한다.

- 최종 결론은 `80:4`와 full-query upper-bound 행이 끝난 뒤 내린다.
- 현재 `40:2`는 success-rate preservation 관점에서는 긍정적이지만, real-time latency claim에는 충분하지 않다.
- 현재 `60:3`은 LR-NODE update frequency가 증가할수록 high-Hz 설정의 안정성 문제가 커질 수 있음을 보여준다.
