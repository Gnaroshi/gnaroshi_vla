# ETRI 발표용 LR-NODE 요약

Archive note: 아래 Ours pre-protocol 결과는 2026-06-16에 archive로 이동했다.

```text
/home/mingyujung/private/seer/seer_node3/archived_experiment_results_20260616/pre_protocol_lrnode
```

## 비교 대상

- Baseline Seer:
  `/home/mingyujung/private/seer/seer_main/eval/sd1_libero_10_100pc_original_settings_20260304`
- Ours:
  `/home/mingyujung/private/seer/seer_node3/scratch_eval_lrnode/sd1_scratch_libero_10_converted_seer_lrnode_student_v2_lw05_aw01_g4_K2`

## 결과 비교

| 항목 | Baseline Seer | Ours |
|---|---:|---:|
| 유효 checkpoint | 30-37 | 32-39 |
| 평균 성공률 | 80.19% | 81.06% |
| 최고 성공률 | 86.0% at ckpt 37 | 83.0% at ckpt 35 |
| 공통 checkpoint 평균 | 80.17% at ckpt 32-37 | 81.08% at ckpt 32-37 |
| 공통 checkpoint 차이 | - | +0.92%p |

Checkpoint별 공통 구간 비교:

| ckpt | Baseline | Ours | 차이 |
|---:|---:|---:|---:|
| 32 | 80.0% | 79.5% | -0.5%p |
| 33 | 78.5% | 82.5% | +4.0%p |
| 34 | 75.5% | 80.5% | +5.0%p |
| 35 | 81.0% | 83.0% | +2.0%p |
| 36 | 80.0% | 80.0% | +0.0%p |
| 37 | 86.0% | 81.0% | -5.0%p |

주의: 현재 Ours 폴더는 이름에 `K2`가 들어가지만, 저장된 JSON 기준으로 `lrnode_eval_skip_full_forward=false`, `lrnode_update_calls=0`이다. 따라서 이 결과는 10Hz skip-forward 효율 평가가 아니라, LR-NODE 모듈이 포함된 모델의 full-forward 성능 sanity check로 해석해야 한다. 실제 효율성 비교는 `lrnode_eval_skip_full_forward=1`, `lrnode_query_interval=2/4` 조건으로 별도 평가가 필요하다.

## ETRI 발표 PPT 작성안

### 가제

LR-NODE: Latent-Reactive Neural ODE for Efficient Vision-Language Robot Policies

### Keyword

- Vision-Language Robot Policy
- Latent Dynamics
- Neural ODE
- Efficient Inference
- Temporal Distillation

### 논문에서 해결할 Problem 한줄

대형 vision-language-action 로봇 정책은 매 제어 step마다 전체 시각 인코더와 transformer를 다시 실행해야 하므로, 연속 제어 환경에서 계산 지연과 중복 연산이 크다.

### 논문에서 제안하는 Solution 한줄

전체 정책을 매번 실행하지 않고, full Seer가 만든 action-relevant latent를 가벼운 visual-delta encoder와 controlled latent ODE로 갱신한 뒤 기존 action head를 재사용한다.

### 핵심 방법론 이름 + 작동원리

핵심 방법론 이름: LR-NODE, Latent-Reactive Neural ODE

작동원리:

1. Full policy가 env step `t`에서 받는 context `C_t`와 env step `t+1`에서 받는 shifted context `C_{t+1}`에서 action-relevant latent를 teacher representation으로 추출한다.
2. 두 context의 경계에 해당하는 RGB/proprio 변화량을 tiny diff-CNN 기반 `FastVisualDeltaEncoder`로 인코딩해 control vector `u_delta`를 만든다.
3. `ControlledLatentNODE`가 fixed Euler update로 `z_next = z_prev + gate * dt * f(z_prev, u_delta, dt, age)`를 계산한다.
4. 학습 시 `z_pred(C_{t+1})`을 teacher latent `z(C_{t+1})`에 맞추는 latent distillation, teacher action에 맞추는 action distillation, smoothness loss를 함께 사용한다.
5. 평가 시에는 full Seer를 K step마다 한 번만 실행해 latent/cache를 갱신하고, skip step에서는 LR-NODE가 latent를 업데이트하여 동일한 기존 action head로 action을 출력한다.

## 현재 결과 해석

현재 full-forward sanity check에서는 LR-NODE가 포함된 모델이 baseline과 유사한 성공률을 보인다. 공통 checkpoint 평균은 Ours가 +0.92%p 높지만, 최고 checkpoint 기준으로는 baseline이 더 높다. 따라서 지금 단계의 핵심 주장은 "성공률 개선"이 아니라, "기존 Seer action head를 유지한 채 action-relevant latent dynamics를 학습해 full-forward를 줄일 수 있는 구조"이며, 효율성 주장은 skip-forward 평가가 완료된 뒤 latency와 success-rate trade-off로 제시하는 것이 맞다.

## Key Highlights 후보

현재 결과로 가장 안전하게 쓸 수 있는 숫자:

| 숫자 | 표현 | 근거 |
|---:|---|---|
| 96.5% | Best Performance Preservation Ratio | Ours best 83.0% / Baseline best 86.0% |
| +0.92%p | Average Success Rate on Shared Checkpoints | ckpt 32-37 평균: Ours 81.08%, Baseline 80.17% |
| 0 | Extra Demonstrations / External Motion Labels | 동일 데이터에서 full Seer latent를 teacher로 사용; 별도 demo, RAFT, CoTracker 미사용 |

발표에서 쓸 문장:

> LR-NODE preserves 96.5% of the best baseline success rate while matching average performance on shared checkpoints, without requiring additional demonstrations or external motion labels.

주의해서 써야 하는 숫자:

| 숫자 | 표현 | 상태 |
|---:|---|---|
| 50% | Target Full-Seer Query Reduction at K=2 | 아직 현재 `_K2` 결과에서는 skip이 꺼져 있어 실측 결과로 쓰면 안 됨 |
| 75% | Target Full-Seer Query Reduction at K=4 | skip-forward 평가 완료 후 latency/success trade-off와 함께 사용 가능 |

현재 ETRI 초안에는 `50% query reduction`을 핵심 실험 결과로 쓰지 않는 것이 안전하다. 대신 "designed to enable K-step full-query reduction" 또는 "evaluation in progress"로만 표현하는 것이 맞다.
