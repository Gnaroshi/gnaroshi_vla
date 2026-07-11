# ChatGPT용 LR-NODE 현재 구현/실험 결과 보고서

작성일: 2026-06-22
저장소 경로: `$SEER_WORKSPACE_ROOT`
대상 연구: LR-NODE, Latent-Reactive Neural ODE for Seer/VLA efficient control

## 0. ChatGPT에게 먼저 알려야 할 핵심 요약

본 연구는 기존 Seer/VLA policy를 매 control step마다 full forward하지 않고, 주기적으로 얻은 action-relevant latent를 cheap visual/proprio delta encoder와 controlled latent ODE로 갱신한 뒤 기존 action head를 재사용하는 방법이다.

현재 가장 유효한 비교는 plain Seer baseline과 절대 SR을 비교하는 것이 아니라, 같은 `scratch_node ckpt36` 안에서 `K=1` full-query reference와 `K>1` LR-NODE skip-query 조건을 비교하는 것이다.

이유:

- `scratch_node`는 LR-NODE 포함 scratch training run이다.
- `K=1`에서는 LR-NODE skip이 발생하지 않고 매 step full Seer를 호출한다.
- 따라서 같은 checkpoint 내부에서 `K=1`은 "해당 모델의 full Seer baseline"이다.
- `K>1`은 같은 weight에서 full Seer 호출 일부를 LR-NODE update로 대체한다.

현재 가장 강한 결과:

- QRED20 `K=3`: SR 82.5%, `K=1` 대비 SR 보존 100.0%, full Seer call 66.6% 감소.
- QRED20 `K=4`: SR 86.5%, full Seer call 74.9% 감소. 단, action jerk p95가 가장 큼.
- HZUP20Q `40Hz,K=2`: 실제 LIBERO control_freq 40Hz, full Seer query rate 20Hz 유지, SR 84.5%. 단, 평균 policy latency 45.1ms로 strict 40Hz real-time budget 25ms는 초과.

## 1. 방법론 한 줄 설명

LR-NODE는 expensive Seer/VLA backbone을 매 step 실행하지 않고, cached action latent를 현재 visual/proprio 변화량으로 갱신하여 기존 action head가 이해할 수 있는 latent trajectory를 생성하는 latent-level reactive control module이다.

## 2. 코드상 구현 요약

### 2.1 Action latent 추출

`models/seer_model.py`에서 Seer action head에 실제로 들어가는 latent를 `action_latent_full`로 노출했다.

```text
action_latent_full: [B, S, action_pred_steps, hidden_dim]
```

이 latent는 기존 action decoder/head 직전의 transformer output이다.

### 2.2 LR-NODE 구성

`models/lrnode_modules.py`:

1. `FastVisualDeltaEncoder`
   - 입력: key/current RGB, optional proprio
   - 이미지 입력: `[key_rgb, cur_rgb, cur_rgb - key_rgb]`
   - 64x64 resize
   - 3-layer ConvNet + pooling + linear
   - proprio가 있으면 `[q_key, q_cur, q_cur-q_key]` MLP feature를 더함

2. `ControlledLatentNODE`
   - fixed Euler update
   - adaptive ODE solver 없음
   - 수식:

\[
\Delta z_t = f_\theta(\mathrm{LN}(z_{t-1}), u_t, \phi(\Delta t), \phi(r_t))
\]

\[
g_t = \sigma(g_\theta(u_t, \phi(r_t)) + b_g)
\]

\[
z_t = z_{t-1} + g_t \Delta t \Delta z_t
\]

여기서 \(r_t\)는 cache age, \(u_t\)는 visual/proprio delta feature다.

### 2.3 Eval skip path

`utils/eval_utils_libero.py`에서 다음 조건이면 full Seer forward를 생략한다.

```text
use_lrnode_latent_update
and lrnode_eval_skip_full_forward
and cached latent exists
and timestep % lrnode_query_interval != 0
```

skip step에서는:

1. cached observation과 current observation으로 `u_delta` 계산
2. cached latent `z_prev`를 `z_next`로 업데이트
3. 기존 `decode_action_from_latent(z_next)` 호출
4. `z_next`와 current observation을 다시 cache

## 3. 학습 protocol 정리

현재 유효하게 보고할 학습 run:

### 3.1 Plain Seer baseline

실행:

```text
sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040
```

특징:

- LR-NODE 없음
- `scratch.sh`
- checkpoint 30-39 평가 완료
- best SR: 83.0%
- best ckpt 후보: `33.pth`, `39.pth`

주의:

- 이 plain baseline은 별도 scratch training run이다.
- LR-NODE 내부 K-sweep의 직접 기준으로는 사용하지 않는다.
- 논문에서 plain Seer baseline으로 보고할 수는 있지만, LR-NODE query reduction의 공정 비교는 같은 checkpoint `K=1` vs `K>1`로 해야 한다.

### 3.2 Scratch-node LR-NODE

실행:

```text
sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053
```

사용 checkpoint:

```text
ckpt36
```

핵심 flags:

```text
use_lrnode_latent_update=1
lrnode_train_latent_distill=1
lrnode_train_protocol=joint
lrnode_detach_input_latent=1
lrnode_detach_teacher_latent=1
lrnode_freeze_action_head_for_lrnode=1
lrnode_latent_weight=0.05
lrnode_action_distill_weight=0.1
lrnode_smooth_weight=0.001
lrnode_gate_init_bias=-4.0
lrnode_use_post_layernorm=0
lrnode_multistep_train=0
```

해석:

- Seer는 scratch로 학습된다.
- LR-NODE branch는 teacher-student latent distillation 구조다.
- LR-NODE loss의 input latent와 target latent는 detach된다.
- LR-NODE branch에서 action head는 frozen으로 사용된다.
- 따라서 LR-NODE는 기존 action head가 해석 가능한 latent를 만들도록 학습된다.

## 4. 현재 완료된 주요 결과

### 4.1 Plain Seer baseline

Eval 조건:

- LIBERO-10
- 200 episodes / ckpt
- control_freq 20Hz
- LR-NODE disabled
- full Seer forward every step

결과:

| ckpt | SR |
|---:|---:|
| 30 | 78.0% |
| 31 | 79.5% |
| 32 | 77.5% |
| 33 | 83.0% |
| 34 | 79.0% |
| 35 | 77.0% |
| 36 | 81.0% |
| 37 | 80.0% |
| 38 | 81.0% |
| 39 | 83.0% |

Plain baseline 최선:

- `ckpt33`: 83.0%
- `ckpt39`: 83.0%

### 4.2 QRED20: 20Hz에서 full Seer query reduction

목적:

> 동일한 20Hz control에서 full Seer call 일부를 LR-NODE로 대체해도 성공률이 유지되는가?

실행 root:

```text
runs_lrnode_protocol_20260616/eval/lrnode_qred20_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_qred20_ckpt36_gpu0123
```

결과:

| control Hz | K | SR | full Seer Hz | LR-NODE Hz | full query reduction | avg policy ms | LR-NODE ms | jerk p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 82.5% | 20.0 | 0.0 | 0.0% | 70.388 | 0.000 | 0.079720 |
| 20 | 2 | 79.0% | 10.0 | 10.0 | 49.939% | 47.900 | 7.205 | 0.216848 |
| 20 | 3 | 82.5% | 6.667 | 13.333 | 66.579% | 36.843 | 6.965 | 0.259359 |
| 20 | 4 | 86.5% | 5.0 | 15.0 | 74.891% | 33.639 | 7.237 | 0.307808 |

주요 해석:

- `K=3`이 가장 보수적이고 강한 evidence:
  - `K=1` 대비 SR 동일
  - full Seer call 66.6% 감소
  - avg policy latency 36.8ms로 20Hz budget 50ms 내
- `K=4`는 SR과 query reduction은 가장 좋지만 jerk p95가 가장 크다.
- K가 커질수록 action smoothness tail이 악화된다.

### 4.3 HZUP20Q: full Seer 20Hz query budget 유지 + control Hz 상승

목적:

> LIBERO control_freq를 실제로 40/60/80Hz로 올리되, expensive full Seer query rate는 기존 20Hz 수준으로 유지할 수 있는가?

실행 root:

```text
runs_lrnode_protocol_20260616/eval/lrnode_hzup20q_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_hzup20q_ckpt36_gpu0123
```

주의:

- experiment tag는 `gpu0123`이지만, `experiment_config.env`에는 `CUDA_VISIBLE_DEVICES=4,5,6,7`로 기록되어 있다.
- 현재 `80Hz,K=4` 실행 중이다.

완료된 결과:

| control Hz | K | SR | full Seer Hz | LR-NODE Hz | full query reduction | avg policy ms | LR-NODE ms | jerk p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 82.5% | 20.0 | 0.0 | 0.0% | 82.818 | 0.000 | 0.079720 |
| 40 | 2 | 84.5% | 20.0 | 20.0 | 49.961% | 45.061 | 6.594 | 0.166171 |
| 60 | 3 | 78.5% | 20.0 | 40.0 | 66.639% | 36.392 | 6.815 | 0.310004 |

해석:

- `40Hz,K=2`는 성공률 관점에서 매우 좋다.
  - 실제 control_freq 40Hz
  - full Seer query rate 20Hz 유지
  - SR 84.5%
- 하지만 strict real-time 40Hz claim은 아직 불가하다.
  - 40Hz budget = 25ms
  - avg policy latency = 45.061ms
  - LR-NODE-only step은 약 6.6ms로 빠르지만 full Seer refresh가 평균 latency를 끌어올림
- `60Hz,K=3`은 full query budget은 유지하지만 SR이 78.5%로 떨어지고 jerk p95가 커진다.
- 따라서 현재 HZUP 결론은 `40Hz,K=2`는 promising, `60Hz,K=3`부터는 stability/smoothness 문제가 보인다는 것이다.

## 5. Real-time / latency 해석

Control Hz가 \(H\)이면 strict per-step wall-clock budget은:

\[
T_{budget}^{ms} = \frac{1000}{H}
\]

예:

- 20Hz: 50ms
- 40Hz: 25ms
- 60Hz: 16.667ms
- 80Hz: 12.5ms

현재 logging에서:

- `avg_policy_step_latency_sec`: preprocessing + policy path + action conversion 평균
- `avg_full_forward_latency_sec`: full Seer forward 평균
- `avg_lrnode_latency_sec`: LR-NODE skip update 평균
- `avg_env_step_latency_sec`: simulator `env.step(action)` 평균

따라서 논문/발표에서 latency를 말할 때는 다음을 구분해야 한다.

1. Query efficiency:
   - full Seer call을 얼마나 줄였는가?
   - QRED20이 직접 측정한다.

2. High control frequency feasibility:
   - 실제 LIBERO control_freq를 올려도 SR이 유지되는가?
   - HZUP20Q가 측정한다.

3. Strict real-time wall-clock:
   - 평균 policy latency가 \(1000/H\) ms보다 작은가?
   - 현재 40Hz에서도 평균 policy latency는 budget 초과다.

## 6. 현재 주장 가능한 문장

안전한 주장:

> LR-NODE는 same scratch-node checkpoint에서 full Seer query interval을 늘려도 20Hz LIBERO control 성능을 유지할 수 있다. 특히 QRED20 `K=3`에서 SR 82.5%를 유지하면서 full Seer call을 66.6% 줄였다.

추가로 가능한 주장:

> 실제 LIBERO control_freq를 40Hz로 올린 HZUP20Q `40:2`에서 full Seer query rate를 20Hz로 유지하면서 SR 84.5%를 얻었다. 이는 LR-NODE가 intermediate action step을 채우는 high-control-rate setting에서 유효할 가능성을 보여준다.

주의해서 써야 할 문장:

> 현재 구현은 아직 전체 policy loop가 40Hz/60Hz strict real-time wall-clock budget을 만족한다고 말할 수 없다. LR-NODE skip step은 빠르지만, 주기적인 full Seer refresh가 평균 latency를 크게 만든다.

## 7. 현재 주장하면 안 되는 것

1. Plain Seer baseline보다 LR-NODE가 절대적으로 성능이 좋다.
   - 현재 핵심 증거는 same checkpoint 내부 `K=1` vs `K>1`이다.

2. 60Hz/80Hz에서도 안정적으로 성능이 유지된다.
   - 60Hz K=3에서 SR 78.5%, jerk p95 0.310004로 악화가 보인다.
   - 80Hz K=4는 아직 실행 중이다.

3. 전체 policy loop가 real-time 40Hz 이상을 만족한다.
   - 평균 policy latency가 40Hz budget 25ms를 넘는다.

## 8. 남은 실험 / 다음 확인 사항

1. HZUP20Q `80Hz,K=4` 완료 결과 확인
2. HZUP20Q full-query upper-bound rows 확인
   - `40Hz,K=1`
   - `60Hz,K=1`
   - `80Hz,K=1`
3. Distill-node run 평가
   - baseline best ckpt를 teacher로 두고 LR-NODE만 distill한 구조가 scratch-node보다 더 공정한 adapter-style evidence가 될 수 있음
4. Smoothness/jerk를 task별로 분석
   - 평균 SR만 보면 K=4가 좋아 보이지만 jerk tail이 커짐
5. Full refresh latency 병목 완화
   - async refresh
   - lower-frequency full Seer scheduling
   - full step과 LR-NODE step을 분리한 latency reporting

## 9. ChatGPT에게 물어볼 만한 질문

1. 현재 QRED20 결과를 논문 claim으로 어떻게 framing하는 것이 가장 안전한가?
2. HZUP20Q에서 `40Hz,K=2`는 SR은 좋지만 real-time budget을 넘는데, 발표에서는 어떤 문장으로 표현해야 하는가?
3. `K=4`의 SR 상승과 jerk 악화를 어떻게 함께 해석해야 하는가?
4. Distill-node 결과가 나오면 scratch-node 결과와 어떤 식으로 비교해야 하는가?
5. Real robot setting에서 camera Hz와 policy Hz가 다를 때, LR-NODE의 claim을 어떻게 formulation하는 것이 좋은가?
