# LR-NODE 실험 원리와 근거 분석

작성일: 2026-06-22
대상 코드: `seer_node3` 현재 구현

## 1. 이 실험이 검증하려는 핵심 주장

이 실험은 "VLA/Seer 전체 forward를 매 control step마다 호출하지 않아도, action-relevant latent를 저비용 모듈로 갱신하면 일정 구간 동안 policy 성능을 유지할 수 있는가"를 검증한다.

즉 비교 대상은 다음 두 가지다.

1. Full Seer query:
   - 현재 observation을 Seer 전체 backbone에 넣는다.
   - action head 직전 latent `z_full`을 얻는다.
   - 기존 Seer action head로 action을 만든다.

2. LR-NODE skip query:
   - 이전 full forward에서 얻은 latent `z_cached`를 들고 있는다.
   - 현재 observation과 cached observation의 visual/proprio delta를 cheap encoder로 읽는다.
   - ControlledLatentNODE로 `z_cached -> z_next`를 갱신한다.
   - 동일한 기존 Seer action head에 `z_next`를 넣어 action을 만든다.

따라서 이 연구의 본질은 새 action head를 만드는 것이 아니라, 기존 action head가 이해하는 latent 공간 위에서 expensive backbone 호출을 일부 대체하는 것이다.

## 2. 코드상 근거

### 2.1 Seer가 실제 action head에 넣는 latent

`models/seer_model.py`에서 action head 입력 latent는 다음 slice다.

```python
action_latent_full = transformer_output[:, :, pred_token_start_idx+this_num_obs_token:pred_token_start_idx+this_num_obs_token+self.action_pred_steps, :]
arm_pred_action, gripper_pred_action = self.decode_action_from_latent(action_latent_full)
```

코드 주석상 shape은 다음과 같다.

```text
action_latent_full: [B, S, action_pred_steps, hidden_dim]
```

여기서 `action_latent_full`은 임의의 auxiliary feature가 아니라 기존 action decoder/head에 실제로 들어가는 transformer output이다. 그래서 LR-NODE가 예측하는 대상이 의미를 가진다.

### 2.2 기존 action head 재사용

`decode_action_from_latent(action_latent)`는 다음 모듈을 그대로 사용한다.

```python
action_pred_feature = self.action_decoder(action_latent)
arm_pred_action = self.arm_action_decoder(action_pred_feature)
gripper_pred_action = self.gripper_action_decoder(action_pred_feature)
```

LR-NODE skip path도 최종적으로 이 same decoder/head를 호출한다. 따라서 LR-NODE는 "새로운 policy"가 아니라 "기존 policy head가 해석 가능한 latent updater"다.

### 2.3 Visual/proprio delta가 control input 역할을 한다

`models/lrnode_modules.py`의 `FastVisualDeltaEncoder`는 다음 입력을 만든다.

```python
x = torch.cat([key_rgb, cur_rgb, cur_rgb - key_rgb], dim=-3)
```

그리고 proprio가 있으면 다음을 더한다.

```python
q_delta = q_cur - q_key
u_delta = u_delta + proprio_proj([q_key, q_cur, q_delta])
```

즉 LR-NODE의 control input은 단순 time index가 아니라 현재 관측 변화다.

수식으로 쓰면:

\[
u_t = E_\Delta(I_{key}, I_t, I_t - I_{key}, q_{key}, q_t, q_t-q_{key})
\]

여기서 `key`는 마지막 full Seer call 또는 마지막 LR-NODE update 후 cache된 기준 observation이다.

### 2.4 ControlledLatentNODE의 Euler update

`ControlledLatentNODE.forward()`는 다음 구조다.

```python
dz = dynamics([LN(z_prev), u_delta, dt_emb, age_emb])
gate = sigmoid(gate([u_delta, age_emb]) + gate_bias)
z_next = z_prev + gate * dt * dz
```

수식:

\[
\Delta z_t = f_\theta(\mathrm{LN}(z_{t-1}), u_t, \phi(\Delta t), \phi(r_t))
\]

\[
g_t = \sigma(g_\theta(u_t, \phi(r_t)) + b_g)
\]

여기서 \(r_t\)는 cache age, 즉 마지막 full refresh 이후 LR-NODE update가 몇 번 누적되었는지를 나타내는 값이다.

\[
z_t = z_{t-1} + g_t \Delta t \Delta z_t
\]

이 구현이 "ODE"라고 불리는 이유는 latent dynamics를

\[
\frac{dz}{dt} = f_\theta(z, u, t)
\]

로 두고, adaptive solver 없이 fixed Euler step으로 한 번 적분하기 때문이다.

MVP에서는 adaptive ODE solver를 쓰지 않는다. 따라서 실험 결과는 "복잡한 solver가 아니라 cheap fixed-step latent update만으로 얼마나 대체 가능한가"를 보여준다.

### 2.5 Skip step에서 실제로 full Seer를 생략한다

`utils/eval_utils_libero.py`의 조건은 다음이다.

```python
use_lrnode_latent_update
and lrnode_eval_skip_full_forward
and lrnode_cached_latent is not None
and timestep % lrnode_query_interval != 0
```

이 조건이 참이면 full model forward 대신 `_update_from_lrnode_cache()`가 호출된다.

그 안에서는:

1. cached image/state와 현재 image/state로 `u_delta` 계산
2. cached latent `z_prev`를 `z_next`로 갱신
3. `decode_action_from_latent(z_next)`로 action 생성
4. `z_next`, 현재 image/state를 다시 cache

즉 skipped step에서는 transformer/perceiver 포함 full Seer forward가 호출되지 않는다.

## 3. 왜 이 실험이 성립하는가

이 실험은 다음 가정 위에 있다.

### 3.1 Action latent는 짧은 시간 간격에서 국소적으로 연속적이다

로봇 제어에서 인접 step의 observation 변화는 보통 작다. 특히 LIBERO의 20Hz 또는 그 이상 control step에서는 한 step 사이에 scene, instruction, object relation이 완전히 바뀌지 않는다.

따라서 Seer가 만드는 action-relevant latent도 짧은 간격에서는 다음처럼 근사할 수 있다고 보는 것이다.

\[
z_{t+1}^{full} \approx z_t^{full} + \Delta z_t
\]

LR-NODE는 이 \(\Delta z_t\)를 full transformer 없이 예측한다.

### 3.2 변화량 예측에는 전체 VLA 재추론보다 작은 모델로 충분할 수 있다

Full Seer forward는 vision encoder, perceiver/resampler, transformer backbone, action decoder를 포함한다. 반면 LR-NODE skip step은 다음만 사용한다.

1. 64x64 diff-CNN
2. small MLP latent dynamics
3. 기존 action head

즉 task/language/long-context grounding은 주기적으로 full Seer가 담당하고, step-to-step correction은 LR-NODE가 담당한다는 분해다.

### 3.3 기존 action head의 latent manifold를 유지해야 한다

LR-NODE가 아무 latent나 만들면 기존 action head가 해석할 수 없다. 그래서 학습 때 teacher-student distillation을 쓴다.

현재 구현에서:

```python
lrnode_z_prev = action_latent_full[:, :-1]
lrnode_z_teacher_next = action_latent_full[:, 1:]
```

그리고 설정상:

```text
lrnode_detach_input_latent=1
lrnode_detach_teacher_latent=1
lrnode_freeze_action_head_for_lrnode=1
```

이 의미는 LR-NODE loss가 teacher latent 자체를 바꾸는 방향으로 흐르지 않게 하고, LR-NODE가 기존 action head가 이해하는 latent 좌표계에 맞추도록 만든다는 것이다.

## 4. QRED20 실험의 원리

QRED20은 `control_freq=20Hz`를 고정하고, full Seer query interval `K`만 바꾼다.

정의:

\[
H = 20
\]

\[
K = \texttt{lrnode\_query\_interval}
\]

\[
\text{full query Hz} = \frac{H}{K}
\]

\[
\text{LR-NODE update Hz} = H - \frac{H}{K}
\]

\[
\text{full query reduction} = 1 - \frac{N_{full}}{N_{policy}}
\]

여기서 `K=1`이면 모든 policy step에서 full Seer를 호출한다. 따라서 같은 `scratch_node ckpt36` 안에서는 `K=1`이 내부 기준 baseline이다.

QRED20이 답하는 질문:

> 20Hz control은 유지하면서 expensive full Seer call을 몇 % 줄여도 성능이 유지되는가?

현재 완료 결과:

| control Hz | K | SR | full query Hz | LR-NODE Hz | full query reduction | avg policy ms | budget ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 82.5% | 20.0 | 0.0 | 0.0% | 70.388 | 50.0 |
| 20 | 2 | 79.0% | 10.0 | 10.0 | 49.939% | 47.900 | 50.0 |
| 20 | 3 | 82.5% | 6.667 | 13.333 | 66.579% | 36.843 | 50.0 |
| 20 | 4 | 86.5% | 5.0 | 15.0 | 74.891% | 33.639 | 50.0 |

해석:

- `K=3`은 가장 깔끔한 evidence다. 같은 ckpt의 `K=1` 대비 SR이 동일하고, full Seer call은 66.6% 줄었다.
- `K=4`는 SR은 더 높지만 action jerk p95가 가장 커서, "무조건 더 좋다"보다 "공격적인 query reduction 조건에서도 성공률은 유지된다" 정도로 해석해야 한다.
- QRED20은 real-time wall-clock claim보다 query efficiency claim에 더 직접적이다.

## 5. HZUP20Q 실험의 원리

HZUP20Q는 실제 LIBERO `control_freq`를 높이되, full Seer query Hz를 약 20Hz로 유지한다.

코드상 `EVAL_CONTROL_HZ`는 다음처럼 실제 env 생성에 들어간다.

```python
env_args = {
    ...
    "control_freq": int(round(control_hz)),
    "horizon": env_horizon,
}
env = OffScreenRenderEnv(**env_args)
```

따라서 HZUP20Q는 단순 metadata 환산이 아니다. LIBERO env의 control rate를 직접 바꾼다.

설정 원리:

\[
H \in \{20, 40, 60, 80\}
\]

\[
K = H / 20
\]

그러면:

\[
\text{full query Hz} = H / K = 20
\]

\[
\text{LR-NODE update Hz} = H - 20
\]

예:

| condition | control Hz | K | full Seer Hz | LR-NODE Hz |
|---|---:|---:|---:|---:|
| 20:1 | 20 | 1 | 20 | 0 |
| 40:2 | 40 | 2 | 20 | 20 |
| 60:3 | 60 | 3 | 20 | 40 |
| 80:4 | 80 | 4 | 20 | 60 |

HZUP20Q가 답하는 질문:

> expensive full Seer는 기존 20Hz 수준으로만 부르면서, cheap LR-NODE update로 더 높은 control frequency를 채울 수 있는가?

현재 완료된 부분 결과:

| control Hz | K | SR | full query Hz | LR-NODE Hz | full query reduction | avg policy ms | step budget ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1 | 82.5% | 20.0 | 0.0 | 0.0% | 82.818 | 50.000 |
| 40 | 2 | 84.5% | 20.0 | 20.0 | 49.961% | 45.061 | 25.000 |
| 60 | 3 | 78.5% | 20.0 | 40.0 | 66.639% | 36.392 | 16.667 |

`80Hz,K=4`는 현재 실행 중이다.

해석:

- `40Hz,K=2`는 성공률 관점에서 baseline `20Hz,K=1`보다 낮지 않다.
- 그러나 strict real-time budget으로 보면 `40Hz`의 step budget은 \(1000/40 = 25\) ms이고, 현재 평균 policy latency는 약 45.1 ms라 wall-clock real-time 조건은 만족하지 못한다.
- 이 병목은 LR-NODE update가 느려서가 아니다. `40Hz,K=2`에서 LR-NODE-only 평균은 약 6.59 ms다. 평균 policy latency가 큰 이유는 매 2 step마다 full Seer refresh가 들어가기 때문이다.
- 따라서 HZUP20Q의 현재 주장은 "simulated high control frequency에서 full Seer query budget을 20Hz로 유지하면서 action을 채울 수 있다"이지, 아직 "전체 policy loop가 40/60Hz real-time wall-clock으로 돈다"는 주장은 아니다.

## 6. Real-time budget 계산 근거

Control frequency가 \(H\) Hz이면 한 control step에 허용되는 wall-clock 시간은:

\[
T_{budget}(H) = \frac{1}{H}\ \mathrm{sec}
\]

ms 단위:

\[
T_{budget}^{ms}(H) = \frac{1000}{H}
\]

예:

- 20Hz: \(1000/20 = 50\) ms
- 40Hz: \(1000/40 = 25\) ms
- 60Hz: \(1000/60 \approx 16.667\) ms
- 80Hz: \(1000/80 = 12.5\) ms

`policy_latency_over_budget`는:

\[
\text{policy latency over budget} =
\frac{\text{avg policy step latency ms}}{1000/H}
\]

이 값이 1보다 작으면 평균 policy 계산 시간이 해당 Hz budget 안에 들어온다. 1보다 크면 strict real-time average 기준으로 초과다.

주의할 점:

- `avg_policy_step_latency_ms`는 preprocessing + full/LR-NODE inference + action conversion을 포함한다.
- `avg_env_step_latency_ms`는 simulator `env.step(action)` 시간을 따로 잰 값이다.
- 따라서 "policy만 real-time인가"와 "simulator까지 포함한 closed-loop wall-clock이 real-time인가"는 별도 문제다.

## 7. 이 실험으로 주장할 수 있는 것과 없는 것

### 주장 가능

1. Same checkpoint 내부에서 `K=1` full Seer 기준 대비 LR-NODE skip이 full query를 줄인다.
2. QRED20에서 `K=3`은 성공률 82.5%를 유지하면서 full Seer 호출을 66.6% 줄였다.
3. LR-NODE skip step 자체는 full Seer보다 훨씬 싸다.
4. HZUP20Q는 LIBERO `control_freq`를 실제로 올리고, full Seer query Hz를 20Hz로 제한하는 실험이다.
5. `40Hz,K=2`에서는 full Seer 20Hz + LR-NODE 20Hz 구성으로 SR 84.5%가 나왔다.

### 아직 주장하면 안 되는 것

1. "전체 policy loop가 40/60/80Hz real-time wall-clock을 만족한다."
   - 현재 평균 policy latency가 40Hz budget 25ms를 초과한다.

2. "LR-NODE가 모든 상황에서 full Seer를 대체한다."
   - K가 커질수록 action jerk가 증가하고, 60Hz K=3에서는 SR이 78.5%로 하락했다.

3. "plain Seer baseline보다 ours가 절대적으로 우수하다."
   - 현재 핵심 비교는 `scratch_node ckpt36` 내부 `K=1` vs `K>1`이다.
   - plain `scratch.sh` baseline과는 다른 학습 run이므로, 절대 SR 비교는 별도 protocol이 필요하다.

## 8. 왜 action hold/interpolation과 다른가

Action hold는:

\[
a_t = a_{t-1}
\]

또는 단순 interpolation이다.

LR-NODE는:

\[
z_t = z_{t-1} + g_t \Delta t f_\theta(z_{t-1}, u_t, \Delta t, r_t)
\]

\[
a_t = H_{action}(z_t)
\]

이다.

차이는 `u_t`가 현재 visual/proprio delta라는 점이다. 즉 LR-NODE는 현재 observation 변화에 반응해서 latent를 바꾸고, 기존 action head를 통해 action을 만든다. 그래서 "state-reactive latent update"라고 부를 수 있다.

## 9. 실험 설계의 타당성 요약

이 실험이 가능한 이유는 다음 논리 사슬 때문이다.

1. Seer action head 직전 latent `z`를 코드에서 명확히 노출했다.
2. 기존 action head를 그대로 재사용하므로 LR-NODE의 출력 공간이 명확하다.
3. 인접 control step에서는 action-relevant latent가 급격히 불연속적으로 바뀌지 않는다는 국소 연속성 가정이 있다.
4. cheap visual/proprio delta encoder가 짧은 구간의 상태 변화 \(u_t\)를 제공한다.
5. ControlledLatentNODE가 \(z_{t-1}\)와 \(u_t\)로 \(z_t\)를 Euler update한다.
6. Eval에서 실제로 full Seer call을 건너뛰고 cached latent를 갱신한다.
7. QRED20은 같은 20Hz 환경에서 query reduction을 검증한다.
8. HZUP20Q는 실제 LIBERO control frequency를 올리면서 full Seer query rate를 20Hz로 고정하는 high-control-rate feasibility를 검증한다.

따라서 본 연구의 정확한 표현은 다음이 가장 안전하다.

> LR-NODE는 expensive VLA/Seer full forward를 매 control step마다 실행하지 않고, 주기적으로 얻은 action latent를 visual/proprio delta-conditioned latent dynamics로 갱신하여 기존 action head를 재사용하는 latent-reactive policy acceleration 방법이다. 현재 결과는 20Hz control에서 full query를 66.6% 줄여도 SR을 유지할 수 있음을 보였고, higher-control-rate 설정에서는 full Seer query budget을 20Hz로 유지한 채 LR-NODE가 intermediate actions를 채울 수 있음을 검증 중이다.
