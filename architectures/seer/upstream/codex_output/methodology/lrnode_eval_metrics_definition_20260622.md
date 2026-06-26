# LR-NODE 평가 지표 정의 - 2026-06-22

이 문서는 QRED20 / HZUP20Q 평가에서 사용하는 지표를 코드 수준 수식과 함께 정의한다.

## 1. 실시간 예산

환경의 제어 주파수가 \(H\) Hz이면, policy/action 한 step에 허용되는 명목상 wall-clock 예산은 다음과 같다.

\[
T_{\text{budget}}(H) = \frac{1}{H} \text{ seconds}
\]

밀리초 단위로 쓰면 다음과 같다.

\[
T_{\text{budget, ms}}(H) = \frac{1000}{H}
\]

예시:

| 제어 Hz | 한 step 실시간 예산 |
|---:|---:|
| 20 Hz | \(1000/20 = 50.0\) ms |
| 40 Hz | \(1000/40 = 25.0\) ms |
| 60 Hz | \(1000/60 = 16.667\) ms |
| 80 Hz | \(1000/80 = 12.5\) ms |

따라서 HZUP20Q `40:2` 행은 policy step당 25 ms 예산을 가진다. 실제 측정된 평균 policy latency가 약 45.1 ms였으므로:

\[
\text{policy/budget} = \frac{45.1}{25.0} \approx 1.80
\]

이 때문에 이 결과는 task 성공률을 보존하고 full-Seer query rate를 20 Hz로 유지하더라도, 엄격한 40 Hz real-time policy loop를 만족한다고 말할 수 없다.

코드 기준:

- `scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh`
  - `budget_ms = 1000.0 / hz`
  - `policy_latency_over_budget = policy_ms / budget_ms`

## 2. 제어 Hz / 환경 step 수

`EVAL_CONTROL_HZ`는 LIBERO의 `OffScreenRenderEnv(control_freq=...)`로 실제 전달된다. 단순 metadata가 아니다.

정의:

\[
H = \texttt{EVAL\_CONTROL\_HZ}
\]

\[
\texttt{env.control\_freq} = \text{round}(H)
\]

`EVAL_SCALE_MAX_STEPS_WITH_HZ=1`이면 20 Hz baseline과 같은 명목상 episode 시간을 유지하기 위해 최대 평가 step 수를 다음처럼 scaling한다.

\[
\texttt{eval\_max\_steps}(H)
= \texttt{base\_eval\_max\_steps} \cdot \frac{H}{\texttt{base\_control\_hz}}
\]

현재 기본값은 다음과 같다.

\[
\texttt{base\_eval\_max\_steps}=600,\quad \texttt{base\_control\_hz}=20
\]

예시:

| Hz | 최대 step 수 |
|---:|---:|
| 20 | 600 |
| 40 | 1200 |
| 60 | 1800 |
| 80 | 2400 |

코드 기준:

- `utils/eval_utils_libero.py`
  - `_eval_control_hz()`
  - `_scaled_step_count()`
  - `OffScreenRenderEnv(control_freq=int(round(control_hz)), horizon=env_horizon)`

## 3. Query interval과 유효 query Hz

다음을 둔다.

- \(H\): control/action 주파수, 단위 Hz
- \(K\): `lrnode_query_interval`

LR-NODE skip mode가 켜져 있을 때:

\[
f_{\text{full}} = \frac{H}{K}
\]

\[
f_{\text{lrnode}} = H - f_{\text{full}}
\]

skip mode가 꺼진 `K=1` full-query baseline에서는:

\[
f_{\text{full}} = H,\quad f_{\text{lrnode}} = 0
\]

예시:

| 행 | 제어 Hz | K | full-Seer query Hz | LR-NODE update Hz |
|---|---:|---:|---:|---:|
| QRED20 `20:1` | 20 | 1 | 20.00 | 0.00 |
| QRED20 `20:2` | 20 | 2 | 10.00 | 10.00 |
| QRED20 `20:3` | 20 | 3 | 6.67 | 13.33 |
| QRED20 `20:4` | 20 | 4 | 5.00 | 15.00 |
| HZUP20Q `40:2` | 40 | 2 | 20.00 | 20.00 |
| HZUP20Q `60:3` | 60 | 3 | 20.00 | 40.00 |
| HZUP20Q `80:4` | 80 | 4 | 20.00 | 60.00 |

코드 기준:

- `utils/eval_utils_libero.py`
  - `effective_full_query_hz = control_hz / query_interval if lrnode_eval_skip_full_forward else control_hz`
  - `effective_lrnode_update_hz = max(0.0, control_hz - effective_full_query_hz)`

## 4. Full-query 감소율

다음을 둔다.

- \(N\): policy/env step 수
- \(N_{\text{full}}\): full Seer forward 호출 수
- \(N_{\text{lrnode}}\): LR-NODE update 호출 수

full-query 감소율은 다음과 같다.

\[
R_{\text{full-query}} = 1 - \frac{N_{\text{full}}}{N}
\]

퍼센트로 쓰면 다음과 같다.

\[
R_{\text{full-query,pct}} = 100 \cdot R_{\text{full-query}}
\]

유효 query interval은 다음과 같다.

\[
K_{\text{effective}} = \frac{N}{N_{\text{full}}}
\]

episode가 일찍 종료될 수 있고 cache 초기화 때문에 full call이 강제로 들어갈 수 있으므로, 측정값은 \(1 - 1/K\)에 가깝지만 항상 정확히 같지는 않다.

QRED20 예시:

- K=4 측정 full-query 감소율: 74.891%
- 이상적 값: \(1 - 1/4 = 75\%\)

코드 기준:

- `utils/eval_utils_libero.py`
  - `full_query_reduction_ratio = 1.0 - (full_forward_calls / num_policy_steps)`
  - `effective_query_interval = num_policy_steps / full_forward_calls`

## 5. Latency 측정

모든 timing은 `time.perf_counter()`를 사용한다. CUDA timing block은 구현된 위치에서 model operation 전후에 `torch.cuda.synchronize()`를 호출하므로, 해당 측정에서는 GPU kernel이 동기화된다.

### 5.1 Full forward latency

full Seer forward를 실행한 step에서만 측정한다.

\[
T_{\text{full, avg}} =
\frac{\sum_i T_{\text{full}, i}}{N_{\text{full}}}
\]

저장되는 key:

- `avg_full_forward_latency_sec`
- sweep summary에서 ms로 변환:

\[
\texttt{avg\_full\_forward\_ms}
= 1000 \cdot \texttt{avg\_full\_forward\_latency\_sec}
\]

중요: 이 값은 model full-forward 시간만 의미한다. preprocessing과 `env.step()` 시간은 포함하지 않는다.

코드 기준:

- full-forward block에서 `self.model(...)` 시간을 `full_ms`로 잰다.
- `self.full_forward_latency_sum += full_ms / 1000.0`
- 평균은 `full_forward_latency_sum / full_forward_calls`

### 5.2 LR-NODE latency

cache를 사용해 LR-NODE로 건너뛴 step에서만 측정한다.

\[
T_{\text{lrnode, avg}} =
\frac{\sum_i T_{\text{lrnode}, i}}{N_{\text{lrnode}}}
\]

저장되는 key:

- `avg_lrnode_latency_sec`
- 변환값:

\[
\texttt{avg\_lrnode\_ms}
= 1000 \cdot \texttt{avg\_lrnode\_latency\_sec}
\]

이 timing은 cached LR-NODE update path 전체를 감싼다.

1. fast visual delta encoding
2. controlled latent update
3. 기존 action head decoding

`env.step()`은 포함하지 않는다. timed block 밖에 있는 모든 preprocessing operation도 포함하지 않는다.

코드 기준:

- LR-NODE branch에서 `_update_from_lrnode_cache(...)` 시간을 `lrnode_ms`로 잰다.
- `self.lrnode_latency_sum += lrnode_ms / 1000.0`

### 5.3 Fast encoder / NODE / action head 세부 latency

이 값들은 LR-NODE path 내부의 debug sub-timer다.

- `avg_fast_encoder_ms`: cheap image/proprio delta encoder 시간
- `avg_node_update_ms`: controlled latent update 시간
- `avg_action_head_ms`: updated latent에 기존 Seer action head를 적용하는 시간

JSON에는 초 단위로 저장된다.

\[
\texttt{avg\_fast\_encoder\_latency\_sec}
= \frac{\sum_i T_{\text{fast},i}}{N_{\text{lrnode}}} \div 1000
\]

sweep summary는 이를 다시 ms로 변환한다.

\[
\texttt{avg\_fast\_encoder\_ms}
= 1000 \cdot \texttt{avg\_fast\_encoder\_latency\_sec}
\]

`node_update`와 `action_head`도 같은 변환을 사용한다.

### 5.4 Policy step latency

real-time action loop 가능성을 판단할 때 가장 중요한 값이다.

각 env step에서 `policy_total_ms`는 `model.step(...)` 시작부터 action을 반환하기 직전까지의 시간으로 측정된다.

\[
T_{\text{policy},t}
= t_{\text{return action}} - t_{\text{start model.step}}
\]

포함하는 것:

- `model.step` 내부 image/text/state preprocessing
- full Seer forward 또는 LR-NODE update
- action sequence 선택 / ensembling logic
- action smoothness metric bookkeeping

포함하지 않는 것:

- `env.step(action)`
- episode 종료 후 video encoding

평균:

\[
T_{\text{policy, avg}}
= \frac{1}{N}\sum_{t=1}^{N} T_{\text{policy},t}
\]

저장되는 key:

- `avg_policy_step_latency_sec`
- 변환값:

\[
\texttt{avg\_policy\_step\_ms}
= 1000 \cdot \texttt{avg\_policy\_step\_latency\_sec}
\]

real-time 예산과의 비교:

\[
\texttt{policy\_latency\_over\_budget}
= \frac{\texttt{avg\_policy\_step\_ms}}{1000/H}
\]

이 값이 \(\le 1\)이면 평균 policy 계산 시간이 제어 주파수 \(H\)의 명목상 real-time 예산 안에 들어온다.

### 5.5 Env step latency

이 값은 simulator stepping 주변에서만 측정된다.

\[
T_{\text{env},t}
= t_{\text{after env.step}} - t_{\text{before env.step}}
\]

평균:

\[
T_{\text{env, avg}}
= \frac{1}{N}\sum_{t=1}^{N} T_{\text{env},t}
\]

저장되는 key:

- `avg_env_step_latency_sec`
- 변환값:

\[
\texttt{avg\_env\_step\_ms}
= 1000 \cdot \texttt{avg\_env\_step\_latency\_sec}
\]

중요: 현재 summary에서 사용하는 real-time policy budget은 policy compute만 비교한다. `policy + env`를 비교하지 않는다. simulator wall-clock throughput을 보고 싶다면 다음 값을 써야 한다.

\[
T_{\text{sim-loop, avg}} = T_{\text{policy, avg}} + T_{\text{env, avg}}
\]

현재 결과에서 `env.step()`은 offscreen rendering과 video collection 때문에 policy compute보다 훨씬 느리다. 따라서 `avg_env_step_ms`를 robot policy inference latency로 해석하면 안 된다.

## 6. 성공률과 성능 보존율

성공률:

\[
SR = \frac{\#\text{successful episodes}}{\#\text{evaluated episodes}}
\]

LIBERO_10은 10개 task와 task당 20개 episode를 사용하므로 full eval은 총 200 episode다.

같은 checkpoint, 같은 Hz에서의 보존율:

\[
\text{Preservation}(H,K)
= 100 \cdot \frac{SR(H,K)}{SR(H,K=1)}
\]

이 값은 같은 checkpoint와 같은 control Hz 기준에서만 정의한다.

QRED20 예시:

\[
\text{Preservation}(20,3)
= 100 \cdot \frac{82.5}{82.5}
= 100.0
\]

코드 기준:

- `same_ckpt_hz_preservation_pct = success_rate / ref * 100`
- `ref`는 같은 checkpoint와 같은 control Hz에서 `K=1` full-forward mode인 행이다.

## 7. Action smoothness 지표

env step \(t\)에서 실제 실행된 action을 \(a_t \in \mathbb{R}^7\)라고 둔다. 이 action은 arm motion 차원과 gripper 차원을 포함한다.

### 7.1 Action delta

\[
\Delta a_t = a_t - a_{t-1}
\]

\[
\text{action\_delta\_l2}(t) = \|\Delta a_t\|_2
\]

첫 step에는 이전 action이 없으므로 zero를 사용한다.

Episode 수준 지표:

\[
\text{avg\_action\_delta\_l2} = \text{mean}_t \|\Delta a_t\|_2
\]

\[
\text{p95\_action\_delta\_l2} = \text{percentile}_{95,t}\|\Delta a_t\|_2
\]

최종 summary:

\[
\text{action\_delta\_l2\_mean}
= \text{mean over episodes of avg\_action\_delta\_l2}
\]

\[
\text{action\_delta\_l2\_p95}
= \text{mean over episodes of p95\_action\_delta\_l2}
\]

중요: 최종 `p95`는 모든 step을 합친 global percentile이 아니다. episode별 p95 값의 평균이다.

### 7.2 Action jerk

현재 코드에서 “jerk”는 action의 discrete second difference를 뜻한다. 초 단위로 정규화된 물리적 jerk가 아니다.

\[
j_t = \Delta a_t - \Delta a_{t-1}
\]

\[
\text{action\_jerk\_l2}(t) = \|j_t\|_2
\]

첫 step에는 이전 delta가 없으므로 zero를 사용한다.

하위 성분:

\[
\text{arm\_action\_jerk}(t)=\|j_t[0:6]\|_2
\]

\[
\text{trans\_action\_jerk}(t)=\|j_t[0:3]\|_2
\]

\[
\text{rot\_action\_jerk}(t)=\|j_t[3:6]\|_2
\]

최종 summary는 action delta와 마찬가지로 episode-level mean / p95 값의 평균을 사용한다.

### 7.3 Gripper switch 비율

각 step에서:

\[
g_t =
\begin{cases}
0, & t=0 \\
1, & a_t[-1] \ne a_{t-1}[-1] \\
0, & a_t[-1] = a_{t-1}[-1]
\end{cases}
\]

Episode 값:

\[
\text{gripper\_switch\_rate}
= \text{mean}_t g_t
\]

최종 summary:

\[
\text{final gripper switch rate}
= \text{mean over episodes of episode gripper switch rate}
\]

## 8. 현재 QRED20 / HZUP20Q latency claim 해석

### QRED20

QRED20은 `control_hz=20`을 사용하므로:

\[
T_{\text{budget}} = 1000/20 = 50\text{ ms}
\]

측정값:

- K=1 policy step: 70.388 ms, 20 Hz 예산 초과
- K=2 policy step: 47.900 ms, 20 Hz 예산 이내
- K=3 policy step: 36.843 ms, 20 Hz 예산 이내
- K=4 policy step: 33.639 ms, 20 Hz 예산 이내

따라서 QRED20에서는 LR-NODE skip mode가 평균 policy 계산 시간을 명목상 20 Hz compute budget 안으로 낮춘다.

### HZUP20Q `40:2`

HZUP20Q `40:2`는 `control_hz=40`을 사용하므로:

\[
T_{\text{budget}} = 1000/40 = 25\text{ ms}
\]

측정값:

- avg policy step = 45.061 ms
- avg LR-NODE-only update = 6.594 ms
- avg full Seer forward = 62.524 ms

평균 policy step에는 env step 두 번마다 한 번 들어가는 비싼 full-Seer step이 포함된다. 따라서:

\[
\frac{45.061}{25.0} \approx 1.80
\]

이 행이 뒷받침하는 claim:

- control frequency를 40 Hz로 올렸다.
- full Seer query rate는 20 Hz로 유지했다.
- 성공률은 보존되거나 개선되었다.

하지만 이 행이 뒷받침하지 못하는 더 엄격한 claim:

- 모든 policy step이 real-time 40 Hz wall-clock budget 안에 들어온다.

그 더 엄격한 claim을 뒷받침하려면 full-forward latency를 낮추거나, asynchronous full-Seer refresh를 구현하거나, 실제 control loop에서 skipped LR-NODE step과 scheduled full-Seer step을 분리해 측정하는 metric이 필요하다.

## 9. 중요한 주의사항

1. `avg_policy_step_ms`는 full-forward step과 LR-NODE skipped step을 모두 포함한 평균이다.
   `K=2`에서는 대략 절반의 step이 비싼 full-Seer call이고 나머지 절반이 cheap LR-NODE update다.

2. `avg_lrnode_ms`는 skipped LR-NODE update step만 측정한다.
   따라서 `avg_policy_step_ms`보다 훨씬 낮을 수 있다.

3. `avg_env_step_ms`는 simulator/rendering 시간이지 robot inference 시간이 아니다.
   policy latency와 분리해서 보고해야 한다.

4. 현재 summary의 `policy_meets_budget`은 average 기준이지 worst-case 기준이 아니다.
   실제 로봇 배포에서는 scheduled full-Seer refresh step 때문에 p95/p99 또는 worst-case latency도 필요하다.

5. 현재 “jerk”는 action 단위의 second difference이며 시간 정규화된 물리적 jerk가 아니다.
   서로 다른 Hz를 비교하려면 시간 정규화 버전이 필요할 수 있다.

\[
j^{\text{time}}_t \approx \frac{a_t - 2a_{t-1} + a_{t-2}}{\Delta t^2}
\]

현재 코드는 이 \(\Delta t^2\) 정규화를 적용하지 않는다.
