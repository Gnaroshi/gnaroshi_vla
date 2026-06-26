# 기존 Seer 병목 분석

작성일: 2026-06-22
대상: `seer_node3` 현재 코드와 LIBERO eval 결과
목적: LR-NODE가 해결하려는 기존 Seer/VLA 병목을 코드와 측정값 기준으로 명확히 분리

## 1. 결론 요약

기존 Seer의 병목은 하나가 아니다. 최소 네 종류로 나뉜다.

1. **Policy inference 병목**
   - 매 control step마다 full Seer forward를 호출한다.
   - baseline plain Seer 기준 full forward 평균은 약 68.8 ms, policy total 평균은 약 88.5 ms다.
   - 20Hz control budget은 50 ms이므로, policy total은 이미 20Hz real-time budget을 넘는다.

2. **Simulator/evaluation wall-clock 병목**
   - `env.step(action)` 평균은 약 279 ms다.
   - 이것은 policy 자체 병목이 아니라 LIBERO offscreen simulation/rendering 병목이다.
   - real robot deployment claim과는 분리해서 해석해야 한다.

3. **Redundant history recomputation 병목**
   - 매 step마다 최근 `sequence_length=7` 전체 window를 다시 구성하고, 두 camera stream을 모두 vision encoder에 다시 넣는다.
   - 즉 full forward 1회마다 image encoder 입력은 `S * 2 cameras = 14` frame equivalent다.
   - 이전 timestep의 vision feature/cache를 재사용하지 않는다.

4. **Obs-pred/image decoder 병목 후보**
   - eval에서도 `--obs_pred`가 켜져 있으면 image prediction branch가 계산된다.
   - action selection에는 action latent/action head만 필요하지만, 현재 forward는 obs prediction branch도 실행한다.
   - 현재 logging은 이 branch의 시간을 따로 분해하지 않으므로 정확한 ms 기여도는 미측정이다.

LR-NODE가 직접 줄이는 병목은 1번과 3번이다. 즉 full Seer query를 줄이고, skipped step에서 full history vision/transformer recomputation을 피한다. 반대로 LR-NODE는 2번 simulator 병목을 해결하지 않는다.

## 2. 측정에 사용한 주요 결과 파일

Plain Seer baseline:

```text
runs_lrnode_protocol_20260616/eval/baseline_sweep_sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040_20260618_132356
```

Scratch-node K=1 full Seer sweep:

```text
runs_lrnode_protocol_20260616/eval/lrnode_scratch_sweep_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_20260620_202559
```

QRED20:

```text
runs_lrnode_protocol_20260616/eval/lrnode_qred20_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_qred20_ckpt36_gpu0123
```

Baseline detailed example:

```text
baseline ckpt33 eval_latency_profile.json
```

## 3. Metric 정의

### 3.1 Full forward latency

코드 위치:

```text
utils/eval_utils_libero.py
```

측정 구간:

```python
t0 = time.perf_counter()
model_outputs = self.model(...)
full_ms = (time.perf_counter() - t0) * 1000.0
```

의미:

- Seer model forward 전체 시간
- vision encoder, perceiver, transformer, obs decoder, action head를 포함
- preprocessing과 env.step은 포함하지 않음

### 3.2 Policy total latency

측정 구간:

```python
policy_t0 = time.perf_counter()
...
total_policy_ms = (time.perf_counter() - policy_t0) * 1000.0
```

의미:

- image/text/state preprocessing
- CPU to GPU transfer
- queue/padding construction
- full Seer 또는 LR-NODE inference
- action post-processing
- video frame collection 일부

을 포함한다.

### 3.3 Env step latency

측정 구간:

```python
env_t0 = time.perf_counter()
obs, reward, done, info = env.step(action)
env_step_ms = (time.perf_counter() - env_t0) * 1000.0
```

의미:

- LIBERO simulator step
- offscreen rendering
- physics/simulation state update

정책 inference 시간과 분리해서 봐야 한다.

### 3.4 Real-time control budget

Control rate가 \(H\) Hz이면 policy step 하나의 budget은:

\[
T_{budget}^{ms}(H) = \frac{1000}{H}
\]

예:

- 20Hz: 50 ms
- 40Hz: 25 ms
- 60Hz: 16.667 ms
- 80Hz: 12.5 ms

## 4. Plain Seer baseline 병목 수치

대상:

```text
sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040
ckpt 30-39
control_freq 20Hz
LR-NODE disabled
```

집계:

| metric | mean | min | max | 해석 |
|---|---:|---:|---:|---|
| SR | 79.9% | 77.0% | 83.0% | ckpt별 성공률 |
| full forward | 68.827 ms | 67.444 | 69.465 | Seer model forward |
| policy total | 88.527 ms | 83.352 | 89.967 | preprocessing 포함 policy step |
| env.step | 279.390 ms | 268.331 | 290.104 | simulator/rendering |

20Hz 기준 budget은 50 ms다.

따라서:

\[
\text{full forward over budget} = \frac{68.827}{50} = 1.377
\]

\[
\text{policy total over budget} = \frac{88.527}{50} = 1.771
\]

즉 plain Seer는 20Hz real-time policy budget 기준으로 full model forward만으로도 budget을 초과한다.

## 5. Baseline ckpt33 latency distribution

대상:

```text
baseline ckpt33
SR = 83.0%
```

Latency profile:

| metric | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| full forward model ms | 69.414 | 68.841 | 73.419 | 74.188 | 81.085 |
| policy total ms | 89.686 | 89.219 | 94.025 | 94.405 | 101.230 |
| env.step ms | 277.437 | 260.005 | 414.570 | 428.260 | 430.646 |

해석:

- full forward는 평균 69.4 ms이고 p95가 74.2 ms다.
- policy total은 평균 89.7 ms이고 p95가 94.4 ms다.
- env.step은 평균 277.4 ms이고 p95가 428.3 ms로 매우 크다.
- 따라서 evaluation 전체 소요시간 병목은 env.step이고, policy real-time 병목은 Seer full forward다.

## 6. Scratch-node K=1 full Seer 기준

대상:

```text
sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053
K=1, skip disabled
```

집계:

| metric | mean | min | max |
|---|---:|---:|---:|
| SR | 78.95% | 75.5% | 82.5% |
| full forward | 68.402 ms | 67.834 | 68.809 |
| policy total | 87.840 ms | 86.907 | 88.413 |
| env.step | 269.378 ms | 261.792 | 276.326 |

이 값도 plain baseline과 같은 결론이다.

- full forward는 20Hz budget 50 ms를 넘는다.
- policy total은 더 크게 넘는다.
- K=1은 LR-NODE skip이 없으므로 사실상 full Seer path다.

## 7. LR-NODE skip이 어떤 병목을 줄였는가

QRED20 ckpt36 결과:

| K | SR | full forward ms | LR-NODE ms | policy total ms | env.step ms | full query reduction | jerk p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.5% | 60.649 | 0.000 | 70.388 | 277.810 | 0.000% | 0.079720 |
| 2 | 79.0% | 66.333 | 7.205 | 47.900 | 308.233 | 49.939% | 0.216848 |
| 3 | 82.5% | 64.220 | 6.965 | 36.843 | 300.683 | 66.579% | 0.259359 |
| 4 | 86.5% | 67.674 | 7.237 | 33.639 | 309.759 | 74.891% | 0.307808 |

해석:

- LR-NODE skip step은 약 7 ms 수준이다.
- K가 커질수록 full Seer call 수가 줄어 average policy total이 줄어든다.
- K=3은 SR을 유지하면서 policy total을 70.388 ms에서 36.843 ms로 줄였다.
- 그러나 jerk p95가 증가한다. 즉 query 병목을 줄이면 action smoothness tail이 악화될 수 있다.

중요:

- `env.step`은 K를 키워도 줄지 않는다.
- 오히려 episode 길이/성공/실패 구성에 따라 env.step 평균은 흔들린다.
- 따라서 LR-NODE는 simulator wall-clock 병목이 아니라 policy query 병목을 겨냥한다.

## 8. 코드 구조상 full Seer forward 내부 병목

현재 full forward는 내부 module별 timing을 따로 기록하지 않는다. 따라서 아래는 직접 ms로 분해된 결과가 아니라, 코드 구조상 확실한 병목 후보다.

### 8.1 두 camera stream의 frozen ViT forward

코드:

```python
image_primary_feature = self.vision_encoder.forward_encoder(image_primary.flatten(0, 1), mask_ratio=0.0)
image_wrist_feature = self.vision_encoder.forward_encoder(image_wrist.flatten(0, 1), mask_ratio=0.0)
```

특징:

- vision encoder는 frozen이지만 eval forward에는 매번 사용된다.
- `sequence_length=7`, camera 2개이므로 full forward 1회마다 `14` frame equivalent가 vision encoder에 들어간다.
- frozen은 gradient를 막을 뿐 inference compute를 없애지 않는다.

### 8.2 History window 전체 재계산

코드:

```python
image_primary = torch.cat(list(self.img_queue), dim=1)
image_wrist = torch.cat(list(self.gripper_queue), dim=1)
state = torch.cat(list(self.state_queue), dim=1)
```

그리고 full Seer forward에는 전체 window가 들어간다.

문제:

- timestep \(t\)에서 이미 계산했던 \(t-1, t-2, ...\) image feature를 cache하지 않는다.
- 매 step마다 sliding window 전체를 다시 vision encoder/perceiver/transformer에 넣는다.
- 따라서 control Hz를 올리면 compute가 선형적으로 증가한다.

### 8.3 Perceiver resampler

코드:

```python
image_primary_feature = self.perceiver_resampler(...)
image_wrist_feature = self.perceiver_resampler(...)
```

현재 설정:

```text
num_resampler_query = 6
camera = 2
sequence_length = 7
```

Per timestep visual tokens:

\[
6 \times 2 = 12
\]

Full window visual resampler tokens:

\[
7 \times 12 = 84
\]

### 8.4 Transformer token count

현재 eval args:

```text
sequence_length = 7
num_resampler_query = 6
num_obs_token_per_image = 9
action_pred_steps = 3
obs_pred = True
```

Per timestep non-prediction tokens:

\[
N_A = 1_{\text{text}} + 1_{\text{state}} + 6 \times 2_{\text{camera resampler}} + 1 \times 2_{\text{camera cls}}
= 16
\]

Prediction tokens:

\[
N_B = 9 \times 2_{\text{obs tokens}} + 3_{\text{action tokens}} = 21
\]

Total tokens per timestep:

\[
N = N_A + N_B = 37
\]

Total transformer input tokens per full forward:

\[
7 \times 37 = 259
\]

즉 매 control step마다 259-token causal transformer forward가 실행된다.

### 8.5 Obs prediction/image decoder branch

코드:

```python
if self.obs_pred:
    obs_pred_feature = transformer_output[:, :, pred_token_start_idx : pred_token_start_idx+self.NUM_OBS_TOKEN, :]
    ...
    image_decoder_output = self.image_decoder(image_decoder_input)
    image_pred = self.image_decoder_pred(image_pred_feature)
```

문제:

- eval action 생성에는 `action_latent_full -> action head`만 필요하다.
- 하지만 `--obs_pred`가 켜져 있으면 image decoder branch도 계산된다.
- 이 branch는 학습에는 필요하지만 eval fast path에서는 생략 가능할 수 있다.
- 현재 코드에는 action-only eval forward가 없다.

### 8.6 Text preprocessing 반복

코드:

```python
text_x = self.text_process_fn([goal])
```

이 코드는 step마다 호출된다. 이후 text queue가 비어 있을 때만 append하지만, tokenization/process 자체는 매 step 수행된다.

문제:

- task language는 episode 동안 변하지 않는다.
- text token은 episode reset 시 한 번만 계산해 cache할 수 있다.
- 현재 policy total과 full forward의 차이 약 19-20 ms 안에 이런 CPU preprocessing/transfer 비용이 포함된다.

### 8.7 Image preprocessing / CPU-GPU transfer

코드:

```python
image = Image.fromarray(image)
image_x = self.image_process_fn([image])
gripper = Image.fromarray(gripper)
gripper = self.image_process_fn([gripper])
...
image_x = image_x.to(device)
gripper = gripper.to(device)
state = state.to(device)
```

문제:

- 매 step 두 camera image를 PIL 변환하고 processor를 통과시킨다.
- CPU tensor를 GPU로 전송한다.
- video 저장이 켜져 있으면 frame copy/concat도 수행한다.
- 이 비용은 full forward latency에는 포함되지 않고 policy total latency에는 포함된다.

## 9. Parameter/memory 관점 병목

Plain baseline parameter snapshot:

| module | params | trainable |
|---|---:|---:|
| total | 331,019,416 | 67,688,462 |
| clip_model | 151,277,313 | 0 |
| vision_encoder | 111,907,840 | 0 |
| transformer_backbone | 42,587,904 | 42,587,904 |
| perceiver_resampler | 18,894,336 | 18,894,336 |
| image_decoder | 3,548,928 | 3,548,928 |
| action_decoder + heads | 112,327 | 112,327 |

해석:

- frozen module도 inference memory와 compute는 차지한다.
- CLIP과 ViT가 frozen이어도 runtime footprint가 크다.
- action head 자체는 매우 작다. 따라서 action head가 병목이 아니다.
- LR-NODE가 action head를 재사용하는 설계는 타당하다. 병목은 action decoding이 아니라 action latent를 만들기 위한 full Seer backbone이다.

## 10. 현재 로깅의 한계

현재 eval logging은 다음을 직접 기록한다.

- full Seer forward 전체 시간
- LR-NODE fast encoder 시간
- LR-NODE dynamics 시간
- LR-NODE action head 시간
- policy total 시간
- env.step 시간

그러나 full Seer 내부를 다음처럼 분해하지 않는다.

- image preprocessing
- CLIP/text processing
- vision encoder
- perceiver resampler
- transformer backbone
- obs prediction/image decoder
- action head

따라서 "full Seer 내부에서 vision encoder가 정확히 몇 ms, transformer가 몇 ms"라는 말은 현재 결과만으로는 할 수 없다.

정확한 내부 bottleneck profiling을 하려면 `models/seer_model.py` forward 안에 CUDA sync 포함 timers를 추가해야 한다.

추천 profiling keys:

```text
preprocess_image_ms
preprocess_text_ms
vision_encoder_primary_ms
vision_encoder_wrist_ms
perceiver_primary_ms
perceiver_wrist_ms
embedding_build_ms
transformer_ms
obs_decoder_ms
action_decoder_ms
```

## 11. LR-NODE가 해결하는 병목과 해결하지 않는 병목

| 병목 | 기존 Seer 상태 | LR-NODE 영향 |
|---|---|---|
| full Seer forward | 매 step 60-70ms 이상 | K>1에서 호출 횟수 감소 |
| policy total latency | baseline 83-90ms | QRED K=3에서 36.8ms |
| history window recomputation | 매 step 7-step window 재계산 | skipped step에서 cache latent update로 대체 |
| action head compute | 매우 작음 | 그대로 재사용 |
| env.step / simulator | 270-300ms 이상 | 해결하지 않음 |
| action smoothness | baseline jerk 낮음 | K 증가 시 jerk tail 증가 가능 |
| high-Hz strict real-time | full Seer로는 불가능 | LR-NODE skip은 빠르지만 full refresh가 남아 있어 별도 최적화 필요 |

## 12. 연구 framing

기존 Seer의 병목을 정확히 표현하면 다음과 같다.

> 기존 Seer는 매 control step마다 최근 history window 전체를 다시 vision encoder/perceiver/transformer에 통과시켜 action latent를 생성한다. 이 full forward는 20Hz control budget 50ms를 초과하며, control Hz를 높이면 expensive full-query rate도 함께 증가한다. LR-NODE는 full Seer를 매 step 호출하는 대신, 주기적으로 얻은 action latent를 cheap visual/proprio delta-conditioned latent dynamics로 갱신하여 full-query bottleneck을 줄인다.

발표/논문에서 피해야 할 표현:

> LR-NODE가 LIBERO simulation wall-clock 전체를 빠르게 만든다.

이 표현은 현재 결과로는 부정확하다. `env.step`이 policy보다 훨씬 크고, LR-NODE는 simulator 병목을 직접 줄이지 않는다.

정확한 표현:

> LR-NODE는 simulator가 아니라 policy query path의 expensive full Seer forward 병목을 줄인다.
