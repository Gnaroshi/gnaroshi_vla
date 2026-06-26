# LR-NODE 학습시간 Weakness 분석 - 2026-06-23

## 결론

현재 LR-NODE 방법은 성능/효율 eval 관점에서는 의미가 있지만, 학습 비용이 약점이다.

핵심 원인은 LR-NODE module parameter 수가 아니라 `shifted_context` teacher target을 만들기 위해 학습 step마다 full Seer teacher forward를 한 번 더 수행한다는 점이다.

현재 구현은 self-KD 계열이다. 더 정확히는 다음과 같이 부르는 것이 맞다.

```text
Latent/action self-distillation from full-query Seer to a cheap latent updater
```

또는 한국어로:

```text
full-query Seer를 자기 teacher로 사용해 LR-NODE latent updater를 학습하는 자기지식증류
```

다만 일반적인 external teacher KD는 아니다. teacher는 별도 큰 모델이 아니라 같은 Seer architecture의 full-forward latent/action이다.

## 1. 학습시간 측정 결과

측정 기준:

- W&B summary의 `_runtime`
- W&B summary의 `step_time`
- checkpoint mtime 간격
- checkpoint mtime은 저장된 checkpoint 구간의 평균 epoch 간격으로 해석한다.

대상 run:

| Protocol | Run |
|---|---|
| baseline scratch | `sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040` |
| scratch_node | `sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053` |
| distill_node | `sd1_distill_node_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_20260620_202533` |

정량 결과:

| Protocol | Runtime | Runtime ratio | W&B step_time | step_time ratio | Avg ckpt gap | ckpt gap ratio | ckpt size |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline scratch | 19.30 h | 1.000x | 6.486 s | 1.000x | 28.11 min | 1.000x | 809.419 MB |
| scratch_node | 28.93 h | 1.499x | 9.709 s | 1.497x | 42.02 min | 1.495x | 815.084 MB |
| distill_node | 27.70 h | 1.436x | 8.363 s | 1.289x | 41.97 min | 1.493x | 5.736 MB |

해석:

- `scratch_node`는 baseline 대비 거의 정확히 1.5배 느리다.
- `distill_node`는 checkpoint size가 5.736 MB뿐인 adapter-only 학습인데도 전체 runtime이 baseline 대비 1.436배다.
- distill은 baseline checkpoint가 이미 있다고 가정하면 추가 비용이 27.70 h다.
- baseline scratch까지 포함한 end-to-end adapter pipeline은 `19.30 + 27.70 = 47.00 h`다.
- 따라서 distill pipeline의 total cost는 baseline scratch 1회 대비 `47.00 / 19.30 = 2.44x`다.

## 2. 왜 module이 작은데 학습이 느린가

LR-NODE module 자체는 작다.

| Checkpoint | Size | 의미 |
|---|---:|---|
| baseline ckpt | 809.419 MB | full Seer trainable checkpoint |
| scratch_node ckpt | 815.084 MB | full Seer + LR-NODE |
| distill_node ckpt | 5.736 MB | LR-NODE adapter-only |

즉 model parameter size overhead는 작다.

하지만 compute overhead는 크다. 이유는 `utils/train_utils.py`의 `shifted_context` teacher target 경로다.

학습 step에서 수행되는 forward:

```text
1. teacher_outputs_next = model(C_{t+1}) under torch.no_grad()
2. model_outputs = model(C_t) with lrnode_compute_loss=True
```

`teacher_outputs_next`는 no-grad이므로 activation/backward memory는 줄지만, forward compute는 여전히 수행한다.

Seer forward 내부에서 teacher forward가 계산하는 주요 블록:

```text
CLIP text encoding
state embedding
MAE vision encoder for primary/wrist
Perceiver resampler
causal transformer
obs prediction image decoder if obs_pred=True
action decoder
```

현재 `obs_pred=True`이기 때문에 teacher no-grad forward에서도 image decoder branch가 실행된다. 즉 LR-NODE teacher target 하나를 만들기 위해 cheap module만 도는 것이 아니라, full Seer inference path 대부분이 한 번 더 돈다.

그 다음 main forward에서는 다음이 추가된다.

```text
normal Seer supervised forward/backward
LR-NODE FastVisualDeltaEncoder
LR-NODE ControlledLatentNODE
existing action head decode for z_pred
latent/action/smooth distillation loss
LR-NODE backward
```

따라서 총 비용은 대략 다음 구조다.

```text
baseline step cost
  = full Seer forward/backward

scratch_node step cost
  = full Seer teacher forward(no-grad)
  + full Seer student forward/backward
  + LR-NODE forward/backward
```

실측상 이 구조가 약 1.5배 overhead로 나타났다.

## 3. scratch_node의 weakness

`scratch_node.sh`는 baseline과 같은 Seer supervised objective를 학습하면서 LR-NODE loss를 추가한다.

구체적으로:

```text
base loss = action loss + 0.1 * image loss
LR-NODE loss = 0.05 * latent MSE
             + 0.1 * action distill L1
             + 0.001 * smooth loss
```

현재 detach/freeze 설정:

```text
lrnode_detach_input_latent=1
lrnode_detach_teacher_latent=1
lrnode_freeze_action_head_for_lrnode=1
```

따라서 LR-NODE loss는 teacher Seer/action head 쪽으로 흐르지 않는다. Seer backbone/action head는 base supervised loss로만 학습된다.

weakness:

1. baseline 대비 학습시간이 약 1.5배다.
2. LR-NODE loss가 Seer를 직접 개선하지 않으므로, 추가 학습시간은 거의 LR-NODE adapter 학습 비용이다.
3. `shifted_context` target 때문에 매 step teacher full forward가 추가된다.
4. teacher forward가 no-grad이어도 full Seer inference path를 돈다.
5. 현재 teacher forward에서 `obs_pred=True` image decoder branch까지 계산된다.
6. K=1 full-forward 성능은 baseline scratch와 반드시 같을 필요가 없다. 같은 initialization parity를 맞췄더라도 optimizer trajectory에는 LR-NODE module/loss/logging/gradient clipping 등이 같이 존재한다. 현재는 “동일한 Seer baseline”이 아니라 “from-scratch Seer + LR-NODE trained run”이다.

따라서 scratch_node는 방법론 proof-of-concept에는 유효하지만, “학습 효율적”이라고 주장하기 어렵다.

## 4. distill_node의 weakness

`distill_node.sh`는 baseline ckpt33을 teacher/base로 로드하고, Seer/action head를 freeze한 뒤 LR-NODE module만 학습한다.

확인된 trainable 상태:

```text
num_lrnode_trainable_tensors = 30
num_non_lrnode_trainable_tensors = 0
```

loss 설정:

```text
loss_image = False
loss_action = False
base_total_loss_without_lrnode = 0
LR-NODE loss만 사용
```

이론적으로는 adapter-only라 훨씬 빨라야 한다. 하지만 실제로는 baseline scratch보다 느리다.

원인:

1. 매 step teacher target을 만들기 위해 frozen Seer full forward를 수행한다.
2. student input latent `z_prev`를 얻기 위해 같은 frozen Seer의 current context forward도 필요하다.
3. 따라서 adapter-only라도 full Seer forward가 최소 2회 들어간다.
4. Seer weight는 freeze되어 backward는 LR-NODE에만 걸리지만, forward compute가 병목이다.
5. 현재 40 epoch를 그대로 돌린다.
6. baseline scratch까지 포함하면 end-to-end cost가 2.44x baseline이다.

즉 distill_node의 현재 가장 큰 weakness는 다음 문장으로 정리된다.

```text
Adapter-only 학습이지만 teacher/student latent 생성을 위해 frozen Seer full forward를 반복하므로, trainable parameter 수 감소가 wall-clock 학습시간 감소로 이어지지 않는다.
```

## 5. self-KD인가?

정확한 답:

```text
Yes, but it is not external teacher KD. It is Seer self-distillation.
```

더 세부적으로:

| Protocol | Self-KD 유형 | Teacher | Student | Teacher 고정 여부 |
|---|---|---|---|---|
| scratch_node | online self-distillation | same training Seer full-forward | LR-NODE latent updater | 고정 아님. Seer가 base loss로 계속 변함 |
| distill_node | frozen self-teacher distillation | frozen baseline Seer ckpt33 full-forward | LR-NODE adapter | 고정 |

`scratch_node`는 online self-KD다. teacher가 같은 training run 내부의 Seer이고, base supervised loss로 계속 바뀐다.

`distill_node`는 frozen self-KD 또는 adapter distillation이다. teacher가 baseline Seer checkpoint로 고정되어 있다.

둘 다 dataset action label을 직접 맞추는 BC가 핵심이 아니다. 현재 `lrnode_bc_weight=0.0`이다. LR-NODE는 주로 teacher latent/action을 맞춘다.

## 6. 논문에서 weakness로 써야 할 부분

정직하게 써야 하는 weakness:

```text
The current training implementation requires an additional full Seer teacher forward to build shifted-context latent targets. As a result, scratch LR-NODE training is about 1.5x slower than baseline Seer training. The adapter-only distillation protocol still remains expensive because frozen Seer forward passes dominate wall-clock time, despite training only a small LR-NODE module.
```

한국어:

```text
현재 학습 구현은 shifted-context latent target을 만들기 위해 추가 full Seer teacher forward를 필요로 한다. 그 결과 scratch LR-NODE 학습은 baseline Seer 학습보다 약 1.5배 느리다. 또한 adapter-only distillation도 학습되는 파라미터는 작지만, frozen Seer forward가 wall-clock time을 지배하기 때문에 충분히 빠르지 않다.
```

## 7. 개선 방향

가장 우선순위가 높은 개선은 teacher forward 비용을 줄이는 것이다.

### A. Teacher latent/action cache

baseline 또는 current Seer로 `z_full`, teacher action을 offline cache한다.

효과:

- distill_node에서 teacher full forward를 없앨 수 있다.
- 학습 중에는 LR-NODE forward/backward만 남는다.

단점:

- scratch_node online self-KD에는 바로 적용하기 어렵다. teacher가 계속 변하기 때문이다.
- cache storage가 필요하다.

### B. distill_node epoch 수 감소

현재 40 epoch는 너무 길다.

권장:

```text
5, 10, 20 epoch adapter distill ablation
```

판단 기준:

```text
DISTILL-LOADPARITY 통과 후 QRED20 K=2/K=3 SR
```

### C. teacher forward에서 obs decoder 제거

teacher target은 action latent/action만 필요하다.

현재 Seer forward는 `obs_pred=True`이면 teacher no-grad에서도 image decoder를 계산한다. teacher-only path에서 image decoder를 skip할 수 있으면 비용을 줄일 수 있다.

주의:

- action latent 위치가 obs token 개수에 의존하므로 transformer input의 obs token 자체를 제거하면 안 된다.
- 다만 transformer output 이후 image decoder branch는 skip 가능하다.

### D. adjacent_sequence fallback

`adjacent_sequence`는 한 번의 forward 안에서 `z_full[:, t] -> z_full[:, t+1]`를 만들 수 있어 teacher extra forward가 없다.

장점:

- 학습시간 overhead가 크게 줄어든다.

단점:

- eval의 cached previous forward latent와 정확히 같은 semantics가 아니다.
- Seer-specific in-window target에 가까워져 방법론 일반성이 약해질 수 있다.

### E. teacher update interval

매 step teacher target을 만들지 않고 N step마다 만들거나, 일부 batch만 LR-NODE loss를 계산한다.

예:

```text
LRNODE_LOSS_EVERY_N_STEPS=2 or 4
```

장점:

- scratch_node overhead를 직접 줄인다.

단점:

- LR-NODE supervision density가 줄어 성능이 떨어질 수 있다.

## 8. 현재 연구 주장에 미치는 영향

현재 LR-NODE의 강점은 inference/query efficiency다.

이미 QRED20에서:

```text
20Hz K=3: SR 82.5%, full Seer query reduction 66.58%, policy latency 36.843 ms
```

이 결과는 유효하다.

하지만 training efficiency는 강점이 아니다. 현재 방법론의 약점으로 명확히 분리해야 한다.

정리:

```text
Inference-time expensive query reduction: strong
Training-time efficiency: current weakness
```
