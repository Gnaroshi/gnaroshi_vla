# LR-NODE 현재 상태 handoff

이 문서는 현재 Seer + LR-NODE 구현 상태를 다른 Codex/ChatGPT 세션에 넘기기 위한 요약이다.
핵심은 “현재 느린 이유는 LR-NODE 모듈 자체가 아니라 shifted-context teacher target을 만들기 위해 Seer full forward를 한 번 더 돌리기 때문”이라는 점이다.

## 핵심 산출물

현재 상황을 한 번에 보여주는 raster 이미지:

```text
codex_output/figures/lrnode_current_state/lrnode_current_state_overview.png
```

이 이미지는 코드로 그린 SVG/vector가 아니라 `image_gen`으로 생성한 PNG raster 이미지다.

## 현재 Seer Action Latent 구조

파일:

```text
models/seer_model.py
```

Seer full forward는 action head에 들어가는 latent를 다음 shape으로 만든다.

```text
action_latent_full: [B, S, action_pred_steps, D]
```

현재 LIBERO 설정에서는 대략:

```text
B: batch size
S: sequence_length = 7
action_pred_steps = 3
D: hidden_dim = 384
```

코드 위치:

```text
models/seer_model.py:624-627
```

의미:

- `S`는 context/window 안의 timestep 위치다.
- `action_pred_steps`는 각 context timestep에서 예측하는 action horizon token 수다.
- eval에서는 full forward 후 보통 selected context step의 `[action_pred_steps, D]` latent를 cache한다.

## LR-NODE 구성

파일:

```text
models/lrnode_modules.py
```

구성:

1. `FastVisualDeltaEncoder`
   - 입력: key/current RGB pair, optional proprio pair
   - 처리: `[key_rgb, cur_rgb, cur_rgb - key_rgb]`를 64x64로 resize 후 작은 CNN
   - 출력: `u_delta`

2. `ControlledLatentNODE`
   - 입력: `z_prev`, `u_delta`, `dt`, `age`
   - fixed Euler update:

```text
dz = f(z_prev, u_delta, time_emb(dt), time_emb(age))
gate = sigmoid(g(u_delta, time_emb(age)) + gate_bias)
z_next = z_prev + gate * dt * dz
```

3. 기존 Seer action head 재사용
   - LR-NODE가 만든 `z_next`를 기존 `action_decoder`, `arm_action_decoder`, `gripper_action_decoder`에 넣어 action을 만든다.

중요:

- 기존 Seer action head는 제거하지 않았다.
- 기존 `use_node_action_head`와 별개의 새 실험 branch다.
- LR-NODE는 CLI flag로 제어되며 기본 off다.

## 학습 target mode

현재 구현은 두 가지 target mode를 가진다.

코드 위치:

```text
utils/arguments_utils.py:215-230
utils/train_utils.py:329-445
models/seer_model.py:665-773
```

### 1. `adjacent_sequence`

한 번의 Seer full forward에서 나온 latent sequence를 사용한다.

```text
Seer([o0, o1, ..., oS-1])
  -> z[0], z[1], ..., z[S-1]

LR-NODE:
  z[i] + delta(o_i, o_{i+1}) -> z[i+1]
```

코드:

```text
models/seer_model.py:755-773
```

장점:

- Seer full forward 1회
- dataset/dataloader 변경 없음
- 학습시간이 baseline 대비 크게 증가하지 않음
- 현재 코드로 바로 사용 가능

단점:

- eval에서 LR-NODE가 실제로 보는 cache latent와 완전히 같은 분포가 아니다.
- train의 `z_prev`는 같은 forward 내부의 이전 context position latent다.
- eval의 `z_prev`는 이전 policy call에서 cache된 selected/last latent다.

해석:

```text
adjacent_sequence = fast proxy / warmup / one-forward baseline
```

### 2. 현재 구현된 `shifted_context`

현재 context와 다음 shifted context를 모두 Seer로 forward한다.

```text
C_t     = [o0, ..., oS-1]
C_{t+1} = [o1, ..., oS]

메인 forward:
  Seer(C_t) -> z_prev

교사 forward:
  Seer(C_{t+1}) -> z_teacher_next

LR-NODE:
  z_prev + delta(oS-1, oS) -> z_teacher_next
```

코드:

```text
utils/train_utils.py:346-400
utils/train_utils.py:402-445
models/seer_model.py:665-710
```

장점:

- eval의 sliding context transition에 더 가까운 teacher target이다.

문제:

- 매 training iteration마다 Seer full forward를 두 번 돌린다.
- `torch.no_grad()` teacher forward도 compute는 그대로 든다.
- 현재 teacher forward는 action latent만 필요하지만 Seer forward 내부에서 image decoder까지 실행될 수 있다.
- 그래서 학습시간이 2.x배까지 증가할 수 있다.

해석:

```text
current shifted_context = correctness-oriented but slow ablation
```

## 사용자가 지적한 핵심 수정 방향

사용자 지적:

```text
shifted_context도 이전 Seer output z_{t-1}를 알고 있으면,
현재 step에서는 Seer(C_t) 한 번만 forward해서 z_t를 teacher로 쓰면 되는 것 아닌가?
```

정답:

맞다. eval과 같은 sequential view에서는 이전 step에서 이미:

```text
z_{t-1} = Seer(C_{t-1})
```

를 계산하고 cache하고 있다.

그러면 ideal cache-based shifted-context training은:

```text
t-1 단계:
  Seer(C_{t-1}) -> cache z_{t-1}

t 단계:
  Seer(C_t) -> z_t
  LR-NODE(z_{t-1}, delta(o_{t-1}, o_t)) -> z_hat_t
  loss(z_hat_t, stopgrad(z_t))
  cache z_t
```

이 방식은 per step full Seer forward 1회로 가능하다.

단, 현재 shuffled batch training loop에서는 “직전 iteration의 z”가 현재 sample의 실제 이전 timestep latent인지 보장되지 않는다.
따라서 cache-based shifted-context를 학습에 쓰려면 최소한 cache key alignment가 필요하다.

필요한 정보:

```text
episode_id
timestep or frame index
```

가능한 구현:

```text
cache[(episode_id, t-1)] -> z_prev
current sample = (episode_id, t)
Seer(C_t) -> z_t
LR-NODE(z_prev, delta) -> z_hat_t
cache[(episode_id, t)] = z_t.detach()
```

cache miss이면 해당 sample의 LR-NODE loss는 skip한다.

## 왜 dataloader 이야기가 나왔는가

“이전 Seer output을 들고 있으면 된다”는 말은 eval에서는 100% 맞다.

학습에서 문제가 되는 것은 previous latent 자체가 아니라:

```text
그 previous latent가 현재 sample의 바로 이전 timestep latent인지 확인해야 한다.
```

현재 일반 shuffled batch에서는 다음 상황이 생길 수 있다.

```text
k번째 반복:
  sample = episode A, timestep 120
  cache z_A120

iteration k+1:
  sample = episode F, timestep 37
  current z_F37

잘못된 사용:
  LR-NODE(z_A120) -> z_F37
```

그래서 dataset/dataloader를 바꾸자는 뜻이 아니라, cache가 맞는 pair인지 확인할 key가 필요하다는 뜻이다.

## 현재 parity 상태

문서:

```text
codex_output/lrnode_parity_audit_20260618.md
```

추가한 재현 script:

```text
scripts/debug/check_lrnode_parity.py
```

검증 결과:

```text
init common tensor diff: 0
base loss diff: 0.0
main output diff: arm/gripper/image/latent all 0.0
common grad diff: 0
common param update diff after AdamW step: 0
eval full-forward action/latent diff: 0.0
LR-NODE nonzero grad tensors: 28
```

중요 수정:

```text
utils/train_utils.py:_preserve_torch_rng()
```

이유:

`shifted_context` teacher forward가 `torch.no_grad()`여도 dropout RNG를 소비해서 main forward dropout mask를 바꿀 수 있었다.
현재는 teacher forward를 RNG preserve context 안에서 실행한다.

## 현재 script 의미

### `scratch.sh`

```text
순수 Seer baseline scratch 학습
LR-NODE 꺼짐
```

### `scratch_node.sh`

```text
Seer scratch 학습 + detached teacher-student LR-NODE 학습
현재 기본 target mode는 shifted_context
정확하지만 teacher full forward 때문에 느림
```

권장 변경:

```text
main fast protocol은 adjacent_sequence 또는 last-pair adjacent로 바꾸는 것이 합리적
shifted_context는 ablation으로 유지
```

### `scratch_node_joint.sh`

```text
coupled joint ablation 설정
lrnode_detach_input_latent=0
lrnode_freeze_action_head_for_lrnode=0
LR-NODE loss가 common Seer/action head에도 영향을 줄 수 있음
baseline parity 기대하면 안 됨
```

### `distill_node.sh`

```text
완료된 baseline checkpoint를 load
non-LR-NODE 동결
LR-NODE만 학습
baseline best checkpoint가 정해진 뒤 실행
```

`finetune_node.sh`는 compatibility wrapper이며 실제로는 `distill_node.sh`로 forwarding한다.

## 현재 논리적 결론

1. 현재 학습시간 2.x 증가는 LR-NODE module 때문이 아니다.
2. 직접 원인은 현재 `shifted_context` 구현이 Seer full forward를 iteration마다 2번 실행하기 때문이다.
3. dataset/dataloader 변경 없이 바로 가능한 one-forward 방식은 `adjacent_sequence`다.
4. 더 eval에 가까운 one-forward 방식은 `last-pair adjacent`다.
   - `z_prev = action_latent_full[:, -2]`
   - `z_teacher = action_latent_full[:, -1]`
5. 가장 이론적으로 정확한 one-forward shifted 방식은 cache-based shifted-context다.
   - 이전 policy/context latent cache를 사용
   - 현재 sample에서 Seer(C_t)만 forward
   - cache alignment key가 필요
6. 논문/발표 관점에서는 다음 구분이 명확해야 한다.

```text
adjacent_sequence:
  fast proxy, cheap, one-forward, train/eval mismatch 존재

last-pair adjacent:
  one-forward, adjacent 중 eval selected-step에 가장 가까움

현재 shifted_context:
  correct shifted teacher, but 2-forward slow ablation

cache 기반 shifted_context:
  ideal main target, one-forward 가능, cache alignment 구현 필요
```

## 다음 권장 작업

우선순위 1:

```text
scratch_node.sh 기본 target을 adjacent_sequence 또는 last-pair adjacent로 바꾼다.
```

우선순위 2:

```text
last-pair adjacent flag 추가:
--lrnode_adjacent_pair_mode all|last
```

예상 코드 위치:

```text
models/seer_model.py
```

현재:

```python
lrnode_z_prev = action_latent_full[:, :-1]
lrnode_z_teacher_next = action_latent_full[:, 1:]
```

last-pair:

```python
lrnode_z_prev = action_latent_full[:, -2]
lrnode_z_teacher_next = action_latent_full[:, -1]
```

우선순위 3:

```text
current shifted_context를 ablation script로 분리하거나 명시적으로 METHOD_TAG에 slow_shifted를 넣는다.
```

우선순위 4:

```text
cache-based shifted_context 설계:
episode_id/timestep key가 현재 dataset sample에서 제공 가능한지 확인
가능하면 cache miss skip 방식으로 구현
```
