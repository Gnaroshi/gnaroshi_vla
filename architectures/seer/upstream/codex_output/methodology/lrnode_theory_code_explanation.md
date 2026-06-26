# LR-NODE 코드 수준 이론 설명

작성일: 2026-06-16 KST

> 2026-06-20 문서 상태: 이 문서는 gate/detach/action-head freeze/loss 구성 등 코드 단위 이론 설명은 여전히 유효하다. 다만 `shifted_context`를 main target으로 강하게 해석한 일부 내용은 최신 논의 전 기준이다. 현재 구현/실험 판단은 먼저 `codex_output/lrnode_current_state_handoff.md`와 `codex_output/README_LRNODE_DOCS.md`를 기준으로 본다. 최신 결론은 `current shifted_context = 정확하지만 느린 2-forward 구현`, `adjacent_sequence/last-pair = dataset 변경 없는 one-forward 대안`, `cache-based shifted_context = ideal one-forward target`이다.

이 문서는 기존 `ours_vs_baseline_pipeline_analysis.md`와 목적이 다르다. 기존 문서가 baseline과 ours의 입출력 pipeline과 결과 비교를 추적했다면, 이 문서는 현재 코드가 구현하는 LR-NODE 방법론을 코드 단위로 해석한다.

핵심 질문은 다음이다.

1. 각 코드가 왜 필요한가?
2. 그 코드가 어떤 이론적 역할을 담당하는가?
3. gate, detach, action head freeze, latent loss, action distillation이 각각 왜 들어갔는가?
4. scratch, finetune, joint, adapter가 방법론적으로 어떻게 다른가?

## 1. 한 줄 정의

LR-NODE는 Seer의 full transformer를 매 control step 실행하지 않기 위해, Seer가 만든 action-relevant latent를 상태 변수처럼 보고, cheap visual/proprio delta와 controlled latent dynamics로 다음 latent를 예측한 뒤, 기존 Seer action head를 그대로 재사용해 action을 생성하는 모듈이다.

수식으로 쓰면 baseline Seer는 매 step마다 다음을 계산한다.

```text
z_t = F_Seer(o_{1:t}, language)
a_t = H_action(z_t)
```

LR-NODE skip step에서는 full Seer `F_Seer`를 생략하고 다음을 계산한다.

```text
u_t = E_delta(o_key, o_t, q_key, q_t)
z_t_hat = z_key + g(u_t, age) * dt * f_theta(z_key, u_t, dt, age)
a_t_hat = H_action(z_t_hat)
```

여기서:

| 기호 | 코드 이름 | 의미 |
|---|---|---|
| `F_Seer` | `SeerAgent.forward` full path | CLIP, ViT, Perceiver, GPT2 transformer를 모두 거치는 비싼 forward |
| `H_action` | `decode_action_from_latent` | 기존 Seer action decoder/head |
| `z_t` | `action_latent_full` 또는 cached latent | action head에 들어가기 직전의 action-token latent |
| `E_delta` | `FastVisualDeltaEncoder` | key frame과 current frame의 싼 변화 인코더 |
| `f_theta` | `ControlledLatentNODE.dynamics` | latent velocity 또는 latent residual 방향 |
| `g` | `ControlledLatentNODE.gate` | update 크기를 제어하는 scalar gate |
| `dt` | `lrnode_dt`, eval에서는 `1.0` | fixed Euler step size |
| `age` | `lrnode_age`, eval cache age | full Seer refresh 이후 몇 번째 LR-NODE update인지 |

## 2. 왜 action latent를 상태 변수로 삼는가?

코드 위치:

- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:601)
- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:606)
- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:608)
- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:609)

Seer의 action은 transformer output 전체에서 바로 나오는 것이 아니라, action prediction token 위치의 hidden state를 action decoder가 읽어서 나온다.

```text
action_latent_full = transformer_output[:, :, action_token_slice, :]
arm_pred_action, gripper_pred_action = decode_action_from_latent(action_latent_full)
```

현재 shape는 다음이다.

```text
action_latent_full: [B, S, action_pred_steps, hidden_dim]
예: [B, 7, 3, D]
```

이 tensor가 중요한 이유는 다음과 같다.

1. 이미 language, visual history, proprio, causal attention 정보가 섞인 action-relevant representation이다.
2. action decoder가 실제로 읽는 입력이므로, 이 latent만 잘 맞추면 기존 action head를 재사용할 수 있다.
3. raw image나 full transformer token 전체를 예측하는 것보다 훨씬 작고 action에 직접 연결되어 있다.
4. skip eval에서는 `z_t`만 업데이트하면 full Seer transformer를 매번 호출하지 않아도 된다.

이론적으로는 Seer의 복잡한 policy를 두 부분으로 나눈다.

```text
pi(o_t) = H_action(F_Seer(o_t))
```

LR-NODE는 `F_Seer` 전체를 근사하지 않고, action head 직전의 latent transition만 근사한다.

```text
z_{t+1} ~= Phi_theta(z_t, delta o_t, delta q_t)
a_{t+1} ~= H_action(z_{t+1})
```

이 선택이 효율성의 핵심이다. full visual-language transformer를 작은 delta encoder와 MLP dynamics로 대체할 수 있기 때문이다.

## 3. 왜 기존 action head를 유지해야 하는가?

코드 위치:

- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:270)
- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:374)
- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:392)

기존 action head는 다음 세 모듈이다.

```text
action_decoder
arm_action_decoder
gripper_action_decoder
```

`decode_action_from_latent`는 latent를 받아 기존 head로 action을 만든다.

```text
action_pred_feature = action_decoder(action_latent)
arm_pred_action = arm_action_decoder(action_pred_feature)
gripper_pred_action = gripper_action_decoder(action_pred_feature)
```

LR-NODE가 기존 action head를 유지해야 하는 이유는 명확하다.

1. 연구 claim이 "새 action policy를 학습했다"가 아니라 "기존 Seer policy의 full query를 줄였다"이기 때문이다.
2. action head를 새로 학습하면 LR-NODE가 Seer latent space를 보존하는지 알 수 없다.
3. 기존 action head가 이해하는 latent manifold 위에 `z_pred`가 있어야 skip step에서도 action이 안정적이다.
4. adapter finetune에서는 baseline full-forward 성능을 보존해야 하므로 action head를 바꾸면 안 된다.

따라서 `decode_lrnode_action_from_latent(..., freeze_action_head=True)`가 있다.

```text
with temporarily_freeze_params(action_head_modules):
    decode_action_from_latent(action_latent)
```

주의할 점:

`temporarily_freeze_params`는 LR-NODE branch에서 action head parameter 업데이트를 막기 위한 장치다. action head를 계산 그래프에서 완전히 제거하는 것은 아니다. latent `z_pred` 쪽으로는 gradient가 흘러야 LR-NODE가 "action head가 원하는 latent"를 만들도록 학습된다.

이론적으로는 다음 최적화다.

```text
min_theta ||H_fixed(Phi_theta(z_t, u_t)) - H_fixed(z_{t+1})||
```

여기서 `H_fixed`는 고정된 action head다.

## 4. FastVisualDeltaEncoder가 왜 필요한가?

코드 위치:

- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:6)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:16)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:38)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:64)

LR-NODE skip step에서 full Seer를 생략하면, current image를 ViT와 transformer에 넣지 않는다. 그래도 로봇과 scene은 변한다. 따라서 latent update에는 "무엇이 변했는지"에 대한 cheap observation signal이 필요하다.

이 역할이 `FastVisualDeltaEncoder`다.

### 4.1 입력

```text
key_rgb: full Seer를 마지막으로 실행했거나 직전 cache에 저장된 RGB
cur_rgb: 현재 RGB
q_key: key proprio/state
q_cur: 현재 proprio/state
```

현재 eval에서는 primary camera와 wrist camera를 둘 다 넣는다.

```text
key_image_primary
key_image_wrist
cur_image_primary
cur_image_wrist
q_key
q_cur
```

### 4.2 왜 `[key, cur, cur - key]`를 concat하는가?

코드:

```text
x = torch.cat([key_rgb, cur_rgb, cur_rgb - key_rgb], dim=-3)
```

RGB가 3채널이므로 concat 결과는 9채널이다.

이론적 의미:

| 구성 | 역할 |
|---|---|
| `key_rgb` | 이전 상태의 기준 appearance |
| `cur_rgb` | 현재 상태의 appearance |
| `cur_rgb - key_rgb` | 픽셀 수준 motion/change cue |

단순히 차이만 쓰면 현재 물체의 절대 위치나 배경 context가 사라진다. 반대로 두 이미지만 concat하면 네트워크가 차이를 직접 학습해야 한다. `[key, cur, diff]`는 absolute context와 first-order visual change를 동시에 준다.

이것은 optical flow, RAFT, CoTracker 같은 비싼 motion model을 쓰지 않는 MVP 설계와 맞는다. 정확한 motion field를 얻는 대신, cheap CNN이 latent update에 필요한 coarse delta만 추출한다.

### 4.3 왜 64x64로 resize하는가?

코드:

```text
x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
```

이론적 의미:

1. LR-NODE는 full visual representation을 만들려는 모듈이 아니다.
2. 필요한 것은 scene change의 coarse signal이다.
3. 고해상도 처리는 latency를 증가시켜 efficiency claim을 약하게 만든다.
4. 64x64는 motion/change cue를 얻는 데 충분히 작고, conv 비용이 낮다.

### 4.4 왜 3-layer CNN인가?

코드:

```text
Conv2d(9, 32, stride=2)
Conv2d(32, 64, stride=2)
Conv2d(64, 128, stride=2)
AdaptiveAvgPool2d(1)
Linear(128, motion_dim)
```

이론적 의미:

1. stride=2 conv를 3번 쓰면 64x64가 대략 8x8 feature map으로 줄어든다.
2. global average pooling은 spatial change를 compact하게 요약한다.
3. `motion_dim=128`은 controlled dynamics에 넣기 충분히 작다.
4. 이 encoder는 full ViT에 비해 훨씬 작다.

즉 `FastVisualDeltaEncoder`는 "perception replacement"가 아니라 "latent dynamics control input generator"다.

### 4.5 왜 multi-camera feature를 평균내는가?

코드:

```text
image_features = [encode(primary), encode(wrist)]
u_delta = torch.stack(image_features, dim=0).mean(dim=0)
```

이론적 의미:

1. primary와 wrist는 같은 latent transition을 설명하는 두 관측이다.
2. concat하면 motion_dim이 camera 수에 따라 변하거나 projection layer가 추가된다.
3. 평균은 간단하고 shape-stable하다.
4. MVP에서는 camera별 복잡한 attention보다 latency와 안정성을 우선한다.

단점도 있다. 평균은 camera별 중요도를 학습하지 못한다. wrist가 중요한 조작 구간과 primary가 중요한 navigation-like 구간을 다르게 weighting하지 못한다. 향후 개선은 camera attention 또는 learned camera gate가 될 수 있다.

### 4.6 왜 proprio도 delta로 넣는가?

코드:

```text
q_delta = q_cur - q_key
u_delta = u_delta + proprio_proj([q_key, q_cur, q_delta])
```

이론적 의미:

로봇의 end-effector pose와 gripper state는 image보다 직접적으로 action-relevant하다. 같은 이미지 변화라도 로봇이 이미 목표에 가까운지, gripper가 열렸는지 닫혔는지에 따라 latent transition이 달라진다.

`[q_key, q_cur, q_delta]`를 모두 넣는 이유는 다음과 같다.

| 구성 | 역할 |
|---|---|
| `q_key` | 시작 proprio context |
| `q_cur` | 현재 proprio context |
| `q_delta` | robot motion/change |

delta만 넣으면 절대 pose 정보가 빠진다. key/current만 넣으면 변화 방향을 모델이 직접 빼야 한다. 셋을 같이 넣으면 state transition learning이 쉬워진다.

### 4.7 왜 `LayerNorm(motion_dim)`을 쓰는가?

코드:

```text
u_delta = self.out_norm(u_delta)
```

이론적 의미:

`u_delta`는 image CNN과 proprio MLP의 합이다. 두 branch의 scale이 다르면 dynamics MLP가 특정 branch에 과도하게 끌릴 수 있다. LayerNorm은 control input의 scale을 안정화해서 gate와 dynamics가 학습 초기에 폭주하지 않게 돕는다.

## 5. ControlledLatentNODE가 담당하는 이론

코드 위치:

- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:91)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:111)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:117)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:124)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:177)

`ControlledLatentNODE`는 이름 그대로 controlled neural ODE를 latent space에 적용한 것이다. 다만 MVP에서는 adaptive ODE solver를 쓰지 않고 fixed Euler 1-step만 쓴다.

연속 시간 관점:

```text
dz / dt = f_theta(z(t), u(t), dt, age)
```

MVP의 discrete Euler:

```text
z_next = z_prev + gate(u, age) * dt * f_theta(z_prev, u, dt, age)
```

현재 코드의 정확한 형태:

```text
dz = dynamics([LayerNorm(z_prev) + token_embedding, u_delta, emb(dt), emb(age)])
gate = sigmoid(gate_mlp([u_delta, emb(age)]) + gate_bias)
update = gate * dt * dz
z_next = z_prev + update
```

## 6. 왜 ODE 형식인가?

일반 MLP transition이라면 다음처럼 쓸 수 있다.

```text
z_next = MLP(z_prev, u_delta)
```

하지만 LR-NODE는 residual update를 쓴다.

```text
z_next = z_prev + delta_z
```

이론적 장점:

1. 인접 timestep latent는 대체로 큰 폭으로 바뀌지 않는다는 inductive bias를 준다.
2. skip step에서 "현재 latent를 조금 이동시킨다"는 해석이 가능하다.
3. `z_prev`를 그대로 유지하는 hold baseline과 비교가 쉽다.
4. gate를 작게 시작하면 초기 학습에서 full policy를 크게 망가뜨리지 않는다.
5. multi-step rollout로 확장하기 쉽다.

즉 이 모듈은 latent를 새로 생성하는 generator가 아니라, cached latent를 controlled residual로 업데이트하는 transition model이다.

## 7. gate는 왜 필요한가?

코드 위치:

- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:124)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:130)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:197)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:200)

gate는 update 크기를 조절하는 scalar다.

```text
gate = sigmoid(MLP_gate([u_delta, age_emb]) + gate_bias)
z_next = z_prev + gate * dt * dz
```

### 7.1 이론적 역할

gate는 "변화가 필요할 때만 latent를 움직이는" 안정화 장치다.

인접 frame 사이에는 다음 경우가 섞여 있다.

| 상황 | 바람직한 update |
|---|---|
| 로봇/물체 변화가 작음 | `z_next ~= z_prev` |
| 큰 end-effector 이동 | 적당한 latent shift |
| gripper open/close 근처 | action-relevant latent 변화 필요 |
| occlusion 또는 시각 노이즈 | update를 과하게 믿으면 위험 |

항상 `z_next = z_prev + dz`로 가면 작은 관측 변화에도 latent가 과도하게 흔들릴 수 있다. gate는 residual update의 step size를 data-dependent하게 만든다.

### 7.2 왜 gate input이 `z_prev`가 아니라 `[u_delta, age]`인가?

현재 gate는 다음을 입력으로 쓴다.

```text
gate_in = motion_dim + time_emb_dim
gate = gate_mlp([u_delta, age_emb])
```

이론적 의미:

1. gate는 "얼마나 움직일지"를 결정하는 역할이다.
2. 움직임의 크기와 신뢰도는 latent 자체보다 관측 변화량 `u_delta`와 cache age에 더 직접적이다.
3. `z_prev`까지 gate에 넣으면 gate가 latent content에 과적합해 특정 task/action mode별 shortcut을 배울 수 있다.
4. 현재 MVP에서는 gate를 단순한 trust controller로 둔다.

### 7.3 왜 `gate_bias=-4.0`인가?

현재 스크립트는 보통 `lrnode_gate_init_bias=-4.0`을 사용한다. sigmoid 값은 대략 다음이다.

```text
sigmoid(-4) ~= 0.018
```

즉 학습 초기에는 update가 거의 0에 가깝다.

```text
z_next ~= z_prev
```

이론적 의미:

1. 초기 LR-NODE는 hold policy에 가깝게 시작한다.
2. 무작위 MLP가 latent를 크게 망가뜨리는 것을 막는다.
3. latent distillation loss가 실제 필요한 방향으로 gate와 dynamics를 서서히 키운다.
4. skip eval에서 sudden jump를 줄이는 prior다.

gate bias가 너무 높으면 초기부터 random residual이 크게 들어가 학습이 불안정할 수 있다. 너무 낮으면 update가 거의 열리지 않아 학습 속도가 느릴 수 있다.

### 7.4 gate가 없으면 어떤 문제가 생기는가?

gate가 없으면 update는 다음이 된다.

```text
z_next = z_prev + dt * dz
```

이 경우 `dz` MLP가 update 크기와 방향을 모두 책임져야 한다. 작은 image delta에서 update를 거의 0으로 만들고, 큰 delta에서 update를 크게 만드는 scaling까지 한 네트워크가 처리해야 한다. gate는 이 scaling 문제를 분리한다.

간단히 말해:

```text
dynamics MLP: 어느 방향으로 latent를 움직일지
gate MLP: 얼마나 믿고 움직일지
```

## 8. 왜 `dt`와 `age`를 넣는가?

코드 위치:

- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:136)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:148)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:185)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:187)

`dt`는 Euler update의 step size다.

```text
z_next = z_prev + gate * dt * dz
```

현재 eval에서는 `dt=1.0`이다. adaptive solver는 없다. 이것은 MVP 요구사항과 일치한다.

`age`는 마지막 full Seer refresh 이후 몇 번 LR-NODE update가 누적되었는지를 뜻한다.

```text
age = 1, 2, 3, ...
```

왜 필요한가?

1. cache가 오래될수록 `z_prev`는 full Seer latent에서 멀어질 수 있다.
2. 같은 image delta라도 age가 1일 때와 age가 5일 때 신뢰도가 다르다.
3. gate가 age를 보고 update를 보수적으로 조절할 수 있다.
4. dynamics가 short-horizon과 long-horizon correction을 다르게 배울 수 있다.

time embedding은 다음 구조다.

```text
emb(x) = [x, sin(x * [1,2,4,8]), cos(x * [1,2,4,8])]
```

따라서 `time_emb_dim=9`다. scalar만 넣는 것보다 비선형 MLP가 step/age 차이를 구분하기 쉽다.

## 9. 왜 action token embedding이 필요한가?

코드 위치:

- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:113)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:166)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:194)

Seer는 `action_pred_steps=3`일 때 3개의 future action token latent를 만든다.

```text
z: [B, S, action_pred_steps, D]
```

각 token은 같은 timestep 안에서도 의미가 다르다.

```text
token 0: 가장 가까운 action
token 1: 다음 action
token 2: 그 다음 action
```

같은 dynamics MLP를 token-wise로 적용하면, token index 정보가 없을 때 모든 token을 동일한 의미로 처리할 위험이 있다. `action_token_embedding`은 token index를 latent에 더해 "이 latent가 몇 번째 future action token인지" 알려준다.

이론적 역할은 horizon embedding이다.

```text
z_dyn = LayerNorm(z_prev) + Emb(action_token_id)
```

이렇게 하면 같은 `u_delta`라도 token 0, 1, 2에 대해 다른 latent velocity를 만들 수 있다.

## 10. 왜 shape alignment가 필요한가?

코드 위치:

- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:154)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:177)

LR-NODE가 지원해야 하는 latent shape는 여러 가지다.

```text
[B, D]
[B, action_pred_steps, D]
[B, S_pair, action_pred_steps, D]
```

반면 `u_delta`는 보통 token dimension이 없다.

```text
[B, motion_dim]
[B, S_pair, motion_dim]
```

`_align_motion`은 `u_delta`를 action token dimension으로 broadcast한다.

```text
u_delta.unsqueeze(-2).expand(..., action_pred_steps, motion_dim)
```

이론적 의미:

같은 visual/proprio delta가 같은 timestep의 모든 action prediction token에 공통 control input으로 들어간다. 다만 token embedding이 있으므로 각 token의 update 방향은 달라질 수 있다.

## 11. post LayerNorm은 현재 어떻게 되어 있는가?

코드 위치:

- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:131)
- [models/lrnode_modules.py](/home/mingyujung/private/seer/seer_node3/models/lrnode_modules.py:204)
- [utils/arguments_utils.py](/home/mingyujung/private/seer/seer_node3/utils/arguments_utils.py:247)

현재 `ControlledLatentNODE`에는 update 뒤 LayerNorm이 있다.

```text
if self.use_post_layernorm:
    z_next = self.post_ln(z_next)
```

하지만 CLI default는:

```text
lrnode_use_post_layernorm = 0
```

현재 대표 실험 스크립트도 보통 0을 사용했다. 즉 실제 실험에서는 post update LayerNorm이 꺼져 있고, 대신 dynamics 입력에는 항상 `input_ln(z_prev)`가 적용된다.

이론적으로 post LayerNorm의 장점은 `z_next` scale drift를 막는 것이다. 단점은 Seer action head가 기대하는 latent norm 정보를 바꿀 수 있다는 점이다. 그래서 현재 구현은 conservative하게 post LN을 optional로 두었다.

중요:

초기 요구사항의 "Use LayerNorm on z_next"를 엄격히 따른다면 `--lrnode_use_post_layernorm 1`을 켜야 한다. 현재 코드에는 기능이 있지만 default는 꺼져 있다.

## 12. training에서 teacher-student가 어떻게 구성되는가?

코드 위치:

- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:315)
- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:690)
- [models/seer_model.py](/home/mingyujung/private/seer/seer_node3/models/seer_model.py:697)
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:413)

현재 기본 구현은 `lrnode_teacher_target_mode=shifted_context`이다. 핵심은 "같은 forward 안의 인접 latent"가 아니라, policy가 실제 평가 step에서 받는 정상 입력 context를 한 step shift한 뒤 teacher latent를 다시 probe한다는 점이다.

일반 VLA 관점에서 쓰면 다음과 같다.

```text
C_t       = policy가 env step t에서 정상적으로 받는 입력 context
C_{t+1}   = policy가 env step t+1에서 정상적으로 받는 입력 context
z_t^T     = Probe(TeacherPolicy(C_t))
z_{t+1}^T = Probe(TeacherPolicy(C_{t+1}))
u_t       = DeltaEncoder(obs_t, obs_{t+1}, q_t, q_{t+1})
z_hat_{t+1} = LR_NODE(stopgrad_or_not(z_t^T), u_t)
```

Seer 코드에서는 `C_t`가 길이 `sequence_length`의 window이고, `C_{t+1}`는 그 window를 한 칸 민 것이다.

```text
C_t     = images[:, 0:S], states[:, 0:S], text[:, 0:S]
C_{t+1} = images[:, 1:S+1], states[:, 1:S+1], text[:, 1:S+1]

selected_step = lrnode_context_selected_step  # default -1, steady-state eval cache 위치
z_prev         = action_latent(C_t)[:, selected_step]
z_teacher_next = action_latent(C_{t+1})[:, selected_step]
z_pred_next    = LR_NODE(z_prev, image[selected_step] -> image[selected_step+1])
```

이 방식이 더 정확한 이유는 eval skip에서 LR-NODE가 근사해야 하는 대상이 `same window 안의 t -> t+1 token`이 아니라, "다음 env step에서 full policy를 다시 호출했을 때의 action-interface latent"이기 때문이다.

기존 구현 방식은 `lrnode_teacher_target_mode=adjacent_sequence`로 남겨 두었다.

```text
z_prev = action_latent_full[:, :-1]
z_teacher_next = action_latent_full[:, 1:]
z_pred_next = LR_NODE(z_prev, image_t, image_{t+1}, q_t, q_{t+1})
```

이 모드는 legacy ablation이다. Seer window 내부 token transition을 학습하지만, 실제 eval의 "policy context shift" target과는 다를 수 있다.

### 12.1 detach input latent

코드:

```text
if lrnode_detach_input_latent:
    lrnode_z_prev = lrnode_z_prev.detach()
```

왜 필요한가?

순수 teacher-student adapter 구조에서는 `z_prev`도 teacher representation이다. `z_prev`가 detach되지 않으면 LR-NODE loss의 gradient가 다음 경로로 흐른다.

```text
LR-NODE loss
-> z_pred_next
-> z_prev
-> Seer transformer
```

그러면 Seer transformer latent 자체가 LR-NODE가 맞추기 쉬운 방향으로 바뀐다. 이것은 "기존 baseline에 LR-NODE만 붙인다"가 아니라 "Seer와 LR-NODE를 joint로 같이 바꾼다"가 된다.

따라서 adapter finetune claim에서는 다음이 맞다.

```text
z_prev = action_latent(C_t)[:, selected_step].detach()
z_teacher_next = action_latent(C_{t+1})[:, selected_step].detach()
```

현재 default는 `lrnode_detach_input_latent=1`, `lrnode_detach_teacher_latent=1`이다.

### 12.2 detach teacher latent

코드:

```text
if lrnode_detach_teacher_latent:
    lrnode_z_teacher_next = lrnode_z_teacher_next.detach()
```

왜 필요한가?

teacher target이 움직이면 LR-NODE가 고정된 target을 맞추는 distillation이 아니라, teacher와 student가 서로 영향을 주는 joint representation learning이 된다.

adapter finetune에서는 teacher는 pretrained Seer의 latent여야 한다. 그래서 target detach가 필수다.

### 12.3 action head freeze

코드:

```text
lrnode_arm_action, lrnode_gripper_action =
    decode_lrnode_action_from_latent(z_pred, freeze_action_head=True)
```

왜 필요한가?

action distillation loss가 다음이면:

```text
L_action = ||H(z_pred) - H(z_teacher)||_1
```

여기서 `H`가 같이 학습되면 loss를 줄이는 쉬운 방법은 두 가지가 된다.

1. LR-NODE가 좋은 latent를 만든다.
2. action head가 LR-NODE가 만든 이상한 latent를 받아들이도록 바뀐다.

우리가 원하는 것은 1번이다. 그래서 LR-NODE branch에서는 action head를 freeze한다.

## 13. loss들이 각각 담당하는 이론적 역할

코드 위치:

- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:413)
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:423)
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:437)
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:477)

전체 loss:

```text
L_total = L_base + L_lrnode

L_base =
    w_arm * L_arm_action
  + w_gripper * L_gripper_action
  + 0.1 * L_image

L_lrnode =
    lambda_z * L_latent
  + lambda_a * L_action_distill
  + lambda_s * L_smooth
  + lambda_bc * L_bc
```

### 13.1 Latent distillation loss

```text
L_latent = MSE(z_pred_next, stopgrad(z_teacher_next))
```

역할:

1. LR-NODE가 Seer full forward의 latent transition을 직접 모방한다.
2. action loss보다 더 dense한 supervision이다.
3. action head가 같은 action을 내더라도 latent manifold에서 벗어나는 것을 막는다.

이 loss가 없으면 LR-NODE는 action만 맞추는 latent를 만들 수 있고, multi-step rollout에서 latent drift가 커질 수 있다.

### 13.2 Action distillation loss

```text
L_action = L1(H_fixed(z_pred_next), stopgrad(H_fixed(z_teacher_next)))
```

역할:

1. latent MSE가 작아도 action-sensitive direction에서 오차가 클 수 있다.
2. action head output 기준으로 중요한 latent 차이를 보정한다.
3. 실제 policy behavior와 더 직접적으로 연결된다.

왜 ground-truth action이 아니라 teacher action인가?

LR-NODE의 목적은 새로운 policy imitation이 아니라 Seer full-forward behavior를 skip step에서 재현하는 것이다. 따라서 우선 teacher는 full Seer action이다.

### 13.3 Smooth loss

```text
L_smooth = MSE(z_pred_next - stopgrad(z_prev), 0)
```

역할:

1. latent update가 불필요하게 커지는 것을 막는다.
2. gate와 함께 residual update를 보수적으로 만든다.
3. skip rollout에서 action jerk 증가를 줄이는 regularizer다.

주의:

smooth loss가 너무 크면 LR-NODE가 hold baseline으로 수렴한다.

```text
z_pred_next ~= z_prev
```

그러면 query는 줄지만 scene 변화에 반응하지 못한다. 그래서 현재 weight는 작게 둔다.

### 13.4 BC loss

```text
L_bc = SmoothL1(action_pred_arm, gt_arm) + BCE(action_pred_gripper, gt_gripper)
```

현재 default는 `lrnode_bc_weight=0.0`이다.

역할:

1. full Seer teacher가 아니라 dataset action에도 LR-NODE action을 맞춘다.
2. teacher가 틀린 경우를 보완할 수 있다.

하지만 이 loss는 연구 claim을 흐릴 수 있다. BC를 크게 주면 LR-NODE가 "Seer skip approximator"가 아니라 별도 policy head 학습에 가까워진다. 그래서 MVP에서는 0이 안전하다.

### 13.5 hold baseline metric

코드:

```text
loss_lrnode_hold_latent = MSE(z_prev, z_teacher_next)
loss_lrnode_hold_action = L1(H(z_prev), H(z_teacher_next))
```

이것은 학습 loss라기보다 sanity metric이다.

의미:

```text
LR-NODE가 아무 update도 하지 않는 hold보다 나은가?
```

좋은 학습이라면 다음이 성립해야 한다.

```text
latent_mse_pred < latent_mse_hold
action_l1_pred < action_l1_hold
```

## 14. scratch, joint, adapter의 정확한 의미

코드 위치:

- [train.py](/home/mingyujung/private/seer/seer_node3/train.py:41)
- [train.py](/home/mingyujung/private/seer/seer_node3/train.py:45)
- [train.py](/home/mingyujung/private/seer/seer_node3/train.py:62)
- [train.py](/home/mingyujung/private/seer/seer_node3/train.py:77)
- [train.py](/home/mingyujung/private/seer/seer_node3/train.py:194)

먼저 용어를 분리해야 한다. 이 프로젝트에서 헷갈리기 쉬운 축은 세 개다.

| 축 | 선택지 | 의미 |
|---|---|---|
| 초기화/학습 방식 | `scratch` vs `finetune` | Seer를 처음부터 학습하는가, 기존 baseline ckpt에서 시작하는가 |
| 목적함수 구성 | `joint x` vs `joint o` | normal Seer base loss만 쓰는가, LR-NODE auxiliary loss도 같이 쓰는가 |
| LR-NODE loss gradient 경로 | detach/freeze on/off | LR-NODE loss가 Seer/action head까지 업데이트하는가 |

현재 정리한 script 기준:

| script | 학습 | LR-NODE loss coupling | 의미 |
|---|---|---:|---|
| `scratch.sh` | scratch | none | plain Seer baseline |
| `scratch_node.sh` | scratch | detached | Seer base loss + LR-NODE teacher-student distillation, LR-NODE loss는 Seer/action head로 직접 흐르지 않음 |
| `distill_node.sh` | checkpoint load | detached distill | baseline ckpt load, Seer/action head freeze, LR-NODE만 distill |
| `scratch_node_joint.sh` | scratch | coupled | Seer base loss + LR-NODE loss, LR-NODE loss가 Seer/action head에도 흐를 수 있음 |

중요한 정정:

`scratch_node.sh`와 `scratch_node_joint.sh`의 차이는 checkpoint load 여부가 아니다. 둘 다 scratch다. 차이는 LR-NODE loss의 gradient coupling이다.

`scratch_node.sh`:

```text
lrnode_detach_input_latent = 1
lrnode_detach_teacher_latent = 1
lrnode_freeze_action_head_for_lrnode = 1
```

따라서 gradient 경로는 다음처럼 분리된다.

```text
base Seer loss
  -> action head
  -> transformer/projectors/perceiver/image decoder 등 Seer trainable modules

LR-NODE loss
  -> lrnode_delta_encoder
  -> lrnode_dynamics
  X  Seer backbone
  X  action head parameters
```

즉 여기서 `joint`라는 말은 "한 training run 안에서 base Seer objective와 LR-NODE objective를 동시에 계산한다"는 뜻이지, "LR-NODE loss가 Seer까지 흘러간다"는 뜻이 아니다.

`scratch_node_joint.sh`:

```text
lrnode_detach_input_latent = 0
lrnode_detach_teacher_latent = 1
lrnode_freeze_action_head_for_lrnode = 0
```

따라서 LR-NODE loss도 다음 경로로 직접 흐를 수 있다.

```text
LR-NODE loss
  -> lrnode_delta_encoder / lrnode_dynamics
  -> z_prev -> Seer transformer/projectors
  -> action head parameters
```

### 14.1 scratch.sh

LR-NODE disabled baseline이다. Seer를 학습하지 않는다는 뜻이 아니다. 오히려 normal Seer를 scratch로 학습한다.

```text
use_lrnode_latent_update = 0
lrnode_train_latent_distill = 0
loss_action = 1
loss_image = 1
```

이 실험은 normal Seer base loss만으로 Seer backbone/action head를 학습하는 control이다.

### 14.2 scratch_node.sh

scratch + LR-NODE teacher-student detached protocol이다.

```text
lrnode_train_protocol = joint
lrnode_teacher_target_mode = shifted_context
lrnode_context_selected_step = -1
lrnode_detach_input_latent = 1
lrnode_freeze_action_head_for_lrnode = 1
loss_action = 1
loss_image = 1
```

이론적 의미:

1. Seer는 base loss로 scratch 학습된다.
2. LR-NODE는 shifted-context teacher latent/action을 distill한다.
3. LR-NODE loss는 Seer/action head를 직접 바꾸지 않는다.
4. 이름에 `_node`가 붙는 이유는 LR-NODE module과 LR-NODE distillation loss가 실제로 존재하기 때문이다.

### 14.3 distill_node.sh

frozen-baseline distill/adapter protocol이다. 현재 실험 의미는 Seer fine-tuning이 아니라 LR-NODE distillation이다.

```text
lrnode_train_protocol = adapter
lrnode_freeze_seer_for_adapter = 1
lrnode_assert_only_lrnode_trainable = 1
lrnode_teacher_target_mode = shifted_context
lrnode_context_selected_step = -1
```

`_apply_lrnode_train_protocol`은 모든 parameter를 freeze한 뒤, 이름이 다음으로 시작하는 parameter만 trainable로 되돌린다.

```text
lrnode_delta_encoder.
lrnode_dynamics.
```

이론적 의미:

1. pretrained Seer policy는 그대로 둔다.
2. action head도 그대로 둔다.
3. LR-NODE만 Seer latent transition을 배우게 한다.
4. 이 protocol에서 K=1 full-forward는 baseline과 같거나 매우 가까워야 한다.

따라서 논문 claim "기존 Seer baseline에 LR-NODE만 붙여 full query를 줄인다"는 이 protocol로 검증해야 한다.

### 14.4 scratch_node_joint.sh

scratch + LR-NODE coupled joint protocol이다.

```text
lrnode_train_protocol = joint
lrnode_teacher_target_mode = shifted_context
lrnode_context_selected_step = -1
lrnode_detach_input_latent = 0
lrnode_freeze_action_head_for_lrnode = 0
loss_action = 1
loss_image = 1
lrnode_train_latent_distill = 1
lrnode_detach_teacher_latent = 1
```

이론적 의미:

1. Seer의 normal BC/image objective가 Seer backbone/action head를 학습한다.
2. LR-NODE objective도 `z_prev`와 action head 경로를 통해 Seer/action head에 영향을 줄 수 있다.
3. 이 script는 기존 baseline ckpt를 보존하는 adapter 실험이 아니라, 별도의 scratch-trained Seer+LR-NODE run이다.
4. 따라서 기존 Seer original baseline과 K=1 full-forward 결과가 달라질 수 있다. 이유는 별도 scratch 학습 ckpt이면서 LR-NODE loss도 원본 policy representation에 결합될 수 있기 때문이다.
5. claim은 "LR-NODE auxiliary objective를 policy 학습에 결합한 scratch-trained variant"가 된다.

즉 joint는 방법론 확장으로는 가능하지만, "기존 baseline에 모듈만 붙였다"는 주장에는 맞지 않는다.

## 15. eval skip mode는 어떻게 작동하는가?

코드 위치:

- [utils/eval_utils_libero.py](/home/mingyujung/private/seer/seer_node3/utils/eval_utils_libero.py:236)
- [utils/eval_utils_libero.py](/home/mingyujung/private/seer/seer_node3/utils/eval_utils_libero.py:244)
- [utils/eval_utils_libero.py](/home/mingyujung/private/seer/seer_node3/utils/eval_utils_libero.py:255)
- [utils/eval_utils_libero.py](/home/mingyujung/private/seer/seer_node3/utils/eval_utils_libero.py:576)
- [utils/eval_utils_libero.py](/home/mingyujung/private/seer/seer_node3/utils/eval_utils_libero.py:662)

eval loop는 매 timestep마다 둘 중 하나를 선택한다.

### 15.1 full refresh step

조건:

```text
timestep % K == 0
or cache empty
```

실행:

```text
model(..., return_action_latent=True)
cache action_latent[:, selected_step]
cache current image/state
```

이론적 의미:

full Seer가 latent를 다시 anchor한다. skip rollout이 오래 누적되어 drift되는 것을 막는다.

### 15.2 LR-NODE skip step

조건:

```text
use_lrnode_latent_update = 1
lrnode_eval_skip_full_forward = 1
cache exists
timestep % K != 0
```

실행:

```text
u_delta = lrnode_encode_delta(cache_image, current_image, cache_state, current_state)
z_next = lrnode_apply_dynamics(cached_z, u_delta, dt=1.0, age=cache_age+1)
action = decode_action_from_latent(z_next)
cache = z_next, current_image, current_state
```

이론적 의미:

Full Seer query는 `1/K` 비율로만 실행된다. 나머지 step은 cheap latent dynamics와 기존 action head만 사용한다.

대략적 full query 감소율:

```text
reduction ~= 1 - 1/K
```

예:

```text
K=2 -> 약 50%
K=3 -> 약 66.7%
K=4 -> 약 75%
```

실제 code는 episode 길이와 cache empty step 때문에 정확히 약간 다를 수 있다.

## 16. 왜 K=1 sanity가 중요한가?

K=1이면 다음 조건 때문에 skip step이 없다.

```text
timestep % 1 == 0
```

즉 매 step full Seer를 실행한다.

adapter finetune protocol에서 K=1 결과가 baseline과 달라지면 이상하다. 왜냐하면:

1. baseline ckpt를 로드한다.
2. Seer/action head는 freeze된다.
3. K=1에서는 LR-NODE dynamics가 action 생성에 쓰이지 않는다.
4. 따라서 full-forward action은 baseline과 같아야 한다.

K=1이 다르면 가능한 원인은 다음이다.

| 원인 | 의미 |
|---|---|
| checkpoint가 baseline과 다름 | fair adapter 비교가 아님 |
| action head가 학습됨 | adapter freeze 실패 |
| eval preprocessing이 다름 | 비교 조건 불일치 |
| LR-NODE full path가 action에 개입 | 구현 bug |
| seed/task split이 다름 | evaluation mismatch |

반대로 scratch joint protocol에서는 K=1이 baseline과 달라질 수 있다. joint 학습 중 Seer 본체가 바뀌기 때문이다.

## 17. 현재 코드에서 disabled-by-default가 왜 중요한가?

코드 위치:

- [utils/arguments_utils.py](/home/mingyujung/private/seer/seer_node3/utils/arguments_utils.py:211)
- [utils/arguments_utils.py](/home/mingyujung/private/seer/seer_node3/utils/arguments_utils.py:212)
- [utils/arguments_utils.py](/home/mingyujung/private/seer/seer_node3/utils/arguments_utils.py:213)

기본값:

```text
use_lrnode_latent_update = 0
lrnode_train_latent_distill = 0
lrnode_eval_skip_full_forward = 0
```

이론적, 실험적 이유:

1. 기존 Seer baseline을 깨지 않아야 한다.
2. LR-NODE는 experimental module이므로 opt-in이어야 한다.
3. baseline training/eval에서 model output type이 바뀌면 기존 코드가 깨질 수 있다.
4. `return_action_latent`도 필요한 경우에만 켜야 한다.

따라서 LR-NODE는 constructor flag와 CLI flag로만 활성화된다.

## 18. 각 CLI flag의 방법론적 의미

| flag | 의미 | 이론적 역할 |
|---|---|---|
| `use_lrnode_latent_update` | LR-NODE module 생성/사용 | latent transition model 활성화 |
| `lrnode_train_latent_distill` | train에서 LR-NODE loss 계산 | teacher-student distillation 수행 |
| `lrnode_eval_skip_full_forward` | eval에서 skip mode 사용 | full Seer query 감소 |
| `lrnode_teacher_target_mode` | `shifted_context` 또는 `adjacent_sequence` | teacher latent target 정의 |
| `lrnode_context_selected_step` | shifted context에서 probe할 context index | eval cache latent와 train target 정렬 |
| `lrnode_query_interval` | K | full refresh 간격 |
| `lrnode_train_protocol` | `joint` 또는 `adapter` | joint learning인지 frozen adapter인지 결정 |
| `lrnode_freeze_seer_for_adapter` | non-LR-NODE freeze | baseline 보존 |
| `lrnode_assert_only_lrnode_trainable` | freeze 검증 | protocol 위반 방지 |
| `lrnode_latent_weight` | `L_latent` weight | latent manifold matching 강도 |
| `lrnode_action_distill_weight` | `L_action` weight | behavior matching 강도 |
| `lrnode_smooth_weight` | `L_smooth` weight | residual update regularization |
| `lrnode_bc_weight` | `L_bc` weight | dataset action supervision 추가 |
| `lrnode_hidden_dim` | dynamics/gate MLP width | transition capacity |
| `lrnode_motion_dim` | `u_delta` dimension | control signal capacity |
| `lrnode_fast_encoder_type` | 현재 `diffcnn`만 지원 | cheap delta encoder 선택 |
| `lrnode_detach_input_latent` | `z_prev.detach()` | Seer teacher로 gradient 역류 방지 |
| `lrnode_detach_teacher_latent` | target detach | moving target 방지 |
| `lrnode_freeze_action_head_for_lrnode` | LR-NODE branch action head freeze | 기존 head가 이해하는 latent 학습 |
| `lrnode_use_post_layernorm` | `z_next` post LN | latent scale drift 제어 |
| `lrnode_multistep_train` | multi-horizon rollout train | long skip robustness |
| `lrnode_train_max_horizon` | rollout horizon max | cache age 대응 학습 |
| `lrnode_gate_init_bias` | gate 초기 bias | 초기 update 크기 제어 |
| `lrnode_trace` | shape trace print | debugging |
| `lrnode_log_sanity` | grad/latent/action stats logging | 학습 진단 |
| `lrnode_eval_shadow_full_forward` | skip step에서 full Seer도 추가 실행 | oracle comparison, latency 오염 주의 |

## 19. logging이 왜 필요한가?

코드 위치:

- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:507)
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:580)
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:603)
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:616)
- [utils/eval_utils_libero.py](/home/mingyujung/private/seer/seer_node3/utils/eval_utils_libero.py:300)
- [utils/eval_utils_libero.py](/home/mingyujung/private/seer/seer_node3/utils/eval_utils_libero.py:604)

LR-NODE는 success rate만 보면 왜 망했는지 알기 어렵다. 다음 failure mode가 가능하다.

| failure mode | 관측해야 하는 metric |
|---|---|
| update가 거의 없음 | `gate_mean`, `update_norm`, `latent_mse_pred ~= latent_mse_hold` |
| update가 너무 큼 | `update_to_latent_norm_ratio`, action jerk |
| visual delta를 안 씀 | `corr_imgdiff_update_norm`, `corr_imgdiff_gate` |
| action head와 안 맞음 | `action_l1_pred`, `action_l1_ratio` |
| latent는 맞는데 action이 안 맞음 | latent MSE vs action L1 비교 |
| long skip에서 drift | shadow by cache age |
| LR-NODE가 full보다 느림 | fast encoder/node/action head latency |

따라서 현재 logging은 다음 층위를 모두 기록한다.

1. latent quality: `latent_mse_pred`, cosine, hold 대비 improvement
2. action quality: `action_l1_pred`, hold 대비 improvement
3. control signal: image diff, `u_delta` norm
4. dynamics behavior: gate, `dz`, update norm
5. gradient sanity: LR-NODE modules, Seer backbone, action head grad norm
6. eval efficiency: full call 수, LR-NODE call 수, latency
7. optional oracle: shadow full forward latent/action error

## 20. shadow full forward는 무엇이고 왜 latency 실험에서는 꺼야 하는가?

`lrnode_eval_shadow_full_forward=1`이면 LR-NODE skip step에서도 full Seer를 추가로 실행한다. 이것은 action 생성에는 쓰지 않고, LR-NODE prediction이 full Seer oracle과 얼마나 다른지 측정하기 위한 것이다.

측정값:

```text
shadow_latent_mse
shadow_latent_cos
shadow_action_l1
shadow_action_hold_l1
pred_vs_hold_improvement
by_cache_age
```

이론적 의미:

```text
LR-NODE update가 hold보다 full Seer에 가까운가?
```

하지만 shadow full forward는 skip step마다 비싼 full Seer를 추가 호출한다. 따라서 policy latency 측정을 오염시킨다. efficiency 실험에서는 꺼야 한다.

## 21. 현재 방법론을 정확히 한 문단으로 설명하면

Full policy가 env step `t`에서 받는 context `C_t`와 env step `t+1`에서 받는 shifted context `C_{t+1}`에서 action decoder 직전 latent를 teacher state로 추출한다. cheap diff-CNN과 proprio delta encoder가 두 context 경계의 visual/proprio change vector `u_delta`를 만들고, controlled latent NODE가 fixed Euler residual update로 `z(C_t)`를 `z(C_{t+1})`에 가깝게 이동시킨다. gate는 observation delta와 cache age를 보고 update 크기를 조절하여 latent drift와 noisy update를 억제한다. 학습에서는 full policy latent와 full policy action을 teacher로 삼아 latent MSE와 action distillation loss를 걸고, adapter/distill protocol에서는 Seer와 action head를 freeze하여 LR-NODE만 기존 latent manifold를 따라가게 만든다. eval에서는 매 K step마다 full Seer를 refresh하고, 사이 step에서는 LR-NODE로 cached latent를 갱신한 뒤 기존 action head로 action을 생성해 full query 수와 policy inference latency를 줄인다.

## 22. 헷갈리기 쉬운 지점 정리

### 22.1 LR-NODE는 기존 NODE action head인가?

아니다. 기존 `use_node_action_head` branch와 다르다. LR-NODE는 action head가 아니라 latent updater다.

```text
기존 action head: latent -> action
LR-NODE: previous latent + delta observation -> next latent
```

마지막 action 생성은 기존 Seer action head가 한다.

### 22.2 LR-NODE가 image를 예측하는가?

아니다. `FastVisualDeltaEncoder`는 image reconstruction을 하지 않는다. key/current/diff image를 작은 CNN으로 압축해 control vector만 만든다.

### 22.3 LR-NODE가 full Seer latent를 매번 완전히 재생성하는가?

아니다. cached latent에 residual update를 더한다.

```text
z_next = z_prev + update
```

### 22.4 gate가 action gate인가?

아니다. gate는 action output을 섞는 gate가 아니다. latent update 크기를 조절하는 gate다.

```text
update = gate * dt * dz
```

### 22.5 K가 커지면 무조건 좋은가?

아니다. K가 커지면 full query는 줄지만 latent rollout이 길어져 drift와 action jerk가 증가할 수 있다. 효율성 frontier를 보려면 success rate, policy latency, full query reduction, action smoothness를 같이 봐야 한다.

### 22.6 finetune과 joint는 둘 다 가능한가?

가능하다. 하지만 claim이 다르다.

```text
distill_node.sh:
  기존 baseline에 LR-NODE만 붙인 효율화 claim

scratch_node_joint.sh:
  LR-NODE auxiliary loss를 포함한 scratch-trained variant claim
```

논문 메인 claim이 efficiency라면 우선순위는 `distill_node.sh`다.

### 22.7 현재 오래된 joint ckpt는 못 쓰는가?

못 쓰는 것은 아니다. 다만 "baseline에 LR-NODE만 붙였다"는 claim에는 쓰면 안 된다. 그 ckpt는 scratch joint variant 결과로만 해석해야 한다.

## 23. 현재 구현의 중요한 주의사항

1. `lrnode_use_post_layernorm` default는 0이다. 즉 update 뒤 LayerNorm은 기본적으로 꺼져 있다.
2. `lrnode_teacher_target_mode` default는 `shifted_context`이다. 기본 학습은 `C_t -> C_{t+1}` policy-context teacher-probe distillation이다.
3. `lrnode_multistep_train` default는 0이다. shifted-context 기본 모드에서는 one-step context transition만 학습한다.
4. `lrnode_bc_weight` default는 0이다. shifted-context 모드에서 BC label alignment는 아직 정의하지 않았기 때문에 `lrnode_bc_weight > 0`은 명시적으로 막는다.
5. eval skip mode는 current history 전체를 full transformer에 넣지 않고, cached single selected-step latent를 갱신한다.
6. multi-camera fusion은 평균이다. camera attention은 없다.
7. adaptive ODE solver는 없다. fixed Euler update만 쓴다.
8. adapter claim은 반드시 freeze snapshot과 K=1 sanity로 검증해야 한다.

## 24. 코드별 책임 요약

| 파일 | 코드 | 책임 |
|---|---|---|
| `models/seer_model.py` | `action_latent_full` slice | Seer action-relevant latent 추출 |
| `models/seer_model.py` | `decode_action_from_latent` | 기존 action head 재사용 |
| `models/seer_model.py` | `decode_lrnode_action_from_latent` | LR-NODE branch에서 action head freeze |
| `models/seer_model.py` | `lrnode_encode_delta` | image/proprio delta를 `u_delta`로 변환 |
| `models/seer_model.py` | `lrnode_apply_dynamics` | controlled Euler latent update 적용 |
| `models/seer_model.py` | `lrnode_compute_loss` branch | legacy adjacent-sequence teacher-student tensor 생성 |
| `models/lrnode_modules.py` | `FastVisualDeltaEncoder` | cheap visual/proprio delta encoder |
| `models/lrnode_modules.py` | `ControlledLatentNODE` | gated residual latent dynamics |
| `utils/train_utils.py` | shifted-context teacher probe | `C_t -> C_{t+1}` latent/action distillation target 생성 |
| `utils/train_utils.py` | LR-NODE losses | latent/action/smooth objective 계산, shifted-context BC는 차단 |
| `utils/train_utils.py` | LR-NODE logs | hold 대비 improvement, gate/update diagnostics |
| `utils/eval_utils_libero.py` | cache/update path | full query skip evaluation |
| `train.py` | `_apply_lrnode_train_protocol` | joint vs adapter trainability 강제 |
| `utils/arguments_utils.py` | LR-NODE flags | disabled-by-default, experiment control |

## 25. 연구 설명용 최종 formulation

논문/발표에서는 다음처럼 정리하는 것이 가장 정확하다.

```text
문제:
Vision-language robot policy는 매 control step마다 full visual-language transformer forward가 필요해 inference cost가 크다.

해결:
Full Seer를 sparse하게 query하고, 사이 step에서는 cheap visual/proprio delta와 gated latent NODE로 action-relevant latent를 업데이트해 기존 action head를 재사용한다.

방법:
Latent-Reactive NODE, LR-NODE.
Seer의 action-token latent를 state로 보고,
u_delta = E_delta(I_key, I_cur, q_key, q_cur)를 control input으로 만들며,
z_next = z_prev + sigmoid(g(u_delta, age)+b) * dt * f(z_prev, u_delta, dt, age)
로 fixed Euler latent update를 수행한다.
학습은 full Seer latent/action distillation으로 하고,
평가는 every-K full refresh와 skip-step latent update로 full query를 줄인다.
```
