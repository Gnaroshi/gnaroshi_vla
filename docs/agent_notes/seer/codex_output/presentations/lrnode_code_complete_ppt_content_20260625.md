# LR-NODE 코드 기준 생략 없는 PPT 내용 구성안 - 2026-06-25

이 문서는 디자인용 deck가 아니라 **슬라이드 내용 원고**다.
목표는 LR-NODE 구현을 코드 기준으로 빠짐없이 설명하는 것이다.

범위는 “LR-NODE 관련 코드 path” 전체다.

- argument/flag 정의
- SeerAgent construction
- action latent 추출
- LR-NODE module 내부
- training teacher target 구성
- LR-NODE loss 계산
- adapter freeze protocol
- adapter-only checkpoint 저장
- eval checkpoint overlay/load guard
- eval cache/full/skip branch
- shadow diagnostic
- metric aggregation/export
- valid result 해석 시 확인해야 하는 artifact

---

## Slide 1. 제목: LR-NODE 코드 기준 전체 실행 흐름

### 핵심 메시지

LR-NODE는 별도 policy가 아니라, 기존 Seer가 만든 **action head 직전 latent**를 cache하고 갱신하는 adapter다.

### 슬라이드 본문

```text
Train:
args -> SeerAgent(+LR-NODE modules)
     -> full Seer action_latent_full
     -> LR-NODE z_pred_next
     -> latent/action/smooth distillation loss
     -> adapter-only checkpoint

Eval:
base ckpt load -> adapter ckpt overlay
     -> full branch caches action latent
     -> skip branch updates cached latent
     -> existing action head decodes action
     -> eval_summary.json
```

### 코드 근거

- `models/seer_model.py:624` - `action_latent_full` 주석
- `models/seer_model.py:626` - action token slice
- `utils/eval_utils_libero.py:316` - full-forward latent cache
- `utils/eval_utils_libero.py:327` - cache 기반 LR-NODE update

### 발표자 설명

이 발표는 성능표가 아니라 구현 해부다. 가장 중요한 관점은 “LR-NODE가 action head를 바꾸지 않고 latent만 업데이트한다”는 것이다.

---

## Slide 2. 파일 맵: LR-NODE 관련 코드가 어디에 있는가

### 핵심 메시지

LR-NODE 구현은 model, train, eval, script, logging으로 나뉜다.

### 슬라이드 본문

| 파일 | 역할 |
|---|---|
| `models/lrnode_modules.py` | `FastVisualDeltaEncoder`, `ControlledLatentNODE` |
| `models/seer_model.py` | Seer forward, action latent, LR-NODE wrapper, training branch |
| `utils/arguments_utils.py` | LR-NODE train/eval flags |
| `train.py` | adapter freeze protocol, optimizer 대상 결정 |
| `utils/train_utils.py` | teacher target 구성, LR-NODE loss, checkpoint 저장 |
| `eval_libero.py` | checkpoint load guard, base+adapter overlay |
| `utils/eval_utils_libero.py` | eval cache, full/skip branch, metrics export |
| `scripts/LIBERO_LONG/Seer/*.sh` | 실험 protocol별 flag 조합 |

### 코드 근거

- `utils/arguments_utils.py:211-290`
- `train.py:41-92`
- `eval_libero.py:56-73`
- `utils/eval_utils_libero.py:139-208`

### 발표자 설명

하나의 클래스만 보면 전체 구조가 안 보인다. 특히 distill adapter 결과를 해석하려면 train-side freeze와 eval-side load overlay를 같이 봐야 한다.

---

## Slide 3. Argument/Flag 구조

### 핵심 메시지

LR-NODE는 build, train, eval, diagnostic flag가 분리되어 있다.

### 슬라이드 본문

| 구분 | flag | 의미 |
|---|---|---|
| build | `use_lrnode_latent_update` | SeerAgent에 LR-NODE module을 붙임 |
| train | `lrnode_train_latent_distill` | LR-NODE loss branch 활성화 |
| train | `lrnode_train_protocol` | `joint` 또는 `adapter` |
| train | `lrnode_teacher_target_mode` | `shifted_context` 또는 `adjacent_sequence` |
| train | `lrnode_detach_input_latent` | student input latent gradient 차단 |
| train | `lrnode_detach_teacher_latent` | teacher target latent gradient 차단 |
| train | `lrnode_freeze_action_head_for_lrnode` | LR-NODE branch에서 action head update 차단 |
| eval | `lrnode_eval_skip_full_forward` | full Seer skip branch 활성화 |
| eval | `lrnode_query_interval` | periodic refresh interval K |
| eval | `lrnode_eval_refresh_policy` | `periodic`, `first_only`, `fixed_budget` |
| diagnostic | `lrnode_eval_shadow_full_forward` | skip step에서 full Seer shadow 비교 |

### 코드 근거

- `utils/arguments_utils.py:211-290`

### 발표자 설명

중요한 점은 `use_lrnode_latent_update=1`만으로 skip eval이 되는 것이 아니라는 점이다. eval에서 실제 full forward를 생략하려면 `lrnode_eval_skip_full_forward=1`과 cache가 필요하다.

---

## Slide 4. SeerAgent constructor: LR-NODE 설정 저장

### 핵심 메시지

SeerAgent는 constructor에서 LR-NODE 관련 flag를 모두 member로 저장한다.

### 슬라이드 본문

```python
# models/seer_model.py:155-167
use_lrnode_latent_update=0
lrnode_hidden_dim=256
lrnode_motion_dim=128
lrnode_fast_encoder_type="diffcnn"
lrnode_detach_input_latent=1
lrnode_detach_teacher_latent=1
lrnode_freeze_action_head_for_lrnode=1
lrnode_gate_init_bias=-4.0
```

```python
# models/seer_model.py:185-197
self.use_lrnode_latent_update = bool(use_lrnode_latent_update)
self.lrnode_motion_dim = lrnode_motion_dim
self.lrnode_freeze_action_head_for_lrnode = bool(...)
```

### 코드 근거

- `models/seer_model.py:155-198`

### 발표자 설명

이 시점은 아직 LR-NODE가 실행되는 것이 아니라 설정이 model object에 저장되는 단계다.

---

## Slide 5. SeerAgent construction: LR-NODE module 부착

### 핵심 메시지

`use_lrnode_latent_update`가 켜져야만 LR-NODE module이 생성된다.

### 슬라이드 본문

```python
# models/seer_model.py:316-317
if self.use_lrnode_latent_update:
    self._build_lrnode_modules_preserving_rng()
```

```python
# models/seer_model.py:329-343
self.lrnode_delta_encoder = FastVisualDeltaEncoder(...)
self.lrnode_dynamics = ControlledLatentNODE(
    latent_dim=self.hidden_dim,
    motion_dim=self.lrnode_motion_dim,
    hidden_dim=self.lrnode_hidden_dim,
    gate_init_bias=self.lrnode_gate_init_bias,
    action_pred_steps=self.action_pred_steps,
)
```

### 코드 근거

- `models/seer_model.py:316-345`
- `models/seer_model.py:347-358`

### 발표자 설명

`_build_lrnode_modules_preserving_rng()`는 torch/numpy/python RNG state를 저장하고 복구한다. 즉 LR-NODE module 추가가 이후 baseline 초기화/환경 설정 random stream을 오염시키지 않게 한다.

---

## Slide 6. 기존 Seer action head는 무엇인가

### 핵심 메시지

기존 action head는 `action_decoder`, `arm_action_decoder`, `gripper_action_decoder` 세 부분이다.

### 슬라이드 본문

```python
# models/seer_model.py:269-284
self.action_decoder = nn.Sequential(...)
self.arm_action_decoder = nn.Sequential(nn.Linear(..., 6), Tanh())
self.gripper_action_decoder = nn.Sequential(nn.Linear(..., 1), Sigmoid())
```

```python
# models/seer_model.py:390-403
def decode_action_from_latent(self, action_latent):
    action_pred_feature = self.action_decoder(action_latent)
    arm_pred_action = self.arm_action_decoder(action_pred_feature)
    gripper_pred_action = self.gripper_action_decoder(action_pred_feature)
    return arm_pred_action, gripper_pred_action
```

### 코드 근거

- `models/seer_model.py:269-284`
- `models/seer_model.py:390-403`

### 발표자 설명

LR-NODE의 claim을 위해 이 부분이 중요하다. action head를 새로 만드는 것이 아니라, 기존 Seer action head가 해석할 수 있는 latent를 만들어야 한다.

---

## Slide 7. LR-NODE branch에서 action head freeze의 의미

### 핵심 메시지

`freeze_action_head_for_lrnode`는 action head 계산을 제거하지 않는다. LR-NODE branch에서 action head parameter update만 막는다.

### 슬라이드 본문

```python
# models/seer_model.py:21-32
@contextmanager
def temporarily_freeze_params(modules):
    old_requires_grad = []
    for module in modules:
        for param in module.parameters():
            old_requires_grad.append((param, param.requires_grad))
            param.requires_grad_(False)
    try:
        yield
    finally:
        ...
```

```python
# models/seer_model.py:408-412
def decode_lrnode_action_from_latent(self, action_latent, freeze_action_head=True):
    if freeze_action_head:
        with temporarily_freeze_params(self.get_action_head_modules()):
            return self.decode_action_from_latent(action_latent)
```

### 코드 근거

- `models/seer_model.py:21-32`
- `models/seer_model.py:405-412`

### 발표자 설명

gradient는 `z_pred_next` 방향으로 흘러야 한다. 그래야 LR-NODE가 action head가 원하는 latent를 만들도록 학습된다. 막는 것은 action head parameter 업데이트다.

---

## Slide 8. FastVisualDeltaEncoder: 입력 구성

### 핵심 메시지

skip step에서 full ViT를 돌리지 않기 위해 cheap delta encoder가 key/current image와 proprio 변화량을 압축한다.

### 슬라이드 본문

```python
# models/lrnode_modules.py:49-52
x = torch.cat([key_rgb, cur_rgb, cur_rgb - key_rgb], dim=-3)
x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])
x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
x = self.image_encoder(x).flatten(1)
```

입력 의미:

| 구성 | 의미 |
|---|---|
| `key_rgb` | 마지막 full Seer/cached 기준 이미지 |
| `cur_rgb` | 현재 이미지 |
| `cur_rgb - key_rgb` | visual change cue |

### 코드 근거

- `models/lrnode_modules.py:38-54`

### 발표자 설명

이 encoder는 semantic perception replacement가 아니다. “latent update에 필요한 변화량”을 cheap하게 만드는 역할이다.

---

## Slide 9. FastVisualDeltaEncoder: network와 multi-camera/proprio 처리

### 핵심 메시지

primary/wrist camera feature를 평균하고, proprio delta를 더한 뒤 LayerNorm을 거쳐 `u_delta`를 만든다.

### 슬라이드 본문

```python
# models/lrnode_modules.py:16-25
Conv2d(9, 32, stride=2)
Conv2d(32, 64, stride=2)
Conv2d(64, 128, stride=2)
AdaptiveAvgPool2d(1)
Linear(128, motion_dim)
```

```python
# models/lrnode_modules.py:68-70
image_features = [self._encode_camera(k, c) for k, c in zip(key_rgb, cur_rgb)]
u_delta = torch.stack(image_features, dim=0).mean(dim=0)
```

```python
# models/lrnode_modules.py:83-86
q_delta = q_cur - q_key
u_delta = u_delta + self.proprio_proj(torch.cat([q_key, q_cur, q_delta], dim=-1))
u_delta = self.out_norm(u_delta)
```

### 코드 근거

- `models/lrnode_modules.py:16-34`
- `models/lrnode_modules.py:64-88`

### 발표자 설명

`last_u_delta`와 `last_camera_features`가 저장되므로 eval/debug에서 delta norm 등을 확인할 수 있다.

---

## Slide 10. ControlledLatentNODE: update 수식과 코드

### 핵심 메시지

이 구현은 adaptive ODE solver가 아니라 fixed-step Euler-style residual update다.

### 슬라이드 본문

```python
# models/lrnode_modules.py:177-203
dt_value = self._scalar(dt, ...)
dt_emb = self._time_embedding(dt, ...)
age_emb = self._time_embedding(age, ...)

z_dyn = self._add_action_token_embedding(self.input_ln(z_prev))
dz = self.dynamics(torch.cat([z_dyn_flat, u_flat, dt_flat, age_flat], dim=-1)).view_as(z_prev)
gate = torch.sigmoid(self.gate(torch.cat([u_flat, age_flat], dim=-1)) + self.gate_bias)
update = gate * dt_value * dz
z_next = z_prev + update
```

수식:

```text
u_t = E_delta(o_key, o_t, q_key, q_t)
z_hat_t = z_prev + sigmoid(g(u_t, age)+b) * dt * f(z_prev, u_t, dt, age)
```

### 코드 근거

- `models/lrnode_modules.py:91-206`

### 발표자 설명

`age`는 cache가 몇 step 오래됐는지를 알려준다. `gate_bias=-4.0`은 초기 update를 작게 만들어 과격한 latent 변화로 시작하지 않게 한다.

---

## Slide 11. SeerAgent wrapper: delta -> dynamics -> next latent

### 핵심 메시지

SeerAgent는 LR-NODE module을 직접 노출하지 않고 wrapper 함수로 device/dtype 정렬 후 호출한다.

### 슬라이드 본문

```python
# models/seer_model.py:414-439
def lrnode_encode_delta(...):
    key_image_primary = key_image_primary.to(device=lrnode_param.device, dtype=lrnode_param.dtype)
    ...
    return self.lrnode_delta_encoder(
        [key_image_primary, key_image_wrist],
        [cur_image_primary, cur_image_wrist],
        q_key=q_key,
        q_cur=q_cur,
    )
```

```python
# models/seer_model.py:441-447
def lrnode_apply_dynamics(self, z_prev, u_delta, dt=1.0, age=1.0):
    z_prev = z_prev.to(device=dynamics_param.device, dtype=dynamics_param.dtype)
    u_delta = u_delta.to(device=dynamics_param.device, dtype=dynamics_param.dtype)
    return self.lrnode_dynamics(z_prev, u_delta, dt=dt, age=age)
```

### 코드 근거

- `models/seer_model.py:414-447`

### 발표자 설명

이 wrapper는 mixed precision/eval runtime에서 dtype mismatch를 줄이는 역할도 한다.

---

## Slide 12. `lrnode_predict_next_latent()` 전체 의미

### 핵심 메시지

LR-NODE의 핵심 runtime API는 “delta encode 후 dynamics 적용”이다.

### 슬라이드 본문

```python
# models/seer_model.py:449-471
u_delta = self.lrnode_encode_delta(
    key_image_primary=key_image_primary,
    key_image_wrist=key_image_wrist,
    cur_image_primary=cur_image_primary,
    cur_image_wrist=cur_image_wrist,
    q_key=q_key,
    q_cur=q_cur,
)
z_next = self.lrnode_apply_dynamics(z_prev, u_delta, dt=dt, age=age)
```

입출력:

```text
Input: z_prev, key/current primary, key/current wrist, key/current proprio
Output: z_next
```

### 코드 근거

- `models/seer_model.py:449-477`

### 발표자 설명

학습 branch와 eval skip branch가 결국 같은 latent update API를 사용한다. 이 점 때문에 train/eval mechanism이 연결된다.

---

## Slide 13. Full Seer forward: action latent가 나오는 위치

### 핵심 메시지

`action_latent_full`은 transformer output에서 action token 위치만 slicing한 tensor다.

### 슬라이드 본문

```python
# models/seer_model.py:600-604
transformer_input = self.embedding_layer_norm(transformer_input)
transformer_output = self.transformer_backbone(
    inputs_embeds=transformer_input,
    attention_mask=self.attention_mask,
)
transformer_output = transformer_output.view(B, S, -1, self.hidden_dim)
```

```python
# models/seer_model.py:624-627
# Shape: [B, S, action_pred_steps, hidden_dim]
action_latent_full = transformer_output[
    :, :,
    pred_token_start_idx+this_num_obs_token:
    pred_token_start_idx+this_num_obs_token+self.action_pred_steps,
    :
]
arm_pred_action, gripper_pred_action = self.decode_action_from_latent(action_latent_full)
```

### 코드 근거

- `models/seer_model.py:600-627`

### 발표자 설명

이 tensor가 LR-NODE의 상태 변수다. full Seer 전체를 예측하지 않고 action head 직전 latent transition만 예측한다.

---

## Slide 14. Train loop: input window와 label action 구성

### 핵심 메시지

train loop는 sequence window를 만들고, LR-NODE teacher target mode를 결정한 뒤 model forward에 LR-NODE 입력을 넘긴다.

### 슬라이드 본문

```python
# utils/train_utils.py:320-327
input_image_primary = images_primary[:, :args.sequence_length, :]
input_image_wrist = images_wrist[:, :args.sequence_length, :]
input_text_token = text_tokens[:, :args.sequence_length, :]
input_state = input_states[:, :args.sequence_length, :]

label_actions = torch.cat([
    actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2)
    for j in range(args.action_pred_steps)
], dim=-2)
```

```python
# utils/train_utils.py:328-334
train_lrnode = bool(args.use_lrnode_latent_update and args.lrnode_train_latent_distill)
lrnode_teacher_target_mode = getattr(args, "lrnode_teacher_target_mode", "shifted_context")
```

### 코드 근거

- `utils/train_utils.py:320-334`

### 발표자 설명

`train_lrnode`가 false면 기존 Seer training처럼 진행된다. true일 때만 LR-NODE branch 출력과 loss가 살아난다.

---

## Slide 15. Teacher target mode 1: shifted_context

### 핵심 메시지

`shifted_context`는 다음 policy context를 teacher로 한 번 더 forward해서 target latent를 만든다.

### 슬라이드 본문

```python
# utils/train_utils.py:346-367
input_image_primary_next = images_primary[:, 1:args.sequence_length + 1, :]
input_image_wrist_next = images_wrist[:, 1:args.sequence_length + 1, :]
input_text_token_next = text_tokens[:, 1:args.sequence_length + 1, :]
input_state_next = input_states[:, 1:args.sequence_length + 1, :]
action_next = actions[:, 1:args.sequence_length + 1, :]
```

```python
# utils/train_utils.py:383-400
with torch.no_grad():
    teacher_outputs_next = model(..., return_action_latent=True, lrnode_compute_loss=False)
lrnode_z_teacher_next_external = teacher_outputs_next["action_latent"][:, lrnode_selected_step]
```

### 코드 근거

- `utils/train_utils.py:340-400`

### 발표자 설명

이 방식은 더 정확한 다음 context teacher를 만들지만 teacher forward가 추가된다. 따라서 학습시간 overhead가 생긴다.

---

## Slide 16. Teacher target mode 2: adjacent_sequence

### 핵심 메시지

`adjacent_sequence`는 같은 `action_latent_full` window 안에서 adjacent latent pair를 target으로 쓴다.

### 슬라이드 본문

```python
# models/seer_model.py:755-762
lrnode_z_prev = action_latent_full[:, :-1]
lrnode_z_teacher_next = action_latent_full[:, 1:]

if lrnode_detach_input_latent:
    lrnode_z_prev = lrnode_z_prev.detach()
if lrnode_detach_teacher_latent:
    lrnode_z_teacher_next = lrnode_z_teacher_next.detach()

lrnode_z_pred_next = self.lrnode_predict_next_latent(...)
```

### 코드 근거

- `models/seer_model.py:711-773`

### 발표자 설명

이 path는 shifted_context처럼 추가 teacher forward를 하지 않는다. 다만 target 정의가 “window 내부 adjacent latent”가 된다.

---

## Slide 17. LR-NODE train branch: validation과 detach

### 핵심 메시지

LR-NODE train branch는 action latent shape와 input 존재 여부를 강하게 검증한다.

### 슬라이드 본문

```python
# models/seer_model.py:631-639
if lrnode_compute_loss:
    if not self.use_lrnode_latent_update:
        raise RuntimeError(...)
    if action_latent_full is None:
        raise RuntimeError(...)
    if action_latent_full.dim() != 4:
        raise RuntimeError(...)
```

```python
# models/seer_model.py:651-659
required_lrnode_inputs = {
    "lrnode_key_image_primary": lrnode_key_image_primary,
    "lrnode_key_image_wrist": lrnode_key_image_wrist,
    "lrnode_cur_image_primary": lrnode_cur_image_primary,
    "lrnode_cur_image_wrist": lrnode_cur_image_wrist,
}
```

### 코드 근거

- `models/seer_model.py:631-659`
- `models/seer_model.py:694-709`
- `models/seer_model.py:755-773`

### 발표자 설명

여기서 missing input이면 바로 RuntimeError가 난다. 즉 LR-NODE loss는 “적당히 None이면 skip”하는 식으로 흐르지 않는다.

---

## Slide 18. LR-NODE predicted action, teacher action, hold action

### 핵심 메시지

training branch는 LR-NODE action뿐 아니라 teacher action과 hold baseline action도 같이 만든다.

### 슬라이드 본문

```python
# models/seer_model.py:782-794
lrnode_arm_action, lrnode_gripper_action = self.decode_lrnode_action_from_latent(
    lrnode_z_pred_next,
    freeze_action_head=bool(lrnode_freeze_action_head_for_lrnode),
)
with torch.no_grad():
    teacher_arm_action, teacher_gripper_action = self.decode_action_from_latent(
        lrnode_z_teacher_next.detach()
    )
    hold_arm_action, hold_gripper_action = self.decode_action_from_latent(
        lrnode_z_prev.detach()
    )
```

### 코드 근거

- `models/seer_model.py:775-794`

### 발표자 설명

`hold_action`은 “latent를 업데이트하지 않고 그대로 action head에 넣었을 때”의 baseline이다. shadow/diagnostic 해석에 중요하다.

---

## Slide 19. Model forward return dictionary

### 핵심 메시지

`return_action_latent=True`이면 full Seer output과 LR-NODE debug tensor가 dictionary로 반환된다.

### 슬라이드 본문

```python
# models/seer_model.py:796-819
return {
    "arm_pred_action": arm_pred_action,
    "gripper_pred_action": gripper_pred_action,
    "action_latent": action_latent_full,
    "lrnode_z_prev": lrnode_z_prev,
    "lrnode_z_teacher_next": lrnode_z_teacher_next,
    "lrnode_z_pred_next": lrnode_z_pred_next,
    "lrnode_arm_action": lrnode_arm_action,
    "lrnode_gripper_action": lrnode_gripper_action,
    "lrnode_teacher_action": lrnode_teacher_action,
    "lrnode_hold_action": lrnode_hold_action,
    "lrnode_gate": lrnode_gate,
    "lrnode_u_delta": ...,
    "lrnode_dz": ...,
    "lrnode_update": ...,
}
```

### 코드 근거

- `models/seer_model.py:796-819`

### 발표자 설명

train/eval에서 필요한 tensor가 이 dictionary를 통해 전달된다. 특히 eval full branch는 `action_latent`를 cache하기 위해 이 반환값이 필요하다.

---

## Slide 20. LR-NODE loss 계산

### 핵심 메시지

LR-NODE loss는 latent matching, action distillation, smooth update regularization으로 구성된다.

### 슬라이드 본문

```python
# utils/train_utils.py:530-557
loss_lrnode_latent = mse_loss(z_pred_next, z_teacher_next)

lrnode_action = torch.cat([lrnode_arm_action, lrnode_gripper_action], dim=-1)
loss_lrnode_action_distill = l1_loss(
    lrnode_action,
    lrnode_teacher_action.detach(),
)

loss_lrnode_smooth = mse_loss(
    z_pred_next - z_prev.detach(),
    zeros_like(z_pred_next),
)
```

### 코드 근거

- `utils/train_utils.py:523-557`

### 발표자 설명

latent loss는 manifold alignment, action distill은 action head output consistency, smooth는 latent update magnitude regularization에 해당한다.

---

## Slide 21. Total loss 조합

### 핵심 메시지

최종 loss는 기존 Seer base loss와 LR-NODE auxiliary/distill loss의 합이다.

### 슬라이드 본문

```python
# utils/train_utils.py:595-606
base_loss_weighted = (
    args.loss_arm_action_ratio * loss_arm_action
    + args.loss_gripper_action_ratio * loss_gripper_action
    + 0.1 * loss_image
)

lrnode_loss_weighted = (
    args.lrnode_latent_weight * loss_lrnode_latent
    + args.lrnode_action_distill_weight * loss_lrnode_action_distill
    + args.lrnode_smooth_weight * loss_lrnode_smooth
    + args.lrnode_bc_weight * loss_lrnode_bc
)

loss_calvin = base_loss_weighted + lrnode_loss_weighted
```

### 코드 근거

- `utils/train_utils.py:595-606`

### 발표자 설명

adapter protocol에서는 base model parameter가 frozen이므로 base loss가 계산되더라도 optimizer update 대상은 LR-NODE parameter뿐이다.

---

## Slide 22. Adapter protocol: 어떤 parameter만 학습되는가

### 핵심 메시지

adapter protocol에서 trainable parameter는 `lrnode_delta_encoder.*`, `lrnode_dynamics.*`뿐이다.

### 슬라이드 본문

```python
# train.py:41-42
def _is_lrnode_parameter(name):
    return name.startswith("lrnode_delta_encoder.") or name.startswith("lrnode_dynamics.")
```

```python
# train.py:62-67
if freeze_for_adapter:
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if _is_lrnode_parameter(name):
            param.requires_grad_(True)
```

```python
# train.py:77-84
if assert_only_lrnode and non_lrnode_trainable:
    raise RuntimeError(...)
```

### 코드 근거

- `train.py:41-92`

### 발표자 설명

이 코드가 distill protocol의 핵심이다. “freeze Seer/action head”는 논리적 설명이 아니라 실제 optimizer 대상에서 제외되는 코드 path다.

---

## Slide 23. Adapter training script flags

### 핵심 메시지

`distill_node.sh`가 adapter training protocol을 구성한다.

### 슬라이드 본문

```bash
# scripts/LIBERO_LONG/Seer/distill_node.sh:69-94
--use_lrnode_latent_update 1
--lrnode_train_latent_distill 1
--lrnode_train_protocol adapter
--lrnode_freeze_seer_for_adapter 1
--lrnode_assert_only_lrnode_trainable 1
--lrnode_detach_input_latent 1
--lrnode_detach_teacher_latent 1
--lrnode_freeze_action_head_for_lrnode 1
```

```bash
# scripts/LIBERO_LONG/Seer/distill_node.sh:113
--finetune_from_pretrained_ckpt "${BASELINE_CKPT}"
```

### 코드 근거

- `scripts/LIBERO_LONG/Seer/distill_node.sh:69-94`
- `scripts/LIBERO_LONG/Seer/distill_node.sh:113`

### 발표자 설명

adapter 학습은 baseline checkpoint에서 시작한다. 이때 Seer/action head는 freeze되고 LR-NODE module만 학습된다.

---

## Slide 24. Optimizer construction: requires_grad만 optimizer에 들어감

### 핵심 메시지

adapter protocol 적용 후 optimizer는 `requires_grad=True` parameter만 받는다.

### 슬라이드 본문

```python
# train.py:192-195
model.clip_model.requires_grad_(False)
model.vision_encoder.requires_grad_(False)
lrnode_protocol_status = _apply_lrnode_train_protocol(model, args)
total_params, trainable_params = count_parameters(model)
```

```python
# train.py:215
optimizer = torch.optim.AdamW(
    [p for p in ddp_model.parameters() if p.requires_grad],
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
)
```

### 코드 근거

- `train.py:192-215`

### 발표자 설명

따라서 adapter protocol에서는 action head forward는 쓰이지만 action head parameter는 optimizer 대상이 아니다.

---

## Slide 25. 왜 distill checkpoint가 adapter-only인가

### 핵심 메시지

checkpoint 저장 함수가 frozen parameter를 삭제하므로 adapter checkpoint에는 LR-NODE weight만 남는다.

### 슬라이드 본문

```python
# utils/train_utils.py:986-993
def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict
```

결과:

```text
adapter protocol:
Seer/action head requires_grad=False -> checkpoint에서 제거
LR-NODE requires_grad=True -> checkpoint에 남음
```

### 코드 근거

- `utils/train_utils.py:986-993`

### 발표자 설명

이게 이전 invalid distill eval의 원인이다. adapter checkpoint만 로드하면 frozen이어야 할 Seer/action head weight가 checkpoint에 없기 때문에 random init 상태가 된다.

---

## Slide 26. Eval checkpoint: adapter-only 감지

### 핵심 메시지

eval loader는 LR-NODE key는 있지만 core Seer key가 없으면 adapter-only checkpoint로 판단한다.

### 슬라이드 본문

```python
# eval_libero.py:56-73
def _is_lrnode_adapter_only_state_dict(state_dict):
    keys = list(state_dict.keys())
    has_lrnode = any("lrnode_" in key or ".lrnode" in key for key in keys)
    has_core_seer = any(
        marker in key
        for key in keys
        for marker in (
            "transformer_backbone",
            "action_decoder",
            "action_pred_token",
            "perceiver_resampler",
            "image_primary_projector",
            "image_wrist_projector",
        )
    )
    return has_lrnode and not has_core_seer
```

### 코드 근거

- `eval_libero.py:56-73`

### 발표자 설명

adapter-only 여부를 파일명으로 판단하지 않는다. state_dict key 구조로 판단한다.

---

## Slide 27. Eval checkpoint guard: invalid eval 차단

### 핵심 메시지

adapter-only checkpoint를 base checkpoint 없이 `--resume_from_checkpoint`로 넣으면 eval이 즉시 실패한다.

### 슬라이드 본문

```python
# eval_libero.py:188-202
if (
    bool(args.use_lrnode_latent_update)
    and args.lrnode_train_protocol == "adapter"
    and args.resume_from_checkpoint is not None
    and args.finetune_from_pretrained_ckpt is None
):
    _, resume_state_dict = _checkpoint_state_dict(args.resume_from_checkpoint)
    if _is_lrnode_adapter_only_state_dict(resume_state_dict):
        raise ValueError(
            "Adapter-only LR-NODE checkpoint was passed ... without base checkpoint"
        )
```

### 코드 근거

- `eval_libero.py:188-202`

### 발표자 설명

이 guard가 들어간 이후의 distill eval만 valid하게 해석할 수 있다.

---

## Slide 28. Valid eval load order: base 먼저, adapter overlay

### 핵심 메시지

valid distill eval은 baseline full Seer checkpoint를 먼저 로드하고, adapter checkpoint를 그 위에 overlay한다.

### 슬라이드 본문

```python
# eval_libero.py:204-218
if args.finetune_from_pretrained_ckpt is not None:
    _load_checkpoint_into_model(ddp_model, args.finetune_from_pretrained_ckpt, "base", args.rank)

if args.resume_from_checkpoint is not None:
    _load_checkpoint_into_model(ddp_model, args.resume_from_checkpoint, "resume_or_adapter", args.rank)
```

```bash
# scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh:254-286
base_ckpt_args=(--finetune_from_pretrained_ckpt "${LRNODE_EVAL_BASE_CKPT}")
...
eval_libero.py ... "${base_ckpt_args[@]}" --resume_from_checkpoint "${ckpt_path}"
```

### 코드 근거

- `eval_libero.py:204-218`
- `scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh:254-286`

### 발표자 설명

이 순서가 baseline ckpt33 + adapter ckpt39 overlay의 코드상 의미다.

---

## Slide 29. Eval ModelWrapper: cache와 counters

### 핵심 메시지

eval runtime state는 `ModelWrapper`가 들고 있다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:176-191
self.lrnode_episode_full_forward_calls = 0
self.lrnode_cached_latent = None
self.lrnode_cached_image_primary = None
self.lrnode_cached_image_wrist = None
self.lrnode_cached_state = None
self.lrnode_cached_age = 0
self.full_forward_calls = 0
self.lrnode_update_calls = 0
self.full_forward_latency_sum = 0.0
self.lrnode_latency_sum = 0.0
self.policy_step_latency_sum = 0.0
self.num_policy_steps = 0
```

### 코드 근거

- `utils/eval_utils_libero.py:139-208`
- `utils/eval_utils_libero.py:218-235`

### 발표자 설명

episode reset 시 cache와 age는 초기화된다. 따라서 skip branch는 첫 full branch 이후에만 가능하다.

---

## Slide 30. Eval skip 여부 결정

### 핵심 메시지

`_should_use_lrnode()`가 현재 timestep에서 full Seer를 쓸지 LR-NODE skip을 쓸지 결정한다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:283-303
if not (
    self.use_lrnode_latent_update
    and self.lrnode_eval_skip_full_forward
    and self.lrnode_cached_latent is not None
):
    return False

if self.lrnode_eval_refresh_policy == "first_only":
    return True

if self.lrnode_eval_refresh_policy == "fixed_budget":
    ...

return timestep % self.lrnode_query_interval != 0
```

### 코드 근거

- `utils/eval_utils_libero.py:283-303`

### 발표자 설명

QRED20은 `periodic` 정책이다. K=3이면 `t % 3 != 0`인 step에서 LR-NODE update를 쓴다.

---

## Slide 31. Eval selected step과 action 변환

### 핵심 메시지

full branch와 skip branch 모두 action sequence를 만든 뒤 env action 7D로 변환한다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:254-257
def _selected_step(self, num_step):
    if num_step < self.history_len:
        return num_step - 1
    return -1
```

```python
# utils/eval_utils_libero.py:259-281
if self.use_ensembling:
    ...
else:
    action = action_seq[:, 0]

action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
action[:, -1] = (action[:, -1] - 0.5) * 2
action = action.detach().cpu().numpy()[-1]
```

### 코드 근거

- `utils/eval_utils_libero.py:254-281`

### 발표자 설명

LR-NODE가 내는 것도 full Seer와 같은 형태의 action sequence다. 최종 env action 변환 path는 동일하다.

---

## Slide 32. Full branch: full Seer 실행과 latent cache

### 핵심 메시지

full branch는 Seer model을 호출하고, 반환된 `action_latent`를 cache한다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:748-781
model_outputs = self.model(
    image_primary=input_image_primary,
    image_wrist=input_image_wrist,
    state=input_state,
    text_token=input_text_token,
    action=torch.zeros(...),
    return_action_latent=self.use_lrnode_latent_update,
)
full_ms = ...
self.full_forward_calls += 1

arm_action = model_outputs["arm_pred_action"]
gripper_action = model_outputs["gripper_pred_action"]
action_latent = model_outputs["action_latent"]
selected_step = self._selected_step(num_step)
action_seq = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1)
self._cache_full_forward_state(action_latent, selected_step, image_x, gripper, state)
```

### 코드 근거

- `utils/eval_utils_libero.py:748-781`

### 발표자 설명

cache되는 latent는 전체 `[B,S,...]`가 아니라 `action_latent[:, selected_step]`이다.

---

## Slide 33. Cache 저장 내용

### 핵심 메시지

full branch 후에는 latent뿐 아니라 기준 image/state도 같이 cache된다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:316-325
def _cache_full_forward_state(self, action_latent, selected_step, image_x, gripper, state):
    if not self.use_lrnode_latent_update:
        return
    if action_latent is None or action_latent.dim() != 4:
        raise RuntimeError(...)
    self.lrnode_cached_latent = action_latent[:, selected_step].detach()
    self.lrnode_cached_image_primary = image_x.detach()
    self.lrnode_cached_image_wrist = gripper.detach()
    self.lrnode_cached_state = state.detach()
    self.lrnode_cached_age = 0
```

### 코드 근거

- `utils/eval_utils_libero.py:316-325`

### 발표자 설명

skip step의 delta encoder는 이 cached image/state와 current image/state를 비교한다.

---

## Slide 34. Skip branch: LR-NODE cache update

### 핵심 메시지

skip branch는 full Seer를 부르지 않고 cached latent를 갱신한다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:327-354
age = self.lrnode_cached_age + 1
z_prev = self.lrnode_cached_latent

u_delta = base_model.lrnode_encode_delta(
    key_image_primary=self.lrnode_cached_image_primary[:, 0],
    key_image_wrist=self.lrnode_cached_image_wrist[:, 0],
    cur_image_primary=image_x[:, 0],
    cur_image_wrist=gripper[:, 0],
    q_key=self.lrnode_cached_state[:, 0],
    q_cur=state[:, 0],
)

z_next = base_model.lrnode_apply_dynamics(
    z_prev=z_prev,
    u_delta=u_delta,
    dt=1.0,
    age=float(age),
)
```

### 코드 근거

- `utils/eval_utils_libero.py:327-356`

### 발표자 설명

여기가 LR-NODE의 deployment-time 핵심이다. full Seer transformer는 호출되지 않는다.

---

## Slide 35. Skip branch: action decode와 cache overwrite

### 핵심 메시지

LR-NODE가 만든 `z_next`는 기존 action head로 decode되고, cache는 current step 기준으로 갱신된다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:358-367
arm_action, gripper_action = base_model.decode_action_from_latent(z_next)
with torch.no_grad():
    hold_arm_action, hold_gripper_action = base_model.decode_action_from_latent(z_prev.detach())

action_seq = torch.concat((arm_action, gripper_action), dim=-1)
hold_action_seq = torch.concat((hold_arm_action, hold_gripper_action), dim=-1)
```

```python
# utils/eval_utils_libero.py:389-393
self.lrnode_cached_latent = z_next.detach()
self.lrnode_cached_image_primary = image_x.detach()
self.lrnode_cached_image_wrist = gripper.detach()
self.lrnode_cached_state = state.detach()
self.lrnode_cached_age = age
```

### 코드 근거

- `utils/eval_utils_libero.py:358-394`

### 발표자 설명

cache가 `z_next`로 overwrite되므로 K가 커질수록 LR-NODE rollout이 길어진다. drift와 jerk가 커질 수 있는 이유가 여기 있다.

---

## Slide 36. Step-level logging: skip/full 공통 metric

### 핵심 메시지

eval step마다 mode, latency, gate/update/action smoothness가 기록된다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:673-689
step_record.update({
    "mode": "lrnode_update",
    "cache_age": ...,
    "fast_encoder_ms": ...,
    "node_update_ms": ...,
    "action_head_ms": ...,
    "total_policy_ms": lrnode_ms + preprocess_ms,
    "gate_mean": ...,
    "u_delta_norm": ...,
    "update_norm": ...,
})
```

```python
# utils/eval_utils_libero.py:791-809
step_record.update({
    "action_norm": ...,
    "action_delta_norm": ...,
    "action_jerk": ...,
    "gripper_switch": ...,
})
self.policy_step_latency_sum += total_policy_ms
self.num_policy_steps += 1
```

### 코드 근거

- `utils/eval_utils_libero.py:662-689`
- `utils/eval_utils_libero.py:783-809`

### 발표자 설명

결과 표의 `policy ms`, `jerk p95`, gate/update diagnostic은 이 step record에서 시작된다.

---

## Slide 37. Shadow full-forward diagnostic

### 핵심 메시지

shadow mode는 skipped step에서 실제 action은 LR-NODE로 실행하되, full Seer를 별도로 돌려 비교한다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:690-720
if self.lrnode_eval_shadow_full_forward:
    shadow_outputs = self.model(..., return_action_latent=True)
    shadow_action = ...
    shadow_latent = shadow_outputs["action_latent"][:, selected_step]
    pred_action = lrnode_debug["action_pred"]
    hold_action = lrnode_debug["action_hold"]
    pred_latent = lrnode_debug["z_pred"]

    latent_mse = mse_loss(pred_latent, shadow_latent)
    action_l1 = l1_loss(pred_action, shadow_action)
    action_hold_l1 = l1_loss(hold_action, shadow_action)
```

### 코드 근거

- `utils/eval_utils_libero.py:690-747`

### 발표자 설명

이 mode는 deployment latency 측정용이 아니다. 실패 원인이 latent drift인지, action drift인지, hold보다 나은지 진단하기 위한 실험이다.

---

## Slide 38. Episode-level metrics

### 핵심 메시지

episode가 끝나면 step records를 모아 episode metric을 만든다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:458-537
full_count = sum(1 for r in records if r.get("mode") == "full")
update_count = sum(1 for r in records if r.get("mode") == "lrnode_update")

metrics = {
    "success": int(success),
    "num_steps": int(steps),
    "mode_full_count": int(full_count),
    "mode_update_count": int(update_count),
    "full_forward_ratio": full_count / max(1, len(records)),
    "skip_ratio": update_count / max(1, len(records)),
    "avg_full_forward_ms": mean("full_forward_ms"),
    "avg_policy_step_ms": mean("total_policy_ms"),
    "p95_action_jerk": percentile("action_jerk", 95),
    "cache_age_at_failure": ...,
    "last_full_forward_step": ...,
}
```

### 코드 근거

- `utils/eval_utils_libero.py:458-537`

### 발표자 설명

failure analysis용 `cache_age_at_failure`, `last_full_forward_step`, `max_action_jerk_before_failure`가 여기서 만들어진다.

---

## Slide 39. Distributed metric merge

### 핵심 메시지

DDP rank별 stats는 rank 0에서 합쳐진 뒤 summary로 저장된다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:967-969
local_lrnode_stats = model.get_lrnode_stats()
all_lrnode_stats = [None for _ in range(device_num)] if rank0 else None
torch.distributed.gather_object(local_lrnode_stats, all_lrnode_stats, dst=0)
```

```python
# utils/eval_utils_libero.py:993-1113
merged["num_env_steps"] += env_steps
merged["full_forward_calls"] += full_calls
merged["lrnode_update_calls"] += lrnode_calls
...
merged["full_query_reduction_ratio"] =
    1.0 - full_forward_calls / num_env_steps
merged["effective_query_interval"] =
    num_env_steps / full_forward_calls
```

### 코드 근거

- `utils/eval_utils_libero.py:967-969`
- `utils/eval_utils_libero.py:993-1113`

### 발표자 설명

summary 수치는 단일 GPU local value가 아니라 rank별 episode 통계를 merge한 값이다.

---

## Slide 40. eval_summary.json 저장

### 핵심 메시지

발표 표에 쓰는 수치는 `save_eval_json()`이 저장한 `eval_summary.json`에서 나온다.

### 슬라이드 본문

```python
# utils/eval_utils_libero.py:1160-1305
success_rate = mean(valid_results)
lrnode_stats = merge_lrnode_stats(lrnode_stats_list)

effective_full_query_hz =
    control_hz * full_forward_calls / num_env_steps

effective_lrnode_update_hz =
    control_hz * lrnode_update_calls / num_env_steps

payload = {
    "success_rate": success_rate,
    "lrnode": {...},
    "query_reduction": {...},
    "action_smoothness": {...},
    "task_results": task_results,
}

eval_summary.json
eval_episode_metrics.csv
eval_latency_profile.json
```

### 코드 근거

- `utils/eval_utils_libero.py:1160-1320`

### 발표자 설명

실험 결과를 발표에 넣을 때는 이 summary 파일 존재 여부를 먼저 확인해야 한다. latest distill QRED20 K=4는 이 파일이 없어서 미완료다.

---

## Slide 41. Source snapshots and logging artifacts

### 핵심 메시지

각 run은 args, flags, loss weights, trainable params, freeze status, git snapshot을 저장한다.

### 슬라이드 본문

```python
# utils/lrnode_logging_utils.py:152-161
save_lrnode_run_snapshots(...)
    args_snapshot.json
    lrnode_flags_snapshot.json
    loss_weights_snapshot.json
    model_trainable_params.json
    freeze_status_snapshot.json
    git_snapshot.json
```

주의:

```text
eval freeze_status_snapshot은 optimizer construction 상태를 직접 의미하지 않을 수 있다.
adapter 학습 claim은 train.py protocol, train status, checkpoint key count, load log로 확인한다.
```

### 코드 근거

- `utils/lrnode_logging_utils.py:7-34`
- `utils/lrnode_logging_utils.py:80-161`

### 발표자 설명

eval snapshot에서 action head가 trainable처럼 보이는 경우가 있어도, eval에는 optimizer가 없다. “학습 시 freeze”는 train protocol과 checkpoint 구조로 확인해야 한다.

---

## Slide 42. Valid distill eval의 필수 확인 로그

### 핵심 메시지

distill adapter 결과를 해석하려면 load log가 base-first, adapter-second인지 확인해야 한다.

### 슬라이드 본문

확인해야 하는 로그:

```text
[CKPT LOAD:base] path=.../baseline/.../33.pth
[CKPT LOAD:base] state_dict_keys=400
[CKPT LOAD:base] adapter_only=False

[CKPT LOAD:resume_or_adapter] path=.../distill_node/.../39.pth
[CKPT LOAD:resume_or_adapter] state_dict_keys=30
[CKPT LOAD:resume_or_adapter] adapter_only=True
```

K=1 parity:

```text
baseline ckpt33 K=1 SR == baseline ckpt33 + adapter ckpt39 K=1 SR
83.0% == 83.0%
```

### 코드 근거

- `eval_libero.py:76-93`
- load parity log:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_loadparity_.../ours_.../lrnode_distill_..._ckpt_39.log`

### 발표자 설명

이 parity가 깨지면 K=2/K=3 skip 결과는 해석하면 안 된다.

---

## Slide 43. 코드 기준 claim boundary

### 핵심 메시지

코드 기준으로 주장 가능한 것과 아닌 것을 분리해야 한다.

### 슬라이드 본문

가능한 claim:

```text
LR-NODE는 기존 Seer action head 앞 latent를 업데이트한다.
adapter protocol은 LR-NODE module만 optimizer 대상으로 둔다.
eval skip branch는 full Seer forward를 생략하고 cached latent를 갱신한다.
valid distill eval은 base ckpt + adapter overlay가 필수다.
```

주의해야 할 claim:

```text
action head를 대체했다 -> 틀림
adapter ckpt 단독 eval 결과 -> invalid
eval freeze_status만으로 train freeze를 결론 -> 부정확
summary 없는 K=4 결과 -> 미완료
strict real-time high-Hz claim -> policy/budget 확인 전 불가
```

### 코드 근거

- `models/seer_model.py:408-412`
- `train.py:62-84`
- `eval_libero.py:188-218`
- `utils/eval_utils_libero.py:1160-1320`

### 발표자 설명

이 슬라이드는 발표자가 질문을 받을 때 방어선 역할을 한다.

---

## Slide 44. 최종 코드 흐름 요약

### 핵심 메시지

LR-NODE 구현은 “latent cache + cheap update + existing action head”로 일관된다.

### 슬라이드 본문

```text
1. Seer full forward
   transformer_output -> action_latent_full -> existing action head

2. LR-NODE train
   z_prev + visual/proprio delta -> z_pred_next
   z_pred_next vs teacher latent/action
   frozen Seer/action head, LR-NODE-only optimizer

3. Adapter checkpoint
   frozen params removed -> LR-NODE-only state_dict

4. Valid eval
   base ckpt first -> adapter overlay
   full step: cache z
   skip step: z <- NODE(z, delta)
   action <- existing action head(z)

5. Metrics
   full calls, skip calls, effective Hz, query reduction, latency, jerk
```

### 발표자 설명

이제 결과표를 볼 때 단순히 K별 SR만 보는 것이 아니라, 어떤 path가 실제로 실행됐는지, 어떤 checkpoint가 로드됐는지, summary artifact가 완성됐는지까지 같이 확인해야 한다.

---

## Appendix A. 코드 참조 빠른 목록

| Topic | File:line |
|---|---|
| LR-NODE flags | `utils/arguments_utils.py:211-290` |
| LR-NODE module construction | `models/seer_model.py:316-345` |
| RNG preserve | `models/seer_model.py:347-358` |
| existing action head | `models/seer_model.py:390-403` |
| LR-NODE action decode freeze | `models/seer_model.py:408-412` |
| delta encoder | `models/lrnode_modules.py:6-88` |
| controlled NODE | `models/lrnode_modules.py:91-206` |
| action latent extraction | `models/seer_model.py:600-627` |
| LR-NODE train branch | `models/seer_model.py:631-819` |
| train teacher target | `utils/train_utils.py:320-447` |
| LR-NODE losses | `utils/train_utils.py:523-606` |
| adapter freeze protocol | `train.py:41-92` |
| optimizer target | `train.py:192-215` |
| adapter checkpoint filtering | `utils/train_utils.py:986-993` |
| adapter-only detection | `eval_libero.py:56-73` |
| invalid eval guard | `eval_libero.py:188-202` |
| base then adapter load | `eval_libero.py:204-218` |
| ModelWrapper state | `utils/eval_utils_libero.py:139-208` |
| skip decision | `utils/eval_utils_libero.py:283-303` |
| full cache | `utils/eval_utils_libero.py:316-325` |
| skip update | `utils/eval_utils_libero.py:327-394` |
| runtime full/skip branch | `utils/eval_utils_libero.py:662-781` |
| shadow diagnostic | `utils/eval_utils_libero.py:690-747` |
| action conversion | `utils/eval_utils_libero.py:259-281` |
| episode metrics | `utils/eval_utils_libero.py:458-537` |
| distributed merge | `utils/eval_utils_libero.py:993-1113` |
| summary save | `utils/eval_utils_libero.py:1160-1320` |

## Appendix B. 발표 구성 권장 순서

코드 기준 발표는 아래 순서가 가장 이해가 쉽다.

1. action head를 먼저 설명한다.
2. action head가 읽는 `action_latent_full` 위치를 설명한다.
3. LR-NODE가 이 latent를 업데이트한다는 관점을 설명한다.
4. delta encoder와 controlled NODE를 설명한다.
5. training branch와 loss를 설명한다.
6. adapter freeze/checkpoint 구조를 설명한다.
7. eval load guard와 base+adapter overlay를 설명한다.
8. eval full/cache/skip branch를 설명한다.
9. metric 저장 path를 설명한다.
10. 마지막에 valid result 해석 조건을 설명한다.
