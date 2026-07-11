# Ours vs Baseline Pipeline Analysis

작성일: 2026-06-15 KST

Archive note: 이 문서가 분석한 pre-protocol LR-NODE 결과는 2026-06-16에 아래 위치로 이동했다. 문서 본문에는 당시 원래 경로가 일부 남아 있을 수 있다.

```text
$SEER_WORKSPACE_ROOT/archived_experiment_results_20260616/pre_protocol_lrnode
```

## 1. 분석 기준

현재 비교 실험의 source of truth는 아래 스크립트와 결과 폴더다.

```text
script:
  scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh

result root:
  $SEER_WORKSPACE_ROOT/scratch_eval_lrnode/lrnode_compare_lrnode_student_v2_lw05_aw01_g4_ckpt35_vs_seer_original_ckpt37_20260615_001039
```

비교 대상:

| 구분 | 값 |
|---|---|
| Baseline name | `seer_original` |
| Baseline run | `sd1_libero_10_100pc_original_settings` |
| Baseline ckpt | `$SEER_BASELINE_ROOT/checkpoints/sd1_libero_10_100pc_original_settings/37.pth` |
| Ours method | `lrnode_student_v2_lw05_aw01_g4` |
| Ours run | `sd1_scratch_libero_10_converted_seer_lrnode_student_v2_lw05_aw01_g4` |
| Ours ckpt | `$SEER_WORKSPACE_ROOT/scratch_checkpoints_lrnode/sd1_scratch_libero_10_converted_seer_lrnode_student_v2_lw05_aw01_g4/35.pth` |

공통 eval 설정:

| 항목 | 값 |
|---|---:|
| suite | `libero_10` |
| seed | 42 |
| sequence_length | 7 |
| action_pred_steps | 3 |
| future_steps | 3 |
| obs_pred | True |
| gripper_width | True |
| eval_libero_ensembling | True |
| multi_step_action | 1 |
| num_resampler_query | 6 |
| transformer_layers | 24 |
| precision | fp32 |
| bf16_module | vision_encoder |
| control_hz used for reporting | 20 |
| videos | success/fail, all ranks, stride 1 |

따라서 현재 비교에서 baseline과 ours의 환경, task, seed, observation preprocessing, history length, action postprocessing은 동일하다. 핵심 차이는 checkpoint, LR-NODE 모듈 활성화 여부, 그리고 skip-forward 시 full Seer를 매 step 실행하는지 여부다.

## 2. 공통 environment/eval input pipeline

모든 run은 LIBERO-10에서 10 tasks x 20 episodes를 평가한다.

각 episode에서 실제 environment input은 다음 순서로 생성된다.

1. `task_id = eval_id // 20`, `exp_id = eval_id % 20`
2. LIBERO BDDL 파일로 `OffScreenRenderEnv` 생성
3. `env.seed(42)`
4. LIBERO init state에서 `exp_id`에 해당하는 state 로드
5. `env.set_init_state(init_state)`
6. 초기 physics settle을 위해 zero action 5회 실행
7. 최대 600 step까지 policy-action loop 실행

매 policy step의 raw observation:

| raw field | 사용 위치 | 의미 |
|---|---|---|
| `obs["agentview_image"]` | primary RGB | 외부 카메라 이미지 |
| `obs["robot0_eye_in_hand_image"]` | wrist RGB | wrist 카메라 이미지 |
| `obs["robot0_eef_pos"]` | state | end-effector position |
| `obs["robot0_eef_quat"]` | state | quaternion, Euler로 변환 |
| `obs["robot0_gripper_qpos"]` | state | gripper width, `gripper_width=True`라서 사용 |
| task language | text | CLIP tokenizer input |

전처리 후 policy input:

| tensor | shape 개념 | 설명 |
|---|---|---|
| `image_x` | `[1, 1, 3, H, W]` | 현재 agentview image 전처리 결과 |
| `gripper` | `[1, 1, 3, H, W]` | 현재 wrist image 전처리 결과 |
| `state` | `[1, 1, 8]` | pos 3 + Euler 3 + gripper qpos 2 |
| `text_x` | `[1, 1, token_dim]` | task language token |

history queue:

- `img_queue`, `gripper_queue`, `state_queue`: 최근 `history_len=sequence_length=7` step 유지
- 초기 step에서 history가 부족하면 마지막 observation으로 padding
- `text_queue`: instruction은 episode 동안 동일하므로 처음 들어온 text를 sequence length만큼 반복

full Seer forward에 들어가는 최종 input:

| tensor | shape 개념 | baseline | ours full | ours skip scheduled full |
|---|---|---:|---:|---:|
| `input_image_primary` | `[1, 7, 3, H, W]` | 사용 | 사용 | 사용 |
| `input_image_wrist` | `[1, 7, 3, H, W]` | 사용 | 사용 | 사용 |
| `input_state` | `[1, 7, 8]` | 사용 | 사용 | 사용 |
| `input_text_token` | `[1, 7, token_dim]` | 사용 | 사용 | 사용 |
| dummy `action` | `[1, 7, 7]` zeros | 사용 | 사용 | 사용 |

## 3. Baseline Seer pipeline

Baseline 실행 설정:

```text
use_lrnode_latent_update = 0
lrnode_eval_skip_full_forward = 0
lrnode_query_interval = 1
```

Baseline은 매 environment step마다 full Seer forward를 실행한다.

### Baseline forward 내부

1. Text:
   - `text_token`을 CLIP text encoder에 넣는다.
   - CLIP encoder는 `torch.no_grad()`로 실행된다.
   - `text_projector`로 hidden dim에 맞춘다.

2. State:
   - `state[..., :6]`은 arm/eef state encoder로 처리
   - `state[..., 6:]`는 gripper width encoder로 처리
   - 두 feature를 concat 후 `state_projector`

3. Vision:
   - primary/wrist image를 MAE ViT encoder에 넣는다.
   - vision encoder도 `torch.no_grad()`로 실행된다.
   - CLS token과 patch token을 분리한다.
   - patch token은 Perceiver Resampler와 projector를 거쳐 image embedding이 된다.
   - CLS token은 별도 projector를 거쳐 cls embedding이 된다.

4. Transformer input:
   - `[text_embedding, state_embedding, image_embedding, image_cls_token_embedding]` concat
   - obs pred token 추가
   - action pred token 3개 추가
   - position embedding 추가
   - causal transformer backbone 실행

5. Action latent:
   - transformer output에서 action token 위치를 slice한다.
   - `action_latent_full` shape:

```text
[B, S, action_pred_steps, hidden_dim]
= [1, 7, 3, D]
```

6. Existing action head:
   - `action_decoder(action_latent_full)`
   - `arm_action_decoder`
   - `gripper_action_decoder`
   - 출력은 3-step action sequence

7. Eval action postprocess:
   - 현재 history 길이에 맞춰 selected step 선택
   - `eval_libero_ensembling=True`라서 multi-step action ensembling 적용
   - 최종 env action shape `[7]`

Baseline은 LR-NODE module을 생성하거나 사용하지 않는다. `return_action_latent=False`이므로 action latent는 외부 cache로 나오지 않는다.

## 4. Ours full-forward K=1 pipeline

Ours full-forward 실행 설정:

```text
use_lrnode_latent_update = 1
lrnode_eval_skip_full_forward = 0
lrnode_query_interval = 1
```

Ours full K=1은 매 step full Seer를 실행한다. 즉 compute path는 baseline full과 거의 같다.

차이점:

| 항목 | Baseline full | Ours full K=1 |
|---|---|---|
| checkpoint | Seer original ckpt37 | LR-NODE student ckpt35 |
| LR-NODE modules | 없음/비활성 | 모델에 존재 |
| full Seer forward | 매 step | 매 step |
| skip update | 없음 | 없음 |
| `return_action_latent` | False | True |
| cache update | 없음 | full forward 후 latent/image/state cache 저장 |

중요한 점:

- `lrnode_eval_skip_full_forward=0`이므로 LR-NODE dynamics는 action 생성에 사용되지 않는다.
- action은 여전히 기존 Seer action head에서 나온다.
- 다만 `use_lrnode_latent_update=1`이므로 full forward 결과에서 `action_latent`를 반환하고 cache를 갱신한다.
- 따라서 Ours full K=1은 "LR-NODE checkpoint 자체의 full-forward sanity"로 봐야 한다.

## 5. Ours skip-forward K>1 pipeline

Ours skip-forward 실행 설정:

```text
use_lrnode_latent_update = 1
lrnode_eval_skip_full_forward = 1
lrnode_query_interval = K
```

현재 sweep:

```text
K = 2, 3, 4, 5, 6, 8
```

### 5.1 Scheduled full-forward step

아래 조건에서는 full Seer를 실행한다.

```text
timestep % K == 0
or cache is empty
```

이때 입력은 baseline과 동일하다.

```text
input_image_primary: [1, 7, 3, H, W]
input_image_wrist:   [1, 7, 3, H, W]
input_state:         [1, 7, 8]
input_text_token:    [1, 7, token_dim]
dummy action:        [1, 7, 7]
```

full Seer 출력:

```text
arm_pred_action
gripper_pred_action
action_latent: [1, 7, 3, D]
```

선택되는 timestep:

- history가 아직 7보다 짧으면 `selected_step = num_step - 1`
- history가 충분하면 `selected_step = -1`

cache에 저장되는 값:

| cache | 값 |
|---|---|
| `lrnode_cached_latent` | `action_latent[:, selected_step]`, shape `[1, 3, D]` |
| `lrnode_cached_image_primary` | 현재 `image_x`, shape `[1, 1, 3, H, W]` |
| `lrnode_cached_image_wrist` | 현재 `gripper`, shape `[1, 1, 3, H, W]` |
| `lrnode_cached_state` | 현재 `state`, shape `[1, 1, 8]` |
| `lrnode_cached_age` | 0 |

### 5.2 Skipped step: LR-NODE update

full Seer를 실행하지 않는 step에서는 아래 값만 사용한다.

| input | 의미 |
|---|---|
| `z_prev` | cached latent `[1, 3, D]` |
| `key_image_primary` | cached primary image |
| `key_image_wrist` | cached wrist image |
| `cur_image_primary` | current primary image |
| `cur_image_wrist` | current wrist image |
| `q_key` | cached proprio state `[1, 1, 8]` |
| `q_cur` | current proprio state `[1, 1, 8]` |
| `age` | cache age + 1 |
| `dt` | 1.0 |

즉 skip step에서는 `[1, 7, ...]` history 전체를 transformer에 넣지 않는다. 현재 frame/state와 cached frame/state/latent만 사용한다.

### 5.3 FastVisualDeltaEncoder

각 카메라별 입력:

```text
[key_rgb, cur_rgb, cur_rgb - key_rgb]
```

채널 기준 concat 후 shape:

```text
[..., 9, H, W]
```

처리:

1. 64x64 resize
2. Conv2d 9->32, stride 2
3. Conv2d 32->64, stride 2
4. Conv2d 64->128, stride 2
5. AdaptiveAvgPool2d(1)
6. Linear 128->128
7. primary/wrist camera feature 평균
8. proprio가 있으면 `[q_key, q_cur, q_cur-q_key]`를 MLP로 투영해 더함
9. LayerNorm

출력:

```text
u_delta: [1, 1, 128] or broadcastable to [1, 3, 128]
```

### 5.4 ControlledLatentNODE

입력:

```text
z_prev:  [1, 3, D]
u_delta: [1, 128] or [1, 1, 128]
dt:      1.0
age:     cache age
```

동작:

```text
z_dyn = LayerNorm(z_prev) + action_token_embedding
dz = MLP([z_dyn, u_delta, time_emb(dt), time_emb(age)])
gate = sigmoid(MLP_gate([u_delta, time_emb(age)]) + gate_bias)
z_next = z_prev + gate * dt * dz
```

현재 설정:

| 항목 | 값 |
|---|---:|
| hidden dim | 256 |
| motion dim | 128 |
| gate init bias | -4.0 |
| post LayerNorm | off |
| adaptive ODE solver | 없음 |

### 5.5 Action decoding

`z_next`는 기존 Seer action head에 바로 들어간다.

```text
decode_action_from_latent(z_next)
```

이때 사용하는 head:

- `action_decoder`
- `arm_action_decoder`
- `gripper_action_decoder`

즉 Ours는 action head를 새로 만들지 않는다. 기존 Seer action head가 이해하는 latent 공간으로 `z_next`를 맞추는 방식이다.

### 5.6 Cache update

skip step 후 cache는 다음 값으로 교체된다.

| cache | 새 값 |
|---|---|
| latent | `z_next.detach()` |
| primary image | current image |
| wrist image | current wrist image |
| state | current state |
| age | previous age + 1 |

따라서 K=3이면 full step 사이에 LR-NODE update가 두 번 연속 일어난다.

## 6. Training pipeline 차이

Baseline training은 일반 Seer BC/obs prediction 학습이다.

Ours training은 full Seer forward를 그대로 수행한 뒤, 추가로 LR-NODE distillation loss를 붙인다.

Training input slicing:

| tensor | 입력 |
|---|---|
| `input_image_primary` | `images_primary[:, :sequence_length]` |
| `input_image_wrist` | `images_wrist[:, :sequence_length]` |
| `input_state` | `input_states[:, :sequence_length]` |
| `input_text_token` | `text_tokens[:, :sequence_length]` |
| `input_image_primary_next` | `images_primary[:, 1:sequence_length+1]` |
| `input_image_wrist_next` | `images_wrist[:, 1:sequence_length+1]` |
| `input_state_next` | `input_states[:, 1:sequence_length+1]` |
| `input_text_token_next` | `text_tokens[:, 1:sequence_length+1]` |

Latent target:

```python
teacher_target_mode = "shifted_context"
selected_step = lrnode_context_selected_step  # default -1

z_prev = action_latent(C_t)[:, selected_step]
z_teacher_next = action_latent(C_{t+1})[:, selected_step]
z_pred_next = LR_NODE(
    z_prev,
    image[selected_step] -> image[selected_step + 1],
    proprio[selected_step] -> proprio[selected_step + 1],
)
```

여기서 `C_t`는 policy가 env step `t`에서 받는 정상 context이고, `C_{t+1}`는 같은 policy 입력 형식을 한 env step 뒤로 민 context다. 즉 target은 같은 sequence window 내부의 `t -> t+1` token이 아니라, 다음 env step에서 full policy를 다시 호출했을 때의 action-interface latent다.

Legacy/ablation용으로는 다음 mode가 남아 있다.

```python
teacher_target_mode = "adjacent_sequence"
z_prev = action_latent_full[:, :-1]
z_teacher_next = action_latent_full[:, 1:]
```

현재 대표 run에서는 아래 flags가 모두 켜져 있다.

```text
lrnode_detach_input_latent = 1
lrnode_detach_teacher_latent = 1
lrnode_freeze_action_head_for_lrnode = 1
```

따라서 실제 학습 의도는 다음과 같다.

```python
z_prev = action_latent_full[:, :-1].detach()
z_teacher_next = action_latent_full[:, 1:].detach()
```

현재 새 protocol에서는 위 코드를 다음처럼 읽어야 한다.

```python
z_prev = action_latent(C_t)[:, selected_step].detach()
z_teacher_next = action_latent(C_{t+1})[:, selected_step].detach()
```

LR-NODE branch의 action head parameter도 임시 freeze된다. 단, `torch.no_grad()`로 action head 전체를 막는 것은 아니다. action head parameter만 freeze하고, gradient는 `action_distill_loss -> z_pred_next -> LR-NODE` 방향으로 흐르게 둔다. 이것이 기존 action head가 이해할 수 있는 latent를 LR-NODE가 만들게 하는 구조다.

Loss:

```text
loss_total = base_loss
           + lrnode_latent_weight * MSE(z_pred_next, z_teacher_next)
           + lrnode_action_distill_weight * L1(action_head(z_pred_next), teacher_action)
           + lrnode_smooth_weight * MSE(z_pred_next - z_prev, 0)
           + lrnode_bc_weight * BC loss
```

현재 대표 training config:

| 항목 | 값 |
|---|---:|
| `lrnode_latent_weight` | 0.05 |
| `lrnode_action_distill_weight` | 0.1 |
| `lrnode_smooth_weight` | 0.001 |
| `lrnode_bc_weight` | 0.0 |
| `lrnode_multistep_train` | 0 |
| `lrnode_train_max_horizon` | 2 |

주의:

- 현재 학습은 one-step latent update 중심이다.
- K=3 이상 eval은 one-step updater를 반복 적용하는 extrapolation 성격이 있다.
- 따라서 K가 커질수록 action smoothness/latent drift를 함께 봐야 한다.

## 7. Baseline vs Ours 차이 요약

| 항목 | Baseline full | Ours full K=1 | Ours skip K>1 |
|---|---|---|---|
| checkpoint | Seer original ckpt37 | LR-NODE ckpt35 | LR-NODE ckpt35 |
| full Seer 입력 | history `[1,7,...]` | history `[1,7,...]` | scheduled step만 history `[1,7,...]` |
| skip step 입력 | 없음 | 없음 | cached latent + current/key image/state |
| full transformer 호출 | 매 step | 매 step | K step마다 1회 |
| action latent cache | 없음 | 저장하지만 skip에 미사용 | 저장 후 skip update에 사용 |
| visual delta CNN | 없음 | 미사용 | 사용 |
| ControlledLatentNODE | 없음 | 미사용 | 사용 |
| action head | 기존 Seer head | 기존 Seer head | 기존 Seer head 재사용 |
| action ensembling | 사용 | 사용 | 사용 |
| 결과 해석 | original baseline | LR-NODE ckpt sanity | efficiency trade-off |

가장 중요한 차이:

- Ours full K=1은 baseline과 입력 pipeline이 거의 같고, checkpoint만 다르다.
- Ours skip K>1은 full Seer 입력을 매 step 넣지 않고, skip step에서 `cached z + image/proprio delta`만 넣는다.
- 이때 action head는 바뀌지 않는다. LR-NODE가 action head에 맞는 latent를 만들어야 한다.

## 8. 평가 용어 및 수식 정의

이 섹션은 아래 결과 표에서 쓰는 모든 축약어의 정확한 의미를 정의한다. 현재 metric의 source of truth는 각 run의 `analysis/eval_summary.json`, `analysis/eval_latency_profile.json`, `analysis/eval_episode_metrics.csv`다.

### 8.1 기본 표기

전체 평가 episode 집합을 \(E\), episode 수를 \(N = |E|\)라고 둔다. episode \(e\)의 policy step 수를 \(T_e\), 전체 policy step 수를 \(M\)이라고 두면:

```text
M = sum_{e in E} T_e
```

여기서 `Env steps`는 이 \(M\)이다. 즉 evaluation policy loop에서 실제 policy action을 낸 step 수이며, episode 시작 전에 수행하는 initial zero-action settle step은 포함하지 않는다.

각 step의 최종 environment action을 \(a_{e,t} \in R^7\)라고 둔다.

```text
a_{e,t} = [x, y, z, roll, pitch, yaw, gripper]
```

이 action은 Seer/LR-NODE action sequence에서 현재 step action을 고른 뒤, action ensembling과 gripper thresholding을 거쳐 `env.step(action)`에 실제로 들어간 값이다. gripper 차원은 최종적으로 `-1` 또는 `+1` 값이다.

### 8.2 Success rate 계열

Episode success indicator:

```text
s_e = 1 if episode e succeeds, else 0
```

Success Rate, `SR`:

```text
SR = (1 / N) * sum_{e in E} s_e
```

Task별 success rate는 task \(g\)에 속한 episode 집합 \(E_g\)에 대해 동일하게 계산한다.

```text
SR_g = (1 / |E_g|) * sum_{e in E_g} s_e
```

Baseline 대비 success 변화량, `Delta SR`:

```text
DeltaSR(run) = 100 * (SR_run - SR_baseline)  [percentage point]
```

성능 보존율, `Preservation`:

```text
Preservation(run) = 100 * SR_run / SR_baseline  [%]
```

예를 들어 baseline SR이 86.0%, run SR이 85.5%이면:

```text
DeltaSR = -0.5 percentage point
Preservation = 85.5 / 86.0 * 100 = 99.4%
```

### 8.3 Query 및 call count 계열

각 policy step의 mode를 \(m_{e,t}\)라고 둔다.

```text
m_{e,t} = full          if full Seer forward is executed
m_{e,t} = lrnode_update if cached latent is updated by LR-NODE
```

Full Seer 호출 수, `Full calls`:

```text
C_full = sum_{e,t} 1[m_{e,t} = full]
```

LR-NODE update 호출 수, `LR calls`:

```text
C_lr = sum_{e,t} 1[m_{e,t} = lrnode_update]
```

현재 skip eval에서는 정상적으로 fallback이 없으면:

```text
M = C_full + C_lr
```

Full-query reduction, `Full-query red.`:

```text
FullQueryReduction = 1 - C_full / M
```

Full step 비율과 LR step 비율:

```text
FullStepRatio = C_full / M
LRStepRatio   = C_lr / M
```

Effective query interval:

```text
EffectiveQueryInterval = M / C_full
```

이 값은 실제 episode 길이와 cache 초기화 때문에 CLI의 `lrnode_query_interval = K`와 아주 미세하게 다를 수 있다.

### 8.4 Hz 계열

평가에서 보고용 control frequency를 \(f_{ctrl}\)라고 둔다. 현재 실험은:

```text
f_ctrl = 20 Hz
```

Environment action rate, `Effective action Hz`:

```text
f_action = f_ctrl
```

즉 LR-NODE skip을 써도 environment에는 매 step action을 낸다.

Nominal full-query frequency, `Full Hz`:

```text
if skip mode is off:
    f_full = f_ctrl
else:
    f_full = f_ctrl / K
```

여기서 \(K =\) `lrnode_query_interval`이다.

Nominal LR-NODE update frequency, `LR Hz`:

```text
f_lr = max(0, f_ctrl - f_full)
```

예를 들어 \(K=3\)이면:

```text
f_full = 20 / 3 = 6.67 Hz
f_lr   = 20 - 6.67 = 13.33 Hz
```

주의할 점은 `Full Hz`와 `LR Hz`는 CLI interval 기반의 nominal 값이고, `Full-query red.`는 실제 call count \(C_full, C_lr, M\)에서 계산된 observed 값이다.

### 8.5 Latency 계열

각 step에서 policy action 생성 시간을 \(\tau^{policy}_{e,t}\)라고 둔다. 코드상 `total_policy_ms`는 `CustomModel.step()` 시작부터 action 반환 직전까지의 시간이다.

포함되는 것:

- image/text/state preprocessing
- full Seer forward 또는 LR-NODE update
- action latent decoding / action head
- action ensembling
- gripper thresholding 및 numpy action 변환

포함되지 않는 것:

- `env.step(action)` 시간
- episode 종료 후 mp4 encoding/write 시간

Policy step mean, `Policy mean`:

```text
PolicyMean = (1 / M) * sum_{e,t} tau_policy_{e,t}
```

Policy reduction, `Policy red.`:

```text
PolicyReduction(run) = 100 * (1 - PolicyMean_run / PolicyMean_baseline)
```

`Policy p95`는 모든 step의 raw p95가 아니다. 현재 `eval_latency_profile.json`은 episode별 평균 latency를 먼저 만들고, 그 episode 평균들의 percentile을 계산한다.

Episode-level policy mean:

```text
PolicyMean_e = (1 / T_e) * sum_t tau_policy_{e,t}
```

표의 `Policy p95`:

```text
PolicyP95 = percentile_95({PolicyMean_e | e in E})
```

Environment step 시간을 \(\tau^{env}_{e,t}\)라고 둔다. 코드상 `env_step_ms`는 `env.step(action)` 호출 시간만 측정한다.

포함되는 것:

- LIBERO/MuJoCo simulation step
- `env.step()` 내부에서 수행되는 observation 생성 및 rendering 비용

포함되지 않는 것:

- policy inference
- episode 종료 후 mp4 encoding/write 시간

Environment step mean, `Env mean`:

```text
EnvMean = (1 / M) * sum_{e,t} tau_env_{e,t}
```

`Env p95` 역시 raw step p95가 아니라 episode 평균들의 p95다.

```text
EnvMean_e = (1 / T_e) * sum_t tau_env_{e,t}
EnvP95    = percentile_95({EnvMean_e | e in E})
```

따라서 현재 표의 `Policy p95`, `Env p95`는 "episode-average latency의 p95"로 해석해야 한다.

### 8.6 Model call latency 계열

Full-call mean:

```text
FullCallMean = TotalFullForwardTime / C_full
```

여기서 `TotalFullForwardTime`은 full Seer `self.model(...)` 호출에 걸린 시간의 합이다. preprocessing과 `env.step()`은 포함하지 않는다.

LR-call mean:

```text
LRCallMean = TotalLRNodeUpdateTime / C_lr
```

LR-NODE skipped step에서 `LRCallMean`은 다음 세 블록을 포함한다.

```text
LRCall = FastDeltaEncoder + ControlledLatentNODE + ActionHeadDecode
```

Fast encoder:

```text
u_delta = lrnode_encode_delta(key_image, current_image, key_state, current_state)
```

NODE update:

```text
z_next = lrnode_apply_dynamics(z_prev, u_delta, dt=1.0, age=cache_age)
```

Action head:

```text
action_seq = decode_action_from_latent(z_next)
```

현재 debug logging에서는 action head timing 안에 hold-action 비교용 `decode_action_from_latent(z_prev)`도 함께 들어간다. 따라서 deployment에서 debug hold decode를 제거하면 action head 부분은 지금보다 약간 더 작아질 수 있다.

LR-call / full-call:

```text
LRCallRatio = 100 * LRCallMean / FullCallMean
```

현재 결과에서 이 값이 약 11%라는 것은 skipped step에서 full Seer 대신 LR-NODE를 쓰면 model call 1회 비용이 약 1/9 수준이라는 뜻이다.

### 8.7 Action smoothness 계열

Action delta는 step-to-step action 변화량이다.

```text
d_{e,0} = 0
d_{e,t} = a_{e,t} - a_{e,t-1},  t >= 1
```

Step action delta L2:

```text
DeltaL2_{e,t} = ||d_{e,t}||_2
```

Action jerk는 물리학의 시간 정규화된 jerk가 아니라, discrete action second-difference다.

```text
j_{e,0} = 0
j_{e,t} = d_{e,t} - d_{e,t-1},  t >= 1
```

따라서 \(t >= 2\)에서는:

```text
j_{e,t} = a_{e,t} - 2 * a_{e,t-1} + a_{e,t-2}
```

Step jerk L2:

```text
JerkL2_{e,t} = ||j_{e,t}||_2
```

중요한 해석:

- 현재 `JerkL2`는 7D action vector 전체에 대한 L2 norm이다.
- gripper 차원도 포함된다.
- \(dt\) 또는 \(f_{ctrl}^2\)로 나누거나 곱하지 않는다.
- 따라서 단위는 "action unit per step^2"에 가까우며, 물리 단위의 jerk가 아니다.
- robot trajectory의 고주파 변화/불연속성을 보기 위한 proxy metric이다.

Episode-level jerk p95:

```text
JerkP95_e = percentile_95({JerkL2_{e,t} | t = 0, ..., T_e - 1})
```

표의 `Mean episode jerk p95`는 모든 step을 합친 global p95가 아니다. 코드상 `eval_summary.json`의 `action_jerk_l2_p95`는 episode별 p95를 먼저 구한 뒤 평균낸 값이다.

```text
ReportedJerkP95 = (1 / N) * sum_{e in E} JerkP95_e
```

즉 문서의 `Mean episode jerk p95 = 0.1843`은 "전체 200 episodes 각각에서 jerk L2의 95th percentile을 구하고, 그 200개 값을 평균낸 값"이다.

Arm/trans/rot jerk:

```text
ArmJerk_{e,t}   = ||j_{e,t}[0:6]||_2
TransJerk_{e,t} = ||j_{e,t}[0:3]||_2
RotJerk_{e,t}   = ||j_{e,t}[3:6]||_2
```

Gripper switch:

```text
g_{e,t} = a_{e,t}[6]

Switch_{e,0} = 0
Switch_{e,t} = 1[g_{e,t} != g_{e,t-1}],  t >= 1
```

Episode-level gripper switch rate:

```text
SwitchRate_e = (1 / T_e) * sum_t Switch_{e,t}
```

Reported gripper switch rate:

```text
ReportedSwitchRate = (1 / N) * sum_{e in E} SwitchRate_e
```

### 8.8 해석상 주의

이 문서에서 efficiency는 세 층으로 분리해서 해석해야 한다.

1. Query efficiency:
   - full Seer 호출 수가 얼마나 줄었는가.
   - `Full-query red.`, `Full Hz`, `LR Hz`로 본다.

2. Policy/model efficiency:
   - policy가 action을 만드는 데 걸리는 시간이 얼마나 줄었는가.
   - `Policy mean`, `Policy p95`, `Full-call mean`, `LR-call mean`으로 본다.

3. Environment/eval wall-clock:
   - `env.step()` 및 simulation/rendering 시간이 포함된 환경 step 시간이 줄었는가.
   - `Env mean`, `Env p95`로 본다.
   - 현재 LIBERO video-on eval에서는 이 값이 줄지 않았으므로, 전체 evaluation runtime이 줄었다고 주장하면 안 된다.

## 9. 현재까지 완료된 결과

Comparison run은 K=8까지 완료됐다.

아래 표의 기준:

- `Policy mean/p95`: policy가 action을 만드는 데 걸린 시간이다. image preprocessing, full Seer 또는 LR-NODE update, action head, action postprocess를 포함한다.
- `Env mean/p95`: `env.step(action)` 호출 시간이다. simulation/rendering은 포함되지만, policy inference와 episode 종료 후 mp4 저장 시간은 포함되지 않는다.
- `Full-call mean`: full Seer forward 1회 평균 시간이다.
- `LR-call mean`: skipped step에서 cheap visual delta encoder + latent NODE update + existing action head를 실행한 1회 평균 시간이다.
- 모든 latency는 video 저장이 켜진 동일 조건에서 측정됐다. 단, per-step `Policy mean`에는 frame append 비용이 일부 들어가지만, episode 종료 후 mp4 encoding/write 시간은 포함되지 않는다.

### 9.1 전체 성능 및 효율

| Run | SR | Delta SR | Preservation | Env steps | Full calls | LR calls | Full-query red. | Full Hz | LR Hz | Policy mean | Policy red. | Policy p95 | Policy p95 red. | Env mean | Env red. | Env p95 | Mean ep. jerk p95 | Videos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline K=1 | 86.0% | +0.0%p | 100.0% | 65642 | 65642 | 0 | 0.0% | 20.00 | 0.00 | 79.2 ms | 0.0% | 85.6 ms | 0.0% | 307.4 ms | 0.0% | 440.8 ms | 0.0707 | 200 |
| Ours full K=1 | 83.0% | -3.0%p | 96.5% | 67708 | 67708 | 0 | 0.0% | 20.00 | 0.00 | 79.2 ms | 0.0% | 85.6 ms | 0.0% | 307.5 ms | -0.0% | 445.8 ms | 0.0859 | 200 |
| Ours skip K=2 | 85.5% | -0.5%p | 99.4% | 65999 | 33043 | 32956 | 49.9% | 10.00 | 10.00 | 49.7 ms | 37.3% | 53.5 ms | 37.5% | 312.9 ms | -1.8% | 452.7 ms | 0.1294 | 200 |
| Ours skip K=3 | 88.0% | +2.0%p | 102.3% | 63200 | 21122 | 42078 | 66.6% | 6.67 | 13.33 | 39.4 ms | 50.3% | 43.2 ms | 49.6% | 312.4 ms | -1.6% | 450.4 ms | 0.1843 | 200 |
| Ours skip K=4 | 86.5% | +0.5%p | 100.6% | 63353 | 15904 | 47449 | 74.9% | 5.00 | 15.00 | 34.7 ms | 56.3% | 38.5 ms | 55.0% | 314.7 ms | -2.4% | 450.5 ms | 0.4108 | 200 |
| Ours skip K=5 | 84.0% | -2.0%p | 97.7% | 64643 | 12991 | 51652 | 79.9% | 4.00 | 16.00 | 31.3 ms | 60.5% | 34.6 ms | 59.7% | 314.0 ms | -2.1% | 450.9 ms | 0.3923 | 200 |
| Ours skip K=6 | 81.0% | -5.0%p | 94.2% | 65096 | 10920 | 54176 | 83.2% | 3.33 | 16.67 | 29.4 ms | 63.0% | 32.2 ms | 62.4% | 315.9 ms | -2.8% | 450.4 ms | 0.4529 | 200 |
| Ours skip K=8 | 83.0% | -3.0%p | 96.5% | 64620 | 8120 | 56500 | 87.4% | 2.50 | 17.50 | 26.9 ms | 66.1% | 30.1 ms | 64.9% | 314.6 ms | -2.3% | 451.6 ms | 0.4599 | 200 |

### 9.2 모델 호출 비용 분해

| Run | Full-call mean | LR-call mean | LR-call / full-call | Fast encoder | NODE update | Action head |
|---|---:|---:|---:|---:|---:|---:|
| Baseline K=1 | 67.9 ms | 0.00 ms | - | 0.00 ms | 0.00 ms | 0.00 ms |
| Ours full K=1 | 67.9 ms | 0.00 ms | - | 0.00 ms | 0.00 ms | 0.00 ms |
| Ours skip K=2 | 68.6 ms | 7.67 ms | 11.2% | 3.20 ms | 1.90 ms | 1.24 ms |
| Ours skip K=3 | 68.4 ms | 7.52 ms | 11.0% | 3.13 ms | 1.87 ms | 1.22 ms |
| Ours skip K=4 | 69.2 ms | 7.62 ms | 11.0% | 3.16 ms | 1.90 ms | 1.25 ms |
| Ours skip K=5 | 68.8 ms | 7.51 ms | 10.9% | 3.12 ms | 1.87 ms | 1.22 ms |
| Ours skip K=6 | 69.0 ms | 7.53 ms | 10.9% | 3.13 ms | 1.87 ms | 1.23 ms |
| Ours skip K=8 | 69.4 ms | 7.56 ms | 10.9% | 3.14 ms | 1.88 ms | 1.23 ms |

핵심은 skipped step의 LR-NODE update가 full Seer forward 1회의 약 11% 비용이라는 점이다. 따라서 model-side compute 관점에서는 LR-NODE update가 충분히 싸다.

반면 `Env mean`은 baseline 307.4 ms에서 K=2~8이 312.4~315.9 ms로 줄지 않았다. 이 값은 `env.step()` 기준이며, 현재 LIBERO simulation/rendering 비용이 이 시간을 지배한다. 따라서 현재 결과에서 정확한 효율 주장은 두 층으로 나눠야 한다.

1. Model/policy-side efficiency:
   - K=2: policy mean 37.3% 감소, p95 37.5% 감소
   - K=3: policy mean 50.3% 감소, p95 49.6% 감소
   - K=4: policy mean 56.3% 감소, p95 55.0% 감소
   - K=5: policy mean 60.5% 감소, p95 59.7% 감소
   - K=6: policy mean 63.0% 감소, p95 62.4% 감소
   - K=8: policy mean 66.1% 감소, p95 64.9% 감소

2. End-to-end eval wall-clock:
   - env step mean은 감소하지 않았다.
   - 이는 policy가 빨라져도 environment simulation/rendering 비용이 per-step eval 시간을 지배하기 때문이다.
   - 별도의 mp4 encoding/write도 전체 job wall-clock에는 영향을 줄 수 있지만, 위 표의 `Policy mean`/`Env mean`에는 포함되지 않는다.
   - 실제 robot deployment처럼 policy inference가 control loop 병목인 환경에서는 policy-side reduction이 의미가 크지만, 현재 video-on LIBERO eval wall-clock만 놓고 보면 전체 실행 시간이 빨라졌다고 주장하면 안 된다.

### 9.3 Query frequency 관점

| Run | Full step 비율 | LR step 비율 | Effective full-query Hz | Effective LR update Hz |
|---|---:|---:|---:|---:|
| K=2 | 50.1% | 49.9% | 10.00 Hz | 10.00 Hz |
| K=3 | 33.4% | 66.6% | 6.67 Hz | 13.33 Hz |
| K=4 | 25.1% | 74.9% | 5.00 Hz | 15.00 Hz |
| K=5 | 20.1% | 79.9% | 4.00 Hz | 16.00 Hz |
| K=6 | 16.8% | 83.2% | 3.33 Hz | 16.67 Hz |
| K=8 | 12.6% | 87.4% | 2.50 Hz | 17.50 Hz |

이 표가 논문에서 가장 직접적인 efficiency metric이다. LR-NODE는 environment action rate는 20 Hz로 유지하면서 expensive full Seer query frequency만 낮춘다.

## 10. 결과 분석

### 10.1 Efficiency 해석

정확한 효율 결론은 다음이다.

```text
LR-NODE는 full Seer 호출 수와 policy inference latency를 크게 줄인다.
하지만 현재 LIBERO eval의 measured env-step time은 simulation/rendering이 지배하므로 줄지 않는다.
```

즉 발표에서 `latency reduction`이라고만 쓰면 안 되고, 반드시 `policy inference latency reduction` 또는 `full Seer query reduction`이라고 써야 한다.

가장 방어적인 setting은 K=2다.

- SR: 85.5%, baseline 86.0% 대비 -0.5%p
- Performance preservation: 99.4%
- Full Seer call reduction: 49.9%
- Policy mean latency reduction: 37.3%
- Policy p95 latency reduction: 37.5%
- Env step mean: baseline보다 1.8% 느림
- mean episode jerk p95: baseline 대비 1.83배

K=3은 현재까지 가장 좋은 success/model-efficiency frontier를 보인다.

- SR: 88.0%, baseline 대비 +2.0%p
- Full Seer call reduction: 66.6%
- Policy mean latency reduction: 50.3%
- Policy p95 latency reduction: 49.6%
- Full query frequency: 20 Hz에서 6.67 Hz로 감소
- 단, mean episode jerk p95가 baseline 대비 2.61배다.

K=4는 SR이 86.5%로 baseline 수준을 유지하면서 call reduction 74.9%를 달성했다.

- Policy mean latency reduction: 56.3%
- Policy p95 latency reduction: 55.0%
- 하지만 mean episode jerk p95가 baseline 대비 5.81배까지 증가한다.
- 정량 SR은 좋지만 action trajectory 품질 측면에서는 aggressive setting이다.

K=5는 model-side 효율은 가장 좋지만 SR이 84.0%로 내려간다.

- Full Seer call reduction: 79.9%
- Policy mean latency reduction: 60.5%
- Policy p95 latency reduction: 59.7%
- SR은 baseline 대비 -2.0%p
- task 6/8/9 같은 취약 task에서 하락이 뚜렷하다.

K=6과 K=8은 효율만 보면 가장 크지만, 성공률과 action smoothness 측면에서 대표 결과로 쓰기 어렵다.

- K=6: SR 81.0%, baseline 대비 -5.0%p, full-query reduction 83.2%, policy mean latency reduction 63.0%
- K=8: SR 83.0%, baseline 대비 -3.0%p, full-query reduction 87.4%, policy mean latency reduction 66.1%
- mean episode jerk p95는 K=6에서 0.4529, K=8에서 0.4599로 baseline 대비 약 6.4~6.5배다.
- 즉 K=6/K=8은 aggressive upper-bound efficiency setting이지, 현재 checkpoint의 main setting으로 쓰면 안 된다.

### 10.2 Task별 패턴

Task별 성공률:

| Task | Baseline | Ours full | K=2 | K=3 | K=4 | K=5 | K=6 | K=8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90% | 95% | 90% | 90% | 90% | 95% | 90% | 100% |
| 1 | 75% | 100% | 95% | 100% | 100% | 90% | 100% | 95% |
| 2 | 95% | 100% | 85% | 100% | 100% | 100% | 95% | 90% |
| 3 | 80% | 95% | 100% | 100% | 100% | 100% | 100% | 95% |
| 4 | 75% | 85% | 80% | 90% | 85% | 85% | 80% | 70% |
| 5 | 100% | 85% | 85% | 100% | 100% | 100% | 95% | 95% |
| 6 | 90% | 70% | 80% | 70% | 70% | 65% | 65% | 70% |
| 7 | 100% | 95% | 95% | 100% | 100% | 100% | 100% | 100% |
| 8 | 75% | 45% | 75% | 60% | 60% | 45% | 30% | 55% |
| 9 | 80% | 60% | 70% | 70% | 60% | 60% | 55% | 60% |

주요 관찰:

- Ours full은 task 6/8/9에서 baseline보다 약하다. 즉 LR-NODE skip 이전에 checkpoint 자체가 특정 task에서 약한 편이다.
- K=2는 task 8을 45%에서 75%로 회복한다. 이 때문에 ours full 83.0%보다 K=2가 85.5%로 올라간다.
- K=3은 task 1/2/3/5/7에서 100%를 기록하며 전체 SR 88.0%가 나온다.
- K=4도 전체 SR은 유지되지만 task 8/9는 60%로 낮다.
- K=5 이후는 task 6/8/9가 더 취약해진다.
- K=6은 task 8이 30%까지 떨어져 전체 SR 81.0%가 된다.
- K=8은 task 0/1/3/5/7은 강하지만 task 4/6/8/9가 낮아 전체 SR 83.0%에 머문다.

### 10.3 Smoothness trade-off

K가 커질수록 action은 더 거칠어진다.

| Run | Mean episode jerk p95 | Gripper switch rate |
|---|---:|---:|
| Baseline | 0.0707 | 0.0108 |
| Ours full | 0.0859 | 0.0117 |
| K=2 | 0.1294 | 0.0140 |
| K=3 | 0.1843 | 0.0155 |
| K=4 | 0.4108 | 0.0179 |
| K=5 | 0.3923 | 0.0186 |
| K=6 | 0.4529 | 0.0183 |
| K=8 | 0.4599 | 0.0185 |

이 결과는 LR-NODE가 success rate는 유지할 수 있지만, K가 커질수록 action trajectory의 고주파 변화가 증가한다는 뜻이다. 특히 K=4 이후는 success만 보면 괜찮아 보여도 실제 로봇 적용이나 real-world transfer에서는 위험할 수 있다.

### 10.4 현재 결론

가장 안전한 main claim:

```text
K=2에서 full Seer query를 약 50% 줄이고 policy inference latency를 약 37% 줄이면서,
LIBERO-10 success rate를 거의 보존한다.
```

가장 좋은 efficiency frontier point:

```text
K=3에서 full Seer query를 약 66.6%, policy inference latency를 약 50% 줄이고,
현재 run에서는 baseline보다 높은 SR을 보인다.
```

공격적인 setting:

```text
K=4 이상은 full-query/policy-latency reduction은 크지만 jerk가 크게 증가한다.
```

현재 checkpoint 기준으로 K=6/K=8은 efficiency upper bound로만 보여주는 것이 맞다. 단, 현재 video-on LIBERO eval의 end-to-end wall-clock은 줄지 않았으므로 `overall runtime reduction`이라고 쓰면 안 된다. 발표/논문용으로는 K=2를 main table의 대표 설정으로 두고, K=3을 best frontier point로 강조하며, K=4 이상은 aggressive ablation으로 분리하는 구성이 가장 방어적이다.

## 11. 앞으로 문서 저장 위치

이 문서부터 Codex가 생성하는 분석 markdown은 아래 디렉터리에 저장한다.

```text
codex_output/
```

이전에 생성했던 markdown도 아래로 이동했다.

```text
codex_output/etri_lrnode_ppt_summary.md
codex_output/lrnode_experiment_brief_for_chatgpt.md
codex_output/ours_vs_baseline_pipeline_analysis.md
```
