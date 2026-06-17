# LR-NODE Training Protocol Definitions

이 문서는 `scripts/LIBERO_LONG/Seer/{scratch_node.sh,distill_node.sh,scratch_node_joint.sh}`의 의미를 코드/gradient 기준으로 고정하기 위한 정의서다.

## 핵심 정정

baseline인 `sd1_libero_10_100pc_original_settings_20260304`는 기존 Seer를 from scratch로 학습한 결과다. 따라서 LR-NODE 연구의 main comparison은 기존 baseline checkpoint에 adapter를 붙이는 것이 아니라, **Seer baseline from scratch vs LR-NODE method from scratch**가 맞다.

다만 `distill_node.sh`는 main comparison이 아니라, 이미 학습된 Seer를 고정했을 때 LR-NODE updater만 따로 학습 가능한지 보는 isolation/distill 실험이다.

## 용어 정의

### scratch

`scratch`는 pretrained Seer policy checkpoint를 load하지 않는다는 뜻이다.

수식적으로는 초기 parameter가 다음과 같다.

```text
theta_Seer, theta_head, theta_LRNode ~ init(seed)
```

여기서 `theta_LRNode`가 없는 경우가 baseline Seer scratch이고, 있는 경우가 LR-NODE scratch다.

### distill

`distill`은 이미 학습된 Seer checkpoint를 teacher로 load하고, 기존 Seer/action head를 freeze한 뒤 LR-NODE updater만 학습한다는 뜻이다.

```text
theta_Seer, theta_head <- checkpoint
theta_LRNode ~ init(seed)
```

현재 `distill_node.sh`는 `theta_Seer`, `theta_head`를 freeze하고 `theta_LRNode`만 학습한다. 따라서 Seer policy를 fine-tune하는 실험이 아니라, frozen teacher에서 LR-NODE updater를 distill하는 isolation experiment다.

### teacher target mode

현재 기본 target은 `shifted_context`다.

```text
C_t       = policy가 env step t에서 정상적으로 받는 context
C_{t+1}   = policy가 env step t+1에서 정상적으로 받는 context
z_t^T     = Probe(TeacherPolicy(C_t))
z_{t+1}^T = Probe(TeacherPolicy(C_{t+1}))
z_hat     = LRNode(z_t^T, Delta(C_t, C_{t+1}))
```

Seer에서는 `C_t = window[0:S]`, `C_{t+1} = window[1:S+1]`로 구현된다. 기본 probe 위치는 `lrnode_context_selected_step=-1`이며, steady-state eval에서 cache되는 마지막 context latent와 맞춘다.

legacy 비교용으로 `adjacent_sequence`도 남아 있다.

```text
z_prev = z_full[:, :-1]
z_teacher = z_full[:, 1:]
```

하지만 이 방식은 같은 policy context 안의 token transition을 target으로 삼으므로, 실제 eval에서 "다음 env step의 full policy latent"를 근사한다는 해석에는 `shifted_context`가 더 정확하다.

### teacher-student detached

LR-NODE가 teacher Seer latent/action을 따라가도록 학습하지만, LR-NODE loss가 Seer backbone/action head로 역전파되지 않는 설정이다.

```text
z_prev = stopgrad(z_t^T)
z_teacher = stopgrad(z_{t+1}^T)
z_pred = LRNode(z_prev, delta_obs)

L_LRNode = MSE(z_pred, z_teacher) + action_distill + smooth
```

gradient:

```text
d L_LRNode / d theta_LRNode != 0
d L_LRNode / d theta_Seer = 0
d L_LRNode / d theta_head = 0
```

단, 같은 training run 안에서 base Seer loss는 여전히 Seer/action head를 학습한다.

```text
d L_base / d theta_Seer != 0
d L_base / d theta_head != 0
```

### coupled joint

LR-NODE loss가 LR-NODE module뿐 아니라 Seer latent path/action head에도 영향을 줄 수 있게 여는 설정이다.

현재 script 기준:

```text
z_prev = z_t^T                         # detach 안 함
z_teacher = stopgrad(z_{t+1}^T)        # teacher target은 detach 유지
action_head_for_lrnode = trainable     # LR-NODE branch에서 action head freeze 안 함
```

gradient:

```text
d L_LRNode / d theta_LRNode != 0
d L_LRNode / d theta_Seer may be != 0
d L_LRNode / d theta_head may be != 0
```

이 설정은 teacher-student purity는 낮지만, 사용자가 말한 의미의 “NODE loss가 원본 Seer 학습에도 같이 고려되는 joint”에 해당한다.

## Script Definitions

### `scratch_node.sh`

정의: **from scratch LR-NODE teacher-student detached**

목적:

```text
baseline Seer와 같은 scratch 조건에서 LR-NODE module도 같이 학습하되,
LR-NODE loss는 Seer backbone/action head를 직접 바꾸지 않는다.
```

주요 flag:

```text
--use_lrnode_latent_update 1
--lrnode_train_latent_distill 1
--lrnode_teacher_target_mode shifted_context
--lrnode_context_selected_step -1
--lrnode_train_protocol joint
--lrnode_detach_input_latent 1
--lrnode_detach_teacher_latent 1
--lrnode_freeze_action_head_for_lrnode 1
--loss_image
--loss_action
```

이름에 `_node`가 붙는 이유:

```text
LR-NODE module과 LR-NODE distillation loss를 학습하기 때문이다.
```

### `distill_node.sh`

정의: **frozen-baseline LR-NODE adapter**

목적:

```text
이미 학습된 Seer baseline checkpoint를 teacher로 고정하고,
LR-NODE module만 학습한다.
```

주요 flag:

```text
--finetune_from_pretrained_ckpt ${BASELINE_CKPT}
--use_lrnode_latent_update 1
--lrnode_train_latent_distill 1
--lrnode_teacher_target_mode shifted_context
--lrnode_context_selected_step -1
--lrnode_train_protocol adapter
--lrnode_freeze_seer_for_adapter 1
--lrnode_assert_only_lrnode_trainable 1
--lrnode_detach_input_latent 1
--lrnode_detach_teacher_latent 1
--lrnode_freeze_action_head_for_lrnode 1
```

이 실험은 main comparison이 아니라 isolation experiment다.

### `scratch_node_joint.sh`

정의: **from scratch LR-NODE coupled joint**

목적:

```text
scratch_node.sh와 달리 LR-NODE loss가 Seer/action head에도 영향을 줄 수 있게 하여,
NODE auxiliary objective를 원본 Seer 학습에 직접 결합한다.
```

주요 flag:

```text
--use_lrnode_latent_update 1
--lrnode_train_latent_distill 1
--lrnode_teacher_target_mode shifted_context
--lrnode_context_selected_step -1
--lrnode_train_protocol joint
--lrnode_detach_input_latent 0
--lrnode_detach_teacher_latent 1
--lrnode_freeze_action_head_for_lrnode 0
--loss_image
--loss_action
```

## 올바른 비교

main research comparison:

```text
Seer baseline scratch
vs
scratch_node.sh
vs
scratch_node_joint.sh
```

여기서 baseline은 기존 `seer_main`의 `sd1_libero_10_100pc_original_settings_20260304`처럼 from scratch로 학습된 Seer다.

adapter/isolation comparison:

```text
same baseline ckpt full-forward
vs
distill_node.sh LR-NODE skip K=2,3,4,...
```

이 비교는 “기존 Seer를 고정했을 때 LR-NODE updater만으로 query를 줄일 수 있는가”를 확인하는 보조 실험이다.

## K=1/full-forward 해석

`eval_skip_full_forward=false` 또는 실질적으로 LR-NODE update call이 0이면 action은 full Seer forward에서 나온다. 따라서 같은 checkpoint와 같은 eval code를 쓰면 결과는 같아야 한다.

하지만 서로 다른 scratch run의 checkpoint는 같은 epoch 번호라도 bitwise identical하다고 가정하면 안 된다. 실제 archived old ours와 `seer_main` baseline은 공통 Seer/action-head tensor가 다르므로, `30-30`, `31-31`, ..., `37-37`의 success rate가 반드시 같아야 한다고 볼 수 없다.
