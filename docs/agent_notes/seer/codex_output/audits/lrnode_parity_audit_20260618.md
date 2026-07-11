# LR-NODE parity 감사 2026-06-18

## 목적

현재 repository에서 `scratch.sh` baseline과 `scratch_node.sh` teacher-student LR-NODE가 공정하게 비교 가능한지 확인했다.
핵심 기준은 다음이다.

1. LR-NODE를 켜도 공통 Seer/action-head 초기 파라미터가 baseline과 동일해야 한다.
2. detached teacher-student 설정에서는 LR-NODE loss가 공통 Seer/action-head gradient를 바꾸지 않아야 한다.
3. LR-NODE 파라미터가 추가되어도 공통 gradient clipping/update가 baseline과 섞이지 않아야 한다.
4. eval K=1 또는 `lrnode_eval_skip_full_forward=0`에서는 LR-NODE skip path가 action을 바꾸지 않아야 한다.

## 확인 및 수정 사항

### 1. LR-NODE 모듈 초기화 RNG 보존

파일: `models/seer_model.py`

문제:
`initialize_weights()` 이후 `_build_lrnode_modules()`가 호출되면 LR-NODE 파라미터 초기화가 torch RNG를 소비한다.
공통 Seer 파라미터는 이미 초기화된 뒤라 직접 바뀌지 않지만, 그 뒤에 이어지는 CLIP/dataset setup의 RNG 상태를 baseline과 다르게 만들 수 있다.

수정:
`_build_lrnode_modules_preserving_rng()`를 추가해 LR-NODE 모듈 생성 전후의 `torch`, `numpy`, `python random` RNG 상태를 복원한다.
LR-NODE 모듈은 `.to(device)` 전에 CPU에서 생성되므로 CUDA RNG는 건드리지 않는다.

결과:
같은 seed에서 baseline/LR-NODE 모델 생성 후 공통 state_dict와 RNG digest가 동일하다.

검증 출력:

```text
COMMON_TENSORS 380
TORCH_EQUAL_UNEQUAL_COUNT 0
RNG_EQUAL True
```

### 2. Detached teacher-student gradient parity

설정:

```text
lrnode_detach_input_latent=1
lrnode_detach_teacher_latent=1
lrnode_freeze_action_head_for_lrnode=1
```

검증:
작은 더미 모델과 동일 입력에서 baseline은 base loss만 backward하고,
LR-NODE 모델은 base loss + LR-NODE latent/action/smooth loss를 backward했다.

결과:
공통 Seer/action-head gradient는 완전히 동일하고, LR-NODE 모듈에만 별도 gradient가 생긴다.

검증 출력:

```text
BASE_LOSS 1.5986560583114624 1.5986560583114624
COMMON_GRAD_TENSORS 110
COMMON_GRAD_DIFF_COUNT 0
COMMON_GRAD_MAX_ABS 0.0
COMMON_GRAD_MISSING 0
LRNODE_NONZERO_GRAD_TENSORS 28
```

해석:
현재 detached `scratch_node.sh` 프로토콜에서 LR-NODE loss 자체는 공통 Seer/action-head gradient를 바꾸지 않는다.

### 3. Gradient clipping 분리

파일: `utils/train_utils.py`

문제:
LR-NODE loss가 공통 gradient로 흐르지 않더라도, 기존 global `clip_grad_norm_(model.parameters(), 0.1)`는 LR-NODE gradient까지 total norm에 포함한다.
그러면 공통 Seer gradient가 baseline보다 더 작게 scaling될 수 있다.

수정:
detached teacher-student 설정에서는 non-LR-NODE 파라미터와 LR-NODE 파라미터를 분리해서 clip한다.
coupled joint 설정은 의도적으로 shared-gradient 실험이므로 global clipping을 유지한다.

### 4. Eval skip path 확인

파일: `utils/eval_utils_libero.py`

skip 조건:

```python
use_lrnode_latent_update
and lrnode_eval_skip_full_forward
and lrnode_cached_latent is not None
and timestep % lrnode_query_interval != 0
```

결론:

- `lrnode_eval_skip_full_forward=0`이면 LR-NODE update path를 타지 않는다.
- `lrnode_query_interval=1`이면 `timestep % 1 != 0`이 항상 false라 LR-NODE update call이 없어야 한다.
- 따라서 K=1 full-forward 결과가 baseline과 다르면 skip path 때문이 아니라 checkpoint weight, eval 설정, 또는 checkpoint load 대상 차이를 먼저 봐야 한다.

### 5. Compile 및 shell syntax

검증 통과:

```text
python -m py_compile models/seer_model.py utils/train_utils.py train.py eval_libero.py utils/eval_utils_libero.py
bash -n scripts/LIBERO_LONG/Seer/scratch.sh scripts/LIBERO_LONG/Seer/scratch_node.sh scripts/LIBERO_LONG/Seer/scratch_node_joint.sh scripts/LIBERO_LONG/Seer/distill_node.sh scripts/LIBERO_LONG/Seer/eval.sh scripts/LIBERO_LONG/Seer/eval_node.sh scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_joint_compare.sh
```

## 현재 실험 해석 기준

### `scratch.sh`

순수 Seer baseline scratch 학습이다.
LR-NODE 모듈과 LR-NODE loss가 없다.

### `scratch_node.sh`

Seer를 scratch로 학습하면서 LR-NODE도 같이 학습한다.
다만 기본 설정은 detached teacher-student다.
즉 base Seer loss는 Seer/action-head를 학습하고, LR-NODE loss는 LR-NODE 모듈만 학습한다.
이번 audit 이후에는 초기화 RNG와 gradient clipping도 baseline parity를 깨지 않도록 보정되어 있다.

### `scratch_node_joint.sh`

Seer를 scratch로 학습하면서 LR-NODE loss가 공통 Seer/action-head에도 영향을 줄 수 있는 coupled joint 실험이다.
이건 baseline parity 실험이 아니라 auxiliary loss coupling 자체를 보는 별도 ablation이다.

### `distill_node.sh`

완료된 `scratch.sh` baseline checkpoint를 load하고 non-LR-NODE 모듈을 freeze한다.
오직 LR-NODE 모듈만 학습한다.
baseline best checkpoint가 정해진 뒤 수행하는 isolation/distill 실험이다.

## 중요한 주의

이 audit 이전에 시작된 `scratch_node.sh` run은 strict parity run으로 주장하면 안 된다.
이유는 당시 global gradient clipping이 LR-NODE gradient까지 포함했기 때문이다.
공통 gradient 자체는 분리되어도 clipping scale이 달라질 수 있었다.

strict detached teacher-student 비교는 이 audit 이후 코드로 새로 시작한 run만 사용해야 한다.

## 20260618_143600 run 재검증 및 폐기

대상 run:

```text
runs_lrnode_protocol_20260616/train/scratch_node/
  sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260618_143600
```

확인 결과:

- run snapshot 생성 시각: `2026-06-18 14:36:19 +0900`
- `models/seer_model.py` RNG 보존 fix 시각: `2026-06-18 14:32:36 +0900`
- `utils/train_utils.py` clipping 분리 fix 시각: `2026-06-18 14:24:32 +0900`

따라서 이 run은 초기화 RNG 보존과 gradient clipping 분리 fix 이후 코드로 시작되었다.

이 run의 LR-NODE flags:

```text
use_lrnode_latent_update=1
lrnode_train_latent_distill=1
lrnode_teacher_target_mode=shifted_context
lrnode_detach_input_latent=1
lrnode_detach_teacher_latent=1
lrnode_freeze_action_head_for_lrnode=1
lrnode_latent_weight=0.05
lrnode_action_distill_weight=0.1
lrnode_smooth_weight=0.001
lrnode_bc_weight=0.0
```

이 run과 동일한 실제 architecture config로 초기화 parity는 통과했다.

```text
sequence_length=7
num_resampler_query=6
num_obs_token_per_image=9
action_pred_steps=3
transformer_layers=24
hidden_dim=384
transformer_heads=12
calvin_input_image_size=224
patch_size=16
```

검증 출력:

```text
EXACT_CONFIG_COMMON_TENSORS 656
EXACT_CONFIG_MISSING_OR_EXTRA 0
EXACT_CONFIG_UNEQUAL_COMMON 0
EXACT_CONFIG_RNG_EQUAL True
```

이 run의 W&B local summary에서도 detached teacher-student clip 분리는 실제로 기록되어 있었다.

```text
train/grad_clip/separate_lrnode_clip=1.0
train/grad_clip/non_lrnode_norm=0.07779133319854736
train/grad_clip/lrnode_norm=0.0006388912443071604
```

이 run의 LR-NODE debug artifact도 `gs_000000`, `gs_001000`, `gs_002000`, `gs_003000`에서 non-finite 값은 없었다.

```text
gs_000000_summary.json nonfinite=0
gs_001000_summary.json nonfinite=0
gs_002000_summary.json nonfinite=0
gs_003000_summary.json nonfinite=0
```

하지만 loss parity를 추가로 비교하면서 별도 문제가 확인되었다.

문제:
`lrnode_teacher_target_mode=shifted_context`에서는 training step 내부에서 teacher target 생성을 위해 no-grad full forward가 main forward 전에 한 번 더 실행된다.
`torch.no_grad()`는 gradient만 끄며 dropout/random stream 소비를 막지 않는다.
따라서 teacher forward가 dropout RNG를 먼저 소비하고, 그 결과 main forward의 dropout mask가 baseline `scratch.sh`와 달라진다.
이 경우 LR-NODE loss가 detached되어도 base action/image loss 자체가 baseline과 달라질 수 있다.

재현 출력:

```text
baseline_total arm grip [0.6068981289863586, 0.5983383059501648, 0.8559852242469788]
ours_no_rng_restore_total arm grip [0.5998631715774536, 0.5920810699462891, 0.7782087922096252]
diff_no_restore [0.007034957408905029, 0.006257236003875732, 0.07777643203735352]
```

수정:
`utils/train_utils.py`에 `_preserve_torch_rng()`를 추가하고, shifted-context teacher forward를 이 context 안에서 실행하도록 변경했다.
teacher forward가 CPU/CUDA torch RNG를 소비하더라도 main forward 전에 원래 RNG state로 복원된다.

수정 후 재현 출력:

```text
baseline_loss [0.6068981289863586, 0.5983383059501648, 0.8559852242469788]
ours_loss_after_patch_path [0.6068981289863586, 0.5983383059501648, 0.8559852242469788]
loss_absdiff [0.0, 0.0, 0.0]
```

조치:
`20260618_143600` run은 loss parity bug가 있는 코드로 시작된 run이므로 strict parity 결과로 사용할 수 없다.
해당 실행 프로세스와 partial output은 중지 및 삭제했다.

결론:
strict detached teacher-student 비교는 `_preserve_torch_rng()` patch 이후 새로 시작한 `scratch_node.sh` run만 사용해야 한다.

## 2026-06-19 전체 재점검

사용자 요청에 따라 static code/script 확인과 재현 가능한 parity test를 다시 수행했다.
이번 확인 범위는 다음이다.

- `models/seer_model.py`
- `models/lrnode_modules.py`
- `utils/train_utils.py`
- `train.py`
- `eval_libero.py`
- `utils/eval_utils_libero.py`
- `utils/data_utils.py`
- `utils/arguments_utils.py`
- `scripts/LIBERO_LONG/Seer/*.sh` 중 baseline/LR-NODE train/eval script

### 추가한 재현 스크립트

파일:

```text
scripts/debug/check_lrnode_parity.py
```

목적:
실제 CLIP/ViT checkpoint, LIBERO, CUDA, dataloader에 의존하지 않고 Seer common path parity만 분리 검증한다.
dummy frozen encoder를 사용하지만, Seer transformer/action head/image decoder/LR-NODE loss path는 실제 코드 경로를 사용한다.

검증 항목:

1. LR-NODE on/off constructor 이후 common `state_dict` equality
2. LR-NODE 모듈 생성 후 torch RNG state equality
3. shifted-context teacher target forward 포함 train main output equality
4. base action/image loss equality
5. detached teacher-student LR-NODE loss 추가 후 common gradient equality
6. separate clipping 후 AdamW 1-step common parameter update equality
7. LR-NODE module gradient가 실제로 생기는지
8. eval full-forward action/latent equality

실행:

```text
$SEER_PYTHON scripts/debug/check_lrnode_parity.py
```

결과:

```text
init_parity:
  common_tensor_count: 392
  unequal_common_tensor_count: 0
  rng_equal_after_constructor: true

train_shifted_teacher_parity:
  base_loss: [0.41673657298088074, 0.30160078406333923, 0.713771641254425, 1.0799806118011475]
  ours_base_loss: [0.41673657298088074, 0.30160078406333923, 0.713771641254425, 1.0799806118011475]
  base_loss_absdiff: 0.0
  main_output_max_absdiff:
    arm: 0.0
    gripper: 0.0
    image: 0.0
    latent: 0.0
  common_grad_checked: 122
  common_grad_diff_count: 0
  common_grad_max_absdiff: 0.0
  common_param_checked_after_step: 392
  common_param_diff_count_after_step: 0
  common_param_max_absdiff_after_step: 0.0
  lrnode_nonzero_grad_tensor_count: 28

eval_full_forward_parity:
  eval_arm_max_absdiff: 0.0
  eval_gripper_max_absdiff: 0.0
  eval_latent_max_absdiff: 0.0
```

해석:

- 현재 코드 기준으로 `scratch_node.sh`의 기본 detached teacher-student 설정은 LR-NODE loss가 common Seer/action-head의 loss, gradient, clipping 후 update를 바꾸지 않는다.
- LR-NODE module에는 gradient가 실제로 생긴다. 즉 LR-NODE는 학습되고, common Seer path만 보존된다.
- eval에서 LR-NODE enabled라도 skip을 쓰지 않는 full-forward path는 baseline full-forward와 action/latent가 동일하다.

### 현재 실행/결과 상태

확인 시점에 실행 중인 Seer 관련 작업:

```text
bash scripts/LIBERO_LONG/Seer/eval.sh
python -m torch.distributed.run ... eval_libero.py ... use_lrnode_latent_update 0 ... ckpt 38
```

즉 실행 중인 것은 baseline eval이며, `scratch_node.sh` 학습 프로세스는 남아 있지 않다.

현재 train protocol 디렉터리:

```text
runs_lrnode_protocol_20260616/train/scratch/
  sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040

runs_lrnode_protocol_20260616/train/scratch_node/
  wandb/
```

`scratch_node`의 invalid partial run은 삭제되어 남아 있지 않다.
baseline scratch 학습 결과는 유지되어 있다.

### shell/python syntax 확인

통과:

```text
$SEER_PYTHON -m py_compile \
  scripts/debug/check_lrnode_parity.py \
  utils/train_utils.py \
  models/seer_model.py \
  models/lrnode_modules.py \
  train.py \
  eval_libero.py \
  utils/eval_utils_libero.py

bash -n \
  scripts/LIBERO_LONG/Seer/scratch.sh \
  scripts/LIBERO_LONG/Seer/scratch_node.sh \
  scripts/LIBERO_LONG/Seer/scratch_node_joint.sh \
  scripts/LIBERO_LONG/Seer/distill_node.sh \
  scripts/LIBERO_LONG/Seer/finetune_node.sh \
  scripts/LIBERO_LONG/Seer/eval.sh \
  scripts/LIBERO_LONG/Seer/eval_node.sh \
  scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh \
  scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh \
  scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_joint_compare.sh
```

### 최종 판단

현재 코드로 새로 시작하는 `scratch_node.sh` run은 strict detached teacher-student LR-NODE 비교용으로 다시 사용할 수 있다.
이전 invalid `scratch_node` run은 사용하면 안 된다.
`scratch_node_joint.sh`는 애초에 common weight가 달라질 수 있는 coupled ablation이므로 baseline parity를 기대하면 안 된다.
