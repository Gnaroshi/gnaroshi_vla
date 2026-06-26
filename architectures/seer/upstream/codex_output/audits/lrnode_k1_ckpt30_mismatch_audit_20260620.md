# LR-NODE K=1 ckpt30 불일치 감사

작성일: 2026-06-20

## 질문

`eval_node.sh`에서 `K=1`이면 LR-NODE skip path를 쓰지 않으므로, baseline ckpt30과 scratch_node ckpt30의 success rate가 같아야 하는가?

결론부터 말하면, **두 checkpoint의 공통 Seer/action-head weight가 같다면 같아야 한다.**
하지만 현재 비교한 두 checkpoint는 공통 weight가 같지 않다. 따라서 `K=1` success rate 차이는 LR-NODE update 때문이 아니라 checkpoint 자체가 다른 모델이기 때문에 발생한다.

## 확인한 eval 경로

대상:

- baseline ckpt30 eval:
  `runs_lrnode_protocol_20260616/eval/baseline_sweep_sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040_20260617_095300/...ckpt_30_K1_20hz/analysis/eval_summary.json`
- scratch_node ckpt30 K=1 eval:
  `runs_lrnode_protocol_20260616/eval/lrnode_scratch_sweep_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_20260620_202559/...ckpt_30_K1_20p00hz/analysis/eval_summary.json`

관측값:

| run | SR | LR-NODE enabled | skip full forward | K | full forward calls | LR-NODE update calls |
|---|---:|---:|---:|---:|---:|---:|
| baseline ckpt30 | 0.780 | false | false | 1 | 69616 | 0 |
| scratch_node ckpt30 K=1 | 0.805 | true | false | 1 | 68093 | 0 |

`eval_node.sh`도 `K=1`이면 `lrnode_eval_skip_full_forward=0`으로 설정한다. 따라서 이 eval에서는 LR-NODE latent update가 호출되지 않았다.

## checkpoint 공통 weight 비교

비교 대상 checkpoint:

- baseline:
  `runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/30.pth`
- scratch_node:
  `runs_lrnode_protocol_20260616/train/scratch_node/sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053/30.pth`

정확한 저장 key는 `model_state_dict`이다.

결과:

```text
ep=30 common=400 equal=14 diff=386 extra=30 extra_lrnode=30
max_abs=0.518978 max_key=module.image_decoder.1.mlp.fc2.weight
rmse=0.0268277 action_diff=9
```

즉 scratch_node checkpoint는 baseline checkpoint 대비 LR-NODE tensor 30개가 추가되어 있고, 공통 tensor 400개 중 386개가 다르다.

Action head 관련 공통 tensor도 다르다:

```text
module.action_decoder.0.bias      max=0.0146698 mean=0.00233186
module.action_decoder.0.weight    max=0.079759  mean=0.005665
module.action_decoder.2.bias      max=0.0714096 mean=0.00828119
module.action_decoder.2.weight    max=0.206286  mean=0.00963431
module.action_pred_token          max=0.0471493 mean=0.00743424
module.arm_action_decoder.0.bias  max=0.00600513 mean=0.00376625
module.arm_action_decoder.0.weight max=0.129626 mean=0.0117511
module.gripper_action_decoder.0.bias max=0.0102082 mean=0.0102082
module.gripper_action_decoder.0.weight max=0.472458 mean=0.0665569
```

저장된 epoch 전체도 같은 패턴이다:

```text
ep=26 common=400 equal=14 diff=386 extra=30 extra_lrnode=30 rmse=0.0262298
ep=27 common=400 equal=14 diff=386 extra=30 extra_lrnode=30 rmse=0.0264283
ep=28 common=400 equal=14 diff=386 extra=30 extra_lrnode=30 rmse=0.0265913
ep=29 common=400 equal=14 diff=386 extra=30 extra_lrnode=30 rmse=0.0267222
ep=30 common=400 equal=14 diff=386 extra=30 extra_lrnode=30 rmse=0.0268277
...
ep=39 common=400 equal=14 diff=386 extra=30 extra_lrnode=30 rmse=0.0270851
```

따라서 `ckpt30 K=1`의 SR 차이는 eval skip path 때문이 아니라, 비교 중인 두 checkpoint가 이미 다른 Seer/action-head model이기 때문이다.

## train args 비교

baseline과 scratch_node의 핵심 train args는 동일했다:

```text
seed 42 42 SAME
batch_size 16 16 SAME
gradient_accumulation_steps 8 8 SAME
learning_rate 0.001 0.001 SAME
weight_decay 0.0001 0.0001 SAME
num_epochs 40 40 SAME
workers 4 4 SAME
precision fp32 fp32 SAME
loss_arm_action_ratio 1.0 1.0 SAME
loss_gripper_action_ratio 0.01 0.01 SAME
loss_image True True SAME
obs_pred True True SAME
action_pred_steps 3 3 SAME
future_steps 3 3 SAME
sequence_length 7 7 SAME
```

차이는 LR-NODE 관련 args와 run/output name이다.

다만 `git_snapshot.json`은 이 디렉토리가 git repository가 아니어서 commit/diff를 기록하지 못했다. 따라서 baseline run이 시작된 2026-06-16 당시의 정확한 코드와 scratch_node run이 시작된 2026-06-19 당시의 정확한 코드 diff는 snapshot만으로 복원할 수 없다.

## 현재 코드의 단위 parity 확인

`scripts/debug/check_lrnode_parity.py` 결과:

```text
init_parity:
  unequal_common_tensor_count = 0
  rng_equal_after_constructor = true

train_shifted_teacher_parity:
  base_loss_absdiff = 0.0
  main_output_max_absdiff arm/gripper/image/latent = 0.0
  common_grad_diff_count = 0
  common_param_diff_count_after_step = 0
  lrnode_nonzero_grad_tensor_count = 28

eval_full_forward_parity:
  eval_arm_max_absdiff = 0.0
  eval_gripper_max_absdiff = 0.0
  eval_latent_max_absdiff = 0.0
```

이 검사는 “같은 초기 common weight와 같은 batch” 조건에서 현재 코드가 baseline common path를 보존하는지를 확인한다.
즉 현재 코드의 controlled parity는 통과하지만, 이미 저장된 long-run checkpoint 두 개가 같은 common weight라는 뜻은 아니다.

## 결론

`K=1`이면 동일한 common checkpoint에 대해서는 success rate가 같아야 한다.

현재 `baseline ckpt30`과 `scratch_node ckpt30 K=1`의 SR이 다른 직접 원인은 다음이다:

1. `K=1` eval에서는 LR-NODE update가 호출되지 않았다.
2. 하지만 두 checkpoint의 공통 Seer/action-head weight가 동일하지 않다.
3. 특히 action decoder와 gripper action decoder weight도 다르다.
4. 따라서 두 eval은 같은 policy를 평가한 것이 아니다.

이 run은 “동일한 Seer baseline에 LR-NODE만 붙였을 때 K=1 parity가 유지된다”는 근거로 사용하면 안 된다.
그 주장을 하려면 동일 common initialization 및 동일 code snapshot에서 baseline/scratch_node를 다시 맞춰 검증하거나, baseline checkpoint를 고정한 distill/adapter 프로토콜로 비교해야 한다.
