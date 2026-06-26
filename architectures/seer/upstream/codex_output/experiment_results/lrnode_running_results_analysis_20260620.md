# LR-NODE 현재 실행 결과 분석

작성일: 2026-06-20 KST

## 1. 현재 실행 상태

현재 실행은 `run_lrnode_post_scratch_all.sh` wrapper가 만든 형태는 아니다.

근거:

```text
runs_lrnode_protocol_20260616/launch_logs
```

위 디렉토리가 존재하지 않는다.

현재 실제로 돌아가는 작업:

1. `eval_node.sh`
   - scratch_node checkpoint K=1 full-forward eval 진행 중
   - 현재 ckpt31 K=1 평가 중
   - 완료된 summary는 ckpt30 K=1 하나
2. `distill_node.sh`
   - baseline ckpt33에서 LR-NODE adapter distill 진행 중
   - 현재 epoch 4/40 근처
   - checkpoint 1, 2, 3 저장 완료

## 2. 완료된 scratch_node eval 결과

완료된 결과:

```text
runs_lrnode_protocol_20260616/eval/lrnode_scratch_sweep_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_20260620_202559/lrnode_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_ckpt_30_K1_20p00hz/analysis/eval_summary.json
```

현재 완료된 scratch_node 결과는 ckpt30 K=1뿐이다. 따라서 아직 scratch_node 전체 checkpoint selection은 끝나지 않았다.

## 3. scratch_node ckpt30 K=1 요약

K=1에서는 LR-NODE skip이 꺼진다.

```text
lrnode_eval_skip_full_forward=0
lrnode_update_calls=0
full_query_reduction_ratio=0.0
```

따라서 이 결과는 LR-NODE skip 효율 결과가 아니라, `scratch_node.sh`로 학습된 Seer full-forward policy 자체의 성능이다.

| run | ckpt | K | SR | success / 200 | env steps | full calls | full ms | policy ms | env ms | videos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline scratch | 30 | 1 | 78.0% | 156 | 69,616 | 69,616 | 68.71 | 88.51 | 275.11 | 200 |
| scratch_node | 30 | 1 | 80.5% | 161 | 68,093 | 68,093 | 68.33 | 87.90 | 276.33 | 200 |
| baseline scratch | 33 | 1 | 83.0% | 166 | 66,564 | 66,564 | 69.13 | 89.34 | 281.20 | 200 |
| baseline scratch | 39 | 1 | 83.0% | 166 | 67,797 | 67,797 | 68.58 | 88.36 | 268.33 | 200 |

해석:

- scratch_node ckpt30은 baseline ckpt30보다 +2.5%p 높다.
- 하지만 현재 baseline best ckpt33/39의 83.0%보다는 -2.5%p 낮다.
- latency는 baseline과 거의 같은 범위다. K=1이라 full forward를 매 step 수행하므로 query reduction은 없다.
- video는 200개 저장됐다.

## 4. Task별 비교

Task별 SR (%):

| task | baseline ckpt30 | scratch_node ckpt30 | delta vs base30 | baseline best ckpt33 | delta vs base33 |
|---:|---:|---:|---:|---:|---:|
| 0 | 65 | 100 | +35 | 80 | +20 |
| 1 | 90 | 75 | -15 | 90 | -15 |
| 2 | 90 | 95 | +5 | 100 | -5 |
| 3 | 80 | 90 | +10 | 100 | -10 |
| 4 | 75 | 90 | +15 | 75 | +15 |
| 5 | 95 | 90 | -5 | 90 | 0 |
| 6 | 80 | 95 | +15 | 75 | +20 |
| 7 | 90 | 85 | -5 | 95 | -10 |
| 8 | 40 | 40 | 0 | 55 | -15 |
| 9 | 75 | 45 | -30 | 70 | -25 |

관찰:

- scratch_node ckpt30은 task 0, 4, 6에서 크게 좋아졌다.
- task 9가 크게 나빠졌다. baseline ckpt30 대비 -30%p, baseline ckpt33 대비 -25%p다.
- task 8은 여전히 약하다. baseline ckpt30과 같은 40%이고, baseline best ckpt33보다 낮다.
- 평균 SR 80.5%는 괜찮지만, task 9 failure가 커서 best checkpoint selection이 아직 필요하다.

## 5. Smoothness / action 변화

| run | ckpt | delta mean | delta p95 | jerk mean | jerk p95 | gripper switch |
|---|---:|---:|---:|---:|---:|---:|
| baseline scratch | 30 | 0.064038 | 0.106529 | 0.062287 | 0.064359 | 0.010795 |
| scratch_node | 30 | 0.065627 | 0.106913 | 0.064880 | 0.075871 | 0.011360 |
| baseline scratch | 33 | 0.066217 | 0.117215 | 0.064881 | 0.087384 | 0.011435 |

해석:

- scratch_node ckpt30의 action delta는 baseline ckpt30과 거의 비슷하다.
- jerk p95는 baseline ckpt30보다 높지만, baseline ckpt33보다는 낮다.
- K=1 full-forward 결과이므로, 여기서의 smoothness는 LR-NODE skip update의 영향이 아니다.

## 6. distill 학습 상태

Distill 실행:

```text
sd1_distill_node_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_20260620_202533
```

Baseline 출처:

```text
runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/33.pth
```

현재 checkpoint:

```text
1.pth
2.pth
3.pth
```

각 checkpoint 크기:

```text
5.7 MB
```

훈련 상태:

- current output 기준 epoch 4/40 근처
- 아직 초반이다.
- distill은 계속 진행 중이다.

## 7. distill freeze 검증

`lrnode_train_protocol_status.json`:

```json
{
  "protocol": "adapter",
  "freeze_for_adapter": true,
  "num_lrnode_trainable_tensors": 30,
  "num_non_lrnode_trainable_tensors": 0,
  "non_lrnode_trainable_preview": []
}
```

파라미터 수:

| module group | params | trainable |
|---|---:|---:|
| total | 331,489,562 | 470,146 |
| lrnode_delta_encoder | 131,456 | 131,456 |
| lrnode_dynamics | 338,690 | 338,690 |
| vision_encoder | 111,907,840 | 0 |
| transformer_backbone | 42,587,904 | 0 |
| action_decoder | 110,976 | 0 |
| arm_action_decoder | 1,158 | 0 |
| gripper_action_decoder | 193 | 0 |
| clip_model | 151,277,313 | 0 |

해석:

- distill은 의도대로 LR-NODE만 학습 중이다.
- Seer backbone, vision encoder, CLIP, action head는 모두 frozen이다.
- 이 실험은 “baseline policy를 바꾸지 않고 LR-NODE updater만 학습”하는 adapter/distill 실험으로 해석 가능하다.

## 8. distill loss 현재 상태

WandB summary의 latest synced 값 기준:

| metric | value |
|---|---:|
| `loss_calvin` | 0.000997 |
| `loss_lrnode_latent` | 0.002718 |
| `loss_lrnode_action_distill` | 0.008598 |
| `loss_lrnode_smooth` | 0.001366 |
| `seer_backbone_grad_norm_total` | 0.0 |
| `action_head_grad_norm_total` | 0.0 |
| `lrnode_fast_encoder_grad_norm` | 0.000180 |
| `lrnode_controlled_node_grad_norm` | 0.001808 |

LR-NODE 개선 지표:

| metric | value |
|---|---:|
| `train/lrnode/action_l1_hold` | 0.014409 |
| `train/lrnode/action_l1_pred` | 0.008434 |
| `train/lrnode/action_l1_improvement` | 0.005975 |
| `train/lrnode/action_l1_ratio` | 0.6017 |
| `train/lrnode/latent_mse_hold` | 0.004594 |
| `train/lrnode/latent_mse_pred` | 0.003147 |
| `train/lrnode/latent_mse_improvement` | 0.001447 |
| `train/lrnode/latent_mse_ratio` | 0.7171 |

해석:

- 현재 단계에서 LR-NODE prediction은 hold baseline보다 action L1을 약 40% 낮춘다.
- latent MSE도 hold보다 약 28-32% 낮다.
- gradient는 Seer/action head로 흐르지 않고 LR-NODE에만 존재한다.
- gate mean은 약 0.12-0.13으로 보수적인 update를 하고 있다.
- 아직 epoch 4/40 근처라 최종 성능 판단은 불가능하다.

## 9. 중요한 문제: distill checkpoint eval 방식

현재 `utils/train_utils.py`의 `get_checkpoint()`는 `requires_grad=False`인 parameter를 checkpoint에서 제거한다.

```python
def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict
```

따라서 distill checkpoint는 LR-NODE adapter parameter만 저장된다.

실제 확인:

```text
distill 3.pth:
num_keys = 30
lrnode_keys = 30
non_lrnode_keys = 0

baseline 33.pth:
num_keys = 400
lrnode_keys = 0
non_lrnode_keys = 400

scratch_node 30.pth:
num_keys = 430
lrnode_keys = 30
non_lrnode_keys = 400
```

이것은 distill 학습 자체에는 문제가 아니다. adapter-only checkpoint로 저장된 것이다.

하지만 distill eval에는 문제가 된다.

현재 `eval_libero.py`는 단일 `--resume_from_checkpoint`만 로드한다.

```text
ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
```

distill checkpoint만 넣으면 LR-NODE 30개 key만 로드되고, baseline Seer/action head 400개 key는 현재 model initialization 상태로 남을 수 있다. 즉 distill eval을 그대로 돌리면 “baseline ckpt33 + LR-NODE adapter”가 아니라 “random initialized Seer + LR-NODE adapter”가 될 위험이 있다.

따라서 distill eval 전에 반드시 다음 중 하나가 필요하다.

1. eval에서 baseline ckpt33을 먼저 로드하고, 그 다음 distill adapter ckpt를 로드한다.
2. distill 학습 저장 시 full merged checkpoint를 저장한다.
3. 별도 merge script로 `baseline ckpt33 + distill adapter ckpt`를 합친 eval용 checkpoint를 만든다.

현재 상태에서는 1번이 가장 안전하다.

## 10. 결론

현재까지 확정적으로 말할 수 있는 것:

- scratch_node ckpt30 K=1은 80.5%로 baseline ckpt30 78.0%보다 높다.
- 하지만 baseline best 83.0%에는 아직 못 미친다.
- scratch_node eval은 아직 ckpt selection이 끝나지 않았다. ckpt31-39 결과를 기다려야 한다.
- distill은 adapter-only 구조로 정상 학습 중이다.
- distill loss는 hold보다 개선되는 방향으로 움직이고 있다.
- distill eval은 현재 코드/script 그대로 하면 안 된다. adapter checkpoint를 baseline checkpoint와 함께 로드하는 경로가 필요하다.

## 11. 다음 액션

1. 현재 `eval_node.sh` K=1 sweep은 계속 둔다.
2. ckpt30-39 K=1이 끝나면 scratch_node best ckpt를 선택한다.
3. selected scratch_node ckpt에 대해서만 K=2,3,4,5,6,8 sweep을 돌린다.
4. distill 학습은 계속 둔다.
5. distill eval 전에 반드시 baseline+adapter 로딩 경로를 수정한다.
6. distill eval은 수정 후 `baseline ckpt33 + distill adapter ckpt` 조합으로 수행한다.
