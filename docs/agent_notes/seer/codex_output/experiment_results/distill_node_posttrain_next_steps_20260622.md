# distill_node 학습 종료 후 다음 단계

작성일: 2026-06-22

## 1. 현재 확인된 상태

Distill 실행:

```text
sd1_distill_node_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_20260620_202533
```

Checkpoint root 경로:

```text
runs_lrnode_protocol_20260616/train/distill_node/sd1_distill_node_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_20260620_202533
```

생성된 checkpoints:

```text
1.pth ... 39.pth
```

Teacher / baseline checkpoint 경로:

```text
runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/33.pth
```

학습 protocol 확인:

```text
lrnode_train_protocol=adapter
lrnode_freeze_seer_for_adapter=1
lrnode_assert_only_lrnode_trainable=1
```

학습 가능 parameter 확인:

```text
전체 params: 331,489,562
학습 가능 params: 470,146
학습 가능 modules:
  - lrnode_delta_encoder
  - lrnode_dynamics
모든 non-LR-NODE modules는 freeze됨
```

즉 이 run은 "baseline Seer ckpt33을 고정하고 LR-NODE adapter만 학습한 distill experiment"로 해석 가능하다.

주의:

- `args_snapshot.json`의 `resume_from_checkpoint=None`은 문제 아님.
- 실제 baseline loading은 `finetune_from_pretrained_ckpt`로 수행됨.
- 확인된 값:

```text
finetune_from_pretrained_ckpt=/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616/train/scratch//sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/33.pth
```

## 2. 지금 바로 해야 할 평가

Distill은 Seer/action head가 frozen이므로 `K=1` full-forward 결과는 baseline ckpt33과 같거나 매우 가까워야 한다. 핵심은 `K>1`에서 LR-NODE adapter가 full Seer query를 줄이면서 성능을 얼마나 보존하는지다.

첫 번째 평가는 final checkpoint `39.pth`로 한다.

추천 명령:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_PORT=12620 \
OURS_CKPT_ID=39 \
RUN_BASELINE=1 \
RUN_OURS_FULL=1 \
LRNODE_QUERY_INTERVALS_STR="2 3 4 5 6 8" \
EXPERIMENT_TAG=distill_ckpt39_full_compare_20260622 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
```

이 명령은 다음을 한 번에 수행한다.

1. baseline Seer ckpt33 full-forward 평가
2. distill ckpt39 full-forward K=1 평가
3. distill ckpt39 LR-NODE skip-forward K=2,3,4,5,6,8 평가
4. eval JSON, latency profile, episode metrics, videos 저장

## 3. 시간이 아까우면 먼저 줄여서 sanity check

baseline은 이미 평가되어 있으므로, 빠른 확인만 하려면 baseline 재평가를 생략하고 K도 줄인다.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_PORT=12621 \
OURS_CKPT_ID=39 \
RUN_BASELINE=0 \
RUN_OURS_FULL=1 \
LRNODE_QUERY_INTERVALS_STR="2 3 4" \
EXPERIMENT_TAG=distill_ckpt39_sanity_20260622 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
```

이 결과에서 봐야 할 것:

- K=1 SR이 baseline ckpt33 SR 83.0%와 맞는가?
- K=2/3/4에서 SR 보존률이 어느 정도인가?
- full query reduction이 기대값과 맞는가?
- avg LR-NODE latency가 scratch_node 결과처럼 6-8ms 근처인가?
- action jerk p95가 scratch_node보다 작아지는가?

## 4. ckpt39 결과가 애매하면 checkpoint 후보 sweep

Distill은 LR-NODE만 학습했기 때문에 최종 ckpt가 반드시 best일 필요는 없다. ckpt39가 좋지 않으면 다음 후보를 먼저 본다.

추천 후보:

```text
30, 33, 36, 39
```

예시:

```bash
for CKPT in 30 33 36 39; do
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  MASTER_PORT=$((12630 + CKPT)) \
  OURS_CKPT_ID="${CKPT}" \
  RUN_BASELINE=0 \
  RUN_OURS_FULL=0 \
  LRNODE_QUERY_INTERVALS_STR="2 3 4" \
  EXPERIMENT_TAG="distill_ckpt${CKPT}_k234_20260622" \
  bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
done
```

이 sweep은 baseline/full-forward를 생략하고 skip 성능만 빠르게 비교한다. 최종 보고용으로 선택한 best distill ckpt에 대해서는 다시 `RUN_BASELINE=1`, `RUN_OURS_FULL=1`로 canonical 비교를 돌리는 것이 좋다.

## 5. Distill 결과가 나오면 비교할 표

Distill 결과는 scratch_node와 다른 질문에 답한다.

| 실험 | 비교 기준 | 의미 |
|---|---|---|
| scratch_node QRED20 | 동일 scratch_node ckpt의 K=1 vs K>1 | LR-NODE를 포함해 scratch 학습한 모델에서 query reduction 가능성 |
| distill_node QRED20 | frozen baseline ckpt33 + LR-NODE adapter | 기존 Seer baseline을 고정하고 adapter만 붙여도 query reduction이 되는지 |

distill_node 결과가 좋으면 논문 claim은 더 깨끗해진다.

사용 가능한 표현:

> 기존 Seer baseline checkpoint를 고정한 상태에서도, LR-NODE adapter만 학습하여 intermediate control steps를 대체할 수 있다.

distill_node 결과가 나쁘면:

> 현재 LR-NODE는 scratch co-training 조건에서는 query reduction 가능성을 보이지만, frozen baseline adapter만으로는 충분하지 않을 수 있다.

## 6. 현재 동시에 돌고 있는 작업 주의

현재 scratch_node HZUP 계열 eval이 아직 실행 중이다.

확인된 실행:

```text
eval_lrnode_highhz_20hz_query_budget.sh
eval_lrnode_scratch_hz_sweep.sh
eval_node.sh --lrnode_query_interval 4
```

현재 표준 GPU set은 `CUDA_VISIBLE_DEVICES=4,5,6,7`이다. distill eval도 4,5,6,7에서 돌린다.

다만 현재 scratch_node HZUP/eval 계열이 4,5,6,7에서 실행 중이면 distill eval을 동시에 시작하지 않는 것이 맞다. GPU를 4,5,6,7만 쓰기로 한 조건에서는, 현재 eval이 끝난 뒤 distill eval을 같은 GPU set에서 이어서 실행한다.
