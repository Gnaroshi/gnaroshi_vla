# LR-NODE 현재 결과와 다음 실험

작성일: 2026-06-21 KST

## 현재 완료된 평가 결과

### Baseline scratch Seer

실행:

```text
runs_lrnode_protocol_20260616/train/scratch/
sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040
```

평가 root:

```text
runs_lrnode_protocol_20260616/eval/baseline_sweep_sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040_20260617_095300
runs_lrnode_protocol_20260616/eval/baseline_sweep_sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040_20260618_132356
```

두 eval 모두 같은 SR을 재현했다.

| ckpt | SR |
|---:|---:|
| 30 | 78.0 |
| 31 | 79.5 |
| 32 | 77.5 |
| 33 | 83.0 |
| 34 | 79.0 |
| 35 | 77.0 |
| 36 | 81.0 |
| 37 | 80.0 |
| 38 | 81.0 |
| 39 | 83.0 |

현재 baseline best는 ckpt33 또는 ckpt39, SR 83.0%다. 이후 distill 기준 teacher/baseline으로는 ckpt33을 쓰는 것이 맞다.

### Scratch-node K=1 full-query Seer reference

실행:

```text
runs_lrnode_protocol_20260616/train/scratch_node/
sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053
```

평가 root:

```text
runs_lrnode_protocol_20260616/eval/lrnode_scratch_sweep_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_20260620_202559
```

현재 완료된 K=1 결과:

| ckpt | K | skip | SR |
|---:|---:|---:|---:|
| 30 | 1 | false | 80.5 |
| 31 | 1 | false | 77.0 |
| 32 | 1 | false | 79.5 |
| 33 | 1 | false | 76.0 |
| 34 | 1 | false | 75.5 |
| 35 | 1 | false | 78.5 |
| 36 | 1 | false | 82.5 |

이후 ckpt37-39까지 완료되어 K=1 full-query reference는 확정됐다.

추가 완료 결과:

| ckpt | K | skip | SR |
|---:|---:|---:|---:|
| 37 | 1 | false | 78.5 |
| 38 | 1 | false | 82.0 |
| 39 | 1 | false | 79.5 |

따라서 scratch-node 내부 full-query best는 ckpt36, SR 82.5%다. 다음 후보는 ckpt38, SR 82.0%다.

해석:

```text
K=1은 LR-NODE update call이 0이므로, 이 값은 같은 scratch_node checkpoint의 full Seer reference다.
```

따라서 scratch_node 기반 논문 표는 plain baseline과 직접 비교하지 말고, 같은 checkpoint의 K=1을 기준으로 K>1 preservation을 계산한다.

## Distill 학습 상태

실행:

```text
runs_lrnode_protocol_20260616/train/distill_node/
sd1_distill_node_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_20260620_202533
```

현재 저장된 checkpoint:

```text
1.pth ... 26.pth
```

분석 snapshot:

```text
analysis/lrnode_train_protocol_status.json
analysis/freeze_status_snapshot.json
analysis/model_trainable_params.json
```

확인된 내용:

```text
protocol = adapter
trainable tensors = LR-NODE only
non-LRNODE trainable tensors = 0
trainable params = 470,146
Seer backbone/action head freeze
```

실제 ckpt26을 확인한 결과, checkpoint에는 LR-NODE 30개 tensor만 들어 있다. 즉 adapter-only checkpoint다.

중요:

```text
현재 eval_lrnode_distill_compare.sh는 adapter ckpt를 --resume_from_checkpoint로 직접 넣는다.
그대로 실행하면 baseline Seer ckpt33과 adapter가 merge되지 않는다.
따라서 distill 평가는 merge/load 경로를 수정한 뒤 돌려야 한다.
```

## 지금 바로 돌릴 실험

### 1. 현재 eval_node.sh는 이미 QRED20 raw sweep으로 진행 중

현재 실행 중:

```text
bash scripts/LIBERO_LONG/Seer/eval_node.sh
```

목적:

```text
scratch_node checkpoint 30-39의 20Hz K sweep
```

현재 K=1은 완료됐고, K=2 ckpt30 평가가 진행 중이다. 이 run은 폴더명은 `qred20`이 아니지만 실험 내용상 QRED20 raw result로 취급할 수 있다.

### 2. QRED20

정식명:

```text
20Hz Query-Reduction / Seer Replacement
```

질문:

```text
기존 20Hz control setting에서도 LR-NODE가 full Seer forward를 얼마나 대체할 수 있는가?
```

실행:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_20hz_query_reduction.sh
```

자동으로 latest scratch_node K=1 eval root에서 best ckpt를 고른다. 현재 확정 best 기준으로는 ckpt36이다.

```bash
CKPT_IDS=36 bash scripts/LIBERO_LONG/Seer/eval_lrnode_20hz_query_reduction.sh
```

우선순위:

```text
가장 먼저 돌릴 것.
20Hz에서 K=2/3/4/5/6/8이 SR을 얼마나 보존하는지 알아야 high-Hz 실험의 K 선택이 정당해진다.
```

### 3. HZUP20Q

정식명:

```text
20Hz full-query budget을 둔 high-Hz control
```

질문:

```text
LIBERO control_freq를 실제 40/60/80Hz로 올리면서 full Seer query rate는 20Hz 수준으로 유지할 수 있는가?
```

실행:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_highhz_20hz_query_budget.sh
```

핵심 rows:

```text
40:2 = control 40Hz, full Seer 20Hz, LR-NODE 20Hz
60:3 = control 60Hz, full Seer 20Hz, LR-NODE 40Hz
80:4 = control 80Hz, full Seer 20Hz, LR-NODE 60Hz
```

우선순위:

```text
QRED20 다음.
논문 핵심 claim인 high-Hz control + fixed full-query budget을 직접 검증한다.
```

## 고친 뒤 돌릴 실험

### Distill adapter 평가

목적:

```text
기존 baseline Seer ckpt33을 고정하고 LR-NODE adapter만 붙였을 때,
K=1 baseline과 K>1 skip 성능을 비교한다.
```

현재 문제:

```text
distill checkpoint는 adapter-only인데 eval script가 baseline ckpt33과 adapter ckpt를 merge하지 않는다.
```

필요한 수정:

```text
eval_libero.py 또는 별도 merge script에서
1. full Seer model 생성
2. baseline ckpt33 full state_dict load
3. distill adapter ckpt LR-NODE state_dict load
4. 그 상태로 K=1/K>1 eval
```

수정 전에는 아래를 돌리지 않는다.

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
```

## 추천 실행 순서

1. 현재 `eval_node.sh` K=1 sweep 완료 대기
2. `QRED20` 실행
3. `HZUP20Q` 실행
4. distill eval merge 경로 수정
5. distill adapter eval 실행
6. 필요하면 `GRID` 실행

`GRID`는 primary result가 아니라 보조 ablation이다.

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_full_grid.sh
```
