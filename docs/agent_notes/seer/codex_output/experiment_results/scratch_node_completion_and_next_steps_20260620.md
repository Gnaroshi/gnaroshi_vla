# scratch_node 완료 확인 및 다음 실험 정리

작성일: 2026-06-20 KST

## 1. scratch_node.sh 완료 상태

현재 `_latest` pointer:

```text
runs_lrnode_protocol_20260616/train/_latest/scratch_node.env
```

내용:

```text
LRNODE_PROTOCOL_SCRIPT=scratch_node.sh
LRNODE_PROTOCOL_KIND=scratch
LRNODE_MODULE=1
LRNODE_COUPLING=teacher_student_detached
LRNODE_JOINT=0
LRNODE_BACKPROP_TO_SEER_FROM_LRNODE=0
LRNODE_TEACHER_TARGET_MODE=shifted_context
LRNODE_CONTEXT_SELECTED_STEP=-1
LRNODE_EXPERIMENT_TAG=20260619_113053
LRNODE_RUN_NAME=sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053
LRNODE_SAVE_CHECKPOINT_PATH=$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/train/scratch_node/
LRNODE_DATASET=libero_10_converted
```

Checkpoint directory 경로:

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/train/scratch_node/sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053
```

확인된 checkpoint:

```text
26.pth
27.pth
28.pth
29.pth
30.pth
31.pth
32.pth
33.pth
34.pth
35.pth
36.pth
37.pth
38.pth
39.pth
```

ckpt 39 생성 시각:

```text
2026-06-20 16:26
```

`torchrun`, `train.py`, `scratch_node.sh` 관련 학습 프로세스는 현재 잡히지 않았다. 따라서 checkpoint 관점에서는 학습이 종료된 것으로 본다.

주의:

- 현재 sandbox에서는 `nvidia-smi`가 driver와 통신하지 못해서 GPU 점유 상태는 확인하지 못했다.
- 실행 전 실제 shell에서 GPU 점유만 한 번 확인하는 것이 좋다.

## 2. eval_node.sh 바로 실행 가능 여부

결론:

```text
실행 가능하다.
```

근거:

- `eval_node.sh`는 기본적으로 `runs_lrnode_protocol_20260616/train/_latest/scratch_node.env`를 source한다.
- 해당 env가 존재한다.
- env가 가리키는 run directory가 존재한다.
- `eval_node.sh` 기본 ckpt 범위인 30-39가 모두 존재한다.
- script 안에서 `eval_summary.json`, `eval_episode_metrics.csv`, `eval_latency_profile.json`, `args_snapshot` 존재를 verify한다.
- `SAVE_VIDEO=1` 기본값으로 video 저장도 verify한다.

하지만 `eval_node.sh`를 그대로 실행하면 매우 큰 sweep이다.

`eval_node.sh` 기본값:

```text
CKPT_IDS=30 31 32 33 34 35 36 37 38 39
LRNODE_QUERY_INTERVALS_STR=1 2 3 4 5 6 8
```

즉 총 eval config 수:

```text
10 checkpoints x 7 K values = 70 eval configs
```

각 config가 200 episodes이고 video를 저장하므로, 기본 실행은 대량의 시간과 저장공간을 쓴다.

현재 수정 사항:

- `eval_node.sh` 기본 `MASTER_PORT`: `12442`
- `distill_node.sh` 기본 `MASTER_PORT`: `12423`
- `eval_lrnode_compare.sh` 기본 `MASTER_PORT`: `12452`
- `distill_node.sh` 기본 `BASELINE_CKPT_ID`: `33`

## 3. 바로 전체 실행용 wrapper

다음 script를 추가했다.

```text
scripts/LIBERO_LONG/Seer/run_lrnode_post_scratch_all.sh
```

이 script는 한 번에 다음을 수행한다.

1. baseline ckpt33에서 `distill_node.sh`를 background로 시작한다.
2. `scratch_node.sh` 결과를 K=1로 ckpt 30-39 전체 평가한다.
3. K=1 결과에서 best scratch_node ckpt를 자동 선택한다.
4. 선택된 best scratch_node ckpt에 대해 K=2,3,4,5,6,8 sweep을 실행한다.
5. background distill이 끝날 때까지 기다린다.

기본 port:

| job | port |
|---|---:|
| scratch_node K=1 eval | 12442 |
| scratch_node best K sweep | 12443 |
| distill train | 12423 |

기본 GPU:

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
```

GPU는 분리하지 않는다. 같은 GPU set에서 eval과 distill이 동시에 돈다. VRAM이 충분하다는 전제에서는 실행 가능하지만, compute contention 때문에 wall-clock 시간은 늘 수 있다.

바로 실행:

```bash
bash scripts/LIBERO_LONG/Seer/run_lrnode_post_scratch_all.sh
```

실행 로그와 run plan:

```text
runs_lrnode_protocol_20260616/launch_logs/<EXPERIMENT_TAG>/
```

주요 override:

```bash
EXPERIMENT_TAG="post_scratch_node_manual_$(date +%Y%m%d_%H%M%S)" \
BASELINE_CKPT_ID=33 \
CKPT_IDS="30 31 32 33 34 35 36 37 38 39" \
BEST_SWEEP_QUERY_INTERVALS="2 3 4 5 6 8" \
bash scripts/LIBERO_LONG/Seer/run_lrnode_post_scratch_all.sh
```

distill을 끄고 scratch_node eval만 하려면:

```bash
RUN_DISTILL=0 bash scripts/LIBERO_LONG/Seer/run_lrnode_post_scratch_all.sh
```

K sweep을 끄고 K=1 checkpoint selection만 하려면:

```bash
RUN_SCRATCH_BEST_K_SWEEP=0 bash scripts/LIBERO_LONG/Seer/run_lrnode_post_scratch_all.sh
```

## 4. 수동 eval 순서

### 4.1 먼저 K=1만 전체 checkpoint 평가

목적:

- scratch_node run 자체의 full-forward Seer 성능을 확인한다.
- LR-NODE skip을 쓰기 전에 어떤 ckpt가 best인지 찾는다.
- K=1에서는 `eval_skip_full_forward=0`이므로 full Seer forward 기준이다.

권장 command:

```bash
CKPT_IDS="30 31 32 33 34 35 36 37 38 39" \
LRNODE_QUERY_INTERVALS_STR="1" \
EXPERIMENT_TAG="scratch_node_k1_full_$(date +%Y%m%d_%H%M%S)" \
bash scripts/LIBERO_LONG/Seer/eval_node.sh
```

해석:

- baseline `scratch.sh` 결과와 비교할 때 first check는 K=1 SR이다.
- baseline primary는 ckpt 33, backup은 ckpt 39다.
- scratch_node의 K=1 성능이 baseline과 얼마나 같은지/다른지를 먼저 봐야 한다.
- K=1 결과가 낮으면 K>1 skip 성능은 논문 claim으로 쓰기 어렵다.

### 4.2 K=1 best checkpoint를 찾은 뒤 K sweep

예를 들어 K=1 평가 후 best가 ckpt 33이면:

```bash
CKPT_IDS="33" \
LRNODE_QUERY_INTERVALS_STR="1 2 3 4 5 6 8" \
EXPERIMENT_TAG="scratch_node_best_ckpt33_k_sweep_$(date +%Y%m%d_%H%M%S)" \
bash scripts/LIBERO_LONG/Seer/eval_node.sh
```

ckpt 39도 같이 확인하려면:

```bash
CKPT_IDS="33 39" \
LRNODE_QUERY_INTERVALS_STR="1 2 3 4 5 6 8" \
EXPERIMENT_TAG="scratch_node_ckpt33_39_k_sweep_$(date +%Y%m%d_%H%M%S)" \
bash scripts/LIBERO_LONG/Seer/eval_node.sh
```

이렇게 나누는 이유:

- 기본 70-config sweep은 너무 크다.
- 먼저 K=1으로 checkpoint selection을 하고, 그 다음 selected checkpoint만 K sweep하는 편이 해석이 명확하다.
- 영상 저장은 유지된다.

## 5. baseline 기준

현재 baseline eval 분석 문서:

```text
codex_output/baseline_eval_analysis_20260620.md
```

기준 결과:

- baseline best SR: ckpt 33, ckpt 39 모두 83.0%
- primary baseline: ckpt 33
- backup baseline: ckpt 39

Primary baseline ckpt 경로:

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/33.pth
```

Backup baseline ckpt 경로:

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616/train/scratch/sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040/39.pth
```

## 6. distill_node.sh 실험 분석

결론:

```text
distill 실험도 지금 시작 가능한 단계가 맞다.
script 기본값도 baseline primary ckpt 33으로 수정했다.
```

이유:

- baseline `scratch.sh` 학습은 끝났다.
- baseline `eval.sh`도 끝났고 best ckpt가 정해졌다.
- `distill_node.sh`는 baseline env를 자동 source한다.
- script 내부 default는 이제 `BASELINE_CKPT_ID=33`이다.
- 현재 best baseline은 ckpt 33/39이고, primary는 ckpt 33이다.

따라서 첫 distill은 이렇게 실행한다:

```bash
EXPERIMENT_TAG="distill_from_baseline_ckpt33_$(date +%Y%m%d_%H%M%S)" \
bash scripts/LIBERO_LONG/Seer/distill_node.sh
```

backup 확인:

```bash
BASELINE_CKPT_ID=39 \
EXPERIMENT_TAG="distill_from_baseline_ckpt39_$(date +%Y%m%d_%H%M%S)" \
bash scripts/LIBERO_LONG/Seer/distill_node.sh
```

## 7. distill의 의미

`distill_node.sh`는 main scratch comparison이 아니다.

목적:

```text
이미 학습된 Seer baseline checkpoint를 고정하고,
LR-NODE module만 추가 학습했을 때,
skip full-forward eval에서 SR을 얼마나 유지하면서 query를 줄일 수 있는지 확인한다.
```

코드 기준:

- `--finetune_from_pretrained_ckpt ${BASELINE_CKPT}`
- `--lrnode_train_protocol adapter`
- `--lrnode_freeze_seer_for_adapter 1`
- `--lrnode_assert_only_lrnode_trainable 1`
- train.py에서 non-LR-NODE parameter는 `requires_grad=False`
- train.py에서 LR-NODE parameter만 `requires_grad=True`

즉 distill은:

- Seer backbone을 바꾸지 않는다.
- action head를 바꾸지 않는다.
- LR-NODE delta encoder와 controlled latent dynamics만 학습한다.

이 실험이 답하는 질문:

```text
baseline policy 자체는 그대로 둔 상태에서,
LR-NODE가 baseline action latent transition만 배워도
full Seer query를 줄일 수 있는가?
```

이 실험이 답하지 않는 질문:

```text
Seer를 scratch부터 LR-NODE와 함께 학습하면 최종 policy가 더 좋아지는가?
```

그 질문은 `scratch_node.sh` 결과가 담당한다.

## 8. distill 실행 시 주의점

### 8.1 GPU와 port

`eval_node.sh`와 `distill_node.sh` 모두 기본 GPU가 같다.

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
```

현재는 GPU를 분리하지 않는 실행을 기준으로 정리했다. 대신 master port를 분리했다.

```text
eval_node.sh                 MASTER_PORT=12442
distill_node.sh              MASTER_PORT=12423
run_lrnode_post_scratch_all  eval K=1 port=12442, eval K sweep port=12443, distill port=12423
eval_lrnode_compare.sh       MASTER_PORT=12452
```

같은 GPU에서 동시에 돌릴 수 있지만, GPU compute contention으로 개별 job의 wall-clock 시간은 늘 수 있다.

필요하면 수동 override도 가능하다.

```bash
MASTER_PORT=12542 bash scripts/LIBERO_LONG/Seer/eval_node.sh
MASTER_PORT=12523 bash scripts/LIBERO_LONG/Seer/distill_node.sh
```

### 8.2 shifted_context 비용

현재 distill도 기본 target mode가 `shifted_context`다.

```text
LRNODE_TEACHER_TARGET_MODE=shifted_context
```

이 mode는 teacher target을 만들기 위해 extra Seer forward가 들어간다. adapter protocol에서는 Seer가 frozen이라 backward는 LR-NODE만 타지만, teacher forward compute 자체는 여전히 든다.

빠른 ablation이 목적이면 다음도 가능하다:

```bash
BASELINE_CKPT_ID=33 \
LRNODE_TEACHER_TARGET_MODE=adjacent_sequence \
EXPERIMENT_TAG="distill_from_baseline_ckpt33_adjacent_$(date +%Y%m%d_%H%M%S)" \
bash scripts/LIBERO_LONG/Seer/distill_node.sh
```

단, `adjacent_sequence`와 `shifted_context`는 teacher target 정의가 다르므로 같은 실험으로 섞어 해석하면 안 된다.

## 9. 지금 권장 액션

한 번에 실행하려면 다음 하나만 실행한다.

```bash
bash scripts/LIBERO_LONG/Seer/run_lrnode_post_scratch_all.sh
```

이 command는 다음을 모두 포함한다.

- scratch_node K=1 ckpt 30-39 평가
- scratch_node best ckpt 자동 선택
- scratch_node best ckpt K=2,3,4,5,6,8 sweep
- baseline ckpt33 distill training
- video 저장
- latency/eval JSON 저장
- master port 분리

수동으로 나눠서 실행해야 하면 다음을 쓴다.

scratch_node K=1:

```bash
CKPT_IDS="30 31 32 33 34 35 36 37 38 39" \
LRNODE_QUERY_INTERVALS_STR="1" \
MASTER_PORT=12442 \
EXPERIMENT_TAG="scratch_node_k1_full_$(date +%Y%m%d_%H%M%S)" \
bash scripts/LIBERO_LONG/Seer/eval_node.sh
```

distill:

```bash
BASELINE_CKPT_ID=33 \
MASTER_PORT=12423 \
EXPERIMENT_TAG="distill_from_baseline_ckpt33_$(date +%Y%m%d_%H%M%S)" \
bash scripts/LIBERO_LONG/Seer/distill_node.sh
```
