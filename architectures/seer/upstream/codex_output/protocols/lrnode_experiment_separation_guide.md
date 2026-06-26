## LR-NODE 실험/평가 분리 저장 가이드

보관 참고: 이 문서 작성 후 pre-protocol LR-NODE 결과는 2026-06-16에 아래 archive로 이동했습니다.

```text
/home/mingyujung/private/seer/seer_node3/archived_experiment_results_20260616/pre_protocol_lrnode
```

요청한 분리 규칙을 반영해서 다음 규칙이 적용됐습니다.

1. `scratch.sh`: plain Seer scratch baseline
2. `scratch_node.sh`: scratch + LR-NODE teacher-student detached
3. `distill_node.sh`: frozen-baseline LR-NODE distill/adapter
4. `scratch_node_joint.sh`: scratch + LR-NODE coupled joint

### 1) 실험(run) 분리

- 각 학습 스크립트는 기본적으로 `EXPERIMENT_TAG`를 생성합니다. (`YYYYMMDD_HHMMSS`)
- run_name 기본값에 태그를 자동 붙여 기존 이름과 충돌하지 않게 저장합니다.
  - `scratch.sh`: `${which_server}_scratch_baseline_..._${EXPERIMENT_TAG}`
  - `scratch_node.sh`: `${which_server}_scratch_node_ts_..._${EXPERIMENT_TAG}`
  - `distill_node.sh`: `${which_server}_distill_node_..._${EXPERIMENT_TAG}`
  - `scratch_node_joint.sh`: `${which_server}_scratch_node_joint_..._${EXPERIMENT_TAG}`
- 따라서 이전 실험 경로를 지우지 않아도 새 실험은 새 폴더/새 run 이름으로 분리됩니다.

### 2) 저장 root

새 protocol은 모두 아래 root에 저장됩니다.

```text
runs_lrnode_protocol_20260616/
```

학습:

```text
runs_lrnode_protocol_20260616/train/scratch/
runs_lrnode_protocol_20260616/train/scratch_node/
runs_lrnode_protocol_20260616/train/distill_node/
runs_lrnode_protocol_20260616/train/scratch_node_joint/
```

최신 run pointer:

```text
runs_lrnode_protocol_20260616/train/_latest/scratch.env
runs_lrnode_protocol_20260616/train/_latest/scratch_node.env
runs_lrnode_protocol_20260616/train/_latest/distill_node.env
runs_lrnode_protocol_20260616/train/_latest/scratch_node_joint.env
```

평가:

```text
runs_lrnode_protocol_20260616/eval/
```

### 3) 실행 시 권장 사용법

태그를 명시하면 추후 추적이 더 쉬워집니다.

```bash
export EXPERIMENT_TAG=20260616A
bash scripts/LIBERO_LONG/Seer/scratch.sh
export EXPERIMENT_TAG=20260616A
bash scripts/LIBERO_LONG/Seer/scratch_node.sh
export EXPERIMENT_TAG=20260616A
bash scripts/LIBERO_LONG/Seer/distill_node.sh
export EXPERIMENT_TAG=20260616A
bash scripts/LIBERO_LONG/Seer/scratch_node_joint.sh

# 평가
export EXPERIMENT_TAG=20260616A
bash scripts/LIBERO_LONG/Seer/eval.sh
bash scripts/LIBERO_LONG/Seer/eval_node.sh
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_joint_compare.sh
```

또는 서로 다른 실험을 완전히 분리하려면 각 run마다 다른 태그를 사용하세요.

### 4) 기록 남기기

- 각 학습 스크립트는 실행 초기에 아래를 콘솔 출력합니다.
  - `[TRAIN INFO] experiment_tag=...`
  - `[TRAIN INFO] run_name=...`
- `eval_node.sh`는 실행 초기에 아래를 출력합니다.
  - `[EVAL INFO] experiment_tag=...`
  - `[EVAL INFO] eval_result_root_base=...`

### 5) 주의

- 새 평가 wrapper는 explicit `*_RUN_NAME`/`*_CKPT`가 없으면 `_latest/*.env`를 읽습니다.
- 특정 checkpoint를 고정 비교할 때만 `BASELINE_CKPT`, `OURS_CKPT`, `CHECKPOINTS_STR`, `LRNODE_QUERY_INTERVALS_STR`를 명시하세요.
# 보관 참고

이 문서에 언급된 pre-protocol LR-NODE 결과는 2026-06-16에 아래 archive로 이동했다.

```text
/home/mingyujung/private/seer/seer_node3/archived_experiment_results_20260616/pre_protocol_lrnode
```
