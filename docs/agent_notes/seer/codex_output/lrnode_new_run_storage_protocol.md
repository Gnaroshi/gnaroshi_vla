# LR-NODE New Run Storage Protocol

작성일: 2026-06-16 KST

새 학습/평가 결과는 기존 결과와 섞이지 않게 아래 root로 분리한다.

```text
$SEER_WORKSPACE_ROOT/runs_lrnode_protocol_20260616
```

## 1. Training Output

| Script | Protocol | Save root |
|---|---|---|
| `scratch.sh` | scratch plain Seer baseline, LR-NODE off | `runs_lrnode_protocol_20260616/train/scratch/` |
| `scratch_node.sh` | scratch + LR-NODE teacher-student detached, shifted-context target | `runs_lrnode_protocol_20260616/train/scratch_node/` |
| `distill_node.sh` | frozen-baseline LR-NODE distill/adapter, shifted-context target | `runs_lrnode_protocol_20260616/train/distill_node/` |
| `scratch_node_joint.sh` | scratch + LR-NODE coupled joint, shifted-context target | `runs_lrnode_protocol_20260616/train/scratch_node_joint/` |

각 script는 timestamp가 들어간 `run_name`을 만들고, 아래 최신 run pointer를 남긴다.

```text
runs_lrnode_protocol_20260616/train/_latest/scratch.env
runs_lrnode_protocol_20260616/train/_latest/scratch_node.env
runs_lrnode_protocol_20260616/train/_latest/distill_node.env
runs_lrnode_protocol_20260616/train/_latest/scratch_node_joint.env
```

## 2. Evaluation Output

공통 eval root:

```text
runs_lrnode_protocol_20260616/eval/
```

Adapter 비교:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh
```

Scratch joint 비교:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_joint_compare.sh
```

두 eval wrapper는 explicit `OURS_CKPT`/`BASELINE_CKPT`가 없으면 `_latest/*.env`를 읽어서 방금 학습한 run을 기본으로 사용한다.

## 3. Archived Existing Results

아래 기존 결과는 삭제하지 않고 archive root로 이동했다.

```text
archived_experiment_results_20260616/pre_protocol_lrnode/scratch_checkpoints_lrnode/
archived_experiment_results_20260616/pre_protocol_lrnode/scratch_eval_lrnode/
```

앞으로 새 protocol 실험은 `runs_lrnode_protocol_20260616/` 아래 결과만 기준으로 정리한다.

기존 결과를 다시 확인해야 할 때는 archive path를 사용한다.
