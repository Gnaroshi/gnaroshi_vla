# LR-NODE Archive Manifest

작성일: 2026-06-16 KST

기존 pre-protocol LR-NODE 실험 결과를 repo root에서 archive 폴더로 이동했다.

## Archive Root

```text
/home/mingyujung/private/seer/seer_node3/archived_experiment_results_20260616/pre_protocol_lrnode
```

## Moved

| 원래 위치 | 현재 위치 | 용량 |
|---|---|---:|
| `scratch_checkpoints_lrnode/` | `archived_experiment_results_20260616/pre_protocol_lrnode/scratch_checkpoints_lrnode/` | 11G |
| `scratch_eval_lrnode/` | `archived_experiment_results_20260616/pre_protocol_lrnode/scratch_eval_lrnode/` | 3.2G |

## Root Separation

| 구분 | 위치 |
|---|---|
| 이전 실험 archive | `archived_experiment_results_20260616/pre_protocol_lrnode/` |
| 새 protocol 학습/평가 | `runs_lrnode_protocol_20260616/` |
| 코드/문서 | repo root, `codex_output/`, `docs/`, `models/`, `utils/`, `scripts/` |

## Notes

- 기존 결과는 삭제하지 않았다.
- `checkpoints/`는 MAE/CLIP 등 기본 asset 가능성이 있어 이동하지 않았다.
- `runs_lrnode_protocol_20260616/`는 새 실험 전용 root로 유지한다.
- legacy `eval_node.sh`와 generic `eval_lrnode_compare.sh`의 기본 old LR-NODE checkpoint path는 archive 위치를 보도록 갱신했다.

