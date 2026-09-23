# Seer LatentLoop paper wrappers

This worktree is the simulation source of truth for the Seer LatentLoop paper rows.

## Active entrypoints

| Scope | Entrypoint | Contract |
|---|---|---|
| Adapter training | `lrnode/distill_node.sh` | Frozen Seer teacher; LatentLoop-only optimization |
| Generic evaluation | `lrnode/eval_lrnode_compare.sh` | Explicit teacher/adapter/K/result root |
| Cross-suite scratch reproduction | `lrnode/run_libero_suite_baseline_latentloop.sh` | Spatial/Object/Goal training and evaluation |
| Spatial selected-teacher reproduction | `lrnode/run_spatial_teacher33_adapter_egl50.sh` | teacher33 + adapter39, seeds 42/43/44 |
| Object final evaluation | `lrnode/run_libero_object_mujoco332_eval.sh` | MuJoCo 3.3.2, seeds 42/43/44 |

The selected checkpoints are immutable inputs under
`/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/artifacts/checkpoints/seer/paper`.
Final compact result evidence is under the sibling shared `results/seer/paper` tree.
New runs use `results/seer/latentloop`, never the compact evidence directory.

The directory name `lrnode` is retained for source and checkpoint compatibility.
The paper-facing method name is **LatentLoop**. Completed FastV, V0/V1/V2,
runtime-aligned, comparison, and one-off resume entrypoints are isolated under
`legacy/`; they are not supported launch commands.
