# LatentLoop for Seer

LatentLoop distills an observation-conditioned latent updater from a frozen Seer
teacher. At evaluation time, Seer performs a full semantic refresh every `K`
control steps; the lightweight updater advances the cached action condition on
intermediate steps, and the existing frozen Seer action head decodes every step.

## Environment and assets

Use the known-working Seer environment:

```bash
conda activate seer_libero
```

Required inputs are:

- a frozen Seer teacher checkpoint;
- the converted LIBERO dataset used to train that teacher;
- the ViT-MAE checkpoint;
- a LIBERO checkout;
- a result root outside the source repository.

The wrappers accept explicit paths. A typical sd1 setup is:

```bash
export LIBERO_PATH=/home/mingyujung/private/LIBERO
export VIT_CHECKPOINT_PATH=/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth
export ROOT_DIR=/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/LIBERO_DATASETS
export LRNODE_PROTOCOL_ROOT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/seer/latentloop/runs
```

`ROOT_DIR` must contain the selected dataset directory. For the default
`DATASET=libero_10_converted`, the loader resolves
`${ROOT_DIR}/libero_10_converted`.

## Adapter training

The stable primitive is:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT=/path/to/teacher.pth \
LIBERO_PATH="${LIBERO_PATH}" \
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
ROOT_DIR="${ROOT_DIR}" \
LRNODE_PROTOCOL_ROOT="${LRNODE_PROTOCOL_ROOT}" \
bash architectures/seer/wrappers/lrnode/distill_node.sh
```

The adapter protocol freezes Seer and trains only the visual-delta encoder and
latent dynamics modules. The checkpoint is therefore paired with the exact
teacher used during distillation; do not combine an adapter with another teacher.

## Evaluation

Use the generic comparison wrapper with explicit checkpoints:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT=/path/to/teacher.pth \
OURS_CKPT=/path/to/latentloop_adapter.pth \
LRNODE_EVAL_BASE_CKPT=/path/to/teacher.pth \
LRNODE_QUERY_INTERVALS_STR="4" \
RUN_BASELINE=1 \
RUN_OURS_FULL=0 \
SAVE_VIDEO=0 \
LIBERO_PATH="${LIBERO_PATH}" \
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
LRNODE_PROTOCOL_ROOT="${LRNODE_PROTOCOL_ROOT}" \
bash architectures/seer/wrappers/lrnode/eval_lrnode_compare.sh
```

For a paper comparison, hold the teacher, adapter, renderer, suite, episode set,
evaluation seed, temporal ensembling, and action protocol fixed. `K=1` is full
Seer. `K=4` performs one full refresh followed by three LatentLoop updates.

## Repository-specific launchers

Paper simulation launchers are isolated in their method worktrees. The main
worktree retains only generic primitives and real-world training/deployment
entrypoints. The ignored `codex_outputs/seer/code_inventory.md` records the
current source-of-truth worktree for each paper baseline.
