# Seer LR-NODE

This document covers the release-facing LR-NODE workflow integrated into
`architectures/seer/upstream`.

LR-NODE trains a lightweight visual-delta encoder and latent update module
against a frozen Seer teacher. Evaluation can periodically run the full Seer
policy and use LR-NODE on intervening environment steps. The existing Seer
action decoder remains shared.

## Prerequisites

Activate the known working environment:

```bash
conda activate seer_libero
```

Provide these machine-specific assets as environment variables:

```bash
export LIBERO_PATH=/path/to/LIBERO
export VIT_CHECKPOINT_PATH=/path/to/mae_pretrain_vit_base.pth
export ROOT_DIR=/path/to/converted-dataset-parent
export LRNODE_PROTOCOL_ROOT=/path/to/seer-lrnode-results
```

`ROOT_DIR` must contain `libero_10_converted/`. The data loader resolves the
training dataset as `${ROOT_DIR}/libero_10_converted`.

Checkpoints and result directories are intentionally excluded from Git.

## Frozen-Teacher Distillation

Set the baseline Seer checkpoint explicitly. The adapter protocol freezes every
non-LR-NODE parameter and trains only `lrnode_delta_encoder` and
`lrnode_dynamics`.

```bash
cd architectures/seer/upstream

CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT=/path/to/seer_baseline.pth \
BASELINE_CKPT_ID=33 \
LIBERO_PATH="${LIBERO_PATH}" \
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
ROOT_DIR="${ROOT_DIR}" \
LRNODE_PROTOCOL_ROOT="${LRNODE_PROTOCOL_ROOT}" \
bash scripts/LIBERO_LONG/Seer/distill_node.sh
```

The default run has 40 epochs and `START_SAVE_CHECKPOINT=25`. Because
`train.py` saves only when `epoch > start_save_checkpoint`, it writes
`26.pth` through `39.pth`.

The latest successful run metadata is written to:

```text
${LRNODE_PROTOCOL_ROOT}/train/_latest/distill_node.env
```

## Evaluation

Evaluate a full-forward baseline, the adapter checkpoint at K=1, and periodic
LR-NODE updates:

```bash
cd architectures/seer/upstream

CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT=/path/to/seer_baseline.pth \
OURS_CKPT=/path/to/lrnode_adapter.pth \
LRNODE_EVAL_BASE_CKPT=/path/to/seer_baseline.pth \
LIBERO_PATH="${LIBERO_PATH}" \
VIT_CHECKPOINT_PATH="${VIT_CHECKPOINT_PATH}" \
LRNODE_PROTOCOL_ROOT="${LRNODE_PROTOCOL_ROOT}" \
LRNODE_QUERY_INTERVALS_STR="2 3 4" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh
```

K=1 uses full Seer forwarding at every environment step. For K greater than
one, the full model runs at refresh steps and LR-NODE updates the cached latent
on skipped steps. Evaluation writes the aggregate summary, per-episode metrics,
latency profile, live progress, and optional videos under the selected result
root. Head metrics distinguish full-forward head calls, skip-path head calls,
and their total.

Useful overrides:

```bash
export RESULT_ROOT=/path/to/evaluation-output
export NODE_NUM=4
export MASTER_PORT=12452
export SAVE_VIDEO=1
export LRNODE_EVAL_STEP_LOG=1
export LRNODE_EVAL_PROFILE_FULL_ACTION_HEAD=1
```

## Official Seer Checkpoint Protocol

The official protocol trains a separate frozen LR-NODE adapter for each
released Seer checkpoint. It does not reuse an adapter trained in another
teacher's latent coordinate system.

Required assets:

```bash
export OFFICIAL_SEER_REPO=/path/to/clean/Seer
export OFFICIAL_SIMVLA_REPO=/path/to/clean/SimVLA
export OFFICIAL_CKPT_ROOT=/path/to/Seer_LIBERO_LONG/checkpoints
export VIT_CHECKPOINT_PATH=/path/to/mae_pretrain_vit_base.pth
export LIBERO_TRAIN_ROOT=/path/to/converted-dataset-parent
export LIBERO_PATH=/path/to/LIBERO
```

Run audit, K=1 baselines, adapter training, and K=1/K=4 evaluation:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
STAGES="audit baseline train eval" \
bash architectures/seer/wrappers/lrnode/official_seer_k4_protocol.sh
```

The baseline success-rate references are diagnostic and non-blocking. File
hashes, checkpoint structure, dataset structure, and adapter-only parameter
contracts remain strict validation checks.

## Checkpoint Loading Contract

Adapter evaluation loads two files:

1. A base Seer checkpoint containing the existing Seer parameters and no
   LR-NODE parameters.
2. An adapter checkpoint containing only
   `module.lrnode_delta_encoder.*` and `module.lrnode_dynamics.*`.

Evaluation rejects missing or unexpected keys for these two checkpoint kinds.
This prevents silently pairing an adapter with an incomplete base model.
