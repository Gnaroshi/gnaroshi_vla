# FastV for SimVLA

This directory implements a training-free FastV comparison for frozen
SimVLA-LIBERO. It is an official-algorithm adaptation, not an official FastV
SimVLA release.

## Evidence contract

The original FastV paper defines visual-token pruning after transformer layer
`K` with ratio `R`. Its principal no-loss setting is `K=2, R=0.5`. Section 4.1
ranks visual tokens using received attention, and Section 4.2 counts both
attention and FFN savings from physically shortening the deeper-layer sequence.

The pinned official Hugging Face implementation differs in one detail: it uses
the last query token at layer `K` after head averaging. SimVLA does not perform
autoregressive text generation during action inference, so that last token is
only the last instruction position. The primary row therefore follows the
documented dual-system VLA adaptation in Latent Bridge Appendix H:

- average text-to-image attention over the first `K` layers;
- `K=2`, `R=0.5`;
- physically compact the LLM sequence after layer 2;
- restore removed positions as zeros before the frozen action head.

`fastv_k2_r50_hf_last_token` is retained as a diagnostic row for the current
official HF-code scoring rule. It is not part of the default paper comparison.

## SimVLA mapping

The frozen SimVLA condition path presents two valid camera views. After the
vision connector this gives 72 visual positions, followed by 50 non-visual
positions, for 122 positions in total. The primary FastV row keeps 36 visual
positions and all 50 non-visual positions. Layers 1-2 therefore process 122
positions and layers 3-32 process 86 positions.

The frozen action transformer was trained on all 122 condition positions and
concatenates them densely with action tokens. Returning only 86 positions would
change that pretrained interface and shift its positional embeddings. The
adapter therefore scatters the 86 final hidden states back to their original
positions and fills the 36 removed visual positions with zero, matching the
dual-system adaptation contract used by Latent Bridge.

This preserves the external SimVLA protocol:

- a current observation is consumed at every policy query;
- action horizon `H=10`;
- execution horizon `R=5`;
- 10 flow-matching evaluations;
- no action-chunk replay and no query skipping.

## What is and is not claimed

The visual sequence is physically shorter in the final 30 language-model
layers, so this implementation can measure real VLM latency reduction. A
zero-only hook that leaves the sequence length unchanged is not used.

The original FastV paper did not evaluate SimVLA or a dense dual-system action
head. Performance preservation is therefore an experimental question. Until a
paired LIBERO run is complete, this adapter makes no success-rate claim.

## Preflight

```bash
cd /home/mingyujung/private/gnaroshi_vla_worktrees/simvla_fastv

SIMVLA_UPSTREAM_ROOT=/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream \
FASTV_UPSTREAM_ROOT=$PWD/architectures/fastv/upstream \
bash architectures/simvla/wrappers/simvla_fastv_eval.sh \
  --contract-only \
  --output /tmp/simvla_fastv_contract
```

## Bounded LIBERO smoke

First run the real-checkpoint condition-only smoke. It does not start LIBERO:

```bash
SIMVLA_FASTV_SMOKE_RUN=1 \
bash architectures/simvla/wrappers/simvla_fastv_smoke.sh \
  --output /non_overlay/results/simvla/fastv/real_checkpoint_smoke \
  --device cuda
```

Only continue when it reports
`SIMVLA_FASTV_REAL_CHECKPOINT_SMOKE_PASS`. Then run the bounded LIBERO smoke.

Use a new output directory for every run.

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SIMVLA_UPSTREAM_ROOT=/path/to/gnaroshi_vla/architectures/simvla/upstream
export LIBERO_ROOT=/path/to/LIBERO
export FASTV_UPSTREAM_ROOT=$PWD/architectures/fastv/upstream

SIMVLA_FASTV_EVAL_RUN=1 \
bash architectures/simvla/wrappers/simvla_fastv_eval.sh \
  --output /non_overlay/results/simvla/fastv/smoke \
  --norm-stats "${SIMVLA_UPSTREAM_ROOT}/norm_stats/libero_norm.json" \
  --suite libero_10 \
  --rows baseline_k1 fastv_k2_r50 \
  --max-tasks 1 \
  --num-trials 1 \
  --device cuda
```

For the paper comparison, remove `--max-tasks 1`, set `--num-trials 20`, and
run the same paired rows for each approved seed. The output records source
hashes, environment metadata, per-episode outcomes, actual token counters,
latencies, and videos when `--save-video` is enabled.
