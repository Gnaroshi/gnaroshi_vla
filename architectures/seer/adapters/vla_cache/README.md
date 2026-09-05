# Seer VLA-Cache adapter

This directory documents the training-free VLA-Cache port for Seer. Runtime
code lives beside Seer's custom GPT-2 implementation because it must execute
the model's exact `Conv1D` QKV projections, custom attention mask, residual
blocks, and output reconstruction without changing checkpoint keys.

## Source lock

- Official repository: <https://github.com/siyuhsu/VLA-Cache>
- Official commit: `a4909880573868dee2769343d52e793c0341678b`
- Official Transformers fork: <https://github.com/siyuhsu/transformers>
- Fork branch: `vla-cache-openvla`
- Fork commit: `2302fce58afa3a4f8461625b1394f9e9c8a7f1ea`
- SimVLA reference port: branch `feat/simvla-real-vla-cache-20260904`,
  implementation commit `9e36d570198b96811292420a39afb2ab130ff566`

## Preserved method contracts

1. The first policy query computes every token and initializes full per-layer
   K/V caches.
2. Later queries select temporally stable visual tokens, remove tokens judged
   task-relevant by the previous decoder attention, and reuse only the
   remaining positions.
3. Reused positions retain prior K/V. Current non-reused positions overwrite
   their K/V. Active queries continue to attend to the complete key sequence.
4. Pruning occurs at layers `2,6,9,11` by default. The amount reused at each
   layer follows the published attention-entropy schedule with growth factor
   `0.55`.
5. Removed final hidden states are restored from the preceding query so Seer's
   action-token output shape and existing action head remain unchanged.
6. Cache state is reset at every LIBERO episode boundary.

## Required Seer adaptation

OpenVLA-OFT and the SimVLA port expose spatially aligned image-grid tokens to
their language decoder. Seer does not: MAE's 196 spatial patches are compressed
into six learned Perceiver latents per camera, and the causal action transformer
receives those six latents plus one CLS token per camera and timestep.

It would therefore be incorrect to label Seer's seven visual-condition token
indices as image patches or to copy SimVLA's 6x6 raw-pixel grid. This port uses:

- **stability:** cosine similarity between the same projected Seer
  visual-condition token slot in consecutive policy queries;
- **task relevance:** previous reference-layer attention from the action-query
  tokens actually selected by the growing seven-step evaluation history;
- **alignment:** fixed relative-history timestep and camera slot;
- **scaled counts:** official `150/256` stable and `100/256` relevant fractions,
  yielding four stable and three protected tokens per seven-token camera group.

This is an explicit Seer architecture adaptation of VLA-Cache, not an
author-provided Seer implementation and not raw-image patch matching.

## Modes

- `off`: unmodified Seer GPT-2 path.
- `matched_full`: indexed cache runtime with all tokens recomputed. This is the
  implementation-overhead and numerical-parity control.
- `reuse`: actual stable-minus-task-relevant token pruning and K/V reuse.

The method is inference-only and introduces no parameters. Existing Seer
checkpoint state-dict keys are unchanged.

## Entry point

Use `architectures/seer/wrappers/vla_cache/run_seer_vla_cache_libero_long.sh`.
Set `PREFLIGHT_ONLY=1` first for asset, import, and configuration checks.

```bash
cd /home/mingyujung/private/gnaroshi_vla_worktrees/seer_vla_cache
source /home/mingyujung/miniconda3/etc/profile.d/conda.sh
conda activate seer_libero

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PREFLIGHT_ONLY=1 \
bash architectures/seer/wrappers/vla_cache/run_seer_vla_cache_libero_long.sh
```

Remove `PREFLIGHT_ONLY=1` to run the same public checkpoint and episode set in
the three modes. Heavy results are written beneath the shared Seer result root;
the worktree contains only code and lightweight documentation.
