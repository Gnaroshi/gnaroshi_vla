# SimVLA Latent Bridge adaptation

This directory ports the released Latent Bridge feature-transition algorithm to
the frozen SimVLA action-condition interface. It is an
**official-algorithm SimVLA adaptation**, not official Latent Bridge SimVLA
code. Both upstream repositories remain unmodified:

- SimVLA: `architectures/simvla/upstream`
- Latent Bridge: `architectures/latent_bridge/upstream`
- pinned Latent Bridge commit: `ed556014aa96bae8ed85768194f02360389b9365`

The integration verifies the Latent Bridge commit and source hashes before
loading its DiT blocks. Datasets and checkpoints also hash this adaptation and
the frozen SimVLA interfaces.

## Evidence hierarchy

The paper, released code, and SimVLA measurements do not define one identical
configuration. This port resolves them explicitly instead of mixing defaults.

| Contract | Evidence | SimVLA decision |
|---|---|---|
| Feature bridge | Released `SingleStepDiT`: width 768, 12 heads, 12 blocks, zero-initialized residual output | Reuse the released blocks and wiring; adapt feature width to 960 |
| Feature recurrence | GR00T feature bridge and its DAgger collection use `f=3` | Collect R1 DAgger data only with `latent_bridge_f3` |
| Latest `f=4` operating point | The current paper/README uses `f=4` for the pi0.5 **KV bridge** | Evaluate the same feature R1 checkpoint at `f=4` as a matched 75% VLM-call-saving point; do not call it official SimVLA code |
| Token scope | GR00T copies text tokens because their adjacent cosine is reported above 0.9999 | Predict all 122 fused SimVLA condition positions: 20-episode SimVLA measurement gives 0.94916 for 72 visual positions and 0.93556 for the other 50 positions |
| Feature loss | Paper feature-bridge recipe: delta MSE plus reconstructed-feature cosine with weight 1 | Use `MSE(delta_pred, delta_target) + (1 - cosine(z_pred, z_target))` |
| Optimizer | Paper appendix: AdamW, weight decay `1e-4`, cosine LR, gradient clipping 1 | Enforce these values |
| R0/R1 | Paper: R0 `200` epochs at `3e-4`; R1 schedule of `100` epochs at `3e-5` | Expose required `--stage r0` and `--stage r1`; initialize R1 from R0 weights and reset optimizer/epoch state |
| Batch | Paper effective batch 64 | Default micro-batch 4 with 16-step accumulation; execution fails if effective batch differs unless explicitly labeled as a deviation |
| Final evaluation | Current official results use seeds 0, 1, 2 and 20 episodes per task | Evaluate paired baseline/f3/f4 rows for each seed with 20 episodes per task |

The pinned repository contains stale/conflicting shell defaults (for example,
100 versus 200 R0 epochs and weight decay `0.01` versus the paper's `1e-4`).
The paper is treated as the scientific hyperparameter contract; the pinned
source supplies implementation mechanics and provenance.

## Runtime contract

- Base policy: frozen `YuankaiLuo/SimVLA-LIBERO`.
- Prediction boundary: all 122 positions of SimVLA's `[B,122,960]` fused action
  condition before action generation.
- Stable context: frozen SmolVLM text-layer 10 output captured by an external
  hook. This candidate index is inherited from the released bridge runtime and
  is accepted only when the SimVLA sync probe verifies adjacent cosine >0.999.
- Bridge: released `DiTCrossBlock` and `DiTFinalLayer`, width 768, 12 heads,
  12 blocks, direct residual-delta prediction.
- Recurrence: `z_next = z_previous + bridge(z_previous, z_stable,
  q_current, a_previous)`.
- Control protocol: every policy query still generates a fresh H=10 action
  chunk using SimVLA's 10 Euler steps, and LIBERO executes R=5 actions. No
  previous action chunk is replayed.
- Released action input semantics are retained: the bridge receives the first
  action of the previously predicted chunk. This is not silently replaced by a
  SimVLA-specific alternative.
- Default all-token bridge size: 183,622,848 trainable parameters.
- Primary RTX comparison runtime: bf16 bridge with eager execution, matching
  the current ours measurement stack. `--compile-bridge` records a separate
  `official_optimized_secondary` axis and must not be mixed with eager latency.

Canonical evaluation rows bind the row name to the refresh period; there is no
global refresh argument that can silently change their meaning:

| Row | Full VLM period | VLM calls saved | Role |
|---|---:|---:|---|
| `baseline_k1` | 1 | 0% | Exact frozen SimVLA condition at every query |
| `latent_bridge_f3` | 3 | 66.7% | Released feature-bridge recurrence / R1 collection point |
| `latent_bridge_f4` | 4 | 75% | Matched `K_C=4` backbone-call budget |

`latent_bridge_f4` matches ours only on the condition-backbone call axis. It
does not reduce SimVLA's action-generation evaluations and therefore is not an
equal-total-compute match to a coupled `(K_C, N_G)` operating point. Reports
must include full VLM calls, action-network calls, latency, and success rate.

## Training protocol

1. Collect 30 full-SimVLA episodes per LIBERO task for R0.
2. Train R0 with the `paper_feature_bridge/r0` recipe.
3. Roll out R0 at `f=3` for 30 episodes per task and query frozen full SimVLA
   only to save teacher conditions.
4. Initialize R1 from R0 weights and train on R0 plus DAgger transitions with
   the `paper_feature_bridge/r1` recipe.
5. Evaluate one R1 checkpoint with paired `baseline_k1`, `latent_bridge_f3`,
   and `latent_bridge_f4` rows for seeds 0, 1, and 2.

Direct delta is the production path. `--flow-matching`, `image_only`, changed
effective batch, shortened epochs, or bounded `--max-steps` require
`--allow-recipe-deviation` and are labeled as ablations/smoke tests in the
checkpoint.

The adaptation splits R0 data by episode (90/10, seed 42), rather than randomly
splitting adjacent transitions, to prevent one rollout from crossing the train
and held-out sets.

## Commands

All heavy entry points require an explicit guard variable.

```bash
# R0 collection: 30 episodes/task.
SIMVLA_LATENT_BRIDGE_COLLECT_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_collect_sync.sh \
  --output /path/to/r0_sync \
  --norm-stats architectures/simvla/upstream/norm_stats/libero_norm.json \
  --suite libero_10 --num-trials 30 --stable-layer-index 10 --device cuda

# R0: defaults resolve to 200 epochs, LR 3e-4, effective batch 64.
SIMVLA_LATENT_BRIDGE_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_train.sh \
  --stage r0 --sync-root /path/to/r0_sync --output /path/to/r0_train \
  --device cuda

# R1 DAgger collection must use f=3.
SIMVLA_LATENT_BRIDGE_EVAL_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_eval.sh \
  --output /path/to/r1_dagger_rollout \
  --norm-stats architectures/simvla/upstream/norm_stats/libero_norm.json \
  --bridge-checkpoint /path/to/r0_train/best.pt \
  --rows latent_bridge_f3 --suite libero_10 --num-trials 30 \
  --collect-dagger-teacher --seed 0 --device cuda

# R1: defaults resolve to 100 epochs, LR 3e-5, effective batch 64.
SIMVLA_LATENT_BRIDGE_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_train.sh \
  --stage r1 --sync-root /path/to/r0_sync \
  --dagger-root /path/to/r1_dagger_rollout/latent_bridge_f3/dagger \
  --resume /path/to/r0_train/best.pt --weights-only-resume \
  --output /path/to/r1_train --device cuda

# Final paired comparison: 3 seeds x 20 episodes/task.
for seed in 0 1 2; do
  SIMVLA_LATENT_BRIDGE_EVAL_RUN=1 \
  bash architectures/simvla/wrappers/simvla_latent_bridge_eval.sh \
    --output "/path/to/final_eval/seed${seed}" \
    --norm-stats architectures/simvla/upstream/norm_stats/libero_norm.json \
    --bridge-checkpoint /path/to/r1_train/best.pt \
    --rows baseline_k1 latent_bridge_f3 latent_bridge_f4 \
    --suite libero_10 --num-trials 20 --seed "${seed}" --device cuda
done
```

No collection, training, DAgger rollout, or LIBERO evaluation runs during
installation or unit tests.
