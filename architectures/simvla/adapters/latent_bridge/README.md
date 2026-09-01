# SimVLA Latent Bridge adaptation

This directory adapts the feature-transition algorithm from the official
Latent Bridge repository to the frozen SimVLA action-condition boundary. Both
upstream repositories remain unmodified, ignored nested clones:

- SimVLA: `architectures/simvla/upstream`
- Latent Bridge: `architectures/latent_bridge/upstream`

The integration verifies the pinned Latent Bridge commit and source hashes
before importing its DiT blocks. Checkpoints and datasets additionally record a
hash of this SimVLA integration. This is an **official-algorithm SimVLA
adaptation**, not official Latent Bridge SimVLA code or a released checkpoint.

## Runtime contract

- Base policy: frozen `YuankaiLuo/SimVLA-LIBERO`.
- Prediction target: SimVLA's 72 image-token positions from the pre-`vlm_proj`
  action condition. The complete condition has shape `[B,122,960]`; the other
  50 positions are copied from the latest full-policy anchor.
- Stable context: output of frozen SmolVLM text layer 10, captured by an
  external forward hook during a full refresh.
- Bridge: official `DiTCrossBlock` and `DiTFinalLayer` wiring with width 768,
  12 heads, and 12 blocks.
- Recurrence: full condition every `K_C=4`; between anchors,
  `z_next = z_previous + bridge(z_previous, z_stable, q_current, a_previous)`.
- Control protocol: every policy query still produces a fresh H=10 action
  chunk using 10 Euler flow steps, and LIBERO executes R=5 actions. No previous
  action chunk is replayed.
- `baseline_k1`: exact full SimVLA condition and action path at every query;
  bridge code and hooks are absent.
- Default bridge size: 183,584,448 trainable parameters (183.584448M).

The primary `image_only` target matches the released Latent Bridge runtime. An
`all` target over all 122 SimVLA condition positions is retained only as a
SimVLA-specific ablation via `--token-mode all`.

## Training protocol

The paper-comparison path follows the released algorithm in two stages:

1. Collect R0 transitions from frozen, full-SimVLA on-policy rollouts. The
   default is 30 episodes per LIBERO task.
2. Train R0 for 100 epochs with batch 64, AdamW, learning rate `3e-4`, weight
   decay `0.01`, and an epoch-wise cosine schedule.
3. Roll out R0 at `K_C=4` while querying frozen full SimVLA only for teacher
   conditions. This produces the DAgger set.
4. Initialize R1 from R0 weights and train on R0 plus DAgger data with learning
   rate `3e-5`.
5. Compare `baseline_k1` and `latent_bridge_k4` on the same LIBERO task,
   initial-state, environment-seed, and action-noise tuples.

The released CLI describes direct feature-delta prediction (`--no_flow`) as
recommended, although its older pipeline script omits that flag. Direct delta
is therefore the default here; `--flow-matching` is an explicit ablation.

Two differences from the old released scripts are deliberate and recorded:

- The released runtime consumes image features from layer 10, while old data
  collection defaults are inconsistent with that layer. Collection, training,
  and inference here all lock layer 10.
- The released trainer randomly splits transitions. This adaptation splits by
  episode (90/10, seed 42) so adjacent transitions from one rollout cannot leak
  across train and held-out sets.

`--precision bf16` and `--gradient-accumulation-steps` are hardware adaptations;
the default remains FP32 with no accumulation. The effective batch is written
to `run_config.json`.

## Bootstrap-only path

`prepare_cache.py` converts an existing training-demonstration cache into a
sidecar without copying RGB or condition tensors. This path is useful for a
bounded implementation smoke test, but it is **not** the primary official-style
R0 dataset and must not be reported as the Latent Bridge comparison.

## Entry points

All heavy entry points require an explicit guard variable:

```bash
# R0 on-policy collection
SIMVLA_LATENT_BRIDGE_COLLECT_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_collect_sync.sh \
  --output /path/to/r0_sync \
  --norm-stats architectures/simvla/upstream/norm_stats/libero_norm.json \
  --suite libero_10 --num-trials 30 --device cuda

# R0 training
SIMVLA_LATENT_BRIDGE_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_train.sh \
  --sync-root /path/to/r0_sync \
  --output /path/to/r0_train \
  --epochs 100 --batch-size 64 --learning-rate 3e-4 --device cuda

# DAgger collection with R0. Use only latent_bridge_k4 for collection.
SIMVLA_LATENT_BRIDGE_EVAL_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_eval.sh \
  --output /path/to/r1_dagger_rollout \
  --norm-stats architectures/simvla/upstream/norm_stats/libero_norm.json \
  --bridge-checkpoint /path/to/r0_train/best.pt \
  --rows latent_bridge_k4 --suite libero_10 --num-trials 30 \
  --refresh-every 4 --collect-dagger-teacher --device cuda

# R1 training
SIMVLA_LATENT_BRIDGE_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_train.sh \
  --sync-root /path/to/r0_sync \
  --dagger-root /path/to/r1_dagger_rollout/latent_bridge_k4/dagger \
  --resume /path/to/r0_train/best.pt --weights-only-resume \
  --output /path/to/r1_train \
  --epochs 100 --batch-size 64 --learning-rate 3e-5 --device cuda

# Paired final comparison. Reserve a trial range not used for DAgger.
SIMVLA_LATENT_BRIDGE_EVAL_RUN=1 \
bash architectures/simvla/wrappers/simvla_latent_bridge_eval.sh \
  --output /path/to/final_eval \
  --norm-stats architectures/simvla/upstream/norm_stats/libero_norm.json \
  --bridge-checkpoint /path/to/r1_train/best.pt \
  --rows baseline_k1 latent_bridge_k4 --suite libero_10 \
  --num-trials 10 --trial-offset 30 --refresh-every 4 --device cuda
```

No collection, training, DAgger rollout, or LIBERO evaluation runs
automatically during installation or tests.
