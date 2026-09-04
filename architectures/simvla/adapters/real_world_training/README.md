# SimVLA real-world baseline and LatentLoop training

This package prepares a controlled comparison on the 40 `stackcupanddoll`
teleoperation trajectories. It does not contain a robot-control entry point.

## Initialization contract

There is no scratch-head or reinitialization ablation in this protocol. Every
baseline parameter is loaded from the complete released
`YuankaiLuo/SimVLA-LIBERO` checkpoint. Loading fails when Hugging Face reports
any missing, unexpected, or shape-mismatched tensor. The released VLM is then
frozen and the already initialized action transformer is fine-tuned on the real
demonstrations. The compact real checkpoint stores only that transformed head
and records the SHA-256 of its complete official parent.

## Data contract

- 40 trajectories are split by episode into 32 train and 8 validation episodes.
- The source and deployment control rate is 15 Hz.
- Exterior and wrist RGB are kept in that order and follow the same
  resize-with-pad-224 then bicubic-384 preprocessing as evaluation.
- State is `[TCP xyz, TCP rotation-vector, +finger opening, -finger opening]`.
- `control[:6]` is an absolute joint command and is deliberately ignored.
- Each action is reconstructed from consecutive TCP poses as
  `inv(T_current) @ T_next`, then represented as local xyz / 0.02 m, XYZ Euler
  / 0.05 rad, and `+1=open,-1=close`.
- Pose-label clipping is rejected by default. The converter writes the measured
  extrema before stopping so that a scale change can be reviewed explicitly.

## Efficient training path

The VLM is frozen, so its exact FP32 action conditions are computed once and
memory-mapped. Baseline fine-tuning then trains only the existing action
transformer from this cache. The selected defaults are 3,000 optimizer steps,
local batch 4, and effective global batch 64. Gradient accumulation is derived
from the requested GPU count so that four GPUs use accumulation 4 and one GPU
uses accumulation 16. These are an engineering protocol for the available 40
demonstrations, not an official SimVLA paper setting.

After the baseline is fixed, the Condition Updater (`K_C=2`) and Generation
Updater (`N_G=3`, full evaluations at solver indices 0, 4, and 8) train in
parallel when at least two GPUs are available and sequentially on one GPU. Both
checkpoints must name the exact real baseline SHA-256 as their teacher. Loss
magnitudes are measured deterministically before training and normalized to
equal contribution; no unexplained hand-selected loss weights are embedded in
the wrapper.

## Comparison contract

Baseline and LatentLoop share all of the following:

- official parent checkpoint and real action-transformer overlay;
- normalization and image/state preprocessing;
- fresh ten-action output on every policy query;
- execution of the first five actions before the next query (`H=10,R=5`);
- deterministic per-query initial flow noise;
- camera order, task instruction, workspace, and safety gates.

Only the internal compute schedule changes: baseline uses `K_C=1,N_G=10`, and
LatentLoop uses `K_C=2,N_G=3`. Real trials should alternate method order within
paired scene resets and record success, policy latency, deadline misses, and
all module-call counters. The deployment manifest starts with live execution
disabled even after training completes.

## Commands

```bash
export SIMVLA_REAL_RAW_DATA=/path/to/stackcupanddoll
bash architectures/simvla/wrappers/train_real_stackcupanddoll.sh --preflight

SIMVLA_REAL_TRAIN_RUN=1 \
bash architectures/simvla/wrappers/train_real_stackcupanddoll.sh --all
```

Before live use, run both artifact preflights and the read-only hardware profile
from `architectures/simvla/wrappers/deploy_latentloop_real.sh`. Live approval is
intentionally not part of the training wrapper.
