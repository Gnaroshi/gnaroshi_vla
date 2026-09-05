# SimVLA real-world baseline and LatentLoop training

This package prepares a controlled comparison on the 40 `stackcupanddoll`
teleoperation trajectories. Live robot control is implemented separately in
`architectures/simvla/adapters/latentloop_real_deploy` and remains disabled
until its artifact, hardware, timing, baseline-canary, and operator approvals
all pass.

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
- The source and deployment control rate is 15 Hz. Source RGB, pose, and
  control records remain in their native synchronized order; the converter
  does not snap an already-15 Hz capture stream onto a second time grid.
- A transition whose capture interval differs from one nominal period by more
  than half a period is excluded. Every H=10 window crossing such a gap is
  omitted from training and counted in the dataset manifest.
- Exterior and wrist RGB are kept in that order. Cache creation, training, and
  deployment call the same JPEG95/subsampling0, resize-with-pad-224 then
  bicubic-384 transform and ImageNet normalization. The processor is used for
  text tokenization, not as an alternative image-normalization path.
- State is `[TCP xyz, TCP rotation-vector, +finger opening, -finger opening]`.
- `control[:6]` is an absolute joint command and is deliberately ignored.
- Each action is reconstructed from consecutive TCP poses as
  `inv(T_current) @ T_next`, then represented as local xyz / 0.02 m, XYZ Euler
  / 0.05 rad. The gripper target is the synchronized current-frame command
  `1 - 2 * command_t`, so it remains continuous with `+1=open,-1=close`.
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
equal initial contribution. This balances numerical scales; it does not prove
that the chosen loss balance is optimal for physical task success.

A subsequent 10,000-step coupling stage freezes both trained updaters and
trains only the Generation Updater's existing 128-by-128 condition-code
projection (16,384 parameters). Its code is the same delta-encoder output used
by the Condition Updater. The objective is local-oracle hidden-state MSE under
the predicted condition. It is not end-to-end joint training and cannot by
itself establish that the approximate condition is correct. Deployment uses
this coupled checkpoint, not the uncoupled generation checkpoint.

Legacy data and checkpoints with next-frame gripper labels are not deployment
inputs. The wrapper requires dataset v3, cache v2, and v2 real checkpoint
formats. Optional cache migration reuses only frozen conditions after checking
image, proprioception, instruction, and record identities. A separate exact
condition check covers one query from each of all 40 episodes before reuse.

Training resume is deliberately not part of the scientific wrapper contract.
An interrupted optimizer run does not preserve the exact distributed sampler
and per-rank random-number state. The wrapper therefore moves an incomplete
run into `quarantine/` and restarts that stage from step zero. Completed stages
with a validated `run_summary.json` are reused.

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

For a machine that receives an already audited compact dataset, set
`SIMVLA_REAL_DATASET=/path/to/converted_dataset`. The wrapper verifies its
manifest and normalization file and skips raw-data conversion; the original
58 GB teleoperation directory does not need to be duplicated on that machine.

Before live use, run both artifact preflights and the read-only hardware profile
from `architectures/simvla/wrappers/deploy_latentloop_real.sh`. Live approval is
intentionally not part of the training wrapper.
