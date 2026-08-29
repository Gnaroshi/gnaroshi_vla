# Seer and LatentLoop real-world deployment

This path deploys either the basketball Seer baseline or its paired LatentLoop
adapter without changing the known-working 3DFlow-Seer GUI, camera, UR5e, or
Robotiq implementation.

## Fixed protocol

- Task: `Pick up the red ball and place it in the basketball hoop`
- Default pair: teacher `37.pth` + the adapter epoch `39.pth` trained from that
  teacher
- Alternative valid pairs: teacher `34.pth` + its adapter `39.pth`, and teacher
  `35.pth` + its adapter `39.pth`
- Baseline mode: the selected teacher only, full Seer every step (`K=1`)
- LatentLoop mode: the same teacher plus its paired adapter (`K=4`)
- Schedule: full Seer at steps `0, 4, 8, ...`; LatentLoop at the intervening
  steps using fresh exterior image, wrist image, and proprioception
- Action decoder and temporal ensemble: the existing Seer implementation
- Default target control rate: `15 Hz` (`66.67 ms` period)
- Optional target control rate: `40 Hz` (`25 ms` period)
- Robot action limits: `max_rel_pos=0.02`, `max_rel_orn=0.05`
- Model environment on the inference computer: the existing `conda activate seer`

Teacher and adapter files are a strict pair. Baseline mode verifies only the
teacher and shared ViT; LatentLoop mode additionally verifies the paired adapter
and architecture before opening the robot or cameras.

## Repository layout

- `architectures/seer/adapters/latentloop_real_deploy/`: LatentLoop controller
  and GUI integration
- `architectures/seer/third_party/3dflow_real_deploy/`: byte-identical snapshot
  of the real deployment source from `/home/jbr/3DFlow-Seer`
- `architectures/seer/upstream/scripts/REAL/deploy_ll_gui.sh`: self-contained
  deployment configuration and launcher
- `artifacts/seer/real_world/basketball/baseline/`: teacher checkpoints
- `artifacts/seer/real_world/basketball/latentloop/`: teacher-specific adapters
- `artifacts/seer/real_world/basketball/shared/`: shared ViT checkpoint
- `real_deploy_results/baseline/`: baseline logs and results
- `real_deploy_results/latentloop/`: LatentLoop logs and results

The artifact directory ignores every file except its README, manifest, and
`.gitignore`. Consequently, teacher, adapter, and ViT checkpoints cannot be
added by a normal `git add`, while JSON deployment results can be shared.

## Required artifact names

Copy the following files into
`~/gnaroshi_vla/artifacts/seer/real_world/basketball/` on the inference computer:

```text
shared/mae_pretrain_vit_base.pth
baseline/teacher_34.pth
baseline/teacher_35.pth
baseline/teacher_37.pth
latentloop/teacher_34/teacher_34_adapter_39.pth
latentloop/teacher_35/teacher_35_adapter_39.pth
latentloop/teacher_37/teacher_37_adapter_39.pth
```

The expected sizes and SHA-256 values are tracked in
`artifacts/seer/real_world/basketball/checkpoint_manifest.json`.

## Inference-computer setup

Clone the deployment branch, then activate the already validated environment:

```bash
ssh -p 9000 jbr@210.107.197.121
git clone --branch exp/seer-latentloop-real-deploy-20260814 \
  https://github.com/Gnaroshi/gnaroshi_vla.git ~/gnaroshi_vla
cd ~/gnaroshi_vla/architectures/seer/upstream
conda activate seer
```

Near the top of `deploy_ll_gui.sh`, select exactly one method and one target
control rate by commenting one assignment and uncommenting the other:

```bash
# deployment_method="baseline"
deployment_method="latentloop"

control_freq=15
# control_freq=40
```

Run a model-only preflight first. It verifies and loads the selected artifacts
but does not initialize the UR5e, gripper, cameras, or Tk GUI:

```bash
bash scripts/REAL/deploy_ll_gui.sh --preflight
```

Run the real GUI with the same self-contained script:

```bash
bash scripts/REAL/deploy_ll_gui.sh
```

No external experiment environment variables are required. To deploy another
validated pair, edit only `teacher_id`; the manifest rejects a teacher/adapter
mismatch.

`control_freq` is a target period for the complete loop consisting of camera and
robot-state acquisition, policy inference, robot command, and any remaining
sleep. Setting `40` requests a `25 ms` period but cannot guarantee 40 Hz when
that complete work takes longer. Runtime summaries therefore record the target
rate, measured inter-policy period, achieved rate, and strict deadline misses.

## Logged evidence

Every launch writes the following under
`real_deploy_results/<baseline|latentloop>/launch_logs/<profile>/<timestamp>/`:

- complete launch configuration
- exact launcher snapshot and shell command
- git commit and dirty state
- checkpoint/manifest SHA-256 values
- complete console output and exit code

Each GUI session is also stored under the selected method and profile. It records
teacher/adapter/K/control-rate metadata, result JSON, rollout media, per-step
full-vs-LatentLoop mode and latency, plus measured control cadence. Warm-up
inference is excluded from rollout statistics.

## Hugging Face consideration

A private or gated Hugging Face artifact repository could later replace manual
transfer and pin files by revision and SHA-256. It is intentionally not used in
this deployment: no checkpoint has been uploaded, and the local ignored artifact
directory remains the source of runtime files.
