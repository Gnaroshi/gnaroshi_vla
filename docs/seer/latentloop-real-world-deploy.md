# Seer and LatentLoop real-world deployment

The real-world path preserves the validated 3DFlow-Seer camera, UR5e, Robotiq,
GUI, action-head, and temporal-ensemble behavior. LatentLoop changes only how
the cached action condition is refreshed between full Seer queries.

## Runtime contract

- Baseline: the selected Seer teacher with full inference at every step.
- LatentLoop: the same teacher and its teacher-specific adapter.
- Default LatentLoop schedule: full Seer at steps `0, 4, 8, ...`, with a fresh
  exterior image, wrist image, and proprioceptive state at every intermediate
  update.
- Environment on the inference computer: `conda activate seer`.
- `control_freq` is a requested loop rate, not a guaranteed achieved rate. The
  complete camera, policy, command, and sleep path must fit within `1000 / Hz`
  milliseconds.
- The launcher logs requested and achieved rates, policy latency, full/skip
  counts, checkpoint hashes, source state, and command status separately for
  baseline and LatentLoop.

Teacher and adapter files are a strict pair. The preflight rejects a missing or
mismatched artifact before cameras or robot control are initialized.

## Source layout

- `architectures/seer/adapters/latentloop_real_deploy/`: controller and GUI
  integration.
- `architectures/seer/third_party/3dflow_real_deploy/`: preserved deployment
  dependency snapshot.
- `architectures/seer/upstream/scripts/REAL/deploy_ll_gui_unified.sh`: the
  single configurable entry point for Basketball, Doll and Cabinet, for both
  baseline and LatentLoop. Earlier task-specific V2 launcher text is retained
  only under `tests/fixtures/seer/deploy/` for argument-parity regression tests,
  not as operational launchers. Use the unified file for new sessions.
- `artifacts/seer/real_world/<task>/`: ignored local checkpoints and manifests.
- `real_deploy_results_v2/<baseline|latentloop>/<task>/`: method- and
  task-separated runtime evidence. Historical Basketball results directly
  under the method folder are not moved or overwritten.

Rings and Stacking Cups adapters use the same training contract. Deployment
launchers should be added only after each task's camera, instruction, checkpoint
pair, and robot workspace have been validated.

## Configuration

Edit the configuration block at the top of `deploy_ll_gui_unified.sh` rather
than relying on persistent shell variables. Leave exactly one preset active:

```bash
deploy_presets=(
    # "basketball_baseline"
    # "basketball_latentloop"
    "doll_baseline"
    # "doll_latentloop"
    # "cabinet_baseline"
    # "cabinet_latentloop"
)
basketball_teacher_id=37       # 34, 35, or 37; Doll/Cabinet use teacher38
adapter_id=39
query_interval=4
execution_mode="live"          # or read_only_profile
camera_mode="sync"
camera_fps=60
control_freq=60
preflight_only=0               # 1 for synthetic model checks, no hardware
```

The preset selects the matching Seer instruction, teacher, task-specific
adapter, manifest and result folder. Multiple active presets are rejected.
Baseline never loads an adapter. All presets use the existing Seer v2 action
decoding, temporal ensembling and camera configuration. The task-specific home
targets below apply equally to baseline and LatentLoop; the existing Seer home
interpolation, duration and servo settings are unchanged. No PI0/PI0.5 action
horizon or policy gripper threshold is imported.

### Task-specific robot home

The `home_pose_json` settings in the launcher's task selection contain six UR5e
joint angles in radians, followed by a normalized gripper target (0 = open).
They match the task targets used by `pi05_relative_deploy_vtp.py`, called from
the reference `efficient_Seer-main/scripts/REAL/deploy_pi05_relative_vtp_baseline.sh`:

| Task | Joint target (rad) | Gripper |
| --- | --- | --- |
| Basketball | `[3.14, -1.57, 1.57, -1.57, -1.57, -1.57]` | Open |
| Doll | `[3.0502887, -1.6030570, 1.8191951, -1.8019783, -1.5417574, -1.6144441]` | Open |
| Cabinet | `[2.9891653, -1.5753395, 1.8866094, -1.8454653, -1.5462163, -1.6641129]` | Open |

Doll uses reference episode `0511_172010`; Cabinet uses `0507_203729`.
The reference stores a measured gripper value of `0.0117647` for those tasks
but its home routine explicitly commands fully open. Seer therefore uses `0.0`
to match the commanded state, not the stored measurement.

Previously all tasks inherited the Basketball home target. New sessions must
use the unified launcher: a missing home setting or a task/manifest mismatch
now fails before model or hardware creation. Home settings are printed at
startup and saved in `launch_config.txt`, the session manifest, and result
metadata. They apply to every existing home call, including rollout start and
return-to-home. The reference image browser remains read-only; browsing another
episode does not change the configured robot home.

For baseline, `baseline_rollout_policy="full"` gives the paper baseline. The
`hold_action` and `hold_latent` modes are mechanism controls and must not be
reported as Seer K=1.

## Preflight and launch

From `architectures/seer/upstream` on the inference computer:

```bash
conda activate seer
bash scripts/REAL/deploy_ll_gui_unified.sh --print-config
bash scripts/REAL/deploy_ll_gui_unified.sh --preflight
bash scripts/REAL/deploy_ll_gui_unified.sh
```

`--print-config` only resolves settings, without loading a model or connecting
to hardware. `--preflight` loads/hashes the selected artifacts and runs
synthetic inference, without opening cameras or connecting to the robot.
The normal command opens the GUI; the existing live environment initialization
and rollout behavior are unchanged. Synthetic checks do not validate new
physical motions. GUI controls for frequency, K and baseline hold modes
remain available as before.

## GUI Layout

The live Primary/Wrist views sit directly above their training-start references.
Start, Stop & home, outcome buttons and recent runtime measurements remain visible
on the right. Session contains notes and result controls; Settings contains
control frequency, query interval and policy selection; Details contains complete
checkpoint paths, home configuration, camera IDs and the latest saved rollout.
Settings are disabled during an active rollout. Record deletion requires confirmation.

`Stop & home` is the existing rollout stop-and-return-home command, not a hardware
emergency stop. No motion, decoding or policy scheduling algorithm is changed by
this layout.

The dashboard's Command Hz uses up to 20 intervals between completed robot commands.
Policy ms is the mean of up to 20 existing inference records, not simulator/camera
time or the complete rollout average. Empty measurements are displayed as `--`.

For legible antialiased text, the inference launcher's `gui_font_backend="system"`
uses the installed system Tcl/Tk 8.6 libraries only in the GUI subprocess. It does
not install or replace conda packages. Use `gui_font_backend="conda"` to retain the
original Tk runtime. Model preflight and read-only profiling do not use this override.

## Training-Start Reference Images

The V2 GUI shows a **Reference images** section directly below Camera previews.
It follows the selected task's checkpoint manifest, for both baseline and
LatentLoop. Basketball, Doll and Cabinet each have all 40 training-episode
starts, with Primary and Wrist views from step `0000` shown together.

- `<` / `>` move through episodes, wrapping at the ends.
- The episode selector jumps directly to any of the 40 starts.
- `Random` chooses a different episode without consuming the policy's RNG.
- `Zoom` or clicking an image opens both views at a larger size.
- Recorded episode name and step are shown below the images.

Images live under `artifacts/seer/real_world/<task>/reference_start_frames_all/`.
Its manifest records source dataset, episode IDs and SHA-256 values. Images are
byte-identical training JPEGs, without re-encoding, flipping or cropping.
The viewer only reads images; it never sets the robot pose or feeds a reference
image to the policy. Missing, mismatched or damaged collections show an error
inside this section rather than silently displaying another task.

No new launcher arguments are needed. After finishing the current rollout,
close and reopen the GUI to load this addition; an already-running GUI is not
hot-reloaded. Keyboard focus inside the reference controls does not trigger
rollout-start/restart shortcuts; the existing `X` stop shortcut remains active.

## Artifact policy

Checkpoint binaries remain ignored by Git. A task artifact directory should
contain a manifest with expected filename, byte size, SHA-256 hash, teacher ID,
adapter ID, instruction, and training-dataset provenance. Runtime JSON and logs
may be versioned separately after removing raw media and private machine data.
