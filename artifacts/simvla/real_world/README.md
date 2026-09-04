# SimVLA real-world deployment artifacts

This directory records the expected layout but excludes binary artifacts from
Git. A deployment computer must receive a real-world-finetuned SimVLA model,
its matching processor and normalization statistics, and Condition/Generation
updaters trained from that exact base checkpoint.

```text
official_base_model/
  config.json
  model.safetensors
processor/
norm_stats/
  real_norm.json
real_action_transformer.pt
updaters/
  condition_updater.pt
  generation_updater.pt
deployment_manifest.local.json
```

The real action checkpoint is a compact overlay on the complete released
SimVLA-LIBERO checkpoint. The loader rejects missing, unexpected, or mismatched
official tensors and applies the overlay strictly to the existing action
transformer; no action-head parameter is reinitialized.

Copy `deployment_manifest.example.json` to the ignored local filename and fill
every placeholder. Record SHA-256 values from the files present on the
inference computer. The current LIBERO checkpoint and LIBERO-trained updaters
are not valid substitutes for real robot control.

The staged checks are:

1. `source-preflight`: verifies the tracked, immutable source snapshots only.
2. `artifact-preflight`: loads local artifacts and checks H=10/R=5, K_C=2, and
   N_G=3 counters using synthetic inputs; it does not initialize hardware.
3. `read-only-profile`: reads cameras and robot state without constructing an
   RTDE control client or sending commands.
4. `live`: remains locked until the manifest review fields and both explicit
   environment confirmations are present.

The optional VLA-Cache comparator is training-free and consumes these same
artifacts. `vla_cache_full` is the matched eager-attention no-reuse control;
`vla_cache` performs actual decoder token pruning and K/V reuse. Neither mode
changes the action horizon, execution horizon, action transformer, or robot
control schedule. Use `condition_loop` (K_C=2,N_G=10) for the direct
backbone-side comparison with ours; use `latentloop` (K_C=2,N_G=3) only for
the full-system comparison against `baseline` (K_C=1,N_G=10).

Do not authorize live mode until the real dataset's state encoding, action
scale, gripper sign, camera order, home pose, and Cartesian workspace have been
checked on the inference computer.
