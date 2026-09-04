# SimVLA deployment source snapshots

This directory contains source copied into the repository for reproducible
SimVLA real-world deployment. The copies are intentionally separate from both
the Seer integration and the ignored SimVLA upstream checkout.

- `3dflow_real_deploy/`: exact copy of the tracked Seer hardware-deployment
  snapshot at commit `36c9d88` before SimVLA-specific adaptation.
- `simvla_upstream_32700d0/`: the official SimVLA model package and license from
  upstream commit `32700d0ad8991996e123e4b685abe370ce6e9aab`.

Large model checkpoints are not stored here or committed to Git.
