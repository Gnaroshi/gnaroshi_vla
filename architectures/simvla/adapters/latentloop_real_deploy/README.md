# SimVLA LatentLoop real-world deployment

This package adapts SimVLA to the proven UR5e/Robotiq/RealSense deployment
surface without editing either original source tree. It uses tracked snapshots
under `architectures/simvla/third_party/` and keeps all SimVLA-specific code in
this directory.

The selected LatentLoop policy preserves a fresh ten-action chunk every five
executed actions. It refreshes the full action condition every two policy
queries (`K_C=2`) and evaluates the full action transformer three times per
ten-step flow trajectory (`N_G=3`). The baseline uses `K_C=1,N_G=10` under the
same observation, action, and H=10/R=5 execution contract.

The real baseline starts from every tensor in the released SimVLA-LIBERO
checkpoint, freezes its VLM, and fine-tunes the existing action transformer;
there is no scratch or reinitialized-head ablation. LatentLoop updaters must be
trained from that exact real baseline. The deployment loader verifies all three
SHA-256 identities before constructing either policy.

## Staged commands

```bash
bash architectures/simvla/wrappers/deploy_latentloop_real.sh source-preflight

bash architectures/simvla/wrappers/deploy_latentloop_real.sh \
  artifact-preflight \
  --manifest artifacts/simvla/real_world/deployment_manifest.local.json \
  --method latentloop

bash architectures/simvla/wrappers/deploy_latentloop_real.sh \
  read-only-profile \
  --manifest artifacts/simvla/real_world/deployment_manifest.local.json \
  --method latentloop \
  --steps 300
```

Live mode intentionally requires three independent approvals:

1. `safety_review` fields in the verified manifest;
2. `SIMVLA_REAL_LIVE_RUN=1`;
3. `SIMVLA_REAL_DEPLOYMENT_ID` equal to the manifest deployment ID.

The wrapper defaults to `source-preflight`; it never defaults to live mode.
