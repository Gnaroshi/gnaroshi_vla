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

All deployment methods use the same resize-with-pad-224 then bicubic-384 image
transform as the real condition cache and action-head training path.

The real baseline starts from every tensor in the released SimVLA-LIBERO
checkpoint, freezes its VLM, and fine-tunes the existing action transformer;
there is no scratch or reinitialized-head ablation. LatentLoop updaters must be
trained from that exact real baseline. The deployment loader verifies the real
baseline, condition updater, parent generation updater, and projection-only
coupled generation checkpoint, together with data/cache/normalization lineage,
before constructing either policy. The `latentloop` method requires coupling;
an uncoupled checkpoint cannot be silently substituted.

This package proves source/artifact consistency and enforces reviewed command
limits. It does not prove real-task success. The frozen-VLM, 3,000-step real
action-head adaptation is a controlled low-data transfer protocol for the 40
available demonstrations, not a published SimVLA real-world recipe. A bounded
baseline canary must pass before LatentLoop can be authorized.

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

The operator-facing Stop/Retry actions raise the abort latch immediately and
stop the UR arm at the current RTDE command boundary. A command already inside
the RTDE call may finish before the software stop takes effect. The rollout is
discarded without an automatic home move. The copied Robotiq API has no stop
primitive, so the software can only stop issuing new gripper commands; the
physical emergency-stop path remains mandatory.
