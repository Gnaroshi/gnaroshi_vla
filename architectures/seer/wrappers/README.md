# Unified Seer method latency

`run_seer_unified_policy_latency.sh` measures Seer K1, LatentLoop K2-K8,
Latent Bridge Large K4, and VLA-Cache in one model-only benchmark. It excludes
simulator and preprocessing time, uses four independent RTX 3090 replicates, and
loads all selected checkpoints from the canonical paper checkpoint registry.

The final immutable aggregate is stored under
`.../gnaroshi_vla/results/seer/paper/latency`; new runs use
`.../gnaroshi_vla/results/seer/latency`.
