# Seer Latent Bridge

Active implementation families:

- `latent_bridge/`: shared contracts and data collection.
- `latent_bridge_efficient/`: compute-matched Large R0/R1 training pipeline.
- `latent_bridge_paper/`: Long and Spatial/Object/Goal paper evaluation launchers.

All paper launchers resolve teacher and selected Large checkpoints from
`.../gnaroshi_vla/artifacts/checkpoints/seer/paper`. New outputs are written to
`.../gnaroshi_vla/results/seer/latent_bridge`. The compact published evidence is
read-only under `.../results/seer/paper/latent_bridge`.

Superseded original-pipeline and inherited LatentLoop wrappers are isolated under
`legacy/`. The two older development worktrees remain untouched because they have
uncommitted provenance; this paper worktree is the only active Latent Bridge source.
