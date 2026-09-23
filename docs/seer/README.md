# Seer

This directory contains release-facing documentation for the Seer instantiation
of LatentLoop. Generated audits, experiment notes, and paper-result ledgers live
under the ignored `codex_outputs/seer/` directory.

## Workflows

- [LatentLoop training and evaluation](latentloop.md)
- [Real-world deployment](latentloop-real-world-deploy.md)

## Source ownership

- `architectures/seer/upstream/`: Seer model, training, and evaluation runtime.
- `architectures/seer/wrappers/`: stable user-facing launchers.
- `architectures/seer/adapters/`: architecture-specific method integration.
- `methods/`: architecture-independent method components.

The historical directory name `lrnode` is retained for checkpoint and command
compatibility. The paper-facing method name is **LatentLoop**.

Checkpoints, datasets, generated results, and W&B state are intentionally not
stored in Git. Supply them through the paths documented by each launcher.
