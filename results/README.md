# Results Schema

All experiments should write into:

```text
results/<architecture>/<method>/<experiment>/<YYYY-MM-DD_HH-MM-SS_runid>/
```

Each run directory should contain:

- `run_manifest.yaml`: architecture, method, env, node, experiment, command,
  working directory, Python, package, CUDA, dataset/checkpoint paths if relevant,
  git status if available, config files, and notes.
- `command.sh`: exact command that was launched.
- `composed_config.yaml`: selected top-level config fields.
- `env_snapshot/`: Python, platform, torch/CUDA, conda list, and pip freeze.
- `git_snapshot/`: root and architecture upstream git status if available.
- `logs/`: stdout and stderr.
- `metrics/`: metrics JSONL and summary JSON placeholders or outputs.
- `checkpoints/README.md`: paths or symlinks for large checkpoints.
- `notes.md`: human notes and run-specific caveats.

Do not copy heavy checkpoint files by default.

