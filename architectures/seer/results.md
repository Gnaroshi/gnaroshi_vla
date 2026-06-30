# Seer Results Notes

Seer results should be saved through the top-level schema:

```text
results/seer/<method>/<experiment>/<timestamp_runid>/
```

Use `method=original` for baseline Seer runs and `method=lrnode` for the current
LR-NODE method. Do not copy large checkpoints by default. Store external
paths or symlinks and record them in `run_manifest.yaml` and `notes.md`.
