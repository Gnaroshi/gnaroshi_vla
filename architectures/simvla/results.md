# SimVLA Results

SimVLA runs should be stored under:

```text
results/simvla/<method>/<experiment>/<YYYY-MM-DD_HH-MM-SS_runid>/
```

For upstream/original LIBERO training, use:

```text
results/simvla/original/simvla_libero_small/
results/simvla/original/simvla_libero_large/
```

Large checkpoints are written under each run directory's `checkpoints/`
subdirectory by default. They are intentionally ignored by git.

