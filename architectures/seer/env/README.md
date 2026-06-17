# Seer Environment

Known working environment:

```bash
conda activate seer_libero
```

This directory stores reproducibility files only. Do not place actual conda
environment directories, `.venv` directories, datasets, checkpoints, or generated
runs here.

Expected files:

- `seer_libero.yaml`: human-readable environment record.
- `environment.seer_libero.yml`: `conda env export --no-builds`.
- `conda-list.seer_libero.txt`: `conda list`.
- `pip-freeze.seer_libero.txt`: `pip freeze`.

