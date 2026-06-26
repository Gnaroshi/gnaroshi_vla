# SimVLA Environment

Known environment:

```bash
conda activate simvla_libero
```

This directory stores reproducibility files only. Do not place actual conda
environment directories, `.venv` directories, datasets, checkpoints, Hugging
Face caches, or generated runs here.

Expected files:

- `simvla_libero.yaml`: human-readable environment record.
- `environment.simvla_libero.yml`: `conda env export --no-builds`.
- `conda-list.simvla_libero.txt`: `conda list`.
- `pip-freeze.simvla_libero.txt`: `pip freeze`.
- `inspection.simvla_libero.txt`: quick import/version inspection.

