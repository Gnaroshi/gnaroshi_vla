# Seer Architecture

Seer is the first architecture integrated into `vla_ours`.

## Directories

- `upstream/`: copied Seer source from `/home/mingyujung/private/seer/seer_node3`.
- `ours/`: our Seer-specific method code and extraction targets.
- `adapters/`: wrappers and integration points between upstream Seer and our method.
- `configs/`: Seer-local config notes and overlays.
- `env/`: Seer environment documentation and export files.
- `scripts/`: Seer helper scripts managed by `vla_ours`.

## Environment

The known working Seer environment is:

```bash
conda activate seer_libero
```

Do not install packages into `seer_libero` without approval. The current
environment snapshot is stored under `env/`.

## Current Method State

The current `seer_node3` code already contains LR-NODE modifications mixed into
Seer training, evaluation, model, and script files. Because the source is not a
valid git repository and no clean upstream baseline is available locally, the
initial integration preserves this working state in `upstream/` and documents
mixed files in `MODIFIED_UPSTREAM_FILES.md`.

Future refactors should move method-owned code into `ours/` and use `adapters/`
for upstream integration.

