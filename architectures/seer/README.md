# Seer Architecture

Seer is the first architecture integrated into `gnaroshi_vla`.

## Directories

- `upstream/`: copied Seer source from `$SEER_WORKSPACE_ROOT`.
- `adapters/`: architecture-specific integration points between upstream Seer and methods.
- `configs/`: Seer-local config notes and overlays.
- `env/`: Seer environment documentation and export files.
- `wrappers/`: Seer launch wrappers managed by `gnaroshi_vla`.

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
mixed files in `docs/architectures/seer/modified_upstream_files.md`.

Future refactors should move method-owned code into `methods/<method>/` and use
`adapters/<method>/` for upstream integration.
