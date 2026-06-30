# Codex Guidance for SimVLA

- SimVLA upstream code lives in `architectures/simvla/upstream/`.
- SimVLA adapter code lives in `architectures/simvla/adapters/`.
- SimVLA wrapper code lives in `architectures/simvla/wrappers/`.
- Method implementations live in `methods/<method>/`.
- `simvla_libero` is the SimVLA LIBERO environment.
- Do not install packages into `simvla_libero` without explicit approval.
- Do not modify upstream SimVLA files casually.
- Prefer adding wrappers, launch scripts, and config overlays outside upstream.
- Do not commit datasets, Hugging Face caches, generated metadata, checkpoints,
  or training runs.
- Do not replace `architectures/simvla/upstream/datasets/`; it contains Python
  package code. Dataset links belong only under
  `architectures/simvla/upstream/datasets/metas/`.
- Do not run full training or evaluation sweeps unless explicitly requested.
