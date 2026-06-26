# Codex Guidance for SimVLA

- SimVLA upstream code lives in `architectures/simvla/SimVLA/`.
- Our SimVLA method code should live in `architectures/simvla/ours/`.
- Adapter and wrapper code should live in `architectures/simvla/adapters/`.
- `simvla_libero` is the SimVLA LIBERO environment.
- Do not install packages into `simvla_libero` without explicit approval.
- Do not modify upstream SimVLA files casually.
- Prefer adding wrappers, launch scripts, and config overlays outside upstream.
- Do not commit datasets, Hugging Face caches, generated metadata, checkpoints,
  or training runs.
- Do not replace `SimVLA/datasets/`; it contains Python package code. Dataset
  links belong only under `SimVLA/datasets/metas/`.
- Do not run full training or evaluation sweeps unless explicitly requested.

