# Seer Adapter Area

Use this directory for architecture-specific glue that connects upstream Seer
code with method code from `methods/<method>/`.

Current integration points to wrap in a future refactor:

- Model construction flags from `upstream/train.py` and `upstream/eval_libero.py`.
- LR-NODE CLI flags from `upstream/utils/arguments_utils.py`.
- LR-NODE train losses and metrics from `upstream/utils/train_utils.py`.
- LR-NODE evaluation controls from `upstream/utils/eval_utils_libero.py`.
- Launch protocol selection from `upstream/scripts/LIBERO_LONG/Seer/`.

Adapters should make method selection explicit and avoid editing upstream files
unless a documented patch is unavoidable.
