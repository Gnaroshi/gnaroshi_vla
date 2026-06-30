# Modified Upstream Files

Initial status: the source directory `/home/mingyujung/private/seer/seer_node3`
is not a valid git repository, so there is no local clean upstream baseline or
diff available. The copied `upstream/` tree therefore preserves the current
known-working state.

The following files contain LR-NODE method markers and should be treated
as mixed upstream files until a clean Seer baseline is cloned and compared:

- `train.py`: LR-NODE train protocol selection, trainable parameter control,
  run snapshots, and LR-NODE arguments passed into the model.
- `eval_libero.py`: LR-NODE evaluation flags, model construction flags, and
  evaluation snapshots.
- `models/seer_model.py`: LR-NODE latent update path and model-side integration.
- `models/lrnode_modules.py`: LR-NODE method modules. This is method-owned code
  and has a reference copy under `methods/lrnode/seer_reference/`.
- `utils/arguments_utils.py`: LR-NODE CLI flags.
- `utils/train_utils.py`: LR-NODE training losses, distillation, metrics, and
  debug artifacts.
- `utils/eval_utils_libero.py`: LR-NODE evaluation behavior and metrics.
- `utils/lrnode_logging_utils.py`: LR-NODE snapshot/logging helper. This is
  method-owned utility code with a reference copy under
  `methods/lrnode/seer_reference/`.
- `scripts/LIBERO_LONG/Seer/*node*.sh`: LR-NODE train/eval launch protocols.
- `scripts/LIBERO_LONG/Seer/eval_lrnode_*.sh`: LR-NODE comparison/eval scripts.
- `docs/lrnode_architecture.svg`: LR-NODE architecture documentation figure.

Generated method analysis and experiment notes were moved out of upstream to
`docs/agent_notes/seer/codex_output/`.

No upstream file was edited during this workspace creation. The separation is
documented rather than behavior-changing so the current experiments remain
reproducible.
