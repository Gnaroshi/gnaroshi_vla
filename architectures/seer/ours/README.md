# Seer Ours Method Area

Use this directory for Seer-specific implementation of our methods.

Current status:

- LR-NODE method code is already mixed into the copied Seer source under
  `../upstream/`.
- Dedicated LR-NODE modules identified in the current source are:
  - `../upstream/models/lrnode_modules.py`
  - `../upstream/utils/lrnode_logging_utils.py`
- Training/evaluation integration is currently in mixed upstream files listed in
  `../MODIFIED_UPSTREAM_FILES.md`.

Refactor target:

1. Move method-owned modules here.
2. Keep upstream Seer behavior reproducible.
3. Add adapters in `../adapters/` to connect upstream Seer to these modules.
4. Update launch scripts only after lightweight checks confirm behavior parity.

