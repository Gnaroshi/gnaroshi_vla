# Seer Regression Fixtures

These are historical source-text fixtures, not user-facing launchers.

- `deploy/`: three task-specific V2 shell sources used to compare model argv and
  robot/camera settings against the current unified deployment launcher.
- `retired_protocols/`: completed comparison/surrogate source snapshots whose
  configuration, gates and scheduling are still checked by unit tests.

The `.txt` suffix intentionally separates them from executable scripts. Tests
read the text; deployment tests run a temporary copy using mocked `torchrun`
and temporary fake checkpoint files, never the physical robot.

All source bytes were preserved during relocation. The original path and SHA-256
are recorded in `codex_outputs/seer/maintenance/cleanup_20260922/manifest.tsv`.
The production launcher remains
`architectures/seer/upstream/scripts/REAL/deploy_ll_gui_unified.sh`.
