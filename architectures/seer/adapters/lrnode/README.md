# Seer LR-NODE Adapter

This directory is reserved for the Seer-specific adapter layer for LR-NODE.

Current state: LR-NODE behavior is still mixed into the current Seer upstream
copy. Reference method-owned files are tracked under
`methods/lrnode/seer_reference/`, and mixed upstream files are documented in
`docs/architectures/seer/modified_upstream_files.md`.

Future work should move Seer-specific integration glue here and leave reusable
method code under `methods/lrnode/`.
