# Seer LR-NODE Adapter

This directory is reserved for the Seer-specific adapter layer for LR-NODE.

Current state: LR-NODE behavior is still mixed into the current Seer upstream
copy. Reference method-owned files are tracked under
`methods/lrnode/seer_reference/`. See
[`docs/seer/lrnode.md`](../../../../docs/seer/lrnode.md) for the public training
and evaluation workflow. Implementation audits and experimental separation
notes remain under the ignored `codex_outputs/` tree.

Future work should move Seer-specific integration glue here and leave reusable
method code under `methods/lrnode/`.
