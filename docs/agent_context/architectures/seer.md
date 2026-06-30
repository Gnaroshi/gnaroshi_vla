# Codex Guidance for Seer

- Seer upstream code lives in `architectures/seer/upstream/`.
- LR-NODE method reference code lives in `methods/lrnode/seer_reference/`.
- Seer adapter code lives in `architectures/seer/adapters/`.
- Seer wrapper code lives in `architectures/seer/wrappers/`.
- `seer_libero` is the known-working Seer environment.
- Do not install packages into `seer_libero` without explicit approval.
- Do not modify upstream Seer files casually.
- If upstream files are modified, document exact files and reasons in
  `docs/architectures/seer/modified_upstream_files.md`.
- Prefer adding wrappers, launch scripts, and config overlays outside upstream.
- Do not run full training or evaluation sweeps unless explicitly requested.
