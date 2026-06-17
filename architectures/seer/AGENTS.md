# Codex Guidance for Seer

- Seer upstream code lives in `architectures/seer/upstream/`.
- Our Seer method code lives in `architectures/seer/ours/`.
- Adapter and wrapper code lives in `architectures/seer/adapters/`.
- `seer_libero` is the known-working Seer environment.
- Do not install packages into `seer_libero` without explicit approval.
- Do not modify upstream Seer files casually.
- If upstream files are modified, document exact files and reasons in
  `MODIFIED_UPSTREAM_FILES.md`.
- Prefer adding wrappers, launch scripts, and config overlays outside upstream.
- Do not run full training or evaluation sweeps unless explicitly requested.

