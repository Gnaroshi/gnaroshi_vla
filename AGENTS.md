# Codex Guidance for gnaroshi_vla

This is a multi-architecture VLA research workspace.

Rules for future sessions:

- Preserve architecture separation. Each architecture owns its own upstream repo,
  environment setup, configs, adapters, and wrappers.
- Preserve method separation. Put method implementations under
  `methods/<method>/` and architecture-specific glue under
  `architectures/<arch>/adapters/<method>/`.
- Keep generated notes, handoff docs, and agent context under `docs/`, not inside
  architecture upstream trees.
- Do not force one environment manager across architectures.
- Do not mutate known-working environments such as `seer_libero` without explicit
  approval.
- Do not edit upstream architecture code unless necessary. Prefer wrappers,
  adapters, config overlays, or documented patch files.
- If an upstream file is modified, document it in that architecture's
  `docs/architectures/<arch>/modified_upstream_files.md`.
- Do not run heavy training, evaluation sweeps, or GPU jobs unless explicitly
  requested.
- Every experiment must record architecture, method, environment, config, git
  state if available, command, logs, and result path.
- Store results under
  `results/<architecture>/<method>/<experiment>/<timestamp_runid>/`.
- Do not copy large checkpoints into the workspace by default. Record paths or
  symlinks instead.
