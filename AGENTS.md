# Codex Guidance for gnaroshi_vla

This is a multi-architecture VLA research workspace.

Rules for future sessions:

- Preserve architecture separation. Each architecture owns its own upstream repo,
  environment setup, configs, adapters, and scripts.
- Preserve method separation. Put our method code under
  `architectures/<arch>/ours/` and integration wrappers under
  `architectures/<arch>/adapters/`.
- Do not force one environment manager across architectures.
- Do not mutate known-working environments such as `seer_libero` without explicit
  approval.
- Do not edit upstream architecture code unless necessary. Prefer wrappers,
  adapters, config overlays, or documented patch files.
- If an upstream file is modified, document it in that architecture's
  `MODIFIED_UPSTREAM_FILES.md`.
- Do not run heavy training, evaluation sweeps, or GPU jobs unless explicitly
  requested.
- Every experiment must record architecture, method, environment, config, git
  state if available, command, logs, and result path.
- Store results under
  `results/<architecture>/<method>/<experiment>/<timestamp_runid>/`.
- Do not copy large checkpoints into the workspace by default. Record paths or
  symlinks instead.
