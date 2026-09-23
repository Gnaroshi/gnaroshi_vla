# Seer VLA-Cache baseline

`run_seer_vla_cache_libero_long.sh` is the sole paper baseline entrypoint. It runs
native Seer (`off`), indexed-full control (`matched_full`), and architecture-adapted
cache reuse (`reuse`) with an explicit public33 checkpoint.

This is an architecture-adapted implementation: Seer's compressed condition tokens
and action-query relevance replace the official method's raw spatial patch tokens and
language attention. The distinction and negative latency result are documented in
`codex_outputs/seer/{paper_results,legacy_results}.md`.
