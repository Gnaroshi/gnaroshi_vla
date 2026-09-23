# Seer Tools

## Current Workflows

| Group | Files | Role |
| --- | --- | --- |
| Freshness | `run_latentloop_freshness*.py` | Current source-locked diagnostics and confirmation |
| Real-world assets | `verify_realworld_latentloop_assets.py`, `export_seer_start_references.py` | Validate training assets; export real starting images |
| GUI development | `preview_real_deploy_ui.py` | Mock preview, not deployment |
| Provisioning | `setup_s4_realworld_latentloop.sh`, `transfer_s4_realworld_assets_from_sd1.sh`, `finalize_s4_realworld_latentloop.sh`, `prepare_sd1_realworld_stacking_rings_assets.sh` | Existing server preparation dependencies |

Training and deployment entry points are indexed in
`architectures/seer/wrappers/README.md`. Do not run provisioning scripts merely
to inspect their status: they transfer files and may configure remote hosts.

## Historical Analysis Dependencies

These tools remain at their original import paths because tests execute their
analysis functions. Previous cleanup had moved them to `seer_legacy/`, leaving
those imports unresolved. This cleanup restores their unchanged bytes:

- `analyze_every_step_latent_filter.py`
- `analyze_independent_k4_confirmation.py`
- `analyze_latentloop_budget_sweep.py`
- `analyze_latentloop_plan_continuation.py`
- `analyze_latentloop_segment_grid.py`
- `analyze_libero_plus_lrnode.py`
- `build_plan_continuation_evidence.py`
- `check_latentloop_k1_parity.py`
- `evaluate_joint_stage_a_action_space_v2.py`
- `evaluate_joint_stage_a_checkpoint_sweep.py`
- `evaluate_latentloop_comparison_offline.py`
- `lock_latentloop_comparison_source.py`
- `register_latentloop_segment_row.py`
- `verify_latentloop_k1_parity_artifact.py`

Retention for testing/analysis does not re-enable a retired training campaign.
Historical source-lock builders can list deleted launchers and are **not**
supported commands for new campaigns. Current launchers are listed above.
Historical outcomes/source records: `codex_outputs/seer/legacy_results.md`
and `codex_outputs/seer/maintenance/cleanup_20260922/`.
