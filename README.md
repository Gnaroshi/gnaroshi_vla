# vla_ours

`vla_ours` is a top-level research workspace for multi-architecture VLA experiments.
It is intentionally not named `vla_seer_ours` because Seer is only the first
architecture being integrated. Future architectures such as OpenPI or
VLA-Adapter should live beside Seer under `architectures/`.

## Design

The workspace separates four axes:

- Architecture: `seer`, `openpi`, `vla_adapter`, and future VLA repositories.
- Method: `original`, `ours`, ablations, and method variants.
- Environment: architecture-specific environments such as `seer_libero`.
- Results: consistent run directories with config, env, git, logs, metrics, and notes.

Do not force all architectures into one dependency manager. Each architecture
keeps its own upstream repository, environment setup, configs, and execution
style. Top-level configs record which environment should be used, but launch
scripts must activate that environment before Python starts.

## Current Architecture: Seer

- Source baseline: `/home/mingyujung/private/seer/seer_node3`
- Integrated copy: `architectures/seer/upstream/`
- Known working environment: `conda activate seer_libero`
- Method currently mixed into the Seer copy: LR-NODE

`seer_node3` was treated as a known-working baseline asset. It was copied, not
renamed or modified.

## Layout

```text
vla_ours/
  architectures/
    registry.yaml
    seer/
      upstream/
      ours/
      adapters/
      configs/
      env/
      scripts/
  configs/
    architecture/
    method/
    env/
    node/
    experiment/
  experiments/
  results/
  scripts/
  tools/
```

## Running Lightweight Checks

From the workspace root:

```bash
conda activate seer_libero
python scripts/sanity_check.py architecture=seer method=ours env=seer_libero node=lrnode
python scripts/sanity_check.py architecture=seer method=original env=seer_libero node=lrnode
bash scripts/run_experiment.sh architecture=seer method=ours env=seer_libero experiment=seer_ours_debug
bash scripts/run_experiment.sh architecture=seer method=original env=seer_libero experiment=seer_original_debug
```

The default `run_experiment.sh` action is `sanity`; it does not start training
or evaluation. To call an upstream Seer script, explicitly set `action=seer_script`
and `SEER_SCRIPT`, for example:

```bash
SEER_SCRIPT=scripts/LIBERO_LONG/Seer/eval.sh \
  bash scripts/run_experiment.sh architecture=seer method=original env=seer_libero \
  experiment=seer_original_debug action=seer_script
```

This is intentionally explicit because many upstream scripts launch distributed
GPU jobs.

## Adding a Future Architecture

1. Create `architectures/<new_arch>/`.
2. Clone or copy the original repository into `architectures/<new_arch>/upstream/`.
3. Add `architectures/<new_arch>/env/` with activation and export files.
4. Add `configs/architecture/<new_arch>.yaml`.
5. Add `configs/env/<new_arch_env>.yaml`.
6. Put our method code in `architectures/<new_arch>/ours/`.
7. Put wrappers or integration glue in `architectures/<new_arch>/adapters/`.
8. Document exact run commands and result locations.

## Results

Runs are organized as:

```text
results/<architecture>/<method>/<experiment>/<YYYY-MM-DD_HH-MM-SS_runid>/
  run_manifest.yaml
  command.sh
  composed_config.yaml
  env_snapshot/
  git_snapshot/
  logs/
  metrics/
  checkpoints/
  notes.md
```

Large checkpoints should not be copied by default. Store paths, symlinks, or a
README pointing to external checkpoint storage.

## Do Not Commit

- Environments: `.venv/`, `.venvs/`, `env/`, `venv/`
- Datasets: `data/`, `datasets/`, `LIBERO_DATASETS/`
- Generated runs/logs: `wandb/`, `lightning_logs/`, `runs/`, `logs/`
- Heavy model files: `checkpoints/`, `ckpts/`, `*.pth`, `*.pt`, `*.ckpt`
- Per-run checkpoint directories under `results/`

