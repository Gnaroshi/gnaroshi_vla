# gnaroshi_vla

`gnaroshi_vla` is a top-level research workspace for multi-architecture VLA
experiments. It is intentionally architecture-neutral because Seer is only the
first architecture being integrated. Future architectures such as SimVLA,
OpenPI, or VLA-Adapter should live beside Seer under `architectures/`.

**Status:** Active public research workspace. Lightweight sanity checks are supported; experiment claims require retained artifacts and owner-reviewed runs.

Upstream architecture code and datasets remain external or isolated under architecture-specific boundaries. This repository must not contain private credentials, datasets, model checkpoints, machine-specific paths in public prose, or unrelated website and publishing code.

## Design

The workspace separates four axes:

- Architecture: `seer`, `simvla`, `openpi`, `vla_adapter`, and future VLA repositories.
- Method: `original`, `lrnode`, ablations, and future method variants.
- Environment: architecture-specific environments such as `seer_libero`.
- Results: consistent run directories with config, env, git, logs, metrics, and notes.

Do not force all architectures into one dependency manager. Each architecture
keeps its own upstream repository, environment setup, configs, and execution
style. Top-level configs record which environment should be used, but launch
scripts must activate that environment before Python starts.

Method implementations live under `methods/<method_name>/`. Architecture
directories keep only upstream code, environment records, wrappers, and thin
architecture-specific adapters.

## Current Architectures

### Seer

- Source baseline: `$SEER_WORKSPACE_ROOT`
- Integrated copy: `architectures/seer/upstream/`
- Known working environment: `conda activate seer_libero`
- Method currently mixed into the Seer copy: LR-NODE

`seer_node3` was treated as a known-working baseline asset. It was copied, not
renamed or modified.

### SimVLA

- Upstream repository: `https://github.com/LUOyk1999/SimVLA.git`
- Upstream clone: `architectures/simvla/upstream/`
- Known environment: `conda activate simvla_libero`
- LIBERO root: `$LIBERO_ROOT`

SimVLA is kept as a clean upstream git clone. Dataset symlinks and training
launchers are managed outside upstream under `architectures/simvla/wrappers/`.

## Layout

```text
gnaroshi_vla/
  architectures/
    registry.yaml
    seer/
      upstream/
      adapters/
      configs/
      env/
      wrappers/
    simvla/
      upstream/
      adapters/
      configs/
      env/
      wrappers/
  methods/
    lrnode/
      seer_reference/
  configs/
    architecture/
    method/
    env/
    node/
    experiment/
  experiments/
  results/
  scripts/
  docs/
  tools/
```

## Running Lightweight Checks

From the workspace root:

```bash
conda activate seer_libero
python scripts/sanity_check.py architecture=seer method=lrnode env=seer_libero node=lrnode
python scripts/sanity_check.py architecture=seer method=original env=seer_libero node=lrnode
bash scripts/run_experiment.sh architecture=seer method=lrnode env=seer_libero experiment=seer_lrnode_debug
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

For SimVLA dataset setup and checks:

```bash
bash architectures/simvla/wrappers/prepare_libero_links.sh
python architectures/simvla/wrappers/check_libero_dataset.py
bash scripts/run_experiment.sh architecture=simvla method=original env=simvla_libero \
  node=simvla_4gpu experiment=simvla_original_libero_small action=simvla_check_data
```

To start SimVLA training with top-level result logging:

```bash
bash scripts/run_experiment.sh architecture=simvla method=original env=simvla_libero \
  node=simvla_4gpu experiment=simvla_original_libero_small action=simvla_train_small
```

The SimVLA wrapper accepts environment overrides such as `CUDA_VISIBLE_DEVICES`,
`SIMVLA_BATCH_SIZE`, `SIMVLA_NUM_PROCESSES`, `SIMVLA_MAIN_PROCESS_PORT`, and
`SIMVLA_ITERS`.

Machine-specific paths are intentionally omitted from public documentation. See
[`docs/public-paths.md`](docs/public-paths.md) for the placeholder variables used
in historical experiment notes. Runtime YAML may still contain local values and
must be migrated only with experiment-owner review.

## Adding a Future Architecture

1. Create `architectures/<new_arch>/`.
2. Clone or copy the original repository into `architectures/<new_arch>/upstream/`.
3. Add `architectures/<new_arch>/env/` with activation and export files.
4. Add `configs/architecture/<new_arch>.yaml`.
5. Add `configs/env/<new_arch_env>.yaml`.
6. Put architecture-specific wrappers in `architectures/<new_arch>/wrappers/`.
7. Put architecture-specific method glue in `architectures/<new_arch>/adapters/<method>/`.
8. Put architecture-neutral method code in `methods/<method>/`.
9. Document exact run commands and result locations.

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
- Upstream nested clone contents unless intentionally converted to a git
  submodule, for example `architectures/simvla/upstream/`

## Licensing And Attribution

Copied third-party code keeps its original license files in place. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This repository does not
currently declare a root license for workspace-owned code; no reuse grant should
be inferred until the owner makes that decision. See [`docs/license-decision.md`](docs/license-decision.md).

## Related Repositories

- [`Gnaroshi/gnaroshi.github.io`](https://github.com/Gnaroshi/gnaroshi.github.io): public research profile and project presentation
- [`Gnaroshi/gnaroshi`](https://github.com/Gnaroshi/gnaroshi): GitHub profile index
