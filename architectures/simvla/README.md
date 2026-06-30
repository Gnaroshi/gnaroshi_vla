# SimVLA Architecture

SimVLA is integrated as a clean upstream clone plus `gnaroshi_vla`-managed
wrappers.

## Directories

- `upstream/`: upstream clone of `https://github.com/LUOyk1999/SimVLA.git`.
- `adapters/`: future integration glue between upstream SimVLA and our methods.
- `configs/`: SimVLA-local notes and overlays.
- `env/`: SimVLA environment documentation and export files.
- `wrappers/`: launch, dataset-link, and validation scripts managed by
  `gnaroshi_vla`.

## Environment

The SimVLA LIBERO environment is:

```bash
conda activate simvla_libero
```

Do not install packages into `simvla_libero` without approval. The current
environment snapshot is stored under `env/`.

## LIBERO Dataset

Use the official LIBERO HDF5 dataset at:

```text
/home/mingyujung/shared/nvme1/mingyujung/datasets/robotics/LIBERO
```

The actual subset directories are under the `datasets/` child:

```text
.../LIBERO/datasets/libero_10
.../LIBERO/datasets/libero_goal
.../LIBERO/datasets/libero_object
.../LIBERO/datasets/libero_spatial
.../LIBERO/datasets/libero_90
```

SimVLA expects those subsets directly below `upstream/datasets/metas/`. Use:

```bash
bash architectures/simvla/wrappers/prepare_libero_links.sh
python architectures/simvla/wrappers/check_libero_dataset.py
```

Do not replace `upstream/datasets/` itself because it is also a Python package.
Only create or update subset symlinks under `upstream/datasets/metas/`.

## Training

Small model:

```bash
bash architectures/simvla/wrappers/train_libero.sh --model-size small
```

Dry-run without starting `accelerate`:

```bash
bash architectures/simvla/wrappers/train_libero.sh --model-size small --dry-run
```

Large model:

```bash
bash architectures/simvla/wrappers/train_libero.sh --model-size large
```

The wrapper stores outputs under:

```text
results/simvla/original/simvla_libero_<small|large>/<timestamp>_<run_id>/
```

It keeps upstream source unchanged and launches `upstream/train_smolvlm.py`
directly with explicit metadata, normalization, output, GPU, and model
arguments.
