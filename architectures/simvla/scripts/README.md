# SimVLA Workspace Scripts

These scripts are managed by `gnaroshi_vla` and keep the upstream SimVLA clone
clean.

- `prepare_libero_links.sh`: links the official LIBERO HDF5 subsets into
  `SimVLA/datasets/metas/`.
- `check_libero_dataset.py`: verifies subset counts, symlinks, metadata paths,
  and representative HDF5 keys.
- `train_libero.sh`: launches upstream `train_smolvlm.py` with explicit
  top-level result paths and configurable small/large model settings.

