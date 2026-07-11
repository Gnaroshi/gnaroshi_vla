# Third-Party Notices

This workspace combines project-owned orchestration with architecture-specific
third-party sources. Do not remove license or attribution files from copied code.

## Seer

The integrated Seer source under `architectures/seer/upstream/` retains its
Apache License 2.0 text at `architectures/seer/upstream/LICENSE`. Nested assets
retain their own notices, including the BSD-licensed Robotiq model under
`real_preprocess/mujoco_menagerie/robotiq_2f85/`.

The current copied tree is documented as a known-working mixed source snapshot.
Its exact clean-upstream revision is not established in this repository, so the
snapshot and modification inventory in `docs/architectures/seer/` must remain.

## SimVLA

SimVLA is attributed to `https://github.com/LUOyk1999/SimVLA`. Its upstream
checkout is gitignored and is expected to keep its own upstream license and
history. Do not vendor it without copying the corresponding license and notices.

## Workspace-Owned Code

No root license has been selected for workspace-owned code. The presence of
third-party licenses does not license unrelated files in this repository.
