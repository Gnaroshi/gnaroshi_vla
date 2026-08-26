#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=${SIMVLA_STABILITY_WORKTREE:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment}
PYTHON=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
UPSTREAM=/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream

SD1_BUNDLE_REMOTE=${SD1_BUNDLE_REMOTE:-/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_stability_aligned_selected}
test -x "$PYTHON"
test -d "$ROOT"
test -d "$UPSTREAM"
export SIMVLA_STABILITY_WORKTREE=$ROOT
export SIMVLA_UPSTREAM_ROOT=$UPSTREAM
export SD1_BUNDLE_REMOTE
export PYTHONPATH="$ROOT:$UPSTREAM:${PYTHONPATH:-}"

cd "$ROOT"
exec "$PYTHON" -m architectures.simvla.adapters.latentloop.stability_alignment.rb2_pipeline
