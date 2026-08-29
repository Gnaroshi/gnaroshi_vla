#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=${SIMVLA_STABILITY_WORKTREE:-/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment}
PYTHON=${SIMVLA_STABILITY_PYTHON:-/home/mingyujung/miniconda3/envs/simvla_libero/bin/python}
UPSTREAM=${SIMVLA_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream}

test -x "$PYTHON"
test -d "$ROOT"
test -d "$UPSTREAM"
export SIMVLA_STABILITY_WORKTREE=$ROOT
export SIMVLA_STABILITY_PYTHON=$PYTHON
export SIMVLA_UPSTREAM_ROOT=$UPSTREAM
export RB2_BUNDLE_DESTINATION=${RB2_BUNDLE_DESTINATION:-rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_stability_aligned_selected}
export PYTHONPATH="$ROOT:$UPSTREAM:${PYTHONPATH:-}"

cd "$ROOT"
exec "$PYTHON" -m architectures.simvla.adapters.latentloop.stability_alignment.sd1_pipeline \
  --gpu-pool "${SD1_GPU_POOL:-4,5,6,7}" \
  --practical-budget-hours "${SIMVLA_STABILITY_BUDGET_HOURS:-12}"
