"""SimVLA DCLD LIBERO evaluation package.

Purpose:
    Split the SimVLA DCLD evaluation wrapper into documented modules while
    preserving the existing command-line behavior.

Inputs/outputs:
    The public CLI still receives the same arguments through
    `architectures/simvla/wrappers/simvla_dcld_eval.py` and writes the same
    dry-run/evaluation artifacts.

Official-match scope:
    Full-mode K=1 paths are intended to match the official SimVLA server/client
    action path at policy-query boundaries, subject to benchmark stochasticity.

DCLD scope:
    This package is SimVLA-specific wrapper code. Architecture-neutral DCLD
    modules remain under `methods/dcld/`.

Caveat:
    `rollout_runner.py` still contains the behavior-preserving rollout core.
    Smaller modules in this package document and centralize invariants for the
    next lower-risk decomposition step.
"""

from .rollout_runner import main

__all__ = ["main"]
