"""Action queue semantics for SimVLA DCLD eval.

Purpose:
    Name and document official-style action chunk queuing: a policy query
    returns a 10-step action chunk and the wrapper executes `replan_steps`
    actions before the next policy query.

Inputs/outputs:
    Queue state is currently owned by `RealSimVLADCLDPolicy` in
    `rollout_runner`. New queue-specific helpers should be added here.

Official-match scope:
    The queue should match official SimVLA client semantics for full-mode rows.

DCLD scope:
    DCLD K is measured over policy-query/action-queue refills, not raw env
    steps.

Caveat:
    This refactor keeps queue behavior in place to preserve row planning and
    rollout behavior.
"""

from __future__ import annotations
