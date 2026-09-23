"""Default-off schedule adapter for the existing Seer LatentLoop cache path.

The adapter does not duplicate Seer preprocessing, temporal ensembling, latent
dynamics, or ``A_psi``. It only decides whether the existing call
``u_t = E_eta(o_{t-1}, o_t)`` receives current observations or a zero feature at
an intermediate segment offset. The existing updater still computes
``z_t^L = U_theta(z_{t-1}^L, u_t, r_t)`` at every intermediate step, and the
existing shared path executes ``a_t = A_psi(z_t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from methods.latentloop_segment_grid.feedback_schedule import (
    FeedbackPlan,
    build_feedback_plan,
)


@dataclass(frozen=True)
class SegmentStepDecision:
    """Execution decision for one environment timestep."""

    timestep: int
    segment_length: int
    segment_offset: int
    full_refresh: bool
    feedback_enabled: Optional[bool]
    planned_feedback_density: Optional[float]

    @property
    def use_zero_feature(self) -> bool:
        """Return whether ``u_t`` must be replaced by zero at this step."""

        return self.feedback_enabled is False


class LatentLoopSegmentExecutor:
    """Map environment timesteps to full refreshes and fixed feedback masks."""

    def __init__(self, segment_length: int, feedback_schedule: str) -> None:
        self.plan: FeedbackPlan = build_feedback_plan(
            segment_length, feedback_schedule
        )

    @property
    def segment_length(self) -> int:
        """Return the number of environment actions in each segment."""

        return self.plan.segment_length

    @property
    def feedback_schedule(self) -> str:
        """Return the predeclared schedule name."""

        return self.plan.schedule

    def decision(self, timestep: int, *, has_latent_cache: bool) -> SegmentStepDecision:
        """Return the schedule decision without changing model or cache state."""

        step = int(timestep)
        if step < 0:
            raise ValueError(f"timestep must be non-negative, got {step}")
        offset = step % self.plan.segment_length
        full_refresh = not has_latent_cache or offset == 0
        enabled: Optional[bool]
        if full_refresh:
            enabled = None
        else:
            enabled = self.plan.enabled_at(offset)
        return SegmentStepDecision(
            timestep=step,
            segment_length=self.plan.segment_length,
            segment_offset=offset,
            full_refresh=full_refresh,
            feedback_enabled=enabled,
            planned_feedback_density=self.plan.density,
        )


def canonical_predicted_horizon_token_indices(
    segment_length: int,
    action_pred_steps: int,
) -> List[int]:
    """Return canonical skip-token indices or reject an unsupported segment.

    The current Seer evaluator uses ``token_idx = skip_age - 1``. Therefore a
    segment with ``L-1`` skipped actions is valid only when all indices fit in
    the existing ``action_pred_steps`` tokens; no future-token semantics are
    invented beyond that horizon.
    """

    skipped_actions = int(segment_length) - 1
    available_tokens = int(action_pred_steps)
    if skipped_actions < 0 or available_tokens <= 0:
        raise ValueError("segment_length and action_pred_steps must be positive")
    if skipped_actions > available_tokens:
        raise ValueError(
            f"L={segment_length} needs {skipped_actions} cached tokens, but Seer "
            f"provides action_pred_steps={available_tokens}"
        )
    return list(range(skipped_actions))


def simulate_segment_execution(
    segment_length: int,
    feedback_schedule: str,
    num_steps: int,
) -> List[Dict[str, object]]:
    """Simulate counters and cache advancement without loading Seer.

    This helper mirrors the production contract: a full Seer call occurs only
    at offset zero; the updater and observation-cache write occur at every
    intermediate step; only the current-observation feature is masked.
    """

    executor = LatentLoopSegmentExecutor(segment_length, feedback_schedule)
    events: List[Dict[str, object]] = []
    has_cache = False
    for timestep in range(int(num_steps)):
        decision = executor.decision(timestep, has_latent_cache=has_cache)
        event = {
            "timestep": timestep,
            "segment_offset": decision.segment_offset,
            "full_forward_called": int(decision.full_refresh),
            "updater_called": int(not decision.full_refresh),
            "feedback_mask": (
                None
                if decision.feedback_enabled is None
                else int(decision.feedback_enabled)
            ),
            "observation_conditioned_updater_called": int(
                decision.feedback_enabled is True
            ),
            "zero_feature_updater_called": int(
                decision.feedback_enabled is False
            ),
            "observation_cache_advanced": 1,
        }
        events.append(event)
        has_cache = True
    return events
