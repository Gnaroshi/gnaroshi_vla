"""Architecture-neutral action-equivalent selective refresh primitives.

The refresh decision predicts the action-space error caused by substituting an
approximate latent update for an exact backbone call.  It never changes the
policy-query cadence, action execution horizon, or action-generation schedule.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ActionFidelityPrediction:
    """Predicted counterfactual action errors for one policy query."""

    arm_q90: Tensor
    direction_q90: Tensor
    gripper_mismatch_logit: Tensor

    @property
    def gripper_mismatch_probability(self) -> Tensor:
        return torch.sigmoid(self.gripper_mismatch_logit)


@dataclass(frozen=True)
class CounterfactualActionTargets:
    """Same-noise exact-versus-approximate action discrepancy targets."""

    arm_normalized_l1: Tensor
    direction_cosine_error: Tensor
    direction_valid: Tensor
    gripper_mismatch: Tensor


class ActionFidelityHead(nn.Module):
    """Small head that estimates the consequence of an approximate latent.

    The head consumes only features available after the cheap latent update.
    Exact conditions and exact actions are training labels, never runtime
    inputs.  Positive error outputs are represented with ``softplus``.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 96,
        bottleneck_dim: int = 48,
        quantile: float = 0.90,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1:
            raise ValueError("input_dim must be positive")
        if int(hidden_dim) < 1 or int(bottleneck_dim) < 1:
            raise ValueError("hidden dimensions must be positive")
        if not 0.5 < float(quantile) < 1.0:
            raise ValueError("risk quantile must be in (0.5,1.0)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.quantile = float(quantile)
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(self.bottleneck_dim, 3)

    def forward(self, features: Tensor) -> ActionFidelityPrediction:
        if features.ndim != 2 or int(features.shape[-1]) != self.input_dim:
            raise ValueError(
                f"features must be [B,{self.input_dim}], got {tuple(features.shape)}"
            )
        raw = self.output(self.trunk(features.float()))
        return ActionFidelityPrediction(
            arm_q90=F.softplus(raw[:, 0]),
            direction_q90=F.softplus(raw[:, 1]),
            gripper_mismatch_logit=raw[:, 2],
        )

    def parameter_audit(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "quantile": self.quantile,
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "generates_actions": False,
            "encodes_images": False,
        }


def _direction_error(
    approximate: Tensor,
    exact: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    exact_norm = exact.norm(dim=-1)
    valid = exact_norm > float(epsilon)
    cosine = F.cosine_similarity(approximate, exact, dim=-1, eps=float(epsilon))
    error = (1.0 - cosine).clamp(min=0.0, max=2.0)
    return error, valid


def counterfactual_action_targets(
    approximate_action: Tensor,
    exact_action: Tensor,
    *,
    arm_scale: Tensor,
    first_r: int = 5,
    direction_epsilon: float = 1e-5,
) -> CounterfactualActionTargets:
    """Build labels from actions decoded with the same initial flow noise.

    ``arm_scale`` must contain six positive, externally declared action scales.
    This avoids mixing translation and rotation units through an implicit
    normalization.  Direction error averages translation and rotation cosine
    errors only when the corresponding exact vector is non-zero.
    """

    if approximate_action.shape != exact_action.shape:
        raise ValueError("approximate and exact actions must have identical shapes")
    if approximate_action.ndim != 3 or int(approximate_action.shape[-1]) != 7:
        raise ValueError("actions must be [B,H,7]")
    horizon = min(int(first_r), int(approximate_action.shape[1]))
    if horizon < 1:
        raise ValueError("first_r must select at least one action")
    scale = torch.as_tensor(
        arm_scale,
        device=approximate_action.device,
        dtype=torch.float32,
    )
    if scale.shape != (6,) or bool((scale <= 0).any()):
        raise ValueError("arm_scale must contain six positive values")

    approximate = approximate_action[:, :horizon].float()
    exact = exact_action[:, :horizon].detach().float()
    arm_error = ((approximate[..., :6] - exact[..., :6]).abs() / scale).mean(
        dim=(1, 2)
    )

    translation_error, translation_valid = _direction_error(
        approximate[..., :3], exact[..., :3], epsilon=direction_epsilon
    )
    rotation_error, rotation_valid = _direction_error(
        approximate[..., 3:6], exact[..., 3:6], epsilon=direction_epsilon
    )
    direction_values = torch.cat((translation_error, rotation_error), dim=1)
    direction_mask = torch.cat((translation_valid, rotation_valid), dim=1)
    direction_count = direction_mask.sum(dim=1)
    direction_mean = (
        (direction_values * direction_mask.float()).sum(dim=1)
        / direction_count.clamp_min(1).float()
    )

    approximate_sign = approximate[..., 6] >= 0.0
    exact_sign = exact[..., 6] >= 0.0
    sign_mismatch = (approximate_sign != exact_sign).any(dim=1)
    if horizon > 1:
        approximate_switch = approximate_sign[:, 1:] != approximate_sign[:, :-1]
        exact_switch = exact_sign[:, 1:] != exact_sign[:, :-1]
        switch_mismatch = (approximate_switch != exact_switch).any(dim=1)
    else:
        switch_mismatch = torch.zeros_like(sign_mismatch)
    gripper_mismatch = (sign_mismatch | switch_mismatch).float()

    return CounterfactualActionTargets(
        arm_normalized_l1=arm_error,
        direction_cosine_error=direction_mean,
        direction_valid=direction_count > 0,
        gripper_mismatch=gripper_mismatch,
    )


def _pinball(prediction: Tensor, target: Tensor, quantile: float) -> Tensor:
    error = target.detach() - prediction
    return torch.maximum(float(quantile) * error, (float(quantile) - 1.0) * error)


def action_fidelity_loss(
    prediction: ActionFidelityPrediction,
    target: CounterfactualActionTargets,
    *,
    quantile: float = 0.90,
) -> dict[str, Tensor]:
    """Equal-task supervision without a hand-tuned routing scalar."""

    arm = _pinball(
        prediction.arm_q90, target.arm_normalized_l1, quantile
    ).mean()
    direction_items = _pinball(
        prediction.direction_q90, target.direction_cosine_error, quantile
    )
    valid = target.direction_valid.to(device=direction_items.device, dtype=torch.bool)
    direction = (
        direction_items[valid].mean()
        if bool(valid.any())
        else direction_items.sum() * 0.0
    )
    gripper = F.binary_cross_entropy_with_logits(
        prediction.gripper_mismatch_logit,
        target.gripper_mismatch.to(prediction.gripper_mismatch_logit),
    )
    total = (arm + direction + gripper) / 3.0
    return {
        "loss": total,
        "arm_q90_pinball": arm,
        "direction_q90_pinball": direction,
        "gripper_bce": gripper,
    }


def _percentile(sorted_reference: Sequence[float], value: float) -> float:
    if not sorted_reference:
        raise ValueError("calibration reference cannot be empty")
    return bisect.bisect_right(sorted_reference, float(value)) / len(sorted_reference)


@dataclass(frozen=True)
class ExactCallBudgetCalibration:
    """Empirical, weight-free score calibration for matched exact-call budgets."""

    arm_reference: tuple[float, ...]
    gripper_reference: tuple[float, ...]
    route_threshold: float
    target_exact_fraction: float
    observed_exact_fraction: float
    max_approximate_age: int
    calibration_queries: int

    def score_values(self, arm_q90: float, gripper_probability: float) -> float:
        return max(
            _percentile(self.arm_reference, arm_q90),
            _percentile(self.gripper_reference, gripper_probability),
        )

    def score(self, prediction: ActionFidelityPrediction) -> Tensor:
        arm = prediction.arm_q90.detach().cpu().reshape(-1).tolist()
        gripper = (
            prediction.gripper_mismatch_probability.detach().cpu().reshape(-1).tolist()
        )
        scores = [
            self.score_values(left, right) for left, right in zip(arm, gripper)
        ]
        return torch.tensor(
            scores,
            device=prediction.arm_q90.device,
            dtype=prediction.arm_q90.dtype,
        ).reshape(prediction.arm_q90.shape)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExactCallBudgetCalibration":
        return cls(
            arm_reference=tuple(float(value) for value in payload["arm_reference"]),
            gripper_reference=tuple(
                float(value) for value in payload["gripper_reference"]
            ),
            route_threshold=float(payload["route_threshold"]),
            target_exact_fraction=float(payload["target_exact_fraction"]),
            observed_exact_fraction=float(payload["observed_exact_fraction"]),
            max_approximate_age=int(payload["max_approximate_age"]),
            calibration_queries=int(payload["calibration_queries"]),
        )


def simulate_exact_fraction(
    scores: Sequence[float],
    episode_first_candidates: Sequence[bool],
    *,
    threshold: float,
    max_approximate_age: int,
) -> tuple[float, tuple[bool, ...]]:
    """Simulate candidate routing including each episode's implicit exact q0."""

    if len(scores) != len(episode_first_candidates) or not scores:
        raise ValueError(
            "scores and episode_first_candidates must be non-empty and aligned"
        )
    if int(max_approximate_age) < 1:
        raise ValueError("max_approximate_age must be positive")
    approximate_age = 0
    exact_count = 0
    query_count = 0
    exact: list[bool] = []
    for score, first_candidate in zip(scores, episode_first_candidates):
        if bool(first_candidate):
            # Compact records begin at q1.  Account for the preceding exact q0
            # before routing q1 itself.
            exact_count += 1
            query_count += 1
            approximate_age = 0
        force = approximate_age >= int(max_approximate_age)
        choose_exact = force or float(score) >= float(threshold)
        exact.append(choose_exact)
        exact_count += int(choose_exact)
        query_count += 1
        approximate_age = 0 if choose_exact else approximate_age + 1
    return exact_count / query_count, tuple(exact)


def fit_exact_call_budget_calibration(
    arm_q90: Sequence[float],
    gripper_probability: Sequence[float],
    episode_first_candidates: Sequence[bool],
    *,
    target_exact_fraction: float,
    max_approximate_age: int = 3,
) -> ExactCallBudgetCalibration:
    """Fit a deterministic threshold closest to a declared exact-call budget.

    Arm and gripper predictions are converted to held-out empirical percentiles
    and combined by maximum.  This preserves either failure mode without an
    arbitrary weighted sum.  Forced first-query/max-age refreshes are included
    when matching the requested budget.
    """

    if not (
        len(arm_q90) == len(gripper_probability) == len(episode_first_candidates)
        and len(arm_q90) > 0
    ):
        raise ValueError("calibration arrays must be non-empty and aligned")
    if not 0.0 < float(target_exact_fraction) <= 1.0:
        raise ValueError("target_exact_fraction must be in (0,1]")
    arm_reference = tuple(sorted(float(value) for value in arm_q90))
    gripper_reference = tuple(sorted(float(value) for value in gripper_probability))
    scores = tuple(
        max(
            _percentile(arm_reference, arm),
            _percentile(gripper_reference, gripper),
        )
        for arm, gripper in zip(arm_q90, gripper_probability)
    )
    candidates = (math.inf, *sorted(set(scores), reverse=True), -math.inf)
    choices: list[tuple[float, bool, float, float]] = []
    for threshold in candidates:
        fraction, _ = simulate_exact_fraction(
            scores,
            episode_first_candidates,
            threshold=threshold,
            max_approximate_age=max_approximate_age,
        )
        choices.append(
            (
                abs(fraction - float(target_exact_fraction)),
                fraction > float(target_exact_fraction),
                -fraction,
                float(threshold),
            )
        )
    _, _, _, threshold = min(choices)
    observed, _ = simulate_exact_fraction(
        scores,
        episode_first_candidates,
        threshold=threshold,
        max_approximate_age=max_approximate_age,
    )
    return ExactCallBudgetCalibration(
        arm_reference=arm_reference,
        gripper_reference=gripper_reference,
        route_threshold=threshold,
        target_exact_fraction=float(target_exact_fraction),
        observed_exact_fraction=float(observed),
        max_approximate_age=int(max_approximate_age),
        calibration_queries=len(scores) + sum(bool(value) for value in episode_first_candidates),
    )


@dataclass(frozen=True)
class RefreshDecision:
    use_exact: bool
    reason: str
    candidate_age: int
    risk_score: float | None


class ActionEquivalentRefreshRouter:
    """State machine that changes only exact-versus-approximate condition compute."""

    def __init__(self, calibration: ExactCallBudgetCalibration) -> None:
        self.calibration = calibration
        self.reset()

    def reset(self) -> None:
        self.query_count = 0
        self.approximate_age = 0

    def candidate_required(self) -> bool:
        return self.query_count > 0 and (
            self.approximate_age < self.calibration.max_approximate_age
        )

    def decide(
        self, prediction: ActionFidelityPrediction | None = None
    ) -> RefreshDecision:
        candidate_age = self.approximate_age + 1
        if self.query_count == 0:
            decision = RefreshDecision(True, "episode_start", 0, None)
        elif self.approximate_age >= self.calibration.max_approximate_age:
            decision = RefreshDecision(True, "max_age", candidate_age, None)
        else:
            if prediction is None:
                raise ValueError("an eligible approximate query requires a risk prediction")
            if prediction.arm_q90.numel() != 1:
                raise ValueError("online router expects one query at a time")
            score = float(self.calibration.score(prediction).item())
            use_exact = score >= self.calibration.route_threshold
            decision = RefreshDecision(
                use_exact,
                "counterfactual_risk" if use_exact else "approximate_safe",
                candidate_age,
                score,
            )
        self.query_count += 1
        self.approximate_age = 0 if decision.use_exact else candidate_age
        return decision

    def contract(self) -> dict[str, Any]:
        return {
            "decision_scope": "exact_condition_or_approximate_condition",
            "changes_policy_query_cadence": False,
            "changes_action_execution_horizon": False,
            "changes_action_generation_schedule": False,
            "max_approximate_age": self.calibration.max_approximate_age,
            "target_exact_fraction": self.calibration.target_exact_fraction,
            "calibrated_exact_fraction": self.calibration.observed_exact_fraction,
        }
