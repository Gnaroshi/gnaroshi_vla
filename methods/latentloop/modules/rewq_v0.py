"""Architecture-neutral primitives for the provisional ``rewq v0`` router.

The router predicts whether a cheaper condition/generation compute mode remains
recoverable after the *next* exact anchor.  It does not alter policy-query
cadence or action execution horizon.  ``exact condition + N_G=3`` is the
unconditional fallback and is never learned away.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CONTINUOUS_TARGET_NAMES = (
    "next_anchor_action_normalized_l1",
    "next_anchor_action_cosine_error",
    "post_recovery_proprio_normalized_l2",
    "post_recovery_ee_normalized_l2",
    "post_recovery_scene_cosine_error",
)
EVENT_TARGET_NAMES = (
    "post_recovery_gripper_mismatch",
    "post_recovery_contact_mismatch",
)


@dataclass(frozen=True)
class ComputeMode:
    """One condition/generation compute choice at a fixed policy query."""

    mode_id: int
    name: str
    exact_condition: bool
    generation_n_g: int
    predicted: bool


COMPUTE_MODES = (
    ComputeMode(0, "exact_condition_ng3", True, 3, False),
    ComputeMode(1, "approx_condition_ng3", False, 3, True),
    ComputeMode(2, "exact_condition_ng2", True, 2, True),
    ComputeMode(3, "approx_condition_ng2", False, 2, True),
)
PREDICTED_MODES = tuple(mode for mode in COMPUTE_MODES if mode.predicted)
MODE_BY_ID = {mode.mode_id: mode for mode in COMPUTE_MODES}
MODE_BY_NAME = {mode.name: mode for mode in COMPUTE_MODES}


@dataclass(frozen=True)
class ComputeCostTable:
    """Measured per-query costs used only to order already-safe modes."""

    exact_condition_ms: float
    approximate_condition_ms: float
    generation_ng3_ms: float
    generation_ng2_ms: float
    router_ms: float = 0.0
    provenance: str = "measured"

    def __post_init__(self) -> None:
        values = (
            self.exact_condition_ms,
            self.approximate_condition_ms,
            self.generation_ng3_ms,
            self.generation_ng2_ms,
            self.router_ms,
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("compute costs must be finite and non-negative")
        if not str(self.provenance).strip():
            raise ValueError("cost provenance must be declared")

    def values(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.tensor(
            (
                self.approximate_condition_ms
                + self.exact_condition_ms
                + self.generation_ng3_ms
                + self.router_ms,
                self.approximate_condition_ms + self.generation_ng3_ms + self.router_ms,
                self.approximate_condition_ms
                + self.exact_condition_ms
                + self.generation_ng2_ms
                + self.router_ms,
                self.approximate_condition_ms + self.generation_ng2_ms + self.router_ms,
            ),
            device=device,
            dtype=dtype,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode_cost_ms"] = {
            mode.name: float(
                self.approximate_condition_ms
                + (self.exact_condition_ms if mode.exact_condition else 0.0)
                + (self.generation_ng3_ms if mode.generation_n_g == 3 else self.generation_ng2_ms)
                + self.router_ms
            )
            for mode in COMPUTE_MODES
        }
        return payload


@dataclass(frozen=True)
class NextAnchorRecoveryTargets:
    """Observable branch differences after one exact recovery anchor."""

    continuous: Tensor
    continuous_valid: Tensor
    events: Tensor
    mode_valid: Tensor

    def validate(self) -> None:
        if self.continuous.ndim != 3:
            raise ValueError("continuous targets must be [B,M,C]")
        batch, modes, channels = self.continuous.shape
        if modes != len(PREDICTED_MODES) or channels != len(CONTINUOUS_TARGET_NAMES):
            raise ValueError("continuous target dimensions violate the rewq v0 contract")
        if self.continuous_valid.shape != self.continuous.shape:
            raise ValueError("continuous_valid must match continuous targets")
        if self.events.shape != (batch, modes, len(EVENT_TARGET_NAMES)):
            raise ValueError("event targets violate the rewq v0 contract")
        if self.mode_valid.shape != (batch, modes):
            raise ValueError("mode_valid must be [B,M]")
        finite = torch.isfinite(self.continuous) | ~self.continuous_valid.bool()
        if not bool(finite.all()):
            raise ValueError("valid continuous recovery targets must be finite")
        if not bool(torch.isfinite(self.events).all()):
            raise ValueError("event recovery targets must be finite")


def _broadcast_reference(reference: Tensor, candidate: Tensor) -> Tensor:
    if candidate.ndim != reference.ndim + 1:
        raise ValueError("candidate must add exactly one compute-mode dimension")
    if candidate.shape[0] != reference.shape[0] or candidate.shape[2:] != reference.shape[1:]:
        raise ValueError("candidate/reference branch shapes do not align")
    return reference.unsqueeze(1).expand_as(candidate)


def _normalized_l2(candidate: Tensor, reference: Tensor, scale: Tensor) -> Tensor:
    scale = torch.as_tensor(scale, device=candidate.device, dtype=torch.float32)
    if scale.shape != candidate.shape[2:]:
        raise ValueError(f"normalization scale must be {tuple(candidate.shape[2:])}")
    if bool((scale <= 0).any()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("normalization scales must be finite and positive")
    difference = (candidate.float() - reference.float()) / scale
    return torch.sqrt(difference.square().mean(dim=-1))


def _cosine_error(candidate: Tensor, reference: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    reference_norm = reference.float().norm(dim=-1)
    candidate_norm = candidate.float().norm(dim=-1)
    valid = (reference_norm > float(epsilon)) & (candidate_norm > float(epsilon))
    cosine = F.cosine_similarity(candidate.float(), reference.float(), dim=-1, eps=float(epsilon))
    return (1.0 - cosine).clamp(min=0.0, max=2.0), valid


def next_anchor_recovery_targets(
    *,
    candidate_next_action: Tensor,
    reference_next_action: Tensor,
    candidate_post_proprio: Tensor,
    reference_post_proprio: Tensor,
    candidate_post_ee: Tensor,
    reference_post_ee: Tensor,
    candidate_post_scene_feature: Tensor,
    reference_post_scene_feature: Tensor,
    candidate_post_gripper_closed: Tensor,
    reference_post_gripper_closed: Tensor,
    candidate_post_contact: Tensor,
    reference_post_contact: Tensor,
    action_scale: Tensor,
    proprio_scale: Tensor,
    ee_scale: Tensor,
    mode_valid: Tensor | None = None,
    first_r: int = 5,
    cosine_epsilon: float = 1e-6,
) -> NextAnchorRecoveryTargets:
    """Build labels from paired branches compared after ten environment actions.

    Candidate actions are decoded at the next exact anchor.  State/scene/event
    tensors are measured after executing the candidate query followed by that
    exact recovery query.  Simulator object state is deliberately absent.
    """

    if candidate_next_action.ndim != 4 or candidate_next_action.shape[-1] != 7:
        raise ValueError("candidate_next_action must be [B,M,H,7]")
    batch, modes, horizon, _ = candidate_next_action.shape
    if modes != len(PREDICTED_MODES):
        raise ValueError("candidate branches must follow PREDICTED_MODES")
    if reference_next_action.shape != (batch, horizon, 7):
        raise ValueError("reference_next_action must be [B,H,7]")
    selected = min(int(first_r), int(horizon))
    if selected < 1:
        raise ValueError("first_r must select at least one action")

    candidate_action = candidate_next_action[:, :, :selected].float()
    reference_action = _broadcast_reference(
        reference_next_action[:, :selected].float(), candidate_action
    )
    arm_scale = torch.as_tensor(
        action_scale, device=candidate_action.device, dtype=torch.float32
    )
    if arm_scale.shape != (6,) or bool((arm_scale <= 0).any()):
        raise ValueError("action_scale must contain six positive values")
    action_l1 = (
        (candidate_action[..., :6] - reference_action[..., :6]).abs() / arm_scale
    ).mean(dim=(2, 3))
    action_cosine, action_cosine_valid = _cosine_error(
        candidate_action[..., :6].reshape(batch, modes, -1),
        reference_action[..., :6].reshape(batch, modes, -1),
        cosine_epsilon,
    )

    reference_proprio = _broadcast_reference(reference_post_proprio, candidate_post_proprio)
    proprio_l2 = _normalized_l2(candidate_post_proprio, reference_proprio, proprio_scale)
    reference_ee = _broadcast_reference(reference_post_ee, candidate_post_ee)
    ee_l2 = _normalized_l2(candidate_post_ee, reference_ee, ee_scale)
    reference_scene = _broadcast_reference(
        reference_post_scene_feature, candidate_post_scene_feature
    )
    scene_cosine, scene_valid = _cosine_error(
        candidate_post_scene_feature, reference_scene, cosine_epsilon
    )

    continuous = torch.stack(
        (action_l1, action_cosine, proprio_l2, ee_l2, scene_cosine), dim=-1
    )
    continuous_valid = torch.ones_like(continuous, dtype=torch.bool)
    continuous_valid[..., 1] = action_cosine_valid
    continuous_valid[..., 4] = scene_valid

    gripper_reference = _broadcast_reference(
        reference_post_gripper_closed.bool(), candidate_post_gripper_closed.bool()
    )
    contact_reference = _broadcast_reference(
        reference_post_contact.bool(), candidate_post_contact.bool()
    )
    events = torch.stack(
        (
            candidate_post_gripper_closed.bool() != gripper_reference,
            candidate_post_contact.bool() != contact_reference,
        ),
        dim=-1,
    ).float()
    valid = (
        torch.ones((batch, modes), device=continuous.device, dtype=torch.bool)
        if mode_valid is None
        else mode_valid.to(device=continuous.device, dtype=torch.bool)
    )
    targets = NextAnchorRecoveryTargets(continuous, continuous_valid, events, valid)
    targets.validate()
    return targets


@dataclass(frozen=True)
class RecoveryPrediction:
    continuous_q90: Tensor
    event_logits: Tensor

    @property
    def event_probability(self) -> Tensor:
        return torch.sigmoid(self.event_logits)


class RecoverabilityHead(nn.Module):
    """Small multi-mode q90/event head; it neither sees images nor generates actions."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        bottleneck_dim: int = 32,
        quantile: float = 0.90,
    ) -> None:
        super().__init__()
        if int(input_dim) < 1 or int(hidden_dim) < 1 or int(bottleneck_dim) < 1:
            raise ValueError("network dimensions must be positive")
        if not 0.5 < float(quantile) < 1.0:
            raise ValueError("quantile must be in (0.5,1.0)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.quantile = float(quantile)
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(bottleneck_dim)),
            nn.GELU(),
        )
        channels_per_mode = len(CONTINUOUS_TARGET_NAMES) + len(EVENT_TARGET_NAMES)
        self.output = nn.Linear(int(bottleneck_dim), len(PREDICTED_MODES) * channels_per_mode)

    def forward(self, features: Tensor) -> RecoveryPrediction:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must be [B,{self.input_dim}]")
        batch = int(features.shape[0])
        raw = self.output(self.trunk(features.float())).reshape(
            batch,
            len(PREDICTED_MODES),
            len(CONTINUOUS_TARGET_NAMES) + len(EVENT_TARGET_NAMES),
        )
        return RecoveryPrediction(
            continuous_q90=F.softplus(raw[..., : len(CONTINUOUS_TARGET_NAMES)]),
            event_logits=raw[..., len(CONTINUOUS_TARGET_NAMES) :],
        )

    def parameter_audit(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "quantile": self.quantile,
            "total_parameters": int(total),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
            ),
            "parameter_ceiling": 100_000,
            "within_parameter_ceiling": total <= 100_000,
            "generates_actions": False,
            "encodes_images": False,
        }


def _pinball(prediction: Tensor, target: Tensor, quantile: float) -> Tensor:
    error = target.detach() - prediction
    return torch.maximum(float(quantile) * error, (float(quantile) - 1.0) * error)


def recoverability_loss(
    prediction: RecoveryPrediction,
    target: NextAnchorRecoveryTargets,
    *,
    normalization: Tensor,
    quantile: float = 0.90,
) -> dict[str, Tensor]:
    """Equal-channel loss after normalizing by the declared safe envelope."""

    target.validate()
    scale = torch.as_tensor(
        normalization,
        device=prediction.continuous_q90.device,
        dtype=prediction.continuous_q90.dtype,
    )
    if scale.shape != (len(CONTINUOUS_TARGET_NAMES),) or bool((scale <= 0).any()):
        raise ValueError("normalization must contain one positive value per continuous target")
    continuous_items = _pinball(
        prediction.continuous_q90 / scale,
        target.continuous.to(prediction.continuous_q90) / scale,
        quantile,
    )
    valid_continuous = target.continuous_valid.to(torch.bool) & target.mode_valid.unsqueeze(-1)
    continuous_loss = (
        continuous_items.masked_select(valid_continuous).mean()
        if bool(valid_continuous.any())
        else continuous_items.sum() * 0.0
    )
    event_items = F.binary_cross_entropy_with_logits(
        prediction.event_logits,
        target.events.to(prediction.event_logits),
        reduction="none",
    )
    valid_events = target.mode_valid.unsqueeze(-1).expand_as(event_items)
    event_loss = (
        event_items.masked_select(valid_events).mean()
        if bool(valid_events.any())
        else event_items.sum() * 0.0
    )
    total = (continuous_loss + event_loss) / 2.0
    return {
        "loss": total,
        "continuous_q90_pinball": continuous_loss,
        "event_bce": event_loss,
    }


def _wilson_upper(successes: Tensor, count: int, z: float = 1.959963984540054) -> Tensor:
    n = float(count)
    rate = successes.float() / n
    denominator = 1.0 + z * z / n
    centre = rate + z * z / (2.0 * n)
    radius = z * torch.sqrt((rate * (1.0 - rate) + z * z / (4.0 * n)) / n)
    return ((centre + radius) / denominator).clamp(max=1.0)


@dataclass(frozen=True)
class RecoverySafetyEnvelope:
    """Empirical safe region derived only from successful K_C=2,N_G=3 rows."""

    continuous_limits: tuple[float, ...]
    event_probability_limits: tuple[float, ...]
    continuous_quantile: float
    confidence: float
    reference_rows: int
    reference_row: str
    provenance: str

    def __post_init__(self) -> None:
        if len(self.continuous_limits) != len(CONTINUOUS_TARGET_NAMES):
            raise ValueError("continuous envelope dimension mismatch")
        if len(self.event_probability_limits) != len(EVENT_TARGET_NAMES):
            raise ValueError("event envelope dimension mismatch")
        if self.reference_rows < 1 or self.reference_row != "condition_kc2_ng3":
            raise ValueError("safe envelope must use successful condition_kc2_ng3 rows")
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in self.continuous_limits
        ):
            raise ValueError("continuous envelope limits must be finite and non-negative")
        if not all(
            math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
            for value in self.event_probability_limits
        ):
            raise ValueError("event envelope limits must be probabilities")
        if not 0.5 < float(self.continuous_quantile) < 1.0:
            raise ValueError("continuous envelope quantile must be in (0.5,1.0)")
        if not 0.5 < float(self.confidence) < 1.0:
            raise ValueError("envelope confidence must be in (0.5,1.0)")
        if not str(self.provenance).strip():
            raise ValueError("safe envelope provenance is required")

    def tensors(self, like: Tensor) -> tuple[Tensor, Tensor]:
        return (
            like.new_tensor(self.continuous_limits),
            like.new_tensor(self.event_probability_limits),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecoverySafetyEnvelope":
        return cls(
            continuous_limits=tuple(float(value) for value in payload["continuous_limits"]),
            event_probability_limits=tuple(
                float(value) for value in payload["event_probability_limits"]
            ),
            continuous_quantile=float(payload["continuous_quantile"]),
            confidence=float(payload["confidence"]),
            reference_rows=int(payload["reference_rows"]),
            reference_row=str(payload["reference_row"]),
            provenance=str(payload["provenance"]),
        )


def fit_recovery_safety_envelope(
    continuous: Tensor,
    events: Tensor,
    *,
    quantile: float = 0.95,
    confidence: float = 0.95,
    provenance: str,
) -> RecoverySafetyEnvelope:
    if continuous.ndim != 2 or continuous.shape[1] != len(CONTINUOUS_TARGET_NAMES):
        raise ValueError("reference continuous values must be [N,C]")
    if events.shape != (continuous.shape[0], len(EVENT_TARGET_NAMES)):
        raise ValueError("reference events must be [N,E]")
    if not 0.5 < float(quantile) < 1.0 or not 0.5 < float(confidence) < 1.0:
        raise ValueError("quantile/confidence must be in (0.5,1.0)")
    if not str(provenance).strip():
        raise ValueError("reference provenance is required")
    if not bool(torch.isfinite(continuous).all()) or not bool(torch.isfinite(events).all()):
        raise ValueError("reference envelope values must be finite")
    count = int(continuous.shape[0])
    if count < 20:
        raise ValueError("at least 20 successful reference rows are required")
    z = NormalDist().inv_cdf((1.0 + float(confidence)) / 2.0)
    event_upper = _wilson_upper(events.float().sum(dim=0), count, z=z)
    return RecoverySafetyEnvelope(
        continuous_limits=tuple(
            float(value)
            for value in torch.quantile(continuous.float(), float(quantile), dim=0).tolist()
        ),
        event_probability_limits=tuple(float(value) for value in event_upper.tolist()),
        continuous_quantile=float(quantile),
        confidence=float(confidence),
        reference_rows=count,
        reference_row="condition_kc2_ng3",
        provenance=str(provenance),
    )


@dataclass(frozen=True)
class SplitConformalCalibration:
    continuous_offsets: tuple[tuple[float, ...], ...]
    event_offsets: tuple[tuple[float, ...], ...]
    alpha: float
    calibration_rows: int
    provenance: str

    def __post_init__(self) -> None:
        if len(self.continuous_offsets) != len(PREDICTED_MODES):
            raise ValueError("conformal continuous mode dimension mismatch")
        if len(self.event_offsets) != len(PREDICTED_MODES):
            raise ValueError("conformal event mode dimension mismatch")
        if any(len(row) != len(CONTINUOUS_TARGET_NAMES) for row in self.continuous_offsets):
            raise ValueError("conformal continuous channel dimension mismatch")
        if any(len(row) != len(EVENT_TARGET_NAMES) for row in self.event_offsets):
            raise ValueError("conformal event channel dimension mismatch")
        values = tuple(value for row in self.continuous_offsets for value in row) + tuple(
            value for row in self.event_offsets for value in row
        )
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("conformal offsets must be finite and non-negative")
        if not 0.0 < float(self.alpha) < 0.5 or self.calibration_rows < 1:
            raise ValueError("invalid conformal alpha or calibration row count")
        if not str(self.provenance).strip():
            raise ValueError("conformal provenance is required")

    def tensors(self, like: Tensor) -> tuple[Tensor, Tensor]:
        return like.new_tensor(self.continuous_offsets), like.new_tensor(self.event_offsets)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitConformalCalibration":
        return cls(
            continuous_offsets=tuple(
                tuple(float(value) for value in row)
                for row in payload["continuous_offsets"]
            ),
            event_offsets=tuple(
                tuple(float(value) for value in row) for row in payload["event_offsets"]
            ),
            alpha=float(payload["alpha"]),
            calibration_rows=int(payload["calibration_rows"]),
            provenance=str(payload["provenance"]),
        )


def _finite_sample_quantile(values: Tensor, alpha: float) -> Tensor:
    count = int(values.shape[0])
    rank = min(count, math.ceil((count + 1) * (1.0 - float(alpha))))
    return values.float().sort(dim=0).values[rank - 1]


def fit_split_conformal_calibration(
    prediction: RecoveryPrediction,
    target: NextAnchorRecoveryTargets,
    *,
    alpha: float = 0.10,
    provenance: str,
) -> SplitConformalCalibration:
    target.validate()
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("conformal alpha must be in (0,0.5)")
    if prediction.continuous_q90.shape != target.continuous.shape:
        raise ValueError("prediction/target continuous shapes differ")
    if prediction.event_logits.shape != target.events.shape:
        raise ValueError("prediction/target event shapes differ")
    continuous_offsets: list[tuple[float, ...]] = []
    event_offsets: list[tuple[float, ...]] = []
    probabilities = prediction.event_probability.detach()
    for mode_index in range(len(PREDICTED_MODES)):
        mode_mask = target.mode_valid[:, mode_index].bool()
        if int(mode_mask.sum()) < 20:
            raise ValueError("each mode needs at least 20 conformal rows")
        channel_offsets: list[float] = []
        for channel in range(len(CONTINUOUS_TARGET_NAMES)):
            mask = mode_mask & target.continuous_valid[:, mode_index, channel].bool()
            if int(mask.sum()) < 20:
                raise ValueError("each continuous channel needs at least 20 conformal rows")
            residual = (
                target.continuous[mask, mode_index, channel]
                - prediction.continuous_q90.detach()[mask, mode_index, channel]
            ).clamp_min(0.0)
            channel_offsets.append(float(_finite_sample_quantile(residual, alpha)))
        continuous_offsets.append(tuple(channel_offsets))
        event_residual = (
            target.events[mode_mask, mode_index]
            - probabilities[mode_mask, mode_index]
        ).clamp_min(0.0)
        event_offsets.append(
            tuple(float(value) for value in _finite_sample_quantile(event_residual, alpha).tolist())
        )
    return SplitConformalCalibration(
        continuous_offsets=tuple(continuous_offsets),
        event_offsets=tuple(event_offsets),
        alpha=float(alpha),
        calibration_rows=int(target.mode_valid.sum()),
        provenance=str(provenance),
    )


@dataclass(frozen=True)
class ModeAssessment:
    selected_mode_id: Tensor
    admissible: Tensor
    continuous_ucb: Tensor
    event_probability_ucb: Tensor
    mode_cost_ms: Tensor


class RecoverabilityRouter:
    """GPU-native cheapest-safe mode selector with exact N_G=3 fallback."""

    def __init__(
        self,
        *,
        envelope: RecoverySafetyEnvelope,
        conformal: SplitConformalCalibration,
        costs: ComputeCostTable,
        max_approximate_age: int = 7,
    ) -> None:
        if int(max_approximate_age) < 1:
            raise ValueError("max_approximate_age must be positive")
        self.envelope = envelope
        self.conformal = conformal
        self.costs = costs
        self.max_approximate_age = int(max_approximate_age)

    def assess(
        self,
        prediction: RecoveryPrediction,
        *,
        candidate_age: Tensor,
        anchor_available: Tensor,
    ) -> ModeAssessment:
        """Return tensor decisions without CPU copies or scalar synchronization."""

        batch = int(prediction.continuous_q90.shape[0])
        if prediction.continuous_q90.shape != (
            batch,
            len(PREDICTED_MODES),
            len(CONTINUOUS_TARGET_NAMES),
        ):
            raise ValueError("continuous prediction shape violates mode contract")
        if prediction.event_logits.shape != (
            batch,
            len(PREDICTED_MODES),
            len(EVENT_TARGET_NAMES),
        ):
            raise ValueError("event prediction shape violates mode contract")
        age = candidate_age.to(
            device=prediction.continuous_q90.device, dtype=torch.long
        ).reshape(batch)
        anchor = anchor_available.to(
            device=prediction.continuous_q90.device, dtype=torch.bool
        ).reshape(batch)
        continuous_offset, event_offset = self.conformal.tensors(
            prediction.continuous_q90
        )
        continuous_ucb = prediction.continuous_q90 + continuous_offset.unsqueeze(0)
        event_ucb = (
            prediction.event_probability + event_offset.unsqueeze(0)
        ).clamp(max=1.0)
        continuous_limit, event_limit = self.envelope.tensors(
            prediction.continuous_q90
        )
        candidate_safe = (
            (continuous_ucb <= continuous_limit).all(dim=-1)
            & (event_ucb <= event_limit).all(dim=-1)
        )

        admissible = torch.zeros(
            (batch, len(COMPUTE_MODES)),
            device=prediction.continuous_q90.device,
            dtype=torch.bool,
        )
        admissible[:, 0] = True
        admissible[:, 1:] = candidate_safe
        approximate_age_safe = anchor & (age <= self.max_approximate_age)
        admissible[:, 1] &= approximate_age_safe
        admissible[:, 3] &= approximate_age_safe
        # Exact N_G=2 still needs a valid runtime prediction.  At episode start
        # or beyond the trained age support, M0 is the only legal mode.
        admissible[:, 2] &= anchor & (age <= self.max_approximate_age)

        costs = self.costs.values(
            device=prediction.continuous_q90.device,
            dtype=prediction.continuous_q90.dtype,
        )
        masked_cost = torch.where(
            admissible,
            costs.unsqueeze(0),
            torch.full_like(admissible, torch.inf, dtype=prediction.continuous_q90.dtype),
        )
        selected = masked_cost.argmin(dim=-1)
        return ModeAssessment(selected, admissible, continuous_ucb, event_ucb, costs)

    def contract(self) -> dict[str, Any]:
        return {
            "provisional_name": "rewq_v0",
            "decision_scope": "condition_compute_and_generation_compute",
            "compute_modes": [asdict(mode) for mode in COMPUTE_MODES],
            "fallback": COMPUTE_MODES[0].name,
            "max_approximate_age": self.max_approximate_age,
            "allows_kc4": self.max_approximate_age >= 3,
            "allows_kc8": self.max_approximate_age >= 7,
            "changes_policy_query_cadence": False,
            "changes_action_execution_horizon": False,
            "action_horizon": 10,
            "execution_horizon": 5,
            "recovery_horizon_actions": 10,
            "costs": self.costs.to_dict(),
            "envelope": self.envelope.to_dict(),
            "conformal": self.conformal.to_dict(),
        }
