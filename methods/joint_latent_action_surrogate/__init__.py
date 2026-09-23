"""Joint latent-anchored action surrogate primitives."""

from .alignment import (
    AlignedActionAnchor,
    ImmutableActionAnchorCache,
    align_immutable_anchor,
)
from .decisions import JointVerdict, apply_joint_decision_rule
from .losses import JointLossWeights, joint_surrogate_loss
from .metrics import ErrorAccumulator, action_surrogate_error, latent_state_error
from .modules import (
    JointLatentAnchoredActionSurrogate,
    SurrogateOutput,
    WideLatentCapacityControl,
    joint_surrogate_parameter_count,
    wide_control_parameter_count,
)
from .schedule import JointExecutionLevel, JointHierarchySchedule

__all__ = [
    "AlignedActionAnchor",
    "ErrorAccumulator",
    "JointExecutionLevel",
    "JointHierarchySchedule",
    "JointLatentAnchoredActionSurrogate",
    "JointLossWeights",
    "JointVerdict",
    "ImmutableActionAnchorCache",
    "SurrogateOutput",
    "WideLatentCapacityControl",
    "action_surrogate_error",
    "align_immutable_anchor",
    "apply_joint_decision_rule",
    "joint_surrogate_loss",
    "joint_surrogate_parameter_count",
    "latent_state_error",
    "wide_control_parameter_count",
]
