"""Cross-query plan-continuation analysis and baseline modules."""

from .action_correction import (
    ActionCorrectionOutput,
    MatchedActionSpaceCorrection,
    find_action_correction_hidden_dim,
    shift_action_horizon,
)
from .anchor_bridge import (
    AnchorBridgeOutput,
    NonRecurrentAnchorToCurrentBridge,
    find_anchor_bridge_hidden_dim,
)
from .cqpc_loss import (
    CQPCLossOutput,
    cqpc_is_enabled,
    cross_query_plan_consistency_loss,
)
from .decisions import apply_plan_continuation_decision
from .feedback_metrics import binned_summary, spearman_correlation
from .feedback_source import FeedbackFeatureBuffer, FeedbackSelection
from .overlap_metrics import aligned_overlap_components, summarize_values

__all__ = [
    "ActionCorrectionOutput",
    "AnchorBridgeOutput",
    "CQPCLossOutput",
    "MatchedActionSpaceCorrection",
    "NonRecurrentAnchorToCurrentBridge",
    "FeedbackFeatureBuffer",
    "FeedbackSelection",
    "aligned_overlap_components",
    "apply_plan_continuation_decision",
    "binned_summary",
    "cross_query_plan_consistency_loss",
    "cqpc_is_enabled",
    "find_action_correction_hidden_dim",
    "find_anchor_bridge_hidden_dim",
    "shift_action_horizon",
    "spearman_correlation",
    "summarize_values",
]
