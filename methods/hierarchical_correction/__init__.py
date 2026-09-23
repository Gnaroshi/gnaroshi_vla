"""Architecture-neutral primitives for Hierarchical Latent-Action Correction."""

from .decisions import (
    HybridGateInputs,
    NativeHybridDecisionInputs,
    RegenerationCandidateGateInputs,
    evaluate_hybrid_gate,
    evaluate_native_hybrid_decision,
    evaluate_r5_regeneration_gate,
    evaluate_regeneration_candidate_gate,
)
from .horizon_provenance import (
    ActionTokenProvenance,
    ProvenanceSchedule,
    corrected_chunk_provenance,
    derive_provenance_schedule,
    first_lineage_exhaustion_query,
    generated_chunk_provenance,
    generator_backed_count,
    original_tokens_remaining,
    r1_hybrid_readiness_verdict,
    simulate_token_provenance,
)
from .metrics import (
    correction_residuals_by_age,
    hierarchical_action_diagnostics,
    paired_outcome_summary,
    trace_metrics_by_age,
)
from .policy_state import (
    HYBRID_TRACE_SCHEMA_VERSION,
    REQUIRED_TRACE_FIELDS,
    HierarchicalPolicyState,
    validate_trace_record,
)
from .provenance import experiment_source_signature, hierarchical_source_manifest
from .schedules import ExecutionLevel, HierarchicalSchedule

__all__ = [
    "ExecutionLevel",
    "ActionTokenProvenance",
    "HYBRID_TRACE_SCHEMA_VERSION",
    "HierarchicalPolicyState",
    "HierarchicalSchedule",
    "HybridGateInputs",
    "NativeHybridDecisionInputs",
    "ProvenanceSchedule",
    "RegenerationCandidateGateInputs",
    "REQUIRED_TRACE_FIELDS",
    "correction_residuals_by_age",
    "evaluate_hybrid_gate",
    "evaluate_native_hybrid_decision",
    "evaluate_r5_regeneration_gate",
    "evaluate_regeneration_candidate_gate",
    "corrected_chunk_provenance",
    "derive_provenance_schedule",
    "experiment_source_signature",
    "first_lineage_exhaustion_query",
    "generated_chunk_provenance",
    "generator_backed_count",
    "hierarchical_action_diagnostics",
    "hierarchical_source_manifest",
    "paired_outcome_summary",
    "original_tokens_remaining",
    "r1_hybrid_readiness_verdict",
    "simulate_token_provenance",
    "trace_metrics_by_age",
    "validate_trace_record",
]
