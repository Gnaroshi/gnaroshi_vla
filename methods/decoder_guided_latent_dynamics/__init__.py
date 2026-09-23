"""Architecture-neutral decoder-guided latent geometry utilities."""

from .geometry import (
    damped_action_to_latent_lift,
    damped_latent_decomposition,
    decoder_jacobian,
)
from .action_delta_predictor import (
    ObservationConditionedPredictor,
    SharedVisualDifferenceEncoder,
    count_trainable_parameters,
)

__all__ = [
    "damped_action_to_latent_lift",
    "damped_latent_decomposition",
    "decoder_jacobian",
    "ObservationConditionedPredictor",
    "SharedVisualDifferenceEncoder",
    "count_trainable_parameters",
]
