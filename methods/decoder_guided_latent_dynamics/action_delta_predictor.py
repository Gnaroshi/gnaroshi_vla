"""Small observation-conditioned predictors for offline latent-geometry studies."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SharedVisualDifferenceEncoder(nn.Module):
    """Encode primary/wrist previous-current image pairs with shared weights."""

    def __init__(self, feature_dim: int = 128, proprio_dim: int = 8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.proprio_dim = proprio_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(9, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.image_projection = nn.Linear(128, feature_dim)
        self.proprio_projection = nn.Sequential(
            nn.Linear(proprio_dim * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def _camera(self, previous: Tensor, current: Tensor) -> Tensor:
        if previous.shape != current.shape or previous.shape[-3] != 3:
            raise ValueError(
                f"Expected matching [...,3,H,W] image tensors, got {previous.shape}, {current.shape}"
            )
        leading = previous.shape[:-3]
        inputs = torch.cat((previous, current, current - previous), dim=-3)
        inputs = inputs.reshape(-1, 9, inputs.shape[-2], inputs.shape[-1])
        if inputs.shape[-2:] != (64, 64):
            inputs = F.interpolate(inputs, (64, 64), mode="bilinear", align_corners=False)
        features = self.image_projection(self.image_encoder(inputs).flatten(1))
        return features.reshape(*leading, self.feature_dim)

    def forward(
        self,
        previous_primary: Tensor,
        current_primary: Tensor,
        previous_wrist: Tensor,
        current_wrist: Tensor,
        previous_proprio: Tensor,
        current_proprio: Tensor,
    ) -> Tensor:
        """Return a fused visual/proprio difference feature."""
        primary = self._camera(previous_primary, current_primary)
        wrist = self._camera(previous_wrist, current_wrist)
        visual = torch.stack((primary, wrist), dim=0).mean(dim=0)
        proprio = torch.cat(
            (
                previous_proprio,
                current_proprio,
                current_proprio - previous_proprio,
            ),
            dim=-1,
        )
        return self.output_norm(visual + self.proprio_projection(proprio))


class ObservationConditionedPredictor(nn.Module):
    """Predict action or latent deltas from fresh observations and current action."""

    def __init__(
        self,
        output_dim: int,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        proprio_dim: int = 8,
        predict_gripper_switch: bool = False,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.predict_gripper_switch = predict_gripper_switch
        self.encoder = SharedVisualDifferenceEncoder(feature_dim, proprio_dim)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + 7 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, output_dim)
        self.switch_head = nn.Linear(hidden_dim, 1) if predict_gripper_switch else None

    def encode(
        self,
        previous_primary: Tensor,
        current_primary: Tensor,
        previous_wrist: Tensor,
        current_wrist: Tensor,
        previous_proprio: Tensor,
        current_proprio: Tensor,
    ) -> Tensor:
        """Expose the shared difference encoder for runtime profiling."""
        return self.encoder(
            previous_primary,
            current_primary,
            previous_wrist,
            current_wrist,
            previous_proprio,
            current_proprio,
        )

    def predict_from_feature(
        self, feature: Tensor, current_action: Tensor, dt: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Predict from a precomputed observation feature."""
        if dt.ndim == 1:
            dt = dt.unsqueeze(-1)
        hidden = self.trunk(torch.cat((feature, current_action, dt), dim=-1))
        switch = self.switch_head(hidden).squeeze(-1) if self.switch_head is not None else None
        return self.delta_head(hidden), switch

    def forward(
        self,
        previous_primary: Tensor,
        current_primary: Tensor,
        previous_wrist: Tensor,
        current_wrist: Tensor,
        previous_proprio: Tensor,
        current_proprio: Tensor,
        current_action: Tensor,
        dt: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Predict a delta and optional gripper-switch logit."""
        feature = self.encode(
            previous_primary,
            current_primary,
            previous_wrist,
            current_wrist,
            previous_proprio,
            current_proprio,
        )
        return self.predict_from_feature(feature, current_action, dt)


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
