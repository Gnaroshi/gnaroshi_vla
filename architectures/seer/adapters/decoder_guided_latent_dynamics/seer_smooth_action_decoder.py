"""Differentiable wrapper around Seer's existing shared action head."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SeerSmoothActionDecoder(nn.Module):
    """Expose first-token or all-token continuous Seer action outputs."""

    def __init__(self, seer_model: nn.Module, output_map: str = "all") -> None:
        super().__init__()
        model = seer_model.module if hasattr(seer_model, "module") else seer_model
        if output_map not in {"first", "all"}:
            raise ValueError(f"output_map must be 'first' or 'all', got {output_map}")
        self.action_decoder = model.action_decoder
        self.arm_action_decoder = model.arm_action_decoder
        self.gripper_action_decoder = model.gripper_action_decoder
        self.action_pred_steps = int(model.action_pred_steps)
        self.hidden_dim = int(model.hidden_dim)
        self.output_map = output_map

    def token_outputs(self, latent: Tensor) -> Tensor:
        """Return smooth per-token outputs ``[..., P, 7]`` without thresholding."""
        if latent.shape[-2:] != (self.action_pred_steps, self.hidden_dim):
            raise ValueError(
                "Expected latent [..., action_pred_steps, hidden_dim] = "
                f"[..., {self.action_pred_steps}, {self.hidden_dim}], got {latent.shape}"
            )
        feature = self.action_decoder(latent)
        arm = self.arm_action_decoder(feature)
        gripper_probability = self.gripper_action_decoder(feature)
        return torch.cat((arm, gripper_probability), dim=-1)

    def forward(self, latent: Tensor) -> Tensor:
        """Return ``[B,7]`` for first-token or ``[B,P*7]`` for all-token mode."""
        outputs = self.token_outputs(latent)
        if outputs.ndim != 3:
            raise ValueError(f"Expected batched latent [B,P,D], got {latent.shape}")
        if self.output_map == "first":
            return outputs[:, 0]
        return outputs.reshape(outputs.shape[0], -1)


class StandaloneSeerSmoothActionDecoder(nn.Module):
    """Small exact copy of a checkpoint's Seer action head for offline analysis."""

    def __init__(self, hidden_dim: int = 384, action_pred_steps: int = 3) -> None:
        super().__init__()
        intermediate = hidden_dim // 2
        self.hidden_dim = hidden_dim
        self.action_pred_steps = action_pred_steps
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, intermediate),
            nn.ReLU(),
        )
        self.arm_action_decoder = nn.Sequential(nn.Linear(intermediate, 6), nn.Tanh())
        self.gripper_action_decoder = nn.Sequential(
            nn.Linear(intermediate, 1), nn.Sigmoid()
        )

    def forward(self, latent: Tensor, output_map: str = "first") -> Tensor:
        """Decode ``[B,P,D]`` into first-token 7D or flattened all-token output."""
        feature = self.action_decoder(latent)
        output = torch.cat(
            (self.arm_action_decoder(feature), self.gripper_action_decoder(feature)),
            dim=-1,
        )
        if output_map == "first":
            return output[:, 0]
        if output_map == "all":
            return output.reshape(output.shape[0], -1)
        raise ValueError(f"Unsupported output_map: {output_map}")

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, hidden_dim: int = 384, action_pred_steps: int = 3
    ) -> "StandaloneSeerSmoothActionDecoder":
        """Load only the exact shared action-head parameters from a Seer checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        prefixes = (
            "action_decoder.",
            "arm_action_decoder.",
            "gripper_action_decoder.",
        )
        selected = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            if key.startswith(prefixes):
                selected[key] = value
        module = cls(hidden_dim, action_pred_steps)
        missing, unexpected = module.load_state_dict(selected, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Action-head checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        module.requires_grad_(False)
        module.eval()
        return module
