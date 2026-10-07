"""Three matched condition generators and a differentiable ridge write.

Only prepare() can consume an original condition. predict() receives current
observations and immutable refresh state, never a future teacher or prediction.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("fixed", "anchor_input", "ridge")


def ridge_write(x, residual, valid, regularization):
    """Minimize masked mean squared error plus lambda * ||W||_F^2."""
    if regularization <= 0 or not math.isfinite(regularization):
        raise ValueError("Ridge regularization must be finite and positive")
    x, residual = x.float(), residual.float()
    weight = valid.to(x.dtype).unsqueeze(-1)
    count = weight.sum(1, keepdim=True).clamp_min(1)
    xt = (x * weight).transpose(1, 2)
    gram = xt @ x / count
    rhs = xt @ residual / count
    eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.linalg.solve(gram + regularization * eye, rhs)


class ObservationFeatures(nn.Module):
    def __init__(self, dim=960, width=64, max_tokens=256):
        super().__init__()
        self.width = width
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
        )
        self.visual = nn.Linear(96, width)
        self.position = nn.Parameter(torch.empty(128, width))
        self.tokens = nn.Embedding(max_tokens, width)
        self.groups = nn.Embedding(8, width)
        self.language = nn.Linear(dim, width)
        self.proprio = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.fusion = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                    nn.Linear(width, width), nn.LayerNorm(width))
        nn.init.normal_(self.position, std=.02)

    def forward(self, images, proprio, language, groups):
        if images.dtype != torch.float32 or images.ndim != 5 or images.shape[1] != 2:
            raise ValueError("Expected two float32 RGB views, normalized on CPU")
        if images.shape[-1] == 3:
            images = images.permute(0, 1, 4, 2, 3)
        if images.shape[2] != 3:
            raise ValueError("Expected RGB channels")
        batch, tokens = groups.shape
        images = F.interpolate(images.flatten(0, 1), (64, 64), mode="bilinear", align_corners=False)
        visual = self.cnn(images).flatten(2).transpose(1, 2)
        visual = self.visual(visual).reshape(batch, 128, self.width) + self.position
        query = self.tokens(torch.arange(tokens, device=groups.device))[None] + self.groups(groups)
        query = query + self.language(language).unsqueeze(1) + self.proprio(proprio.float()).unsqueeze(1)
        attention = torch.softmax(query @ visual.transpose(1, 2) / math.sqrt(self.width), dim=-1)
        features = torch.tanh(self.fusion(query + attention @ visual)) / math.sqrt(self.width)
        return torch.cat((features, features.new_ones(batch, tokens, 1)), dim=-1)


@dataclass
class RefreshState:
    anchor: torch.Tensor
    valid: torch.Tensor
    groups: torch.Tensor
    language: torch.Tensor
    correction: torch.Tensor | None
    anchor_code: torch.Tensor | None
    reference_features: torch.Tensor | None = None
    reference_prediction: torch.Tensor | None = None


class RefreshCalibratedCondition(nn.Module):
    def __init__(self, variant, dim=960, width=64, ridge_lambda=.01, anchor_exact=False):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        if ridge_lambda <= 0 or not math.isfinite(ridge_lambda):
            raise ValueError("Invalid ridge regularization")
        self.variant, self.ridge_lambda = variant, ridge_lambda
        self.anchor_exact = bool(anchor_exact)
        self.features = ObservationFeatures(dim, width)
        self.readout = nn.Linear(width + 1, dim, bias=False)
        if variant == "anchor_input":
            self.anchor_encoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU())
            self.anchor_readout = nn.Sequential(nn.Linear(2 * width + 1, width), nn.GELU(),
                                                nn.Linear(width, dim))
            nn.init.zeros_(self.anchor_readout[-1].weight)
            nn.init.zeros_(self.anchor_readout[-1].bias)

    def direct(self, images, proprio, language, groups):
        return self.readout(self.features(images, proprio, language, groups))

    def fit_correction(self, x, anchor, valid):
        return ridge_write(x, anchor.float() - self.readout(x), valid, self.ridge_lambda)

    def prepare(self, anchor, images, proprio, language, valid, groups):
        correction = code = x = None
        if self.variant == "ridge":
            x = self.features(images, proprio, language, groups)
            correction = self.fit_correction(x, anchor, valid)
        elif self.variant == "anchor_input":
            code = self.anchor_encoder(anchor.float())
        state = RefreshState(anchor, valid, groups, language, correction, code)
        if self.anchor_exact:
            state.reference_features = x if x is not None else self.features(images, proprio, language, groups)
            self.update_reference(state)
        return state

    def update_reference(self, state):
        if self.anchor_exact:
            state.reference_prediction = self.mapping(state, state.reference_features)

    def mapping(self, state, x):
        condition = self.readout(x)
        if self.variant == "ridge":
            condition = condition + x @ state.correction
        elif self.variant == "anchor_input":
            condition = condition + self.anchor_readout(torch.cat((x, state.anchor_code), -1))
        return condition

    def predict(self, state, images, proprio):
        x = self.features(images, proprio, state.language, state.groups)
        condition = self.mapping(state, x)
        if self.anchor_exact:
            condition = state.anchor.float() + (condition - state.reference_prediction)
        return torch.where(state.valid.unsqueeze(-1), condition, state.anchor)

    def initialize_common(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected = {k for k in self.state_dict() if k.startswith(("anchor_encoder.", "anchor_readout."))}
        if set(missing) != expected or unexpected:
            raise RuntimeError(f"Incompatible shared initialization: {missing}, {unexpected}")
