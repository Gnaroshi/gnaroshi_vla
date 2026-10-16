"""Retain within-cell feature geometry at the trained visual predictor input."""
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('mean_pool_control', 'spatial_moments', 'subpatch_projection')


class LocalGeometryPool(nn.Module):
    """Map [B,C,24,24] to [B,C,6,6], with exact average-pool initialization.

    Moments encode horizontal/vertical feature placement inside each 4x4 cell.
    Subpatch projection retains all sixteen mean-centered feature vectors.
    Both corrections are computed independently for each camera and cell.
    """
    def __init__(self, channels, mode, block_size=4):
        super().__init__()
        if mode not in ARMS[1:]:
            raise ValueError(mode)
        self.mode, self.channels, self.block_size = mode, channels, block_size
        width = 2 if mode == 'spatial_moments' else block_size ** 2
        self.projection = nn.Conv2d(channels * width, channels, 1, bias=False)
        nn.init.zeros_(self.projection.weight)
        coordinate = torch.linspace(-1., 1., block_size)
        self.register_buffer('horizontal', coordinate.repeat(block_size).view(1, 1, -1, 1, 1))
        self.register_buffer('vertical', coordinate.repeat_interleave(block_size).view(1, 1, -1, 1, 1))

    def forward(self, x):
        k = self.block_size
        mean = F.avg_pool2d(x, k, k)
        local = F.pixel_unshuffle(x, k).reshape(x.shape[0], self.channels, k*k,
                                                mean.shape[2], mean.shape[3])
        centered = local - mean.unsqueeze(2)
        if self.mode == 'spatial_moments':
            geometry = torch.cat(((centered * self.horizontal).mean(2),
                                  (centered * self.vertical).mean(2)), dim=1)
        else:
            geometry = centered.flatten(1, 2)
        return mean + self.projection(geometry)


def configure_model(model, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    encoder = model.visual.predictor.encoder
    if not isinstance(encoder[-1], nn.AvgPool2d):
        raise RuntimeError('Expected the original mean-pooling visual predictor')
    if arm != ARMS[0]:
        weight = encoder[3].weight
        encoder[-1] = LocalGeometryPool(64, arm).to(device=weight.device, dtype=weight.dtype)
    model.requires_grad_(False)
    model.visual.predictor.requires_grad_(True)
    return model
