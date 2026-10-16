"""Matched capacity and attention-target controls for the local geometry study."""
from torch import nn
from torch.nn import functional as F

from .local_geometry_input import LocalGeometryPool

ARMS = ('capacity_only', 'mean_original_target', 'subpatch_original_target')


class MeanCapacityPool(nn.Module):
    """Keep spatial averaging, adding the same 65,536 weights as the geometry arm."""
    def __init__(self, channels=64, block_size=4):
        super().__init__()
        self.block_size = block_size
        self.correction = nn.Sequential(nn.Conv2d(channels, 8*channels, 1, bias=False),
                                        nn.GELU(), nn.Conv2d(8*channels, channels, 1, bias=False))
        nn.init.zeros_(self.correction[-1].weight)

    def forward(self, x):
        pooled = F.avg_pool2d(x, self.block_size, self.block_size)
        return pooled + self.correction(pooled)


def configure_model(model, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    encoder = model.visual.predictor.encoder
    if not isinstance(encoder[-1], nn.AvgPool2d):
        raise RuntimeError('Expected the fixed parent mean-pooling architecture')
    model.arm = 'student_query_control' if arm == 'capacity_only' else 'original_query_target'
    if arm != 'mean_original_target':
        pool = MeanCapacityPool() if arm == 'capacity_only' else LocalGeometryPool(64, 'subpatch_projection')
        weight = encoder[3].weight
        encoder[-1] = pool.to(device=weight.device, dtype=weight.dtype)
    model.requires_grad_(False)
    model.visual.predictor.requires_grad_(True)
    return model
