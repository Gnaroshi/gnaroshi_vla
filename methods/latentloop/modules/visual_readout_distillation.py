"""Warm retention and age-4-anchored distillation of visual attention readouts.

The caller supplies a completed ``VisualMidpointCondition(..., 'late_every')``.
Checkpoint provenance, original-vision targets, losses and optimization belong
to the training main. Distilled readout uses normalized key/value MSE at ages
5..7; joint readout adds action L1 and 0.05 condition MSE, with zero KV loss at
age 4. Both start from the same parent and zero-head predictor initialization;
main resets the seed and supplies matched batches for their independent runs.
"""

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .native_simvla_v0 import NativeV0ObservationPair
from .visual_memory import SpatialPairEncoder, VisualMemoryCondition
from .visual_midpoint import VisualMidpointCondition, VisualReference


ARMS = ('warm_retain', 'distilled_readout', 'joint_readout')


class ReadoutPredictor(nn.Module):
    """Predict per-token key/value deltas from the fixed age-4 reference.

    Reference key/value are each [B,72,64], projected from age-4 vision tokens.
    Ordered age-4/current RGB [B,2,H,W,3] produce a local [B,72,64] grid.
    Concatenating these and (age-4)/3 gives [B,72,193]; the MLP returns
    [B,72,128], split into deltas to add to the age-4 key and value.
    """

    def __init__(self):
        super().__init__()
        self.spatial = SpatialPairEncoder(rank=64)
        self.mlp = nn.Sequential(nn.Linear(193, 128), nn.GELU(), nn.Linear(128, 128))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, reference_key, reference_value, reference_images, current_images, age):
        local = self.spatial(reference_images, current_images)
        normalized_age = reference_key.new_full((*reference_key.shape[:2], 1), (age - 4) / 3)
        inputs = torch.cat((reference_key, reference_value, local, normalized_age), dim=-1)
        return self.mlp(inputs).chunk(2, dim=-1)


class DistilledVisualReference(nn.Module):
    """Fixed parent attention with a trainable, nonrecurrent readout predictor.

    ``features`` always contains original age-4 vision [B,72,condition_dim].
    ``query`` is the current late updater's [B,T,64] query, incorporating the
    previous predicted condition and the current observation-pair code.
    ``forward`` returns the parent's [B,T,64] attention feature for that updater.
    Predicted/target readouts exclude positions; forward adds the fixed parent
    position to each key and value before the original attention computation.
    """

    def __init__(self, parent_visual):
        super().__init__()
        if not isinstance(parent_visual, VisualReference) or parent_visual.key.out_features != 64:
            raise ValueError('Expected a parent VisualReference with rank 64')
        for name in ('key', 'value', 'query', 'position', 'output'):
            setattr(self, name, copy.deepcopy(getattr(parent_visual, name)))
        self.requires_grad_(False)
        self.predictor = ReadoutPredictor().to(self.key.weight)

    def _project(self, features):
        if features.ndim != 3 or features.shape[1:] != (72, self.key.in_features):
            raise ValueError('Expected 72 original two-camera vision tokens [B,72,condition_dim]')
        source = F.layer_norm(features, (features.shape[-1],))
        return self.key(source), self.value(source)

    def predicted_readout(self, features, reference_images, current_images, age):
        """Return current [B,72,64] key/value estimates, without positions.

        Age 4 bypasses the predictor exactly. Ages 5..7 add predicted deltas
        to the same age-4 projections; earlier predicted readouts are not inputs.
        Age is a Python policy-query age, as in the surrounding predict API.
        """
        if type(age) is not int or not 4 <= age <= 7:
            raise ValueError('Readout age must be a Python int in 4..7')
        key, value = self._project(features)
        if age == 4:
            return key, value
        delta_key, delta_value = self.predictor(key, value, reference_images, current_images, age)
        return key + delta_key, value + delta_value

    def target_readout(self, currentfeatures):
        """Detach fixed parent projections of main-supplied current vision.

        Input: original current-age vision [B,72,condition_dim]. Output: the
        current key/value targets, each [B,72,64], excluding position embeddings.
        This method does not request vision features or change the reference.
        """
        with torch.no_grad():
            key, value = self._project(currentfeatures)
        return key.detach(), value.detach()

    def forward(self, query, features, valid, reference_images, current_images, age):
        key, value = self.predicted_readout(features, reference_images, current_images, age)
        positions = self.position(torch.arange(features.shape[1], device=features.device))[None]
        key, value = key + positions, value + positions
        weights = (self.query(query) @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])).softmax(-1)
        return self.output(weights @ value) * valid.unsqueeze(-1)


class ReadoutDistillationCondition(VisualMemoryCondition):
    """Three arms sharing the completed late-every parent's original weights.

    warm_retain: unchanged nonspatial VisualMemoryCondition, late/visual train.
    distilled_readout: only visual.predictor trains, including its spatial CNN.
    joint_readout: late delta encoder/updater and visual.predictor train.

    Construction copies the supplied parent's full state, never just early.
    Main resets the same seed before constructing distilled_readout and
    joint_readout so both start with identical new predictor weights.
    ``prepare``, ``experts``, ``uses_vision`` and ``feature_provider`` retain
    their existing interfaces, including attach_visual's forward wrapper.
    """

    def __init__(self, trained_late_every, arm):
        if arm not in ARMS:
            raise ValueError(f'Unknown readout arm: {arm}')
        self._check_parent(trained_late_every)
        super().__init__(trained_late_every.early, spatial=False)
        self.arm = arm
        self.visual = self.visual.to(trained_late_every.visual.key.weight)
        if arm != 'warm_retain':
            self.visual = DistilledVisualReference(trained_late_every.visual)
        self.load_parent(trained_late_every)

    @staticmethod
    def _check_parent(parent):
        if not isinstance(parent, VisualMidpointCondition) or parent.arm != 'late_every':
            raise ValueError('Expected the completed late_every VisualMidpointCondition parent')

    def _configure_trainable(self):
        self.requires_grad_(False)
        if self.arm in ('warm_retain', 'joint_readout'):
            self.late.requires_grad_(True)
        if self.arm == 'warm_retain':
            self.visual.requires_grad_(True)
        else:
            self.visual.predictor.requires_grad_(True)

    def load_parent(self, parent):
        """Copy a parent module or bare state dict, allowing only new predictor keys.

        Warm retention uses strict=True. Readout arms require exact names and
        shapes for all original tensors; the returned missing_keys contains
        exactly visual.predictor.* because the parent has no new predictor.
        Predictor values are left untouched when explicitly reloading a parent.
        """
        if isinstance(parent, nn.Module):
            self._check_parent(parent)
            state = parent.state_dict()
        elif isinstance(parent, Mapping):
            state = parent
        else:
            raise TypeError('Expected a late_every parent module or its state dict')
        destination = self.state_dict()
        new_keys = {name for name in destination if name.startswith('visual.predictor.')}
        original_keys = set(destination) - new_keys
        missing, unexpected = original_keys - set(state), set(state) - original_keys
        mismatched = [name for name in original_keys & set(state)
                      if destination[name].shape != state[name].shape]
        if missing or unexpected or mismatched:
            raise RuntimeError(f'Parent state mismatch: missing={sorted(missing)}, '
                               f'unexpected={sorted(unexpected)}, shapes={sorted(mismatched)}')
        result = self.load_state_dict(state, strict=not new_keys)
        if set(result.missing_keys) != new_keys or result.unexpected_keys:
            raise RuntimeError('Only new visual.predictor keys may be missing from the parent')
        self._configure_trainable()
        return result

    def predict(self, context, age, images, proprio):
        if self.arm == 'warm_retain':
            return super().predict(context, age, images, proprio)
        if age < 4:
            return self.early.predict(context, age, images, proprio)
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive')
        if age == 4:
            if self.feature_provider is None:
                raise RuntimeError('Missing current vision provider')
            context.visual_memory = self.feature_provider(age, images)
            context.visual_images = images
        code = self.late.delta_encoder(NativeV0ObservationPair(
            context.previous_images, images, context.previous_proprio, proprio))
        updater = self.late.condition_updater
        positions = updater.token_embedding(
            torch.arange(context.previous.shape[1], device=images.device))[None]
        query = updater.down(updater.norm(context.previous)) + positions + updater.group_embedding(context.groups)
        query = query + updater.delta_projection(code)[:, None]
        feature = self.visual(query, context.visual_memory, context.valid,
                              context.visual_images, images, age)
        prediction = updater(context.previous, code, valid_mask=context.valid,
                             group_ids=context.groups, age=age, token_feature=feature)
        candidate = context.previous + prediction.residual
        output = context.previous + prediction.gate * (candidate - context.previous)
        diagnostics = dict(base=output, carried=output, addition=output-context.previous,
                           gate=prediction.gate, candidate=candidate)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
