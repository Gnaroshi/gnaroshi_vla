"""Learn the age4 decoder Condition while preserving the frozen recurrent path."""
import copy
from types import MethodType

import torch
from torch import nn

from .attention_query_contract import make_query
from .native_simvla_v0 import NativeV0ObservationPair
from .visual_midpoint import VisualReference

ARMS = ('condition_target', 'action_target', 'joint_target')


def install(model):
    if hasattr(model, 'current_head'):
        raise ValueError('Current-action head already installed')
    model.requires_grad_(False)
    reader = VisualReference(model.visual.key.in_features, model.visual.key.out_features,
                             model.visual.position.num_embeddings)
    for name in ('key', 'value', 'query', 'position', 'output'):
        getattr(reader, name).load_state_dict(getattr(model.visual, name).state_dict())
    model.current_head = nn.ModuleDict(dict(updater=copy.deepcopy(model.late.condition_updater), reader=reader))
    model.current_head.requires_grad_(True)
    original_prepare, original_predict = model.prepare, model.predict

    def prepare(self, anchor, images, proprio, valid, groups, interval=8):
        if interval == 1:
            return None
        return original_prepare(anchor, images, proprio, valid, groups, interval)

    def predict(self, context, age, images, proprio):
        previous = context.previous
        previous_images, previous_proprio = context.previous_images, context.previous_proprio
        with torch.no_grad():
            carried, diagnostics = original_predict(context, age, images, proprio)
        if age != 4:
            return carried, diagnostics
        with torch.no_grad():
            code = self.late.delta_encoder(NativeV0ObservationPair(
                previous_images, images, previous_proprio, proprio))
        updater = self.current_head['updater']
        query = make_query(updater, previous.detach(), code, context.groups)
        feature = self.current_head['reader'](query, context.visual_memory.detach(), context.valid)
        prediction = updater(previous.detach(), code, valid_mask=context.valid,
            group_ids=context.groups, age=age, token_feature=feature)
        candidate = previous.detach() + prediction.residual
        output = previous.detach() + prediction.gate * (candidate - previous.detach())
        # context.previous remains the original frozen recurrent prediction.
        return output, dict(base=output, carried=carried, addition=output-previous.detach(),
                            gate=prediction.gate, candidate=candidate)

    model.prepare, model.predict = MethodType(prepare, model), MethodType(predict, model)
    return model


def parameter_counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        current_updater=sum(p.numel() for p in model.current_head['updater'].parameters()),
        current_reader=sum(p.numel() for p in model.current_head['reader'].parameters()))


def assert_frozen(model, original):
    for name, p in model.named_parameters():
        if not name.startswith('current_head.') and (p.requires_grad or p.grad is not None):
            raise RuntimeError('Frozen recurrent parameter received a gradient: ' + name)
    if any(p.requires_grad or p.grad is not None for p in original.parameters()):
        raise RuntimeError('Frozen original model received a gradient')


def frozen_hash(model):
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    return state_hash(nn.ModuleDict({k: v for k, v in model._modules.items() if k != 'current_head'}))
