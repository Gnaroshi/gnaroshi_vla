"""Share frozen parameters while truncating a separate text-model module tree."""
import copy

from torch import nn


def module_view(module):
    result = copy.copy(module)
    result._modules = module._modules.copy()
    result._parameters = module._parameters.copy()
    result._buffers = module._buffers.copy()
    result._forward_hooks = module._forward_hooks.copy()
    result._forward_pre_hooks = module._forward_pre_hooks.copy()
    return result


class FrozenPrefix:
    def __init__(self, model, depth):
        text = model.vlm.model.text_model
        if not 1 <= depth <= len(text.layers):
            raise ValueError('Prefix depth exceeds the original transformer')
        self.depth, self.total_layers = depth, len(text.layers)
        self.original_text = text
        self.model = module_view(model)
        self.model.vlm = module_view(model.vlm)
        self.model.vlm.model = module_view(model.vlm.model)
        truncated = module_view(text)
        truncated.layers = nn.ModuleList(list(text.layers[:depth]))
        truncated.config = copy.deepcopy(text.config)
        truncated.config.num_hidden_layers = depth
        truncated.config.use_cache = False
        self.model.vlm.model.text_model = truncated
        self.captured = None
        self.capture_enabled = False
        self.handle = text.layers[depth - 1].register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        if self.capture_enabled:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self.captured = hidden.detach()

    def start_capture(self):
        self.captured = None
        self.capture_enabled = True

    def finish_capture(self):
        self.capture_enabled = False
        if self.captured is None:
            raise RuntimeError('Full refresh did not execute the prefix')
        result = self.original_text.norm(self.captured).float()
        self.captured = None
        return result

    def encode(self, batch):
        return self.model.forward_vlm_efficient(batch['image_input'], batch['image_mask'],
                                                batch['input_ids'])['vlm_features'].float()

    def close(self):
        self.handle.remove()
