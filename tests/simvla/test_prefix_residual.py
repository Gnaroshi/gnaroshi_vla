from types import SimpleNamespace

import pytest
import torch
from torch import nn

from methods.prefix_residual.model import PrefixResidual
from architectures.simvla.adapters.prefix_residual.prefix import FrozenPrefix


def inputs():
    torch.manual_seed(7)
    return torch.randn(2,6,16),torch.randn(2,6,16),torch.randn(2,6,16),torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,0]],dtype=torch.bool)


def test_initial_equals_analytic_control_and_mask():
    model=PrefixResidual(16,8)
    a,p,q,valid=inputs()
    result=model(a,p,q,valid)
    assert torch.equal(result,model(a,p,q,valid,False))
    assert torch.equal(result[~valid],a[~valid])
    torch.testing.assert_close(result[valid],(a+q-p)[valid])


def test_repeated_input_is_identity_even_after_learning():
    model=PrefixResidual(16,8)
    for parameter in model.parameters(): nn.init.normal_(parameter)
    a,p,_,valid=inputs()
    assert torch.equal(model(a,p,p,valid),a)


def test_delta_changes_output_and_has_trainable_gradient():
    model=PrefixResidual(16,8)
    a,p,q,valid=inputs()
    result=model(a,p,q,valid)
    result[valid].square().mean().backward()
    assert model.up.weight.grad.abs().sum()>0
    assert not torch.equal(result,model(a,p,p,valid))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    optimizer.step(); optimizer.zero_grad()
    model(a,p,q,valid)[valid].square().mean().backward()
    assert model.delta_down.weight.grad.abs().sum()>0


class Block(nn.Linear):
    def forward(self,x):
        return x+super().forward(x)


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(num_hidden_layers=6,use_cache=False)
        self.layers=nn.ModuleList(Block(16,16) for _ in range(6))
        self.norm=nn.LayerNorm(16)
    def forward(self,x):
        for layer in self.layers[:self.config.num_hidden_layers]: x=layer(x)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm=nn.Module(); self.vlm.model=nn.Module(); self.vlm.model.text_model=Text()
    def forward_vlm_efficient(self,images,mask,ids):
        return {'vlm_features':self.vlm.model.text_model(images)}


def test_prefix_matches_original_capture_without_suffix_calls_or_mutation():
    original=Model().eval()
    original.requires_grad_(False)
    before={k:v.clone() for k,v in original.state_dict().items()}
    prefix=FrozenPrefix(original,2)
    batch=dict(image_input=torch.randn(1,6,16),image_mask=None,input_ids=None)
    count=[0]*6
    handles=[]
    for i,layer in enumerate(original.vlm.model.text_model.layers):
        def count_call(m,a,index=i): count[index]+=1
        handles.append(layer.register_forward_pre_hook(count_call))
    prefix.start_capture(); full=original.forward_vlm_efficient(batch['image_input'],None,None)['vlm_features']
    expected=prefix.finish_capture(); partial=prefix.encode(batch)
    torch.testing.assert_close(expected,partial,rtol=0,atol=0)
    assert count==[2,2,1,1,1,1]
    assert len(original.vlm.model.text_model.layers)==6
    assert original.vlm.model.text_model.config.num_hidden_layers==6
    assert prefix.model.vlm.model.text_model.layers[0] is original.vlm.model.text_model.layers[0]
    assert all(torch.equal(v,before[k]) for k,v in original.state_dict().items())
    prefix.close()
    for h in handles: h.remove()
    complete=FrozenPrefix(original,6)
    torch.testing.assert_close(complete.encode(batch),full,rtol=0,atol=0)
    complete.close()


def test_invalid_depth():
    for depth in (0,7):
        with pytest.raises(ValueError): FrozenPrefix(Model(),depth)


def test_permutation_of_current_tokens_is_observable():
    a,p,q,valid=inputs()
    model=PrefixResidual(16,8)
    assert not torch.equal(model(a,p,q,valid),model(a,p,q.flip(1),valid))
