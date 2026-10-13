"""Keep exact midpoint vision tokens separate from recurrent Condition estimates."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .native_simvla_v0 import NativeV0ObservationPair
from .visual_midpoint import VisualMidpointCondition, VisualReference


class SpatialPairEncoder(nn.Module):
    def __init__(self, rank=64):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(9,32,5,stride=2,padding=2),nn.GroupNorm(4,32),nn.GELU(),
            nn.Conv2d(32,rank,3,stride=2,padding=1),nn.GroupNorm(8,rank),nn.GELU(),nn.AdaptiveAvgPool2d((6,6)))

    @staticmethod
    def rgb(images):
        if images.ndim!=5 or images.shape[1]!=2 or images.shape[-1]!=3:
            raise ValueError('Expected ordered front/wrist RGB [B,2,H,W,3]')
        if images.dtype not in (torch.uint8,torch.float32):
            raise ValueError('Expected cache uint8 or live normalized float32')
        x=images.permute(0,1,4,2,3).flatten(0,1).float()
        if images.dtype==torch.uint8:x=x/255.
        return F.interpolate(x,size=(96,96),mode='bilinear',align_corners=False)

    def forward(self,reference,current):
        a,b=self.rgb(reference),self.rgb(current)
        features=self.net(torch.cat((a,b,b-a),dim=1))
        return features.flatten(2).transpose(1,2).reshape(reference.shape[0],72,-1)


class MemoryReference(VisualReference):
    def __init__(self,condition_dim,rank,spatial):
        super().__init__(condition_dim,rank)
        self.spatial=SpatialPairEncoder(rank) if spatial else None
        if spatial:
            self.spatial_key=nn.Linear(rank,rank)
            self.spatial_value=nn.Linear(rank,rank)
            for module in (self.spatial_key,self.spatial_value):
                nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)

    def forward(self,query,features,valid,reference_images,current_images):
        if self.spatial is None:
            return super().forward(query,features,valid)
        if features.shape[1]!=72:raise ValueError('Expected72 original two-camera vision tokens')
        positions=self.position(torch.arange(72,device=features.device))[None]
        source=F.layer_norm(features,(features.shape[-1],))
        local=self.spatial(reference_images,current_images)
        key=self.key(source)+positions+self.spatial_key(local)
        value=self.value(source)+positions+self.spatial_value(local)
        weights=(self.query(query)@key.transpose(-1,-2)/math.sqrt(query.shape[-1])).softmax(-1)
        return self.output(weights@value)*valid.unsqueeze(-1)


class VisualMemoryCondition(VisualMidpointCondition):
    def __init__(self,trained_short,spatial=False):
        super().__init__(trained_short,'late_every')
        original=self.visual
        self.visual=MemoryReference(trained_short.condition_dim,trained_short.rank_dim,spatial)
        missing,unexpected=self.visual.load_state_dict(original.state_dict(),strict=False)
        if unexpected or any(not name.startswith('spatial') for name in missing):
            raise RuntimeError('Visual initialization mismatch')

    def uses_vision(self,age):
        return age==4

    def predict(self,context,age,images,proprio):
        if age<4:return self.early.predict(context,age,images,proprio)
        if age!=context.age+1 or not 1<=age<context.interval:
            raise ValueError('Queries must be consecutive')
        if age==4:
            if self.feature_provider is None:raise RuntimeError('Missing current vision provider')
            context.visual_memory=self.feature_provider(age,images)
            context.visual_images=images
        code=self.late.delta_encoder(NativeV0ObservationPair(
            context.previous_images,images,context.previous_proprio,proprio))
        updater=self.late.condition_updater
        positions=updater.token_embedding(torch.arange(context.previous.shape[1],device=images.device))[None]
        query=updater.down(updater.norm(context.previous))+positions+updater.group_embedding(context.groups)
        query=query+updater.delta_projection(code)[:,None]
        feature=self.visual(query,context.visual_memory,context.valid,context.visual_images,images)
        prediction=updater(context.previous,code,valid_mask=context.valid,group_ids=context.groups,
            age=age,token_feature=feature)
        candidate=context.previous+prediction.residual
        output=context.previous+prediction.gate*(candidate-context.previous)
        diagnostics=dict(base=output,carried=output,addition=output-context.previous,
            gate=prediction.gate,candidate=candidate)
        context.previous=output
        context.previous_images,context.previous_proprio,context.age=images,proprio,age
        return output,diagnostics
