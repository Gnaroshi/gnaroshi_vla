"""Extend a trained short-interval updater without changing its early outputs."""
import copy

from torch import nn


class AgeSpecializedCondition(nn.Module):
    def __init__(self, trained_short, protected):
        super().__init__()
        if type(protected) is not bool:
            raise ValueError('protected must be bool')
        self.protected = protected
        self.early = copy.deepcopy(trained_short)
        self.late = copy.deepcopy(trained_short) if protected else None
        self.early.requires_grad_(not protected)
        if self.late is not None:
            self.late.requires_grad_(True)

    def prepare(self, *args, **kwargs):
        return self.early.prepare(*args, **kwargs)

    def expert(self, age):
        if type(age) is not int or not 1 <= age <= 7:
            raise ValueError('Expected a physical age in 1..7')
        return self.late if self.protected and age >= 4 else self.early

    def predict(self, context, age, images, proprio):
        return self.expert(age).predict(context, age, images, proprio)

    def experts(self):
        return [('early', self.early)] + ([('late', self.late)] if self.protected else [])
