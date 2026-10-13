"""Predict a midpoint Condition, then reuse a frozen short-interval updater."""
from .visual_midpoint import VisualMidpointCondition
from .visual_reanchor import VisualReanchorCondition


class VisualShortReset(VisualReanchorCondition):
    def __init__(self, trained_short, anchor_reset=False):
        super().__init__(trained_short, 'midpoint', False)
        self.anchor_reset = bool(anchor_reset)

    def predict(self, context, age, images, proprio):
        if age != context.age + 1 or not 1 <= age < context.interval:
            raise ValueError('Queries must be consecutive')
        if age < 4:
            return self.early.predict(context, age, images, proprio)
        if age == 4:
            fn = VisualReanchorCondition.predict if self.anchor_reset else VisualMidpointCondition.predict
            output, diagnostics = fn(self, context, age, images, proprio)
            # Retain the gradient from future actions through the frozen updater.
            context.short_context = self.early.prepare(output, images, proprio,
                context.valid, context.groups, interval=4)
            return output, diagnostics
        output, diagnostics = self.early.predict(context.short_context, age-4, images, proprio)
        context.previous = output
        context.previous_images, context.previous_proprio, context.age = images, proprio, age
        return output, diagnostics
