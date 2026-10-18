"""Intervene on the current decoder input and next recurrent input separately."""
from types import MethodType

ARMS = ('current_action', 'next_prediction', 'both')


def route(predicted, original, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    if predicted.shape != original.shape:
        raise ValueError('Original and predicted Condition shapes differ')
    action = original if arm in ('current_action', 'both') else predicted
    carried = original if arm in ('next_prediction', 'both') else predicted
    return action, carried


def install(model, arm):
    if arm not in ARMS or hasattr(model, 'routing_arm'):
        raise ValueError('Invalid or duplicate Condition routing installation')
    original_predict = model.predict
    original_prepare = model.prepare
    model.routing_arm = arm
    model.routing_reference = None

    def prepare(self, anchor, images, proprio, valid_mask, group_ids, interval=8):
        if type(interval) is int and interval == 1:
            return None
        return original_prepare(anchor, images, proprio, valid_mask, group_ids, interval)

    def predict(self, context, age, images, proprio):
        predicted, diagnostics = original_predict(context, age, images, proprio)
        if age == 4:
            if self.routing_reference is None:
                raise RuntimeError('Missing current original Condition at age4')
            output, carried = route(predicted, self.routing_reference, arm)
            context.previous = carried
            diagnostics = dict(diagnostics, carried=carried, base=output)
            self.routing_reference = None
            return output, diagnostics
        return predicted, diagnostics

    model.predict = MethodType(predict, model)
    model.prepare = MethodType(prepare, model)
    return model
