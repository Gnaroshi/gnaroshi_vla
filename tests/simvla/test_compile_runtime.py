import torch
from torch import nn

from tools.simvla.compile_runtime import ActionStep, Compiler, compile_bridge_predict_next


class Bridge(nn.Module):
    def forward(self, x):
        return x.sin() * 2

    def predict_next(self, x):
        return x + self(x)


def test_bound_entry_is_really_compiled():
    torch._dynamo.reset()
    model = Bridge()
    x = torch.ones(3)
    expected = model.predict_next(x)
    compiler = Compiler(True, backend=lambda graph, inputs: graph.forward)
    compile_bridge_predict_next(model, compiler)
    torch.testing.assert_close(model.predict_next(x), expected, rtol=0, atol=0)
    assert compiler.graph_count() > 0


def test_original_module_wrapper_reproduces_bypass():
    torch._dynamo.reset()
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    original = Bridge()
    compiled = torch.compile(original, backend=backend)
    compiled.predict_next(torch.ones(3))
    assert not graphs
    compiled(torch.ones(3))
    assert graphs


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.action_decoder = nn.Linear(8, 2)

    def forward(self, vlm_features, action_with_noise, proprio, t):
        return self.action_decoder(self.hidden(vlm_features))


def test_action_hidden_does_not_mutate_original():
    original = Transformer().eval()
    decoder = original.action_decoder
    args = (torch.randn(2, 4), None, None, None)
    expected = original(*args)
    step = ActionStep(original)
    hidden, velocity = step(*args)
    assert original.action_decoder is decoder
    assert step.hidden_model.hidden is original.hidden
    torch.testing.assert_close(hidden, original.hidden(args[0]), rtol=0, atol=0)
    torch.testing.assert_close(velocity, expected, rtol=0, atol=0)


def test_disabled_compiler_is_identity():
    model = Bridge()
    method = model.predict_next
    compiler = Compiler(False)
    assert compiler.wrap("predict", method) is method
    assert compiler.graph_count() == 0
