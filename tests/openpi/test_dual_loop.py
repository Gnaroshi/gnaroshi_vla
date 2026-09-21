import importlib.util
import sys
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from methods.latentloop.modules.flow_hidden_update import FlowHiddenConfig, FlowHiddenUpdater
from architectures.openpi.adapters.latentloop.dual_loop import generate


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=10)
        self.action_out_proj = torch.nn.Linear(8, 4)
        self.calls = 0
        self.times = []

    def denoise_step(self, state, mask, cache, x, t):
        self.calls += 1
        self.times.append(t[0].item())
        hidden = torch.cat((x, x), -1) + t[:, None, None]
        return self.action_out_proj(hidden)


def inputs():
    model = Model().requires_grad_(False)
    prefix = SimpleNamespace(values=(torch.zeros(1, 1, 3, 2),), pad_mask=torch.ones(1, 3, dtype=torch.bool))
    hook = SimpleNamespace(rebuild_cache=lambda p: None)
    updater = FlowHiddenUpdater(FlowHiddenConfig(8, 4, 2, 4, rank=8))
    return model, hook, prefix, torch.zeros(1, 4), torch.randn(1, 10, 4), updater


def test_exact_grid_and_output():
    model, hook, prefix, state, noise, updater = inputs()
    result = generate(model, hook, prefix, state, noise, updater, n_g=10)
    assert model.calls == 10
    x = noise.clone()
    t, dt = torch.tensor(1.), torch.tensor(-0.1)
    for _ in range(10):
        x = x + dt * model.denoise_step(state, None, None, x, t[None])
        t = t + dt
    assert torch.equal(x, result.actions)
    assert result.metrics["generation_updater_calls"] == 0


def test_ng3_counters_and_gradients():
    model, hook, prefix, state, noise, updater = inputs()
    result = generate(model, hook, prefix, state, noise, updater, n_g=3, train=True)
    assert result.metrics["action_expert_calls"] == 3
    assert result.metrics["generation_updater_calls"] == 7
    assert result.metrics["oracle_calls"] == 7
    assert result.metrics["flow_iterations"] == 10
    result.loss.backward()
    assert updater.core.hidden_up.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_inference_has_no_teacher_oracle():
    model, hook, prefix, state, noise, updater = inputs()
    with torch.no_grad():
        result = generate(model, hook, prefix, state, noise, updater, n_g=3)
    assert model.calls == 3
    assert result.loss is None
    assert result.metrics["oracle_calls"] == 0
    assert len(model.action_out_proj._forward_pre_hooks) == 0


def test_zero_residual_initialization():
    updater = FlowHiddenUpdater(FlowHiddenConfig(8, 4, 2, 4, rank=8))
    hidden = torch.randn(1, 10, 8)
    x = torch.randn(1, 10, 4)
    out = updater(hidden, x, x + .1, torch.zeros(1, 2), torch.zeros(1, 4),
                  torch.tensor(1.), torch.tensor(.9), 1)
    assert torch.equal(hidden, out)


def test_scheduler_and_report_recovery(tmp_path):
    tools = Path(__file__).resolve().parents[2] / "tools/openpi"
    sys.path.insert(0, str(tools))
    from train_pi05_generation import learning_rate
    from run_pi05_dual_loop import aggregate, ROWS
    assert learning_rate(199, 10000) == 1e-4
    assert abs(learning_rate(9999, 10000) - 1e-5) < 1e-12
    for row in ROWS:
        target = tmp_path / "eval" / row
        target.mkdir(parents=True)
        # Synthetic fixture only, never written under an experiment results directory.
        (target / "summary.json").write_text(json.dumps({
            "complete": True, "episodes": 500, "successes": 250,
            "success_rate": .5, "policy_ms_per_actual_action": 10.0}))
    aggregate(tmp_path)
    assert (tmp_path / "results_for_chatgpt.zip").is_file()
    assert len(json.loads((tmp_path / "combined_summary.json").read_text())) == 5
