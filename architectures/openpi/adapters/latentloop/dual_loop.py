"""External pi0.5 condition and generation adapters; upstream stays unchanged."""

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from .prefix_kv_hook import PrefixKVHook


ROWS = {
    "baseline": (1, 10),
    "condition_k2": (2, 10),
    "generation_ng3": (1, 3),
    "dual_k2_ng3": (2, 3),
    "naive_nfe3": (1, 3),
}
ANCHORS = {10: tuple(range(10)), 3: (0, 4, 7)}


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def condition_summary(prefix):
    values = prefix.values[-1].float()
    mask = prefix.pad_mask[:, None, :, None].float()
    return ((values * mask).sum(2) / mask.sum(2).clamp_min(1)).flatten(1)


def exact_hidden(model, robot_state, prefix_mask, cache, x, t):
    captured = []

    def capture(_module, inputs):
        captured.append(inputs[0])

    handle = model.action_out_proj.register_forward_pre_hook(capture)
    try:
        velocity = model.denoise_step(robot_state, prefix_mask, cache, x, t.expand(x.shape[0]))
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("expected one frozen action_out_proj invocation per exact expert evaluation")
    return captured[0], velocity


@dataclass
class GenerationResult:
    actions: torch.Tensor
    metrics: dict
    loss: torch.Tensor | None


def generate(model, hook, prefix, robot_state, noise, updater, *, n_g=3, train=False):
    if n_g not in ANCHORS or (n_g != 10 and updater is None):
        raise ValueError("generation supports N_G=3 or exact N_G=10")
    if model.config.action_horizon != 10:
        raise ValueError("this checkpoint contract requires H=10")
    device = noise.device
    sync(device)
    started = time.perf_counter()
    cache = hook.rebuild_cache(prefix)
    sync(device)
    metrics = {"cache_rebuild_ms": (time.perf_counter() - started) * 1000,
               "action_expert_ms": 0.0, "generation_updater_ms": 0.0,
               "action_decoder_ms": 0.0, "integration_ms": 0.0,
               "action_expert_calls": 0, "generation_updater_calls": 0,
               "flow_iterations": 10, "oracle_calls": 0}
    context = condition_summary(prefix)
    x = noise
    dt = torch.tensor(-0.1, dtype=torch.float32, device=device)
    t = torch.tensor(1.0, dtype=torch.float32, device=device)
    previous_hidden = previous_x = previous_t = None
    last_anchor = 0
    losses, hidden_losses, velocity_losses = [], [], []
    for step in range(10):
        sync(device)
        started = time.perf_counter()
        if step in ANCHORS[n_g]:
            with torch.no_grad():
                hidden, velocity = exact_hidden(model, robot_state, prefix.pad_mask, cache, x.detach(), t)
            last_anchor = step
            sync(device)
            metrics["action_expert_ms"] += (time.perf_counter() - started) * 1000
            metrics["action_expert_calls"] += 1
        else:
            hidden = updater(previous_hidden, previous_x, x, context, robot_state,
                             previous_t, t, step - last_anchor)
            sync(device)
            metrics["generation_updater_ms"] += (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            velocity = model.action_out_proj(hidden)
            sync(device)
            metrics["action_decoder_ms"] += (time.perf_counter() - started) * 1000
            metrics["generation_updater_calls"] += 1
            if train:
                # Local oracle at the student's x_t, not a different teacher trajectory.
                with torch.no_grad():
                    target_hidden, target_velocity = exact_hidden(
                        model, robot_state, prefix.pad_mask, cache, x.detach(), t)
                h_loss = F.mse_loss(F.layer_norm(hidden.float(), (hidden.shape[-1],)),
                                    F.layer_norm(target_hidden.float(), (hidden.shape[-1],)))
                v_loss = F.mse_loss(velocity, target_velocity) / target_velocity.square().mean().clamp_min(1e-4)
                losses.append(h_loss + v_loss)
                hidden_losses.append(h_loss.detach())
                velocity_losses.append(v_loss.detach())
                metrics["oracle_calls"] += 1
        started = time.perf_counter()
        previous_hidden, previous_x, previous_t = hidden, x, t
        x = x + dt * velocity
        t = t + dt
        sync(device)
        metrics["integration_ms"] += (time.perf_counter() - started) * 1000
    loss = torch.stack(losses).mean() if losses else None
    if losses:
        metrics["hidden_normalized_mse"] = float(torch.stack(hidden_losses).mean())
        metrics["velocity_relative_mse"] = float(torch.stack(velocity_losses).mean())
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite generation output")
    return GenerationResult(x, metrics, loss)


class DualLoopPolicy:
    def __init__(self, model, condition_updater, generation_updater, row):
        self.model, self.condition_updater, self.generation_updater = model, condition_updater, generation_updater
        self.row = row
        self.k_c, self.n_g = ROWS[row]
        self.hook = PrefixKVHook(model)
        self.reset()

    def reset(self):
        self.query_index = 0
        self.prefix = None
        self.previous_embeddings = None

    @torch.no_grad()
    def query(self, observation, noise, executed_actions=None):
        if self.row in {"baseline", "naive_nfe3"}:
            steps = 10 if self.row == "baseline" else 3
            sampler = getattr(self.model.sample_actions, "_torchdynamo_orig_callable", self.model.sample_actions)
            sync(noise.device)
            start = time.perf_counter()
            actions = sampler(noise.device, observation, noise=noise, num_steps=steps)
            sync(noise.device)
            metrics = {"original_sampler_ms": (time.perf_counter() - start) * 1000,
                       "full_prefix_calls": 1, "condition_updater_calls": 0,
                       "action_expert_calls": steps, "generation_updater_calls": 0, "flow_iterations": steps}
        else:
            full = self.prefix is None or self.query_index % self.k_c == 0
            if full:
                extraction = self.hook.extract(observation)
                self.prefix = extraction.state
                robot_state = extraction.robot_state
                metrics = {"prefix_embedding_ms": extraction.prefix_embedding_ms,
                           "prefix_transformer_ms": extraction.full_prefix_ms,
                           "condition_updater_ms": 0.0}
            else:
                current, robot_state, embedding_ms = self.hook.embed(observation)
                if executed_actions is None or tuple(executed_actions.shape[1:]) != (5, 7):
                    raise ValueError("condition update requires the previous five executed physical actions")
                sync(noise.device)
                start = time.perf_counter()
                update = self.condition_updater(
                    self.prefix, current, self.previous_embeddings, executed_actions, robot_state,
                    delta_q=1, delta_a=5, full_refresh_age=self.query_index % self.k_c,
                    executed_action_lengths=torch.full((noise.shape[0],), 5, device=noise.device, dtype=torch.long))
                self.prefix = update.state
                sync(noise.device)
                metrics = {"prefix_embedding_ms": embedding_ms, "prefix_transformer_ms": 0.0,
                           "condition_updater_ms": (time.perf_counter() - start) * 1000}
            self.previous_embeddings = self.prefix.embeddings
            result = generate(self.model, self.hook, self.prefix, robot_state, noise,
                              self.generation_updater, n_g=self.n_g)
            actions = result.actions
            metrics.update(result.metrics)
            metrics.update(full_prefix_calls=int(full), condition_updater_calls=int(not full))
        metrics.update(query_index=self.query_index, k_c=self.k_c, n_g=self.n_g, row=self.row)
        self.query_index += 1
        return actions, metrics
