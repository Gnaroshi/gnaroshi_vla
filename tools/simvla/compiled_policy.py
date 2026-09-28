"""Use the audited compiled components inside the existing LIBERO policies."""

from __future__ import annotations

from types import MethodType


def base_row(row):
    return "latent_bridge_f2" if row in {"latent_bridge_f2", "latent_bridge_f3", "latent_bridge_f4"} else row


def expected_counts(row, queries):
    kc = int(row[-1]) if row.startswith("latent_bridge_f") else (2 if row in {
        "condition_naive3", "condition_nfe10", "ours_kc2_ng3"} else 1)
    full = (queries + kc - 1) // kc
    generation = row in {"generation_ng3", "ours_kc2_ng3"}
    nfe = 3 if generation or row in {"naive_nfe3", "condition_naive3"} else 10
    return {"num_full_vlm_calls": full, "num_condition_updater_calls": queries - full,
        "num_action_transformer_calls": queries * nfe,
        "num_generation_decoder_only_steps": queries * (7 if generation else 0)}


def attach_policy(replay, config, row, manifest):
    from tools.simvla.compile_audit import make_reference
    if replay.native is not None:
        from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import use_normalized_float_delta_inputs
        use_normalized_float_delta_inputs(replay.native)
    if replay.hook is not None:
        replay.hook.close()
        replay.hook = None
    policy = make_reference(replay, {**config, "seed": manifest["action_noise_seed_base"]})
    policy.suite = manifest["suite"]
    policy.row_name = row
    policy.log_action_chunks = False
    if row.startswith("latent_bridge_f"):
        policy.refresh_every = int(row[-1])
    # Component timers are diagnostic only. The online worker synchronizes at
    # the same outer policy.act boundary for every method.
    policy._sync = lambda: None
    if replay.loop is not None:
        def decode(self, condition, proprio, *, policy_query_index):
            noise, seed = self._paired_initial_noise(condition, proprio, policy_query_index)
            if noise is None:
                raise RuntimeError("Explicit paired noise is required")
            normalized = self.action_adapter.normalize_proprio(proprio)
            def full_step(x, tau):
                return replay.step(condition, x, normalized, tau)
            trace = self.generation_loop(noise, full_step=full_step,
                full_step_indices=self.full_step_indices, proprio=normalized, condition=condition,
                condition_valid_mask=None,
                condition_change_code=condition.new_zeros(condition.shape[0], self.generation_loop.updater.condition_code_dim))
            action = self.action_adapter.action_space.postprocess(trace.final_noisy_action)
            self.metrics.counters["num_action_transformer_calls"] += self.n_g
            self.metrics.counters["num_action_transformer_decodes"] += 1
            self.metrics.counters["num_generation_decoder_only_steps"] += 10 - self.n_g
            return action, seed
        policy._decode = MethodType(decode, policy)
    return policy


def check_policy(policy, row):
    queries = int(policy.metrics.counters["num_policy_queries"])
    observed = policy.metrics.counters
    for name, expected in expected_counts(row, queries).items():
        if int(observed.get(name, 0)) != expected:
            raise RuntimeError(f"{row}: {name}={observed.get(name, 0)}, expected {expected}")
    if queries != (policy.step_index + 4) // 5:
        raise RuntimeError("H=10/R=5 action queue contract changed")


def check_reset(policy):
    policy.reset()
    if policy.query_index or policy.step_index or policy.action_queue:
        raise RuntimeError("Episode queue/index reset failed")
    for name in ("cached_condition", "cached_raw_rgb", "cached_proprio", "cached_action_chunk", "cached_stable", "condition_layout"):
        if getattr(policy, name, None) is not None:
            raise RuntimeError(f"Stale episode state: {name}")
