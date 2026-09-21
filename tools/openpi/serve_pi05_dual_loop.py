#!/usr/bin/env python3
"""Serve paired pi0.5 rows through the unchanged OpenPI websocket transport."""

import argparse
from pathlib import Path
import time

from dual_loop_runtime import atomic_json, load_components, load_generation, read_json
import numpy as np
import torch

from architectures.openpi.adapters.latentloop.dual_loop import DualLoopPolicy, ROWS, sync
from architectures.openpi.adapters.latentloop.policy_io import (
    explicit_policy_noise, policy_noise_seed, prepare_policy_observation, postprocess_policy_actions)
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class ServingPolicy:
    def __init__(self, config, generation_path):
        self.config = config
        self.base, model, condition = load_components(config)
        generation, payload = load_generation(generation_path)
        if payload["config_id"] != config["config_id"]:
            raise ValueError("generation checkpoint belongs to a different experiment contract")
        self.policies = {row: DualLoopPolicy(model, condition, generation, row) for row in ROWS}
        self.model = model
        self.episode = None
        self.previous_actions = None
        self.metadata = {"config_id": config["config_id"], "rows": list(ROWS), "action_horizon": 10,
                         "execution_horizon": 5, "integration_steps": 10,
                         "generation_step": payload["step"], "generation_checkpoint": str(generation_path)}

    def reset(self):
        self.episode = None
        self.previous_actions = None
        for policy in self.policies.values():
            policy.reset()

    @torch.no_grad()
    def infer(self, raw):
        raw = dict(raw)
        request = raw.pop("latentloop")
        row = str(request["policy_path"])
        if row not in ROWS or request["suite"] != "libero_10":
            raise ValueError("unexpected policy row or suite")
        identity = (row, int(request["task_id"]), int(request["episode_id"]))
        if request.get("reset") or identity != self.episode:
            if int(request["query_index"]) != 0:
                raise ValueError("an episode must start at query zero")
            self.reset()
            self.episode = identity
        policy = self.policies[row]
        if int(request["query_index"]) != policy.query_index:
            raise ValueError("noncontiguous policy queries")
        sync("cuda")
        started = time.perf_counter()
        observation, transformed = prepare_policy_observation(self.base, raw)
        sync("cuda")
        preprocessing_ms = (time.perf_counter() - started) * 1000
        seed = policy_noise_seed(self.config["noise_seed"], "libero_10", identity[1], identity[2], policy.query_index)
        noise = explicit_policy_noise((1, 10, self.model.config.action_dim), seed=seed, device="cuda")
        sync("cuda")
        forward_start = time.perf_counter()
        normalized, metrics = policy.query(observation, noise, self.previous_actions)
        sync("cuda")
        metrics["model_forward_ms"] = (time.perf_counter() - forward_start) * 1000
        output = postprocess_policy_actions(self.base, transformed["state"], normalized)
        actions = np.asarray(output["actions"])
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid physical action chunk {actions.shape}")
        self.previous_actions = torch.as_tensor(actions[:5], device="cuda", dtype=torch.float32)[None]
        sync("cuda")
        metrics.update(infer_ms=(time.perf_counter() - started) * 1000, preprocessing_ms=preprocessing_ms,
                       noise_seed=seed, task_id=identity[1], episode_id=identity[2],
                       peak_vram_bytes=torch.cuda.max_memory_allocated())
        return {"actions": actions, "latentloop_metrics": metrics}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--generation", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ready", required=True)
    args = p.parse_args()
    policy = ServingPolicy(read_json(args.config), Path(args.generation))
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata)
    atomic_json(args.ready, policy.metadata)
    server.serve_forever()
