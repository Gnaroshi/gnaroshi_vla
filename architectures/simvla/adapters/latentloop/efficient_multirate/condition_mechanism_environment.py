"""Paired object displacement with identical prefix actions and full continuations."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import (
    PRIMARY, completed_unit, tensor_hash, write_json,
)
from architectures.simvla.adapters.latentloop.native_v0_policy import RealSimVLANativeV0Policy
from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs, get_libero_env
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair


METHODS = ("baseline", *PRIMARY)


class InterventionPolicy(RealSimVLANativeV0Policy):
    """H=10/R=5/NFE10; only the intermediate K_C=2 condition rule changes."""

    def __init__(self, *, variant: str, **kwargs: Any):
        if variant not in METHODS:
            raise ValueError(variant)
        self.variant = variant
        super().__init__(**kwargs)
        self.refresh_every = 1 if variant == "baseline" else 2
        self.first_intervention_chunk = None

    def _refill_action_queue(self, batch):
        query = self.query_index
        self.metrics.counters["num_policy_queries"] += 1
        age = query % self.refresh_every
        if age == 0:
            condition, chunk, seed = self._full_refresh(batch, policy_query_index=query)
            source = "full_refresh"
        else:
            if self.variant == "hold":
                condition = self.cached_condition
            else:
                if self.variant == "zero_feature":
                    code = self.cached_condition.new_zeros((1, self.native_v0.delta_dim))
                else:
                    pair = NativeV0ObservationPair(self.cached_raw_rgb, batch["raw_rgb"], self.cached_proprio, batch["proprio"])
                    code = self.native_v0.delta_encoder(pair)
                update = self.native_v0.condition_updater(
                    self.cached_condition, code, valid_mask=self.condition_layout.valid_mask,
                    group_ids=self.condition_layout.group_ids, age=1,
                )
                condition = update.condition
            chunk, seed = self._decode(condition, batch["proprio"], policy_query_index=query)
            self.cached_condition = condition.detach()
            self.cached_raw_rgb = batch["raw_rgb"].detach()
            self.cached_proprio = batch["proprio"].detach()
            self.cached_action_chunk = chunk.detach()
            source = self.variant
        self.action_queue.clear()
        for action in chunk[0, :5]:
            self.action_queue.append((action.detach(), source))
        if query == 1:
            self.first_intervention_chunk = chunk.detach().cpu().numpy()[0]
        self.query_trace.append({"query": query, "source": source, "noise_seed": seed, "H": 10, "R": 5, "NFE": 10})
        self.query_index += 1
        return {"refreshed": age == 0, "age": age, "queue_mode": source, "action_noise_seed": seed}


def movable_target(env: Any) -> tuple[str, str] | None:
    """Use task-declared objects of interest, never an outcome-selected object."""
    core = env.env
    for name in core.obj_of_interest:
        obj = core.objects_dict.get(name)
        if obj is None:
            continue
        for joint in obj.joints:
            pose = np.asarray(env.sim.data.get_joint_qpos(joint))
            if pose.shape == (7,):
                return name, joint
    return None


def robot_touching_object(env: Any, joint: str) -> bool:
    body = int(env.sim.model.jnt_bodyid[env.sim.model.joint_name2id(joint)])
    parents = np.asarray(env.sim.model.body_parentid)
    def target_body(candidate):
        while candidate > 0:
            if candidate == body:
                return True
            candidate = int(parents[candidate])
        return False
    target_geoms = {g for g, b in enumerate(env.sim.model.geom_bodyid) if target_body(int(b))}
    robot_geoms = set()
    for robot in env.robots:
        for model in (robot.robot_model, robot.gripper):
            for name in getattr(model, "contact_geoms", []):
                robot_geoms.add(env.sim.model.geom_name2id(name))
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        if (contact.geom1 in target_geoms and contact.geom2 in robot_geoms) or (contact.geom2 in target_geoms and contact.geom1 in robot_geoms):
            return True
    return False


def simulator_signature(env: Any) -> dict:
    sim = env.sim
    return {"state": np.asarray(env.get_sim_state()).copy(), "ctrl": np.asarray(sim.data.ctrl).copy()}


def assert_same_prefix(reference: dict, observed: dict) -> None:
    for name in reference:
        if not np.allclose(reference[name], observed[name], atol=1e-10, rtol=0):
            raise RuntimeError(f"Prefix replay changed simulator {name}; causal pairing invalid")


def displace_object(env: Any, joint: str, offset: np.ndarray) -> tuple[dict, dict]:
    qpos_before = np.asarray(env.sim.data.qpos).copy()
    qvel_before = np.asarray(env.sim.data.qvel).copy()
    ctrl_before = np.asarray(env.sim.data.ctrl).copy()
    time_before = float(env.sim.data.time)
    pose = np.asarray(env.sim.data.get_joint_qpos(joint)).copy()
    changed = pose.copy()
    changed[:3] += offset
    env.sim.data.set_joint_qpos(joint, changed)
    address = env.sim.model.get_joint_qpos_addr(joint)
    start = address[0] if isinstance(address, tuple) else int(address)
    allowed = np.zeros_like(qpos_before, dtype=bool)
    allowed[start:start + 3] = True
    if not np.array_equal(qpos_before[~allowed], np.asarray(env.sim.data.qpos)[~allowed]):
        raise RuntimeError("Object displacement changed non-target joints")
    if not np.array_equal(qvel_before, env.sim.data.qvel) or not np.array_equal(ctrl_before, env.sim.data.ctrl):
        raise RuntimeError("Object displacement changed velocity or control")
    obs = env.regenerate_obs_from_state(env.get_sim_state())
    if float(env.sim.data.time) != time_before:
        raise RuntimeError("Observation regeneration advanced physics")
    return obs, {"joint": joint, "before_pose": pose.tolist(), "after_pose": changed.tolist(),
                 "offset_m": offset.tolist(), "physics_steps_during_intervention": 0}


def policy_for(config: dict, manifest: dict, model: Any, processor: Any, adapter: Any, task: int, trial: int, variant: str):
    return InterventionPolicy(
        variant=variant, model=model, processor=processor, adapter=adapter,
        checkpoint_id=config["checkpoint"], device=next(adapter.parameters()).device,
        suite="libero_10", task_id=task, trial_id=trial,
        action_noise_seed_base=manifest["action_noise_seed_base"],
        client_resize_size=manifest["client_resize_size"], image_size=manifest["model_image_size"], flow_steps=10,
    )


def reset_episode(env: Any, initial: Any, seed: int, wait: int) -> dict:
    env.seed(seed)
    env.reset()
    observation = env.set_init_state(initial)
    for _ in range(wait):
        observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
    return observation


@torch.no_grad()
def run_environment(config: dict, manifest: dict, output: Path, identity: str, model: Any, processor: Any, adapter: Any) -> dict:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    cases = [e for e in manifest["episodes"] if e["trial_id"] < config["intervention_trials_per_task"] and e["task_id"] in config["intervention_task_ids"]]
    cases.sort(key=lambda e: (-e["task_id"], e["trial_id"]))
    write_json(output / "selection.json", {
        "identity": identity, "cases": cases, "methods": METHODS, "worlds": ["nominal", "displaced"],
        "offset_rule": "+x for even trials, -x for odd trials; fixed before observing outcomes",
        "shift_m": config["object_shift_m"], "injection_after_actions": 5,
        "note": "Exploratory environment intervention, not a standard LIBERO benchmark SR. No outcome filtering.",
    })
    progress = tqdm(total=len(cases) * len(METHODS) * 2, desc="물체 위치 개입·완주 평가", mininterval=1.0)
    completed = invalid = 0
    started = time.monotonic()
    for spec in cases:
        task_id, trial = int(spec["task_id"]), int(spec["trial_id"])
        task = suite.get_task(task_id)
        initial = suite.get_task_init_states(task_id)[int(spec["init_state_index"])]
        unit_root = output / "units" / f"task_{task_id:02d}_trial_{trial:02d}"
        case_path = unit_root / "case.json"
        old_case = completed_unit(case_path, identity)
        if old_case and old_case.get("invalid_reason"):
            invalid += 1
            progress.update(len(METHODS) * 2)
            continue
        branch_paths = [unit_root / f"{world}_{method}.json" for world in ("nominal", "displaced") for method in METHODS]
        if all(completed_unit(p, identity) is not None for p in branch_paths):
            completed += len(branch_paths)
            progress.update(len(branch_paths))
            continue
        env, prompt = get_libero_env(task, manifest["environment_resolution"], manifest["environment_seed"])
        try:
            observation = reset_episode(env, initial, manifest["environment_seed"], manifest["num_wait_steps"])
            base = policy_for(config, manifest, model, processor, adapter, task_id, trial, "baseline")
            batch = base.preprocess(*build_env_obs(observation), prompt)
            condition, chunk, _ = base._full_refresh(batch, policy_query_index=0)
            prefix = chunk[0, :5].cpu().numpy()
            for action in prefix:
                observation, _, done, _ = env.step(action.tolist())
            # env.step() caches some sensors before the final forward. Compare
            # canonical snapshot observations on BOTH branches, including sham.
            before_forward = simulator_signature(env)
            observation = env.regenerate_obs_from_state(env.get_sim_state())
            assert_same_prefix(before_forward, simulator_signature(env))
            reference_signature = simulator_signature(env)
            pre_q = build_env_obs(observation)[2]
            target = movable_target(env)
            invalid_reason = "no_task_relevant_free_joint" if target is None else ("task_completed_before_intervention" if done else None)
            if target is not None and robot_touching_object(env, target[1]):
                invalid_reason = "target_already_in_robot_contact"
            case = {"identity": identity, "complete": True, "task_id": task_id, "trial_id": trial,
                    "prompt": prompt, "prefix_actions": prefix.tolist(), "prefix_sha256": tensor_hash(prefix),
                    "sim_state_sha256": tensor_hash(reference_signature["state"]), "condition_sha256": tensor_hash(condition),
                    "target": None if target is None else list(target), "invalid_reason": invalid_reason,
                    "environment": {"control_freq": env.env.control_freq,
                                    "camera_names": list(env.env.camera_names),
                                    "controller": type(env.env.robots[0].controller).__name__}}
            if old_case and old_case != case:
                raise RuntimeError("Case prefix identity changed on resume")
            write_json(case_path, case)
            if invalid_reason:
                invalid += 1
                progress.update(len(METHODS) * 2)
                continue
            cached = {name: getattr(base, name) for name in ("cached_condition", "cached_raw_rgb", "cached_proprio", "cached_action_chunk", "condition_layout")}
        finally:
            env.close()
        for world in ("nominal", "displaced"):
            for method in METHODS:
                branch_path = unit_root / f"{world}_{method}.json"
                if completed_unit(branch_path, identity):
                    completed += 1
                    progress.update()
                    continue
                branch_start = time.monotonic()
                env, _ = get_libero_env(task, manifest["environment_resolution"], manifest["environment_seed"])
                try:
                    observation = reset_episode(env, initial, manifest["environment_seed"], manifest["num_wait_steps"])
                    for action in prefix:
                        observation, _, _, _ = env.step(action.tolist())
                    observation = env.regenerate_obs_from_state(env.get_sim_state())
                    assert_same_prefix(reference_signature, simulator_signature(env))
                    if not np.array_equal(pre_q, build_env_obs(observation)[2]):
                        raise RuntimeError("Robot state changed across paired branches")
                    offset = np.zeros(3)
                    if world == "displaced":
                        offset[0] = config["object_shift_m"] * (1 if trial % 2 == 0 else -1)
                    observation, perturbation = displace_object(env, target[1], offset)
                    if not np.array_equal(pre_q, build_env_obs(observation)[2]):
                        raise RuntimeError("Object intervention changed robot proprioception before action")
                    if method == "baseline":
                        Image.fromarray(observation["agentview_image"][::-1, ::-1]).save(unit_root / f"{world}.png")
                    policy = policy_for(config, manifest, model, processor, adapter, task_id, trial, method)
                    for name, value in cached.items():
                        setattr(policy, name, value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
                    policy.query_index = 1
                    policy.step_index = 5
                    done = bool(env.check_success())
                    immediate_success = done
                    actions = []
                    first_obs_hash = tensor_hash(observation["agentview_image"])
                    remaining = min(config["continuation_actions"], manifest["max_policy_actions"] - 5)
                    for _ in range(remaining):
                        if done:
                            break
                        result = policy.act(*build_env_obs(observation), prompt)
                        actions.append(result.action.tolist())
                        observation, _, done, _ = env.step(result.action.tolist())
                        if len(actions) % 25 == 0:
                            progress.set_postfix(task=task_id, trial=trial, method=method, world=world, actions=5 + len(actions))
                    row = {
                        "identity": identity, "complete": True, "task_id": task_id, "trial_id": trial,
                        "method": method, "world": world, "success": bool(done),
                        "success_immediately_after_intervention": immediate_success,
                        "episode_actions": 5 + len(actions), "continuation_limit": remaining,
                        "full_episode_horizon": remaining == manifest["max_policy_actions"] - 5,
                        "perturbation": perturbation, "first_obs_sha256": first_obs_hash,
                        "proprio_sha256": tensor_hash(pre_q), "previous_condition_sha256": tensor_hash(condition),
                        "first_action_chunk": None if policy.first_intervention_chunk is None else policy.first_intervention_chunk.tolist(),
                        "executed_actions": actions, "query_trace": policy.query_trace,
                        "seconds": time.monotonic() - branch_start,
                        "latency_scope": "instrumented diagnostic wall time; not paper policy latency",
                    }
                    write_json(branch_path, row)
                finally:
                    env.close()
                completed += 1
                progress.update()
                progress.set_postfix(task=task_id, trial=trial, method=method, world=world, success=int(row["success"]))
    progress.close()
    result = {"identity": identity, "complete": True, "planned_cases": len(cases), "invalid_cases": invalid,
              "completed_branches": completed, "seconds_this_invocation": time.monotonic() - started,
              "performance_threshold": None, "no_outcome_filtering": True}
    write_json(output / "summary.json", result)
    return result
