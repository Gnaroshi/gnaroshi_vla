"""Install a calibrated condition generator without changing H10/R5/NFE3."""
from types import MethodType

import torch

from methods.refresh_calibration.model import RefreshCalibratedCondition
from .train import language_key, path_for
from tools.simvla.error_compensation_common import identity, sha, read_json


def load_payload(c, row, k, smoke=False):
    if "candidate" in c:
        spec = c["candidate"]
        path, expected_hash = spec["path"], spec["sha256"]
        expected_identity = spec["training_identity"]
    else:
        path = path_for(c, row, k, smoke)
        expected_hash = read_json(path.parent / "summary.json")["checkpoint_sha256"]
        expected_identity = identity(c)
    if sha(path) != expected_hash:
        raise RuntimeError("Condition checkpoint checksum changed")
    p = torch.load(path, map_location="cpu", weights_only=False)
    expected_step = c["smoke_steps"] if smoke and "candidate" not in c else c["steps"]
    if (p["format"] != "simvla_refresh_calibration_v1" or p["variant"] != row
            or p["identity"] != expected_identity or p["k_c"] != k or p["step"] != expected_step
            or p["contract"]["nfe"] != 3):
        raise RuntimeError("Condition checkpoint method/horizon/step mismatch")
    return p


def attach(policy, payload, row, k, compiler=None):
    model = RefreshCalibratedCondition(row, **payload["contract"]["model"]).to(policy.device).eval()
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False)
    model.condition_updater = None
    bank = {key: value.to(policy.device) for key, value in payload["language_bank"].items()}
    if compiler is not None:
        model.features.forward = compiler.wrap("refresh_observation", model.features.forward)
        model.readout.forward = compiler.wrap("refresh_readout", model.readout.forward)
        if row == "ridge":
            model.fit_correction = compiler.wrap("refresh_fit", model.fit_correction)
        elif row == "anchor_input":
            model.anchor_encoder.forward = compiler.wrap("refresh_anchor_encoder", model.anchor_encoder.forward)
            model.anchor_readout.forward = compiler.wrap("refresh_anchor_readout", model.anchor_readout.forward)
    policy.native_v0 = model
    policy.k_c = policy.refresh_every = k
    policy.row_name = policy.mode = row
    old_full, old_reset, old_preprocess = policy._full_refresh, policy.reset, policy.preprocess

    def reset(self):
        old_reset()
        self._refresh_state = None
        self._refresh_language = None
        self._refresh_prompt = None

    def preprocess(self, image0, image1, proprio, prompt):
        batch = old_preprocess(image0, image1, proprio, prompt)
        key = language_key(prompt)
        if key not in bank:
            raise RuntimeError(f"Unregistered instruction embedding: {key}")
        self._refresh_language, self._refresh_prompt = bank[key], key
        return batch

    def full(self, batch, *, policy_query_index):
        condition, action, noise = old_full(batch, policy_query_index=policy_query_index)
        self._refresh_state = model.prepare(condition, batch["raw_rgb"], batch["proprio"],
            self._refresh_language, self.condition_layout.valid_mask, self.condition_layout.group_ids)
        self.metrics.counters["num_refresh_preparations"] += 1
        self.metrics.counters["num_calibration_writes"] += int(row == "ridge")
        self.metrics.counters["num_observation_encoder_calls"] += int(row == "ridge" or model.anchor_exact)
        return condition, action, noise

    def update(self, batch, *, age, policy_query_index):
        if self._refresh_state is None or age != policy_query_index % k:
            raise RuntimeError("Missing or inconsistent refresh state")
        condition = model.predict(self._refresh_state, batch["raw_rgb"], batch["proprio"])
        self.metrics.counters["num_condition_updater_calls"] += 1
        self.metrics.counters["num_observation_encoder_calls"] += 1
        action, noise = self._decode(condition, batch["proprio"], policy_query_index=policy_query_index)
        self.cached_condition, self.cached_action_chunk = condition.detach(), action.detach()
        return condition, action, noise

    policy.reset = MethodType(reset, policy)
    policy.preprocess = MethodType(preprocess, policy)
    policy._full_refresh = MethodType(full, policy)
    policy._v0_update = MethodType(update, policy)
    policy.reset()
    return policy


def make_sd1_policy(c, row, *, smoke=False, k_c=4):
    from tools.simvla.error_compensation_eval import make_policy
    parent = make_policy(c, "condition_naive3", k_c=min(k_c, 4))
    return attach(parent, load_payload(c, row, k_c, smoke), row, k_c)


def check_counts(policy, row, calls=None, k_c=None):
    q = int(policy.metrics.counters["num_policy_queries"])
    k = k_c or policy.k_c
    full = (q + k - 1) // k
    expected = dict(num_full_vlm_calls=full, num_condition_updater_calls=q-full,
        num_action_transformer_calls=3*q, num_generation_decoder_only_steps=0,
        num_refresh_preparations=full, num_calibration_writes=full if row == "ridge" else 0,
        num_observation_encoder_calls=q if row == "ridge" or policy.native_v0.anchor_exact else q-full)
    for name, value in expected.items():
        if int(policy.metrics.counters.get(name, 0)) != value:
            raise RuntimeError(f"{row}: {name} invocation mismatch")
    if q != (policy.step_index + 4) // 5 or policy.replan_steps != 5:
        raise RuntimeError("Query/action queue cadence changed")
    if calls is not None and calls.get("transformer", 0) != 3*q:
        raise RuntimeError("Measured action transformer calls differ from NFE3")
    return dict(queries=q, **expected)


def replay_factory(c, row, compiler, samples):
    from tools.simvla.compile_benchmark import Replay
    return Replay(c, "condition_naive3", compiler, samples)


def make_rb2_policy(replay, c, row, manifest):
    from tools.simvla.compiled_policy import attach_policy
    base = attach_policy(replay, c, "condition_naive3", manifest)
    return attach(base, load_payload(c, row, c["condition_interval"]), row,
                  c["condition_interval"], replay.compiler)


def check_compiler(compiler, row):
    required = {"vlm", "action_transformer", "action_decoder", "refresh_observation", "refresh_readout"}
    if row == "ridge": required.add("refresh_fit")
    if row == "anchor_input": required.update(("refresh_anchor_encoder", "refresh_anchor_readout"))
    missing = sorted(k for k in required if not compiler.records.get(k, {}).get("graphs", 0))
    if missing:
        raise RuntimeError(f"Compiled component was bypassed: {missing}")


def check_reset(policy):
    from tools.simvla.compiled_policy import check_reset as base
    base(policy)
    if policy._refresh_state is not None or policy._refresh_language is not None:
        raise RuntimeError("Calibration state leaked across episodes")
