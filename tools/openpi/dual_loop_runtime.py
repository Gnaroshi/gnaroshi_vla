"""Shared data/provenance helpers for the bounded pi0.5 dual-loop campaign."""

from pathlib import Path
import hashlib
import json
import os
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get("OPENPI_UPSTREAM_ROOT", ROOT / "architectures/openpi/upstream"))
for path in (ROOT, UPSTREAM / "src", UPSTREAM / "packages/openpi-client/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_components(config, condition_path=None):
    import torch
    from openpi.policies import policy_config
    from openpi.training import config as config_api
    from architectures.openpi.adapters.latentloop.aligned_condition import load_condition
    deterministic(config["train_seed"])
    torch.cuda.set_per_process_memory_fraction(0.90)
    base = policy_config.create_trained_policy(
        config_api.get_config("pi05_libero_lora_pytorch"), Path(config["checkpoint"]),
        pytorch_device="cuda", sample_kwargs={"num_steps": 10})
    model = base._model
    model.requires_grad_(False).eval()
    condition = None
    if condition_path is not None:
        condition, _ = load_condition(condition_path, "cuda", config_id=config["config_id"])
        condition.requires_grad_(False).eval()
    if model.config.action_horizon != 10:
        raise ValueError("expected the reproduced H=10 pi0.5 policy")
    if model.config.discrete_state_input:
        raise ValueError("the reproduced baseline uses a fixed instruction, without state tokenization")
    return base, model, condition


class TrainingPairs:
    def __init__(self, config, policy):
        import torch
        from openpi.training import config as config_api, data_loader
        from architectures.openpi.adapters.latentloop.streaming_teacher import build_streaming_episode_plan
        self.config, self.policy = config, policy
        train_config = config_api.get_config("pi05_libero_lora_pytorch")
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        self.dataset = data_loader.create_torch_dataset(data_config, 10, train_config.model)
        episodes = build_streaming_episode_plan(read_json(config["split_contract"]))
        self.roles = {role: [e for e in episodes if e.suite == "libero_10" and e.role == role]
                      for role in ("train", "checkpoint_validation")}
        underlying = getattr(self.dataset, "_dataset", self.dataset)
        for role, rows in self.roles.items():
            if not rows:
                raise ValueError(f"no LIBERO-Long demonstrations for {role}")
            for e in rows:
                if (int(underlying.episode_data_index["from"][e.episode_id]),
                    int(underlying.episode_data_index["to"][e.episode_id])) != (e.dataset_frame_start, e.dataset_frame_stop):
                    raise ValueError("dataset frame boundaries changed")
        self.final_manifest = read_json(config["final_manifest"])
        train_ids = {e.episode_id for e in self.roles["train"]}
        heldout_ids = {e.episode_id for e in self.roles["checkpoint_validation"]}
        if train_ids & heldout_ids:
            raise ValueError("train/heldout episode overlap")
        for role, rows in self.roles.items():
            if {e.benchmark_task_index for e in rows} != set(range(10)):
                raise ValueError(f"{role} must cover all ten LIBERO-Long tasks")

    @property
    def validation_count(self):
        return 3 * len(self.roles["checkpoint_validation"])

    def window(self, index, role="train"):
        import numpy as np
        import torch
        from architectures.openpi.adapters.latentloop.streaming_teacher import _raw_policy_observation
        from architectures.openpi.adapters.latentloop.policy_io import prepare_policy_observation
        from architectures.openpi.adapters.latentloop.cache_contract_v2 import resolve_task_identity
        rows = self.roles[role]
        if role == "checkpoint_validation":
            if not 0 <= index < self.validation_count:
                raise IndexError("validation index outside the frozen episode/stage manifest")
            e = rows[index // 3]
            start = round((index % 3) * (e.query_count - 4) / 2)
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.config["train_seed"], index]))
            e = rows[int(rng.integers(len(rows)))]
            start = int(rng.integers(e.query_count - 3))
        observations = []
        for q in range(start, start + 4):
            sample = self.dataset[e.dataset_frame_start + q * 5]
            task = resolve_task_identity(int(sample["task_index"]), str(sample["prompt"]), self.final_manifest)
            if (task["suite"], int(task["benchmark_task_index"])) != (e.suite, e.benchmark_task_index):
                raise ValueError("demonstration task identity mismatch")
            if int(sample["frame_index"]) != q * 5 or bool(torch.as_tensor(sample["actions_is_pad"][:5]).any()):
                raise ValueError("window crosses an episode boundary")
            observations.append(prepare_policy_observation(self.policy, _raw_policy_observation(sample))[0])
        return observations, {"episode_id": e.episode_id, "task_id": e.benchmark_task_index,
                              "query_index": start, "role": role}

    def pair(self, index, role="train"):
        import numpy as np
        import torch
        from architectures.openpi.adapters.latentloop.streaming_teacher import _raw_policy_observation
        from architectures.openpi.adapters.latentloop.policy_io import prepare_policy_observation
        from architectures.openpi.adapters.latentloop.cache_contract_v2 import resolve_task_identity
        rows = self.roles[role]
        # Step-indexed sampling makes interrupted training resume the same data/noise sequence.
        rng = np.random.default_rng(np.random.SeedSequence([self.config["train_seed"], index,
                                                           0 if role == "train" else 1]))
        if role == "checkpoint_validation":
            rows = [e for e in rows if e.benchmark_task_index == index % 10]
            if not rows:
                raise ValueError("heldout split is missing a LIBERO-Long task")
        e = rows[int(rng.integers(len(rows)))]
        q = int(rng.integers(1, e.query_count))
        frames = [e.dataset_frame_start + (q - 1) * 5, e.dataset_frame_start + q * 5]
        samples = [self.dataset[f] for f in frames]
        for sample, frame in zip(samples, frames):
            if int(sample["frame_index"]) != frame - e.dataset_frame_start:
                raise ValueError("misaligned demonstration frame")
            if bool(torch.as_tensor(sample["actions_is_pad"][:5]).any()):
                raise ValueError("training sample crosses episode boundary")
            task = resolve_task_identity(int(sample["task_index"]), str(sample["prompt"]), self.final_manifest)
            if (task["suite"], int(task["benchmark_task_index"])) != (e.suite, e.benchmark_task_index):
                raise ValueError("demonstration task identity mismatch")
        observations = [prepare_policy_observation(self.policy, _raw_policy_observation(s))[0] for s in samples]
        executed = torch.as_tensor(samples[0]["actions"][:5, :7], device="cuda", dtype=torch.float32)[None]
        return observations, executed, {"episode_id": e.episode_id, "query_index": q, "role": role}


def training_prefix(model, condition, observations, executed, approximate):
    import torch
    from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook
    hook = PrefixKVHook(model)
    with torch.no_grad():
        if not approximate:
            extraction = hook.extract(observations[1])
            return extraction.state, extraction.robot_state
        raise ValueError("Generation is trained independently on full teacher conditions")


def load_generation(path, device="cuda"):
    import torch
    from methods.latentloop.modules.flow_hidden_update import FlowHiddenConfig, FlowHiddenUpdater
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "pi05_simvla_core_generation_v1":
        raise ValueError("not a method-aligned Generation checkpoint")
    module = FlowHiddenUpdater(FlowHiddenConfig(**payload["updater_config"]))
    module.load_state_dict(payload["updater"], strict=True)
    return module.to(device).eval(), payload
