from types import SimpleNamespace
import copy
import json
import threading
import os
import subprocess
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from architectures.simvla.adapters.real_world_training import train_joint_baseline as joint
from architectures.simvla.adapters.real_world_training import model_io
from tools.simvla import launch_doll_baseline as launcher


def test_validation_spans_all_episodes_and_final_window():
    samples = [(f"episode{i}", n) for i in range(8) for n in range(13 + i)]
    selected = joint.validation_indices(samples, 5)
    for i in range(8):
        frames = [samples[x][1] for x in selected if samples[x][0] == f"episode{i}"]
        assert frames[0] == 0
        assert frames[-1] == 12 + i
    assert len(joint.validation_indices(samples, 1)) == len(samples)


def test_noise_is_independent_of_batching():
    all_noise, all_t = joint.sample_noise(["a", "b"], [1, 99], 42, "cpu")
    single_noise, single_t = joint.sample_noise(["b"], [99], 42, "cpu")
    assert torch.equal(all_noise[1], single_noise[0])
    assert torch.equal(all_t[1], single_t[0])


def test_warmup_and_scheduler_have_no_3000_step_lock():
    assert joint.schedule(0, warmup=200, total=5000, peak=1e-4, mode="constant") == 5e-7
    assert joint.schedule(4999, warmup=200, total=5000, peak=1e-4, mode="constant") == 1e-4
    assert joint.schedule(4999, warmup=200, total=5000, peak=1e-4, mode="cosine") == pytest.approx(1e-5)


class ActionSpace:
    def normalize_state(self, x):
        return x

    def normalize_action(self, x):
        return x

    def postprocess(self, x):
        return x


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, vlm_features, action_with_noise, proprio, t):
        return action_with_noise * self.weight + vlm_features[:, :1, :1]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = torch.nn.Linear(1, 1)
        self.transformer = TinyTransformer()
        self.action_space = ActionSpace()

    def forward_vlm_efficient(self, image_input, image_mask, input_ids):
        return {"vlm_features": self.vlm(image_input[:, :1, :1])}

    def forward(self, image_input, image_mask, input_ids, proprio, action):
        assert self.training and self.vlm.training and self.transformer.training
        condition = self.forward_vlm_efficient(image_input, image_mask, input_ids)["vlm_features"]
        noise = torch.randn_like(action)
        prediction = self.transformer(condition, noise, proprio, torch.ones(len(action)))
        return {"velocity_loss": (prediction - (noise - action)).square().mean()}


class TinyDataset(Dataset):
    def __init__(self, *args, split="train", **kwargs):
        self.samples = [(f"episode{i}", frame) for i in range(2 if split == "train" else 8) for frame in range(3)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode, frame = self.samples[index]
        return {"episode_id": episode, "frame_index": frame, "language_instruction": "doll",
                "image_input": torch.ones(1, 1) * (index + 1) / 25,
                "image_mask": torch.ones(1, dtype=torch.bool),
                "proprio": torch.ones(8), "action": torch.ones(10, 7) * 0.3}


PROCESSOR = SimpleNamespace(encode_language=lambda text: {"input_ids": torch.ones(len(text), 1, dtype=torch.long)})


def test_preflight_preserves_real_launcher_exit_code(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    status = logs / "launcher.exit_code"
    status.write_text("130\n")
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "SIMVLA_REAL_PYTHON": "/bin/true",
           "SIMVLA_DOLL_JOINT_OUTPUT": str(tmp_path), "SIMVLA_REAL_GPU_IDS": "4"}
    subprocess.run(["bash", str(root / "architectures/simvla/wrappers/train_doll_joint.sh"),
                    "--preflight"], env=env, check=True, capture_output=True)
    assert status.read_text() == "130\n"


def test_action_evaluation_includes_eight_episodes_and_is_repeatable(tmp_path):
    model = TinyModel().train()
    loader = DataLoader(TinyDataset(split="validation"), batch_size=4)
    first = joint.validate(model, PROCESSOR, loader, torch.device("cpu"), tmp_path, 0, 42)
    second = joint.validate(model, PROCESSOR, loader, torch.device("cpu"), tmp_path, 1, 42)
    assert first == second
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        nested = joint.validate(model, PROCESSOR, loader, torch.device("cpu"), tmp_path, 2, 42)
    assert first == nested
    assert len(first["episodes"]) == 8
    assert first["windows"] == 24
    assert model.training
    assert first["robot_success_measured"] is False
    assert first["precision"] == "float32_no_autocast_matches_real_deployment"
    assert first["selection_metric"] == "episode_macro_first5_action_l1"


def test_joint_checkpoint_restores_vlm_and_head(tmp_path):
    norm = tmp_path / "norm.json"
    norm.write_text("{}")
    base = model_io.OfficialBaseIdentity("base", "sha", "processor", "libero_joint", 10, 1024, 24)
    model = TinyModel()
    expected = copy.deepcopy(model.state_dict())
    path = tmp_path / "joint.pt"
    model_io.save_real_joint_checkpoint(path, model=model, official_base=base,
        norm_stats_path=norm, dataset_identity_sha256="dataset", optimizer_step=271,
        training_config={}, validation={})
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
    report = model_io.apply_real_action_checkpoint(model, path, expected_base_sha256="sha",
        expected_norm_sha256=model_io.sha256_file(norm), expected_dataset_identity_sha256="dataset",
        expected_optimizer_step=271)
    assert report["vlm_overlay_loaded"] is True
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])
    payload = model_io.load_real_action_payload(path)
    del payload["vlm_state_dict"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="fine-tuned VLM"):
        model_io.load_real_action_payload(path)


@pytest.mark.parametrize("loaded_parent_training", [True, False])
def test_train_save_interrupt_resume_matches_uninterrupted(tmp_path, monkeypatch, loaded_parent_training):
    from architectures.simvla.adapters.real_world_training import artifact_validation
    from architectures.simvla.adapters.real_world_training.distributed import DistributedContext
    norm = tmp_path / "real_norm.json"
    norm.write_text("{}")
    manifest = {"norm_stats": {"path": norm.name}, "dataset_identity_sha256": "dataset"}
    monkeypatch.setattr(artifact_validation, "validate_real_dataset_manifest", lambda *a, **kw: manifest)
    monkeypatch.setattr(joint, "RealSimVLADataset", TinyDataset)
    monkeypatch.setattr(joint, "initialize_distributed", lambda _: DistributedContext(0, 0, 1, torch.device("cpu"), False))
    def load_model(**kwargs):
        model = TinyModel().train(loaded_parent_training)
        # HF sets the parent to eval; the shared loader enables its children.
        model.vlm.train()
        model.transformer.train()
        return model, PROCESSOR, {}

    monkeypatch.setattr(joint, "load_exact_official_model", load_model)
    monkeypatch.setattr(joint, "enable_checkpointing", lambda _: None)
    monkeypatch.setattr(joint, "official_base_identity", lambda *a: model_io.OfficialBaseIdentity("base", "sha", "processor", "libero_joint", 10, 1024, 24))
    args = joint.parser().parse_args(["--dataset", str(tmp_path), "--checkpoint", "base", "--processor", "processor",
                                     "--output", str(tmp_path / "full"), "--device", "cpu", "--num-workers", "0",
                                     "--max-steps", "3", "--accumulation", "2", "--validation-interval", "1"])
    joint.run(args)
    expected = model_io.load_real_action_payload(tmp_path / "full/checkpoints/joint_step_000003.pt")
    original_validate = joint.validate

    def interrupt(*a, **kw):
        if a[5] == 2:
            raise KeyboardInterrupt
        return original_validate(*a, **kw)

    args.output = str(tmp_path / "resumed")
    monkeypatch.setattr(joint, "validate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        joint.run(args)
    monkeypatch.setattr(joint, "validate", original_validate)
    args.resume = True
    joint.run(args)
    observed = model_io.load_real_action_payload(tmp_path / "resumed/checkpoints/joint_step_000003.pt")
    for key in ("vlm_state_dict", "action_transformer_state_dict"):
        for name, value in expected[key].items():
            assert torch.equal(value, observed[key][name])
    assert json.loads((tmp_path / "resumed/joint_gradient_audit.json").read_text())["vlm"]["nonzero"]
    assert (tmp_path / "resumed/best_checkpoint.txt").exists()


def test_runtime_settings_do_not_change_action_or_model_contract():
    payload = {"runtime": {"training_sample_hz": 15}, "hardware": {"cameras": {}},
               "policy": {"action_horizon": 10, "execution_horizon": 5}, "artifacts": {"model": "same"}}
    args = SimpleNamespace(control_hz=60, camera_fps=60, num_rollouts=15, warmup_steps=3)
    result = launcher.apply_runtime_options(copy.deepcopy(payload), args)
    assert result["policy"] == payload["policy"]
    assert result["artifacts"] == payload["artifacts"]
    assert result["runtime"]["control_frequency_hz"] == 60
    assert result["runtime"]["training_sample_hz"] == 15


@pytest.mark.parametrize("cancel", [None, "stop", "retry"])
def test_outcome_home_permit_is_only_enabled_by_worker_after_save(monkeypatch, cancel):
    from architectures.simvla.adapters.latentloop_real_deploy import deploy_gui as gui
    calls = []
    app = object.__new__(gui.SimVLADeployGuiApp)
    app._outcome_lock = threading.RLock()
    app._outcome_home_pending = False
    app.emergency_stop_reports = []
    app.env = SimpleNamespace(
        emergency_stop=lambda: {"stopped": True},
        cancel_pending_policy_step=lambda: calls.append("cancel_policy"),
        allow_outcome_home=lambda: calls.append("allow_home"),
        disarm_policy_commands=lambda: None,
    )
    app.current_events = {name: threading.Event() for name in ("stop", "retry", "success", "failure")}
    app.set_status = lambda *a: None
    monkeypatch.setattr(gui.legacy_gui.DeployGuiApp, "set_run_state", lambda *a: None)
    app.signal_current("success")
    assert calls == ["cancel_policy"]  # no robot motion in Tk event callback
    calls.append("saved")              # legacy worker saves result before MOVING HOME
    if cancel:
        app.signal_current(cancel)
    app.set_run_state("MOVING HOME")
    assert ("allow_home" in calls) == (cancel is None)
    if cancel is None:
        assert calls.index("saved") < calls.index("allow_home")
