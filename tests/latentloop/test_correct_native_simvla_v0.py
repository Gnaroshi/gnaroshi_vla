from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    NativeV0Config,
    load_native_v0_checkpoint,
    save_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_condition_hook import (
    GROUP_IMAGE_VIEW_0,
    GROUP_IMAGE_VIEW_1,
    GROUP_LANGUAGE_PAD_ID,
    GROUP_PADDING,
    build_condition_token_layout,
    source_hook_audit,
)
from architectures.simvla.adapters.latentloop.native_v0_long_eval import MANIFEST_SCHEMA
from architectures.simvla.adapters.latentloop.native_v0_aggregate import merge_row
from architectures.simvla.adapters.latentloop.native_v0_dataset import NativeV0SequenceDataset
from architectures.simvla.adapters.latentloop.native_v0_prepare import (
    _official_training_image_inputs,
)
from architectures.simvla.adapters.latentloop.native_v0_training_cache import (
    build_training_cache,
    validate_training_cache,
)
from architectures.simvla.wrappers.simvla_two_gpu_guard import parse_selected_gpu_ids
from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0DeltaEncoder,
    NativeV0ObservationPair,
    NativeV0UnrollOutput,
    NativeV0UpdateOutput,
    TokenSharedConditionUpdater,
)
from methods.latentloop.training.native_simvla_v0 import (
    NativeV0LossWeights,
    decode_age_conditions,
    lr_multiplier,
    native_v0_raw_losses,
    weighted_native_v0_loss,
)


def test_condition_layout_matches_two_view_fused_contract() -> None:
    condition = torch.zeros(2, 122, 960)
    image_mask = torch.tensor([[True, True, False], [True, False, False]])
    input_ids = torch.tensor([[1] * 49 + [0], [2] * 50])
    # Mixed valid-view counts imply max sequence 122 and 36 image tokens/view.
    layout = build_condition_token_layout(
        condition=condition,
        image_mask=image_mask,
        input_ids=input_ids,
        pad_token_id=0,
        special_token_ids=(1,),
    )
    assert layout.image_tokens_per_view == 36
    assert layout.valid_mask[0].all()
    assert layout.valid_mask[1, :86].all()
    assert not layout.valid_mask[1, 86:].any()
    assert torch.all(layout.group_ids[0, :36] == GROUP_IMAGE_VIEW_0)
    assert torch.all(layout.group_ids[0, 36:72] == GROUP_IMAGE_VIEW_1)
    assert layout.group_ids[0, 121].item() == GROUP_LANGUAGE_PAD_ID
    assert torch.all(layout.group_ids[1, 86:] == GROUP_PADDING)


def test_condition_layout_rejects_nonprefix_image_mask() -> None:
    with pytest.raises(ValueError, match="prefix-valid"):
        build_condition_token_layout(
            condition=torch.zeros(1, 122, 960),
            image_mask=torch.tensor([[True, False, True]]),
            input_ids=torch.ones(1, 50, dtype=torch.long),
        )


def test_exact_condition_hook_is_before_original_vlm_projection() -> None:
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.transformer_smolvlm import SmolVLMActionTransformer

    audit = source_hook_audit(SmolVLMVLA, SmolVLMActionTransformer)
    assert audit["verdict"] == "SOURCE_EXACT_PRE_VLM_PROJ"
    assert audit["checks"]["concat_path_owns_vlm_projection"] is True
    assert audit["checks"]["external_hook_is_pre_projection"] is True


def test_cache_reencode_uses_official_training_image_transform() -> None:
    raw_rgb = (
        torch.arange(2 * 13 * 17 * 3, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(2, 13, 17, 3)
    )
    observed = _official_training_image_inputs(raw_rgb)
    official_transform = transforms.Compose(
        [
            transforms.Resize(
                (384, 384),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                inplace=True,
            ),
        ]
    )
    expected_views = [
        official_transform(Image.fromarray(image.numpy())) for image in raw_rgb
    ]
    expected = torch.stack([*expected_views, torch.zeros_like(expected_views[0])]).unsqueeze(0)
    assert torch.equal(observed["image_input"], expected)
    assert torch.equal(observed["image_mask"], torch.tensor([[True, True, False]]))


def test_cache_reencode_does_not_use_processor_image_path() -> None:
    source = inspect.getsource(__import__(
        "architectures.simvla.adapters.latentloop.native_v0_prepare",
        fromlist=["_processed_cache_record"],
    )._processed_cache_record)
    assert "encode_image" not in source
    assert "processor(images=" not in source


def test_padding_is_copied_bitwise() -> None:
    updater = TokenSharedConditionUpdater(condition_dim=16, delta_dim=8, rank_dim=64, max_tokens=8)
    with torch.no_grad():
        updater.up.bias.fill_(1.0)
    previous = torch.randn(2, 5, 16)
    valid = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    result = updater(
        previous,
        torch.randn(2, 8),
        valid_mask=valid,
        group_ids=torch.zeros(2, 5, dtype=torch.long),
        age=torch.tensor([1, 2]),
    )
    assert torch.equal(result.condition[~valid], previous[~valid])
    assert torch.equal(result.residual[~valid], torch.zeros_like(result.residual[~valid]))


def test_multiview_delta_encoder_preserves_order() -> None:
    torch.manual_seed(5)
    encoder = NativeV0DeltaEncoder(num_views=2, image_size=16)
    view0_prev = torch.zeros(1, 3, 16, 16)
    view0_cur = torch.ones(1, 3, 16, 16)
    view1_prev = torch.full((1, 3, 16, 16), 0.25)
    view1_cur = torch.full((1, 3, 16, 16), 0.75)
    q0 = torch.zeros(1, 8)
    q1 = torch.ones(1, 8)
    first = encoder(NativeV0ObservationPair([view0_prev, view1_prev], [view0_cur, view1_cur], q0, q1))
    swapped = encoder(NativeV0ObservationPair([view1_prev, view0_prev], [view1_cur, view0_cur], q0, q1))
    assert not torch.allclose(first, swapped)


def test_v0_api_has_no_executed_actions_or_teacher_input() -> None:
    update_parameters = set(inspect.signature(NativeSimVLAV0.update_once).parameters)
    unroll_parameters = set(inspect.signature(NativeSimVLAV0.unroll_k4).parameters)
    forbidden = {"executed_actions", "executed_subchunk", "teacher_condition", "teacher_conditions"}
    assert not (update_parameters & forbidden)
    assert not (unroll_parameters & forbidden)


def test_token_transition_is_shared_and_parameter_cap_passes() -> None:
    model = NativeV0Config().build()
    audit = model.parameter_audit()
    assert audit == {
        "observation_change_encoder": 433024,
        "token_transition": 134080,
        "gates": 65,
        "embeddings": 17152,
        "total": 584321,
        "under_hard_cap_1000000": True,
        "in_target_range_500000_1000000": True,
    }
    assert isinstance(model.condition_updater.down, torch.nn.Linear)
    assert model.condition_updater.down.in_features == 960
    assert model.condition_updater.down.out_features == 64


def test_recursive_ages_consume_previous_prediction() -> None:
    model = NativeSimVLAV0(condition_dim=16, delta_dim=8, rank_dim=64, max_tokens=8)
    with torch.no_grad():
        model.condition_updater.up.bias.fill_(0.5)
    captured: list[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().clone())

    handle = model.condition_updater.register_forward_pre_hook(pre_hook)
    result = model.unroll_k4(
        torch.zeros(1, 5, 16),
        torch.rand(1, 4, 2, 3, 16, 16),
        torch.rand(1, 4, 8),
        valid_mask=torch.ones(1, 5, dtype=torch.bool),
        group_ids=torch.ones(1, 5, dtype=torch.long),
    )
    handle.remove()
    assert len(captured) == 3
    assert torch.equal(captured[1], result.conditions[0])
    assert torch.equal(captured[2], result.conditions[1])


def test_mode_a_b_use_identical_explicit_noise() -> None:
    conditions = tuple(torch.randn(2, 3, 4, requires_grad=True) for _ in range(3))
    proprio = tuple(torch.randn(2, 8) for _ in range(3))
    noises = tuple(torch.randn(2, 10, 7) for _ in range(3))

    def decoder(condition: torch.Tensor, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        signal = condition.mean(dim=(1, 2)) + state.mean(dim=1)
        return noise + signal[:, None, None]

    mode_a = decode_age_conditions(decoder, conditions, proprio, noises, mode="A")
    mode_b = decode_age_conditions(decoder, conditions, proprio, noises, mode="B")
    assert all(torch.equal(left, right) for left, right in zip(mode_a, mode_b))


def test_same_noise_action_loss_and_teacher_target_contract() -> None:
    conditions = tuple(torch.randn(1, 4, 8, requires_grad=True) for _ in range(3))
    updates = tuple(
        NativeV0UpdateOutput(
            condition=condition,
            residual=torch.zeros_like(condition),
            gate=torch.zeros(1, 4, 1),
        )
        for condition in conditions
    )
    unroll = NativeV0UnrollOutput(
        conditions=conditions,
        delta_features=tuple(torch.zeros(1, 2) for _ in range(3)),
        updates=updates,
    )
    explicit_noise_actions = tuple(torch.randn(1, 10, 7) for _ in range(3))
    raw = native_v0_raw_losses(
        unroll=unroll,
        teacher_conditions=tuple(condition.detach().clone() for condition in conditions),
        predicted_actions=explicit_noise_actions,
        teacher_actions=tuple(action.detach().clone() for action in explicit_noise_actions),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    total, _ = weighted_native_v0_loss(
        raw,
        NativeV0LossWeights(1.0, 1.0, 1.0, 1.0, 0.0),
    )
    assert total.item() == pytest.approx(0.0, abs=1e-8)
    total.backward()
    assert all(condition.grad is not None for condition in conditions)


def test_scheduler_contract_values() -> None:
    assert lr_multiplier(0) == 0.0
    assert lr_multiplier(7_500) == 1.0
    assert lr_multiplier(150_000) == pytest.approx(0.1)


def test_two_gpu_guard_rejects_configured_forbidden_and_wrong_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMVLA_FORBIDDEN_GPU_IDS", raising=False)
    assert parse_selected_gpu_ids("2,3") == (2, 3)
    assert parse_selected_gpu_ids("4,7") == (4, 7)
    with pytest.raises(ValueError, match="exactly two"):
        parse_selected_gpu_ids("2")
    monkeypatch.setenv("SIMVLA_FORBIDDEN_GPU_IDS", "4,7")
    with pytest.raises(ValueError, match="forbidden"):
        parse_selected_gpu_ids("4,6")
    with pytest.raises(ValueError, match="forbidden"):
        parse_selected_gpu_ids("6,7")


def test_final_checkpoint_policy_and_serialization(tmp_path: Path) -> None:
    model = NativeV0Config().build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = tmp_path / "final.pt"
    save_native_v0_checkpoint(
        checkpoint,
        model=model,
        config=NativeV0Config(),
        global_step=150_000,
        optimizer=optimizer,
        scheduler_state={"optimizer_step": 150_000},
        sampler_state={"next_optimizer_step": 150_000},
        source_lock={"combined_sha256": "test"},
        training_config={"mode": "B"},
        final=True,
    )
    reloaded, payload = load_native_v0_checkpoint(checkpoint, device="cpu", require_final_150k=True)
    assert payload["scientific_primary_checkpoint"] is True
    assert reloaded.parameter_audit()["total"] == 584321
    assert json.loads(json.dumps({"schema": MANIFEST_SCHEMA, "episodes": 500}))["episodes"] == 500


def test_scientific_manifest_contract_is_500_not_100() -> None:
    source = inspect.getsource(__import__(
        "architectures.simvla.adapters.latentloop.native_v0_long_eval",
        fromlist=["create_manifest"],
    ).create_manifest)
    assert "range(10)" in source
    assert "range(50)" in source
    assert '"episodes_per_row": 500' in source
    assert '"episodes_per_row": 100' not in source


def test_long_row_serialization_preserves_v0_parameter_count(tmp_path: Path) -> None:
    row_root = tmp_path / "row"
    manifest = {
        "manifest_sha256": "manifest",
        "source_combined_sha256": "source",
        "selected_physical_gpu_ids": [2, 3],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fieldnames = [
        "row",
        "task_id",
        "trial_id",
        "success",
        "num_policy_queries",
        "num_full_vlm_calls",
        "num_condition_updater_calls",
        "num_action_transformer_flow_iterations",
        "num_action_transformer_decodes",
        "fallback_full_calls",
        "episode_length",
        "normalized_second_difference",
        "short_reversal",
        "switch_disagreement",
    ]
    for rank, task_ids in ((0, range(5)), (1, range(5, 10))):
        shard = row_root / f"shard_rank{rank}_tasks_{task_ids.start}_{task_ids.stop - 1}"
        shard.mkdir(parents=True)
        (shard / "shard_summary.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "manifest_sha256": "manifest",
                    "peak_vram_bytes": 123,
                    "v0_module_parameters": 584321,
                }
            ),
            encoding="utf-8",
        )
        with (shard / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for task_id in task_ids:
                for trial_id in range(50):
                    writer.writerow(
                        {
                            "row": "native_v0_k4",
                            "task_id": task_id,
                            "trial_id": trial_id,
                            "success": True,
                            "num_policy_queries": 4,
                            "num_full_vlm_calls": 1,
                            "num_condition_updater_calls": 3,
                            "num_action_transformer_flow_iterations": 40,
                            "num_action_transformer_decodes": 4,
                            "fallback_full_calls": 0,
                            "episode_length": 20,
                            "normalized_second_difference": 0.1,
                            "short_reversal": 0.0,
                            "switch_disagreement": 0.0,
                        }
                    )
        (shard / "latency_records.jsonl").write_text("", encoding="utf-8")
        (shard / "query_metrics.jsonl").write_text("", encoding="utf-8")
    output = tmp_path / "merged"
    summary = merge_row(
        argparse.Namespace(
            output=str(output),
            row_root=str(row_root),
            manifest=str(manifest_path),
        )
    )
    assert summary["episodes"] == 500
    assert summary["v0_module_parameters"] == 584321
    assert json.loads((output / "row_summary.json").read_text())["v0_module_parameters"] == 584321


def test_official_training_cache_compaction_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "datasets"
    hdf5_path = dataset_root / "libero_10" / "task_demo.hdf5"
    hdf5_path.parent.mkdir(parents=True)
    with h5py.File(hdf5_path, "w") as h5:
        demo = h5.create_group("data/demo_0")
        obs = demo.create_group("obs")
        front = np.arange(16 * 8 * 8 * 3, dtype=np.uint8).reshape(16, 8, 8, 3)
        wrist = np.flip(front, axis=1).copy()
        obs.create_dataset("agentview_rgb", data=front)
        obs.create_dataset("eye_in_hand_rgb", data=wrist)

    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()
    records = []
    for timestep in (0, 5, 10, 15):
        raw_ref = {
            "hdf5_path": f"/obsolete/libero_10/{hdf5_path.name}",
            "demo_key": "demo_0",
            "timestep": timestep,
            "camera_names": ["agentview_rgb", "eye_in_hand_rgb"],
            "rotate_180": True,
        }
        records.append(
            {
                "episode_id": "0000:demo_0",
                "timestep": timestep,
                "language_instruction": "test instruction",
                "raw_rgb_ref": raw_ref,
                "proprio": torch.full((1, 8), float(timestep)),
                "condition": torch.full((1, 122, 960), float(timestep)),
            }
        )
    torch.save(records, teacher_root / "shard_000000.pt")
    (teacher_root / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "architecture": "simvla",
                    "checkpoint": "YuankaiLuo/SimVLA-LIBERO",
                    "extra": {"suite": "libero_10", "action_horizon": 10},
                },
                "shards": ["shard_000000.pt"],
            }
        ),
        encoding="utf-8",
    )
    norm = tmp_path / "norm.json"
    norm.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("SIMVLA_GPU_IDS", "2,3")
    compact = tmp_path / "compact"
    result = build_training_cache(
        output=compact,
        teacher_cache=teacher_root,
        dataset_root=dataset_root,
        checkpoint="YuankaiLuo/SimVLA-LIBERO",
        norm_stats=norm,
        action_noise_seed_base=20260822,
    )
    assert result["verdict"] == "NATIVE_V0_TRAINING_CACHE_BUILT"
    assert result["sequences"] == 1
    assert validate_training_cache(compact)["passed"] is True
    dataset = NativeV0SequenceDataset(compact, split="all")
    item = dataset[0]
    assert item["image_sequence"].shape == (4, 2, 8, 8, 3)
    assert item["proprio_sequence"][:, 0].tolist() == [0.0, 5.0, 10.0, 15.0]
    assert item["teacher_conditions"].shape == (3, 122, 960)
    assert item["explicit_noises"].shape == (3, 10, 7)
    assert len(dataset.contract()["split_sha256"]) == 64
