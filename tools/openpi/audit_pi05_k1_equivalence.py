#!/usr/bin/env python3
"""GPU/user-run hard gate for full-prefix extraction/rebuild and K_q=1 bypass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from _common import (
    DEFAULT_CHECKPOINT,
    load_local_policy,
    refuse_nonempty_output,
    require_run,
    require_source_lock_v2,
)
from architectures.openpi.adapters.latentloop.policy_io import (
    explicit_policy_noise,
    postprocess_policy_actions,
    prepare_policy_observation,
)
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook, tensor_sha256
from architectures.openpi.adapters.latentloop.recurrent_policy import OpenPILatentLoopPolicy
from architectures.openpi.adapters.latentloop.transition_core import OpenPIKVLatentLoop


def _rng_snapshot(device: torch.device) -> dict[str, str]:
    result = {"cpu": tensor_sha256(torch.random.get_rng_state())}
    if device.type == "cuda":
        result["cuda"] = tensor_sha256(torch.cuda.get_rng_state(device))
    return result


def _real_dataset_observation(policy, index: int):
    from openpi.training import config as config_api
    from openpi.training import data_loader

    config = config_api.get_config("pi05_libero_lora_pytorch")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    sample = dataset[index]

    def image_uint8(value):
        array = np.asarray(value)
        if array.shape[0] == 3:
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
        return array

    raw = {
        "observation/image": image_uint8(sample["image"]),
        "observation/wrist_image": image_uint8(sample["wrist_image"]),
        "observation/state": np.asarray(sample["state"]),
        "prompt": str(sample["prompt"]),
    }
    return prepare_policy_observation(policy, raw)[0]


def _postprocess(policy, observation, actions):
    return postprocess_policy_actions(policy, observation.to_dict()["state"], actions)["actions"]


def _observation_hash(observation) -> str:
    digest = hashlib.sha256()
    payload = observation.to_dict()
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, dict):
            for inner_key in sorted(value):
                digest.update(f"{key}/{inner_key}".encode("utf-8"))
                digest.update(tensor_sha256(torch.as_tensor(value[inner_key])).encode("ascii"))
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            digest.update(key.encode("utf-8"))
            digest.update(tensor_sha256(torch.as_tensor(value)).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-lock")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=20260820)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--episode-smoke-json")
    parser.add_argument("--tensor-report")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()

    if args.merge_only:
        if not args.tensor_report:
            raise ValueError("--merge-only requires --tensor-report and a new --output directory")
        output = refuse_nonempty_output(output)
        report_path = Path(args.tensor_report).resolve()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        source_lock_verification = require_source_lock_v2(payload["source_lock"])
        if payload.get("source_lock_id") != source_lock_verification["source_lock_id"]:
            raise RuntimeError("source mismatch: tensor report is stale")
    else:
        require_run(args.run, "OPENPI_LATENTLOOP_K1_RUN")
        if not args.source_lock:
            raise ValueError("--source-lock is required for the tensor parity audit")
        source_lock_verification = require_source_lock_v2(args.source_lock)
        source_lock = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
        if Path(source_lock["checkpoint"]["directory"]).resolve() != Path(args.checkpoint).resolve():
            raise ValueError("K1 checkpoint does not match the frozen source lock")
        output = refuse_nonempty_output(output)
        device = torch.device(args.device)
        policy = load_local_policy(args.checkpoint, args.device, flow_steps=args.flow_steps)
        model = policy._model  # noqa: SLF001
        observation = _real_dataset_observation(policy, args.dataset_index)
        noise = explicit_policy_noise(
            (1, model.config.action_horizon, model.config.action_dim),
            seed=args.noise_seed,
            device=device,
        )
        hook = PrefixKVHook(model)
        # Keep an adapter resident while proving that the K_q=1 branch bypasses it.
        # This is the pre-training form of the adapter-loaded parity gate.
        bypass_adapter = OpenPIKVLatentLoop().to(device).eval()
        before = _rng_snapshot(device)
        sampler = getattr(model.sample_actions, "_torchdynamo_orig_callable", model.sample_actions)
        with torch.no_grad():
            path_a = sampler(device, observation, noise=noise, num_steps=args.flow_steps)
            after_a = _rng_snapshot(device)

            extraction_a = hook.extract(observation)
            extraction_b = hook.extract(observation)
            cache = hook.cache_allclose(
                extraction_b.state,
                extraction_b.source_cache,
                atol=args.atol,
                rtol=args.rtol,
            )
            path_b, timing = hook.sample_actions_from_state(
                extraction_b.state,
                extraction_b.robot_state,
                noise,
                num_steps=args.flow_steps,
            )
            after_b = _rng_snapshot(device)
            k1 = OpenPILatentLoopPolicy(
                model,
                bypass_adapter,
                k_q=1,
                num_flow_steps=args.flow_steps,
            )
            path_c = k1.query(observation, noise).normalized_actions
            after_c = _rng_snapshot(device)

            post_a, post_b, post_c = (
                _postprocess(policy, observation, value) for value in (path_a, path_b, path_c)
            )
        embedding_equal = torch.equal(extraction_a.state.embeddings, extraction_b.state.embeddings)
        action_ab = torch.equal(path_a, path_b)
        action_ac = torch.equal(path_a, path_c)
        post_ab = np.array_equal(post_a, post_b)
        post_ac = np.array_equal(post_a, post_c)
        rng_unchanged = before == after_a == after_b == after_c
        action_parity_pass = bool(
            embedding_equal
            and action_ab
            and action_ac
            and post_ab
            and post_ac
            and k1.runtime.updater_calls == 0
            and rng_unchanged
        )
        tensor_pass = bool(
            cache["passed"]
            and action_parity_pass
        )
        payload = {
            "schema_version": 2,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "source_lock": str(Path(args.source_lock).resolve()),
            "source_lock_id": source_lock_verification["source_lock_id"],
            "markers": sorted(
                (["REAL_KV_ROUNDTRIP_PASS"] if cache["passed"] else [])
                + (["K1_ACTION_PARITY_PASS"] if action_parity_pass else [])
            ),
            "REAL_KV_ROUNDTRIP_PASS": bool(cache["passed"]),
            "K1_ACTION_PARITY_PASS": action_parity_pass,
            "dataset_index": args.dataset_index,
            "explicit_noise_seed": args.noise_seed,
            "explicit_noise_hash": tensor_sha256(noise),
            "observation_hash": _observation_hash(observation),
            "flow_steps": args.flow_steps,
            "tensor_parity_pass": tensor_pass,
            "prefix_embedding_equal": embedding_equal,
            "prefix_embedding_hash": tensor_sha256(extraction_a.state.embeddings),
            "prefix_pad_mask_hash": tensor_sha256(extraction_a.state.pad_mask),
            "prefix_position_ids_hash": tensor_sha256(extraction_a.state.position_ids),
            "cache_rebuild": cache,
            "normalized_action_exact_a_b": action_ab,
            "normalized_action_exact_a_c": action_ac,
            "normalized_action_max_abs_a_b": float((path_a - path_b).abs().max().item()),
            "postprocessed_action_exact_a_b": post_ab,
            "postprocessed_action_exact_a_c": post_ac,
            "updater_calls_k1": k1.runtime.updater_calls,
            "adapter_resident_during_k1": True,
            "adapter_trainable_parameters": bypass_adapter.trainable_parameters,
            "k1_ours_call_counters": {
                "latentloop_sequential_calls": k1.runtime.updater_calls,
                "latentloop_direct_calls": 0,
                "direct_reanchor_events": 0,
            },
            "rng_unchanged": rng_unchanged,
            "timing_ms": timing,
            "episode_smoke": None,
            "hard_gate_pass": False,
        }

    if args.episode_smoke_json:
        smoke = json.loads(Path(args.episode_smoke_json).read_text(encoding="utf-8"))
        smoke_pass = bool(
            smoke.get("complete")
            and smoke.get("tasks") == 1
            and smoke.get("episodes") == 2
            and smoke.get("all_query_actions_exact")
            and smoke.get("paired_outcomes_identical")
            and smoke.get("updater_calls") == 0
        )
        payload["episode_smoke"] = smoke
        payload["episode_smoke_pass"] = smoke_pass
        payload["hard_gate_pass"] = bool(payload.get("tensor_parity_pass") and smoke_pass)
        payload["K1_EPISODE_PARITY_PASS"] = smoke_pass
        markers = set(payload.get("markers", []))
        if smoke_pass:
            markers.add("K1_EPISODE_PARITY_PASS")
        payload["markers"] = sorted(markers)

    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "pi05_k1_equivalence.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = output / "pi05_k1_equivalence_report.md"
    markdown.write_text(
        "\n".join(
            (
                "# pi0.5 K1 equivalence report",
                "",
                f"- Tensor parity: `{payload.get('tensor_parity_pass')}`",
                f"- Real 1-task x 2-episode smoke: `{payload.get('episode_smoke_pass', False)}`",
                f"- Hard gate: `{payload.get('hard_gate_pass')}`",
                f"- K1 updater calls: `{payload.get('updater_calls_k1')}`",
                f"- Normalized action A/B exact: `{payload.get('normalized_action_exact_a_b')}`",
                f"- Postprocessed action A/B exact: `{payload.get('postprocessed_action_exact_a_b')}`",
                "",
                "Cache generation and training wrappers require `hard_gate_pass=true`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(json_path)
    if not payload.get("hard_gate_pass"):
        print("K1 hard gate remains closed", file=sys.stderr)


if __name__ == "__main__":
    main()
