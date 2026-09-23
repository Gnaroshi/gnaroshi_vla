"""Export every training episode's unchanged step-0000 camera pair."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(dataset, output, expected_episodes=40, data_info=None):
    dataset, output = Path(dataset), Path(output)
    episodes = sorted(p for p in dataset.glob("*/*") if (p / "steps").is_dir())
    if len(episodes) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {len(episodes)}")
    if data_info:
        info = json.loads(Path(data_info).read_text())
        if {str(p.relative_to(dataset)) for p in episodes} != {record[0] for record in info}:
            raise ValueError("Dataset episode IDs differ from training data_info")
    sampling_file = dataset / "episode_sampling_manifest.json"
    sampling = json.loads(sampling_file.read_text())
    names = {row["new_episode_name"]: row["original_episode_name"]
             for row in sampling["sampled_mapping"]}
    if set(names) != {episode.name for episode in episodes} or len(names) != len(episodes):
        raise ValueError("Sampling manifest does not cover every training episode")
    samples, instruction = [], None
    for episode in episodes:
        step = episode / "steps/0000"
        with np.load(step / "other.npz", allow_pickle=True) as payload:
            value = payload["language_instruction"].item()
            value = value.decode() if isinstance(value, bytes) else str(value)
        if instruction is not None and instruction != value:
            raise ValueError("Training instructions differ between episodes")
        instruction = value
        sample = {"dataset_episode": str(episode.relative_to(dataset)),
                  "original_episode": names[episode.name], "step": "0000"}
        for camera in ("primary", "wrist"):
            source = step / f"image_{camera}.jpg"
            with Image.open(source) as image:
                if image.mode != "RGB" or image.size != (640, 480):
                    raise ValueError(f"Unexpected source image format: {source}")
                image.verify()
            filename = f"episode_{episode.parent.name}_{episode.name}_step_0000_{camera}.jpg"
            sample[camera] = {"filename": filename, "sha256": digest(source),
                              "source_relative_path": str(source.relative_to(dataset))}
        samples.append(sample)
    output.mkdir(parents=True, exist_ok=False)
    for sample in samples:
        for camera in ("primary", "wrist"):
            entry = sample[camera]
            shutil.copyfile(dataset / entry["source_relative_path"], output / entry["filename"])
            assert digest(output / entry["filename"]) == entry["sha256"]
    manifest = {
        "schema_version": 2, "task": dataset.name, "instruction": instruction,
        "episode_count": len(samples), "source_host": socket.gethostname(),
        "source_dataset_root": str(dataset), "selection_rule": "All training episodes, step 0000; byte-identical JPEG copies; no flip, crop or color conversion",
        "sampling_manifest_sha256": digest(sampling_file),
        "data_info_path": str(data_info) if data_info else None,
        "data_info_sha256": digest(Path(data_info)) if data_info else None,
        "samples": samples,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"task": dataset.name, "episodes": len(samples),
                      "images": 2*len(samples), "instruction": instruction, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-info", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=40)
    args = parser.parse_args()
    export(args.dataset, args.output, args.expected_episodes, args.data_info)
