"""Resumable rb2 Condition analysis campaign; scientific negatives never stop it."""

from __future__ import annotations

import argparse
import codecs
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = "architectures.simvla.adapters.latentloop.efficient_multirate."


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def configure(config):
    env = {
        "PYTHONPATH": os.pathsep.join((str(ROOT), config["upstream"], config["libero_root"])),
        "SIMVLA_UPSTREAM_ROOT": config["upstream"], "SIMVLA_LIBERO_ROOT": config["libero_root"],
        "LIBERO_CONFIG_PATH": config["libero_config"], "HF_HOME": config["hf_home"],
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": str(config["physical_gpu"]), "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(config["physical_gpu"]), "PYTHONHASHSEED": "20260815",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NVIDIA_TF32_OVERRIDE": "0",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_CACHE_DIR": "/tmp/numba_simvla_mechanism", "MPLCONFIGDIR": "/tmp/matplotlib_simvla_mechanism",
        "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONUNBUFFERED": "1",
    }
    for name in ("GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE", "LP_NUM_THREADS", "EGL_DEVICE_ID"):
        os.environ.pop(name, None)
    os.environ.update(env)
    for path in reversed(env["PYTHONPATH"].split(os.pathsep)):
        if path not in sys.path:
            sys.path.insert(0, path)


def validate_config(c):
    if c["host"] != platform.node() or c["physical_gpu"] != 0:
        raise ValueError("This campaign is restricted to rb2 GPU0, not sd1 or another GPU")
    for field in ("calibration_windows", "heldout_windows", "intervention_trials_per_task", "continuation_actions", "gpu_wait_seconds"):
        if not isinstance(c[field], int) or c[field] < 1:
            raise ValueError(f"Invalid positive integer {field}")
    if c["intervention_trials_per_task"] > 50 or c["continuation_actions"] > 895:
        raise ValueError("Episode limits exceed the existing LIBERO-Long manifest")
    if not 0 < c["object_shift_m"] <= 0.05:
        raise ValueError("Object shift must be in (0, 0.05] meters")
    if not c["intervention_task_ids"] or any(i not in range(10) for i in c["intervention_task_ids"]):
        raise ValueError("Invalid Long task IDs")


def manifest_contract(c):
    m = json.loads(Path(c["episode_manifest"]).read_text())
    expected = {"suite": "libero_10", "action_horizon": 10, "execution_horizon": 5, "flow_steps": 10,
                "checkpoint_revision": c["checkpoint_revision"], "num_wait_steps": 10, "max_policy_actions": 900}
    for key, value in expected.items():
        if m.get(key) != value:
            raise RuntimeError(f"Manifest mismatch {key}: {m.get(key)} != {value}")
    ids = [(e["task_id"], e["trial_id"]) for e in m["episodes"]]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise RuntimeError("Expected the existing 500-episode reference manifest")
    return m


def provenance(c):
    from huggingface_hub import snapshot_download
    files = {
        "condition_checkpoint": c["condition_checkpoint"], "norm_stats": c["norm_stats"],
        "cache_manifest": str(Path(c["cache"]) / "manifest.json"), "episode_manifest": c["episode_manifest"],
    }
    hashes = {name: digest(path) for name, path in files.items()}
    for key, expected in c["expected_sha256"].items():
        if hashes[key] != expected:
            raise RuntimeError(f"Input identity mismatch: {key}")
    snapshot = snapshot_download(c["checkpoint"], revision=c["checkpoint_revision"], local_files_only=True)
    backbone = snapshot_download(c["smolvlm_model"], local_files_only=True)
    model_hashes = {}
    for prefix, directory in (("simvla", snapshot), ("smolvlm", backbone)):
        for path in sorted(Path(directory).iterdir()):
            if path.is_file() and path.suffix in (".json", ".safetensors", ".bin"):
                model_hashes[f"{prefix}/{path.name}"] = digest(path)
    source_paths = [Path(__file__), *ROOT.glob("architectures/simvla/adapters/latentloop/efficient_multirate/condition_mechanism*.py")]
    source_paths += [ROOT / p for p in (
        "methods/latentloop/modules/native_simvla_v0.py",
        "architectures/simvla/adapters/latentloop/native_v0_policy.py",
        "architectures/simvla/adapters/dcld/simvla_action_adapter.py",
        "architectures/simvla/wrappers/dcld_eval/rollout_runner.py",
        "architectures/simvla/adapters/latentloop/efficient_multirate/exact_teacher_cache.py",
        "architectures/simvla/adapters/latentloop/efficient_multirate/efficient_delta.py",
        "architectures/simvla/adapters/latentloop/efficient_multirate/latent_fidelity_analysis.py",
        "architectures/simvla/adapters/latentloop/native_v0_condition_hook.py",
        "architectures/simvla/adapters/latentloop/native_v0_runtime.py",
        "architectures/simvla/adapters/latentloop/native_v0_checkpoint.py",
    )]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    upstream_hashes = {str(p.relative_to(c["upstream"])): digest(p) for p in Path(c["upstream"]).glob("models/*.py")}
    packages = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "mujoco", "robosuite", "h5py")}
    if packages["torch"] != "2.7.1" or packages["mujoco"] != "2.3.7" or packages["transformers"] != "4.57.3":
        # Torch metadata can include the CUDA suffix depending on wheel metadata.
        if not (packages["torch"] == "2.7.1+cu128" and packages["mujoco"] == "2.3.7" and packages["transformers"] == "4.57.3"):
            raise RuntimeError(f"Unexpected runtime: {packages}")
    committed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"config": c, "input_sha256": hashes, "source_sha256": source_hashes,
            "upstream_source_sha256": upstream_hashes, "packages": packages, "git_commit": committed,
            "checkpoint_snapshot": snapshot, "backbone_snapshot": backbone, "model_sha256": model_hashes,
            "libero_source_sha256": {str(p.relative_to(c["libero_root"])): digest(p) for p in Path(c["libero_root"]).glob("libero/libero/envs/**/*.py")},
            "host": platform.node()}


def preflight(c, output):
    print("[1/4] 경로·환경·모델·cache 검증", flush=True)
    validate_config(c)
    for key in ("python", "upstream", "libero_root", "libero_config", "cache", "condition_checkpoint", "norm_stats", "episode_manifest"):
        if not Path(c[key]).exists():
            raise FileNotFoundError(f"{key}: {c[key]}")
    if os.statvfs(c["storage"]).f_bavail * os.statvfs(c["storage"]).f_frsize < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free storage is required")
    m = manifest_contract(c)
    p = provenance(c)
    identity = signature(p)
    contract = output / "metadata" / "contract.json"
    if contract.exists() and json.loads(contract.read_text())["identity"] != identity:
        raise RuntimeError("Config/source/input changed since previous run. Use a different --output; old results are preserved.")
    write_json(contract, {"identity": identity, **p})
    write_json(output / "metadata" / "episode_manifest.json", m)
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import validate_exact_cache
    cache_report = validate_exact_cache(c["cache"], verify_checksums=False)
    if not cache_report["passed"]:
        raise RuntimeError(cache_report)
    write_json(output / "metadata" / "cache_validation.json", cache_report)
    # CPU-only import and one HDF5 read catch missing linked data before waiting for a GPU.
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import make_datasets
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism_environment import InterventionPolicy
    _, payload = load_native_v0_checkpoint(c["condition_checkpoint"], device="cpu", require_final_150k=True)
    train, heldout = make_datasets(c, payload)
    sample = heldout[0]
    if sample["image_sequence"].dtype != torch.uint8:
        raise RuntimeError("Expected uint8 RGB source images")
    write_json(output / "metadata" / "dataset_splits.json", {"train": train.contract(), "heldout": heldout.contract()})
    print("PREFLIGHT_PASS: 학습/평가 split, HDF5 접근, 환경 import 확인", flush=True)
    return identity


def wait_gpu(c, output):
    while True:
        free = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        pids = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if not pids and free >= c["minimum_free_mib"]:
            return
        write_json(output / "pipeline_status.json", {"state": "WAITING_FOR_GPU", "free_mib": free, "pids": pids})
        print(f"GPU0 대기: free={free} MiB, pids={pids or '없음'}; {c['gpu_wait_seconds']}초 뒤 확인", flush=True)
        time.sleep(c["gpu_wait_seconds"])


def run_stage(stage, c, output, identity):
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism, freeze_module, load_frozen_simvla
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import run_offline
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism_environment import run_environment
    contract = json.loads((output / "metadata" / "contract.json").read_text())
    configure_strict_torch_determinism(manifest_contract(c)["determinism_seed"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(max(0.1, (total - 2 * 1024**3) / total), device)
    adapter, payload = load_native_v0_checkpoint(c["condition_checkpoint"], device=device, require_final_150k=True)
    freeze_module(adapter)
    model, processor, action = load_frozen_simvla(checkpoint=contract["checkpoint_snapshot"], norm_stats=c["norm_stats"], smolvlm_model=contract["backbone_snapshot"], device=device)
    freeze_module(model)
    if stage == "offline":
        from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
        from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import _drop_unused_vlm
        install_exact_uint8_delta_path(adapter)
        _drop_unused_vlm(model)
        run_offline(c, output / stage, identity, adapter, payload, action)
    else:
        run_environment(c, manifest_contract(c), output / stage, identity, model, processor, adapter)


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        keys = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def validate_completion(output, identity):
    """Never call a partial campaign complete solely because a summary exists."""
    expected_offline = output / "offline/summary.json"
    expected_environment = output / "environment/summary.json"
    if not expected_offline.exists() or not expected_environment.exists():
        return False
    for path in (expected_offline, expected_environment):
        data = json.loads(path.read_text())
        if data.get("identity") != identity or data.get("complete") is not True:
            raise RuntimeError(f"Incompatible summary {path}")
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import REGIMES, VARIANTS
    expected_keys = {(r, a, v) for r in REGIMES for a in (1, 2, 3) for v in VARIANTS}
    units = list((output / "offline/units").glob("*.json"))
    offline = json.loads(expected_offline.read_text())
    if len(units) != offline["windows"]:
        raise RuntimeError("Offline result units are missing")
    for path in units:
        data = json.loads(path.read_text())
        keys = {(r["regime"], r["age"], r["variant"]) for r in data["rows"]}
        if keys != expected_keys or len(data["rows"]) != len(expected_keys):
            raise RuntimeError(f"Missing or duplicate intervention rows: {path}")
    selection = json.loads((output / "environment/selection.json").read_text())
    for case in selection["cases"]:
        directory = output / "environment/units" / f"task_{case['task_id']:02d}_trial_{case['trial_id']:02d}"
        info = json.loads((directory / "case.json").read_text())
        if info.get("identity") != identity or info.get("complete") is not True:
            raise RuntimeError("Environment case identity changed")
        if info.get("invalid_reason"):
            continue
        for world in ("nominal", "displaced"):
            for method in ("baseline", "hold", "zero_feature", "full_update"):
                p = directory / f"{world}_{method}.json"
                if not p.exists():
                    raise RuntimeError(f"Missing environment branch {p}")
    return True


def recover_stage_summary(output, stage, identity):
    """Repair a final-write failure from complete units, without loading a model."""
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import REGIMES, VARIANTS, completed_unit
    directory = output / stage
    final = directory / "summary.json"
    if final.exists():
        return completed_unit(final, identity) is not None
    selection = directory / "selection.json"
    if not selection.exists():
        return False
    selected = json.loads(selection.read_text())
    if selected["identity"] != identity:
        raise RuntimeError("Cannot recover a foreign selection")
    result = {"identity": identity, "complete": True, "recovered_from_units_without_gpu": True}
    if stage == "offline":
        units = list((directory / "units").glob("*.json"))
        if len(units) != len(selected["heldout"]):
            return False
        expected_keys = {(r, a, v) for r in REGIMES for a in (1, 2, 3) for v in VARIANTS}
        for path in units:
            unit = completed_unit(path, identity)
            keys = {(r["regime"], r["age"], r["variant"]) for r in unit["rows"]}
            if keys != expected_keys or len(unit["rows"]) != len(expected_keys):
                return False
        result.update(windows=len(units), rows=len(units) * len(expected_keys))
    else:
        invalid = branches = 0
        for case in selected["cases"]:
            unit = directory / "units" / f"task_{case['task_id']:02d}_trial_{case['trial_id']:02d}"
            record = completed_unit(unit / "case.json", identity)
            if record is None:
                return False
            if record.get("invalid_reason"):
                invalid += 1
                continue
            for world in ("nominal", "displaced"):
                for method in ("baseline", "hold", "zero_feature", "full_update"):
                    if completed_unit(unit / f"{world}_{method}.json", identity) is None:
                        return False
                    branches += 1
        result.update(planned_cases=len(selected["cases"]), invalid_cases=invalid, completed_branches=branches)
    write_json(final, result)
    return True


def aggregate(output, identity):
    import numpy as np
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import action_metrics
    import torch

    report = output / "report"
    report.mkdir(exist_ok=True)
    rows = []
    for path in sorted((output / "offline" / "units").glob("*.json")):
        unit = json.loads(path.read_text())
        if unit["identity"] != identity or not unit["complete"]:
            raise RuntimeError(f"Invalid unit: {path}")
        rows.extend(unit["rows"])
    csv_file(report / "offline_all_rows.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["age"], row["variant"])].append(row)
    summary = []
    metric_keys = ("first5_action_l1", "translation_l1", "rotation_l1", "continuous_gripper_l1", "gripper_sign_disagreement",
                   "condition_mse", "condition_cosine", "updater_layernorm_mse", "action_input_projection_mse", "gate_mean", "residual_rms", "update_rms")
    for (regime, age, variant), values in sorted(groups.items()):
        entry = {"regime": regime, "age": age, "variant": variant, "windows": len(values)}
        for key in metric_keys:
            entry[key + "_mean"] = float(np.mean([r[key] for r in values]))
            entry[key + "_p95"] = float(np.quantile([r[key] for r in values], .95))
        summary.append(entry)
    csv_file(report / "offline_summary.csv", summary)
    branches, responses, cases = [], [], []
    for case_path in sorted((output / "environment" / "units").glob("*/case.json")):
        case = json.loads(case_path.read_text())
        cases.append({k: case[k] for k in ("task_id", "trial_id", "target", "invalid_reason")})
        for method in ("baseline", "hold", "zero_feature", "full_update"):
            pair = {}
            for world in ("nominal", "displaced"):
                path = case_path.parent / f"{world}_{method}.json"
                if not path.exists():
                    continue
                value = json.loads(path.read_text())
                if value["identity"] != identity or not value["complete"]:
                    raise RuntimeError(f"Invalid branch {path}")
                pair[world] = value
                branches.append({k: value[k] for k in ("task_id", "trial_id", "method", "world", "success", "episode_actions", "seconds", "full_episode_horizon", "success_immediately_after_intervention")})
            if len(pair) == 2 and all(p["first_action_chunk"] is not None for p in pair.values()):
                a, b = pair["nominal"], pair["displaced"]
                if a["proprio_sha256"] != b["proprio_sha256"] or a["previous_condition_sha256"] != b["previous_condition_sha256"]:
                    raise RuntimeError("Paired response does not share proprioception and previous condition")
                if a["query_trace"][0]["noise_seed"] != b["query_trace"][0]["noise_seed"]:
                    raise RuntimeError("Paired response action noise differs")
                response = action_metrics(torch.tensor([a["first_action_chunk"]]), torch.tensor([b["first_action_chunk"]]))
                if method in ("hold", "zero_feature") and response["first5_action_l1"] > 1e-6:
                    raise RuntimeError("An image-independent first query changed its action under an image-only intervention")
                baseline_pair = {}
                for world in ("nominal", "displaced"):
                    baseline_path = case_path.parent / f"{world}_baseline.json"
                    if baseline_path.exists():
                        baseline_pair[world] = json.loads(baseline_path.read_text())
                if len(baseline_pair) == 2 and all(r["first_action_chunk"] is not None for r in baseline_pair.values()):
                    nominal = np.asarray(a["first_action_chunk"])
                    displaced = np.asarray(b["first_action_chunk"])
                    original_nominal = np.asarray(baseline_pair["nominal"]["first_action_chunk"])
                    original_displaced = np.asarray(baseline_pair["displaced"]["first_action_chunk"])
                    change = (displaced[:5] - nominal[:5]).reshape(-1)
                    original_change = (original_displaced[:5] - original_nominal[:5]).reshape(-1)
                    denominator = np.linalg.norm(change) * np.linalg.norm(original_change)
                    response.update(
                        baseline_response_l1=float(np.abs(original_change).mean()),
                        response_alignment_cosine=None if denominator < 1e-12 else float(np.dot(change, original_change) / denominator),
                        response_error_vs_baseline_l1=float(np.abs(change - original_change).mean()),
                        nominal_first5_error_vs_baseline=float(np.abs(nominal[:5] - original_nominal[:5]).mean()),
                        displaced_first5_error_vs_baseline=float(np.abs(displaced[:5] - original_displaced[:5]).mean()),
                    )
                responses.append({"task_id": case["task_id"], "trial_id": case["trial_id"], "method": method,
                                  "same_proprio": a["proprio_sha256"] == b["proprio_sha256"],
                                  "same_previous_condition": a["previous_condition_sha256"] == b["previous_condition_sha256"],
                                  "image_changed": a["first_obs_sha256"] != b["first_obs_sha256"], **response})
    csv_file(report / "intervention_episode_outcomes.csv", branches)
    csv_file(report / "intervention_all_cases.csv", cases)
    csv_file(report / "paired_first_query_responses.csv", responses)
    complete = validate_completion(output, identity)
    text = ["# SimVLA Condition 원인 분석", "", f"완료 상태: {'모든 단계 완료' if complete else '일부 단계 미완료: pipeline_status.json 확인'}", "",
            "## 설정과 해석 범위", "- 학습 없이 기존 Condition 150K와 공식 SimVLA checkpoint를 고정했다.",
            "- 모든 action 비교는 NFE=10, H=10, R=5다. Generation updater는 사용하지 않았다.",
            "- held-out window는 Condition 학습 checkpoint의 episode 분할과 일치한다.",
            "- 고정/age-only 보정량은 학습 episode에서 zero-feature update를 평균했다. 평가 정답으로 맞추지 않았다.",
            "- shared_teacher_previous는 같은 이전 teacher condition, recursive는 방법별 이전 예측을 사용한다.",
            "- condition 표준화 오차는 진단 지표다. decoder가 그 표준화를 수행한다고 해석하면 안 된다.",
            "- 환경 개입은 selection.json에 고정된 사례의 탐색 분석이다. 기존 500-episode 논문 SR이나 3-seed 평균을 대체하지 않는다.",
            "- 물체 변위는 첫 5개 action 후 주며, 양쪽에 동일한 prefix action을 실행하고 simulator state를 비교한다.",
            "- baseline이 회복하지 못한 사례, 관측 반응이 없는 사례도 제외하지 않는다.",
            "- 두 세계 간 첫 query action 차이는 반응의 크기이지 올바른 수정 방향 또는 인과 기여도의 증명은 아니다.",
            "- 반응 방향 cosine은 같은 상황의 baseline 반응과 비교한다. 정답 행동이라고 가정하지 않으며 무반응일 때는 빈 값이다.",
            "- 계측 중 wall time은 논문용 policy latency가 아니다.", "", "## 동일 입력 첫 갱신", "",
            "| 방법 | 첫 5 action L1 | Condition cosine | Gate 평균 |", "|---|---:|---:|---:|"]
    for row in summary:
        if row["regime"] == "shared_teacher_previous" and row["age"] == 1:
            text.append(f"| {row['variant']} | {row['first5_action_l1_mean']:.6f} | {row['condition_cosine_mean']:.6f} | {row['gate_mean_mean']:.6f} |")
    text += ["", "## 환경 개입 결과", "", "| 방법 | 환경 | 성공 / 완료 분기 |", "|---|---|---:|"]
    for method in ("baseline", "hold", "zero_feature", "full_update"):
        for world in ("nominal", "displaced"):
            selected = [r for r in branches if r["method"] == method and r["world"] == world]
            if selected:
                text.append(f"| {method} | {world} | {sum(r['success'] for r in selected)} / {len(selected)} |")
    invalid_cases = [r for r in cases if r["invalid_reason"]]
    text += ["", f"개입 불가 사례: {len(invalid_cases)} / {len(cases)}. 사유는 intervention_all_cases.csv에 전부 기록했다.",
             "물체가 이미 로봇과 접촉했거나 이동 가능한 task 물체가 없는 사례는 결과를 만들지 않고 별도 표시한다."]
    text += ["", "## 다음 판단", "- zero-feature update가 유지보다 action 오차도 줄이는지 확인한다.",
             "- 단순 고정 보정이 비슷하면 동역학 학습으로 과장하지 않는다.",
             "- gate 교체로 오차가 복구되는지와 최신 이미지 경로의 효과를 구분한다.",
             "- 외부 변화에 대한 baseline 반응, Ours 반응, 최종 성공을 함께 본다.",
             "- 현재 결과만으로 시각 정보 불필요, 새 논문 기여 입증, Generation Loop 무용을 결론내리지 않는다."]
    (report / "analysis_report_ko.md").write_text("\n".join(text) + "\n")
    write_json(report / "summary.json", {"identity": identity, "complete": complete, "offline_rows": len(rows), "environment_branches": len(branches)})
    package = output / "simvla_condition_mechanism_results.zip"
    with zipfile.ZipFile(package.with_suffix(".tmp"), "w", compression=zipfile.ZIP_DEFLATED) as z:
        for folder in ("report", "metadata", "offline", "environment"):
            for path in sorted((output / folder).rglob("*")):
                if path.is_file() and path.suffix in (".json", ".csv", ".md", ".png"):
                    z.write(path, path.relative_to(output))
    package.with_suffix(".tmp").replace(package)
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "architectures/simvla/configs/condition_mechanism_rb2.json"))
    parser.add_argument("--output")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", choices=("offline", "environment"))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    if args.output:
        c["output"] = args.output
    if args.smoke:
        c.update(calibration_windows=2, heldout_windows=1, intervention_task_ids=[0], intervention_trials_per_task=1, continuation_actions=15)
        if not args.output:
            c["output"] += "_smoke"
    configure(c)
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    status = output / "pipeline_status.json"
    if args.stage:
        contract = json.loads((output / "metadata/contract.json").read_text())
        run_stage(args.stage, c, output, contract["identity"])
        return 0
    lock_path = Path(c["storage"]) / "locks/simvla_condition_mechanism.lock"
    lock_path.parent.mkdir(exist_ok=True)
    child = None
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("이미 같은 분석이 실행 중입니다.", flush=True)
            return 2
        def interrupt(signum, frame):
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            write_json(status, {"state": "INTERRUPTED", "signal": signum, "completed_units_preserved": True})
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        try:
            identity = preflight(c, output)
            if args.preflight:
                write_json(status, {"state": "PREFLIGHT_PASS", "identity": identity})
                return 0
            failed = []
            for stage in ("offline", "environment"):
                recover_stage_summary(output, stage, identity)
            if not args.aggregate_only:
                for number, stage in ((2, "offline"), (3, "environment")):
                    final = output / stage / "summary.json"
                    if final.exists():
                        value = json.loads(final.read_text())
                        if value.get("identity") == identity and value.get("complete"):
                            print(f"[{number}/4] {stage}: 완료 결과 재사용", flush=True)
                            continue
                    for attempt in range(c["stage_retry_limit"] + 1):
                        wait_gpu(c, output)
                        print(f"[{number}/4] {stage} 시작 (시도 {attempt + 1})", flush=True)
                        write_json(status, {"state": "RUNNING", "stage": stage, "attempt": attempt + 1, "identity": identity})
                        cmd = [c["python"], str(Path(__file__)), "--config", str(Path(args.config).resolve()), "--output", str(output), "--stage", stage]
                        if args.smoke:
                            cmd.append("--smoke")
                        with (output / "logs" / f"{stage}.log").open("a") as log:
                            log.write("\nCOMMAND " + json.dumps(cmd) + "\n")
                            log.flush()
                            child = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                            # Preserve tqdm carriage returns at the terminal while
                            # keeping only complete log lines on disk.
                            pending = ""
                            decoder = codecs.getincrementaldecoder("utf-8")("replace")
                            while True:
                                chunk = os.read(child.stdout.fileno(), 4096)
                                if not chunk:
                                    break
                                sys.stdout.buffer.write(chunk)
                                sys.stdout.buffer.flush()
                                pending += decoder.decode(chunk)
                                while "\n" in pending:
                                    line, pending = pending.split("\n", 1)
                                    log.write(line.rsplit("\r", 1)[-1] + "\n")
                                    log.flush()
                            rc = child.wait()
                            child = None
                        if recover_stage_summary(output, stage, identity):
                            break
                        print(f"{stage} 실행 오류 rc={rc}. 완료 단위는 보존합니다.", flush=True)
                    else:
                        failed.append(stage)
            print("[4/4] 전체 CSV·한국어 보고서·ZIP 생성", flush=True)
            complete = aggregate(output, identity)
            state = "COMPLETE" if complete and not failed else "INCOMPLETE"
            write_json(status, {"state": state, "failed_stages": failed, "identity": identity, "output": str(output)})
            print(f"{state}: {output}/report/analysis_report_ko.md", flush=True)
            print(f"ZIP: {output}/simvla_condition_mechanism_results.zip", flush=True)
            return 0 if state == "COMPLETE" else 1
        except KeyboardInterrupt:
            return 130
        except Exception:
            write_json(status, {"state": "ERROR", "traceback": traceback.format_exc()})
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
