"""Frozen SimVLA eager/Inductor comparison on identical recorded query windows.

This is not a LIBERO rollout and must not replace paper policy ms/action. CPU
image processing, simulation, queues and network transport are outside timing.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "architectures/simvla/configs/compile_benchmark_rb2.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def source_identity(c):
    directories = [ROOT / "architectures/simvla/adapters/latentloop",
        ROOT / "architectures/simvla/adapters/dcld", ROOT / "methods/latentloop",
        Path(c["upstream"]) / "models",
        Path(c["bridge_adapter_root"]) / "architectures/simvla/adapters/latent_bridge",
        Path(c["vla_cache_adapter_root"]) / "architectures/simvla/adapters/vla_cache",
        Path(c["bridge_upstream"]) / "qcvla/model"]
    files = {str(p.resolve()): sha(p) for directory in directories for p in directory.rglob("*.py")}
    for p in (Path(__file__), ROOT / "tools/simvla/compile_runtime.py"):
        files[str(p.resolve())] = sha(p)
    return files


def configure(c):
    os.environ["SIMVLA_UPSTREAM_ROOT"] = c["upstream"]
    os.environ["HF_HOME"] = c["hf_home"]
    os.environ["LATENT_BRIDGE_UPSTREAM_ROOT"] = c["bridge_upstream"]
    os.environ.setdefault("USE_TF", "0")
    for path in (str(ROOT), c["upstream"]):
        if path not in sys.path:
            sys.path.insert(0, path)
    import architectures.simvla.adapters as adapters
    adapters.__path__ = list(adapters.__path__)
    for key in ("bridge_adapter_root", "vla_cache_adapter_root"):
        path = str(Path(c[key]) / "architectures/simvla/adapters")
        if path not in adapters.__path__:
            adapters.__path__.append(path)


def preflight(c):
    required = [c[key] for key in (
        "python", "upstream", "norm_stats", "condition_checkpoint", "generation_checkpoint",
        "bridge_checkpoint", "bridge_adapter_root", "vla_cache_adapter_root", "bridge_upstream",
    )]
    required.append(str(Path(c["cache"]) / "manifest.json"))
    snapshot = Path(c["hf_home"]) / "hub/models--YuankaiLuo--SimVLA-LIBERO/snapshots" / c["checkpoint_revision"]
    required.append(str(snapshot / "config.json"))
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    if shutil.disk_usage(c["storage"]).free < 10 * 1024**3:
        raise RuntimeError("At least 10 GiB free storage is required for compiler caches")
    configure(c)
    from architectures.simvla.adapters.latent_bridge.checkpoint import load_bridge_checkpoint
    from architectures.simvla.adapters.vla_cache.smolvlm_runtime import SimVLAVLACacheBackbone
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import ExactTeacherSequenceDataset
    from tools.simvla.compile_runtime import Compiler
    import torch
    return {"verdict": "CPU_PREFLIGHT_PASS", "torch": torch.__version__,
            "checkpoint_snapshot": str(snapshot), "gpu_jobs_started": False}


def prepare_inputs(c, output):
    import torch
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import ExactTeacherSequenceDataset
    from architectures.simvla.adapters.latentloop.efficient_multirate.latent_fidelity_analysis import _balanced_indices
    from architectures.simvla.adapters.latentloop.native_v0_prepare import _official_training_image_inputs
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    path = output / "recorded_inputs.pt"
    if path.exists():
        return
    dataset = ExactTeacherSequenceDataset(c["cache"], split="heldout")
    selected = _balanced_indices(dataset.identities, limit=c["windows"], seed=c["seed"])
    dataset.store._loaded.clear()
    processor = SmolVLMVLAProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    samples = []
    for index in selected:
        sequence = dataset[index]
        prompt = sequence["language_instruction"]
        text = processor.encode_language([prompt])
        relevance = processor.tokenizer([prompt], return_tensors="pt", padding="max_length",
            max_length=processor.language_max_length, truncation=True)["attention_mask"]
        queries = []
        for q in range(4):
            processed = _official_training_image_inputs(sequence["image_sequence"][q],
                image_size=processor.image_size, num_views=processor.num_views)
            queries.append({**processed, "input_ids": text["input_ids"],
                "text_attention_mask": relevance,
                "raw_rgb": sequence["image_sequence"][q:q + 1],
                "proprio": sequence["proprio_sequence"][q:q + 1],
                "noise": sequence["explicit_noises"][max(0, q - 1):max(0, q - 1) + 1]})
        samples.append({"identity": dataset.identities[index], "queries": queries,
            "valid_mask": sequence["valid_mask"].unsqueeze(0),
            "group_ids": sequence["group_ids"].unsqueeze(0)})
        dataset.store._loaded.clear()
        print(f"INPUT {len(samples)}/{len(selected)} task={sequence['task_id']}", flush=True)
    torch.save(samples, path)
    write_json(output / "input_contract.json", {
        "sha256": sha(path), "samples": len(samples), "queries_per_sample": 4,
        "identities": [s["identity"] for s in samples],
        "cache_manifest_sha256": sha(Path(c["cache"]) / "manifest.json"),
        "preprocessing": "existing no-augmentation cache transform; excluded from timing",
        "noise": "cached explicit noise; anchor q0 reuses q1 noise, identically for all rows",
        "scope": "recorded teacher observations, not live policy trajectories or success evaluation",
    })


class Replay:
    def __init__(self, c, row, compiler, samples):
        import torch
        from architectures.simvla.adapters.latentloop.native_v0_runtime import load_frozen_simvla, freeze_module
        from tools.simvla.compile_runtime import ActionStep, compile_bridge_predict_next

        self.torch, self.row, self.compiler = torch, row, compiler
        checkpoint = str(Path(c["hf_home"]) / "hub/models--YuankaiLuo--SimVLA-LIBERO/snapshots" / c["checkpoint_revision"])
        self.model, _, self.action = load_frozen_simvla(checkpoint=checkpoint,
            norm_stats=c["norm_stats"], smolvlm_model="HuggingFaceTB/SmolVLM-500M-Instruct",
            device=torch.device("cuda"))
        self.step = ActionStep(self.model.transformer).eval()
        first = samples[0]["queries"][0]
        condition = self.model.forward_vlm_efficient(first["image_input"], first["image_mask"], first["input_ids"])["vlm_features"]
        args = dict(vlm_features=condition, action_with_noise=first["noise"],
            proprio=self.action.normalize_proprio(first["proprio"]), t=first["noise"].new_ones(1))
        expected = self.model.transformer(**args)
        hidden, actual = self.step(**args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.hookless_parity = {"bitwise_velocity_equal": True, "hidden_shape": list(hidden.shape)}
        self.step.forward = compiler.wrap("action_transformer", self.step.forward)
        self.step.decoder.forward = compiler.wrap("action_decoder", self.step.decoder.forward)
        self.model.transformer.forward = lambda **kwargs: self.step(**kwargs)[1]
        self.native = self.loop = self.bridge = self.cache = self.hook = None

        if row in ("condition_naive3", "condition_nfe10", "ours_kc2_ng3"):
            from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
            from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
            self.native, _ = load_native_v0_checkpoint(c["condition_checkpoint"], device="cuda", require_final_150k=True)
            freeze_module(self.native)
            install_exact_uint8_delta_path(self.native)
            self.native.delta_encoder.forward = compiler.wrap("observation_encoder", self.native.delta_encoder.forward)
            self.native.condition_updater.forward = compiler.wrap("condition_updater", self.native.condition_updater.forward)
        if row in ("ours_kc2_ng3", "generation_ng3"):
            from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import load_generation_checkpoint
            from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop
            updater, _ = load_generation_checkpoint(c["generation_checkpoint"], device="cuda")
            freeze_module(updater)
            updater.forward = compiler.wrap("generation_updater", updater.forward)
            self.loop = SimVLAGenerationLoop(updater, self.step.decoder).eval()
        if row == "latent_bridge_f2":
            from architectures.simvla.adapters.latent_bridge.checkpoint import load_bridge_checkpoint
            from architectures.simvla.adapters.latent_bridge.condition_hook import SimVLAConditionWithStableHook
            self.bridge, payload = load_bridge_checkpoint(c["bridge_checkpoint"], device="cuda")
            self.bridge = self.bridge.to(dtype=torch.bfloat16).eval()
            freeze_module(self.bridge)
            if self.bridge.config.token_mode != "all":
                raise ValueError("The paper's full-token Large bridge checkpoint is required")
            self.hook = SimVLAConditionWithStableHook(self.model,
                stable_layer_index=self.bridge.config.stable_layer_index)
            compile_bridge_predict_next(self.bridge, compiler)
        if row == "vla_cache":
            from architectures.simvla.adapters.vla_cache.smolvlm_runtime import SimVLAVLACacheBackbone
            from architectures.simvla.adapters.vla_cache.official_contract import VLACacheConfig
            self.cache = SimVLAVLACacheBackbone(self.model, VLACacheConfig(),
                enable_reuse=True, optimized=True, diagnostics=False)
            self.cache.encode_condition = compiler.wrap("vla_cache_backbone", self.cache.encode_condition)
        else:
            self.model.forward_vlm_efficient = compiler.wrap("vlm", self.model.forward_vlm_efficient)

    def __call__(self, sample):
        from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
        from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import GENERATION_SCHEDULES
        torch = self.torch
        previous = condition = stable = chunk = None
        outputs = []
        if self.cache is not None:
            self.cache.reset()
        for q, batch in enumerate(sample["queries"]):
            if self.cache is not None:
                condition = self.cache.encode_condition(input_ids=batch["input_ids"],
                    image_input=batch["image_input"], image_mask=batch["image_mask"],
                    text_attention_mask=batch["text_attention_mask"])
            elif q % 2 and self.native is not None:
                pair = NativeV0ObservationPair(previous["raw_rgb"], batch["raw_rgb"],
                    previous["proprio"], batch["proprio"])
                code = self.native.delta_encoder(pair)
                condition = self.native.condition_updater(condition, code,
                    valid_mask=sample["valid_mask"], group_ids=sample["group_ids"], age=1).condition
            elif q % 2 and self.bridge is not None:
                condition = self.bridge.predict_next(condition.to(torch.bfloat16), stable.to(torch.bfloat16),
                    batch["proprio"].to(torch.bfloat16), chunk[:, 0].to(torch.bfloat16)).to(condition.dtype)
            elif self.hook is not None:
                captured = self.hook.encode(input_ids=batch["input_ids"],
                    image_input=batch["image_input"], image_mask=batch["image_mask"])
                condition, stable = captured.condition, captured.stable
            else:
                condition = self.model.forward_vlm_efficient(batch["image_input"],
                    batch["image_mask"], batch["input_ids"])["vlm_features"]
            if self.loop is None:
                steps = 3 if self.row in ("naive_nfe3", "condition_naive3") else 10
                chunk = self.action.decode_action_from_condition(condition, batch["proprio"],
                    steps=steps, initial_noise=batch["noise"])
            else:
                proprio = self.action.normalize_proprio(batch["proprio"])
                def full_step(x, tau):
                    return self.step(vlm_features=condition, action_with_noise=x, proprio=proprio, t=tau)
                trace = self.loop(batch["noise"], full_step=full_step,
                    full_step_indices=GENERATION_SCHEDULES[3], proprio=proprio,
                    condition=condition, condition_valid_mask=None,
                    condition_change_code=condition.new_zeros(1, self.loop.updater.condition_code_dim))
                chunk = self.action.action_space.postprocess(trace.final_noisy_action)
            outputs.append(chunk.clone())
            previous = batch
        return torch.cat(outputs, dim=0)


def stats(values):
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values), "p95": ordered[min(len(ordered)-1, int(.95*len(ordered)))]}


def worker(c, output, row, mode):
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from tools.simvla.compile_runtime import Compiler

    torch.set_num_threads(1)
    configure_strict_torch_determinism(c["seed"])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total - 2 * 1024**3) / total)
    compiler = Compiler(mode == "compile")
    directory = output / row / mode
    directory.mkdir(parents=True, exist_ok=True)
    cpu = torch.load(output / "recorded_inputs.pt", map_location="cpu", weights_only=False)
    def move(x):
        if torch.is_tensor(x): return x.cuda()
        if isinstance(x, dict): return {k: move(v) for k,v in x.items()}
        if isinstance(x, list): return [move(v) for v in x]
        return x
    samples = move(cpu)
    with torch.inference_mode():
        model = Replay(c, row, compiler, samples)
        warm_started = time.perf_counter()
        for cycle in range(c["warmup_cycles"]):
            for i, sample in enumerate(samples):
                model(sample)
                torch.cuda.synchronize()
                print(f"WARMUP row={row} mode={mode} cycle={cycle+1}/{c['warmup_cycles']} window={i+1}/{len(samples)} graphs={compiler.graph_count()}", flush=True)
                write_json(directory / "progress.json", {"phase": "warmup", "window": i+1,
                    "cycle": cycle+1, "compiler": compiler.records})
        warm_seconds = time.perf_counter() - warm_started
        before = compiler.graph_count()
        reference = torch.load(output / row / "eager/actions.pt", map_location="cpu", weights_only=True) if mode == "compile" else None
        predictions, comparisons = [], []
        for i, sample in enumerate(samples):
            actual = model(sample).cpu()
            if not torch.isfinite(actual).all():
                raise RuntimeError(f"Non-finite action output: {row}/{mode}/window{i}")
            predictions.append(actual)
            if reference is not None:
                diff = (actual.float() - reference[i].float()).abs()
                comparisons.append({"window": i, "bitwise_equal": bool(torch.equal(actual, reference[i])),
                    "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                    "finite": bool(torch.isfinite(actual).all()),
                    "first5_gripper_sign_changes": int(((actual[:, :5, 6] > 0) != (reference[i][:, :5, 6] > 0)).sum())})
        torch.save(predictions, directory / "actions.pt")
        elapsed = []
        for cycle in range(c["measured_cycles"]):
            # Reverse alternate cycles to reduce input-order/thermal bias.
            order = list(range(len(samples)))[::(-1 if cycle % 2 else 1)]
            for i in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                model(samples[i])
                torch.cuda.synchronize()
                milliseconds = (time.perf_counter() - started) * 1000
                elapsed.append({"cycle": cycle, "window": i, "ms_per_four_queries": milliseconds,
                    "ms_per_query": milliseconds / 4})
            print(f"MEASURE row={row} mode={mode} cycle={cycle+1}/{c['measured_cycles']} mean_ms/query={statistics.mean(v['ms_per_query'] for v in elapsed):.3f}", flush=True)
        after = compiler.graph_count()
        if mode == "compile" and before == 0:
            verdict = "COMPILE_BYPASS_FAIL"
        elif mode == "compile" and row == "latent_bridge_f2" and not compiler.records["bridge_predict_next"]["graphs"]:
            verdict = "BRIDGE_COMPILE_BYPASS_FAIL"
        elif after != before:
            verdict = "TIMING_RECOMPILED_NEEDS_REVIEW"
        else:
            verdict = "RECORDED_INPUT_BENCHMARK_COMPLETE"
        report = {"row": row, "mode": mode, "verdict": verdict,
            "scope": "preprocessed GPU-resident 4-query replay; no simulator/network/CPU preprocessing",
            "paper_policy_latency": False, "success_rate_measured": False,
            "warmup_seconds_excluded": warm_seconds, "graphs_before_measurement": before,
            "graphs_after_measurement": after, "compiler": compiler.records,
            "options": {"max_autotune": True, "triton.cudagraphs": False, "dynamic": False},
            "component_compile_includes_eager_control_flow": True,
            "hookless_action_readout_parity": model.hookless_parity,
            "latency_ms_per_query": stats([v["ms_per_query"] for v in elapsed]),
            "measurements": elapsed, "output_comparisons": comparisons,
            "bitwise_equal_on_all_recorded_inputs": all(v["bitwise_equal"] for v in comparisons) if comparisons else None,
            "max_output_abs_diff": max((v["max_abs"] for v in comparisons), default=None),
            "gripper_sign_changes": sum(v["first5_gripper_sign_changes"] for v in comparisons),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "next": "live SR re-evaluation required before using compiled SR; this is only finite-input parity"}
        write_json(directory / "result.json", report)
        print(json.dumps({k: report[k] for k in ["row", "mode", "verdict", "latency_ms_per_query", "max_output_abs_diff"]}), flush=True)


def aggregate(c, output):
    results = {}
    lines = ["# SimVLA compile 검증", "", "고정된 실제 입력의 모델 계산시간 비교입니다. 논문의 policy ms/action 또는 성공률이 아닙니다.", "",
        "| 구성 | eager ms/query | compile ms/query | eager/compile | 출력 최대 차이 | 상태 |",
        "|---|---:|---:|---:|---:|---|"]
    for row in c["rows"]:
        pair = {mode: read_json(output / row / mode / "result.json")
            for mode in ("eager", "compile") if (output / row / mode / "result.json").exists()}
        results[row] = pair
        e, compiled = pair.get("eager", {}), pair.get("compile", {})
        a = e.get("latency_ms_per_query", {}).get("mean")
        b = compiled.get("latency_ms_per_query", {}).get("mean")
        if a and b:
            lines.append(f"| {row} | {a:.3f} | {b:.3f} | {a/b:.3f}x | {compiled.get('max_output_abs_diff')} | {compiled.get('verdict')} |")
        else:
            lines.append(f"| {row} | - | - | - | - | {compiled.get('verdict', e.get('verdict', 'NOT_RUN'))} |")
    complete = all(len(pair) == 2 and all(v.get("verdict") == "RECORDED_INPUT_BENCHMARK_COMPLETE" for v in pair.values()) for pair in results.values())
    verdict = "BENCHMARK_COMPLETE" if complete else "BENCHMARK_FINISHED_WITH_ITEMS_TO_REVIEW"
    write_json(output / "summary.json", {"verdict": verdict, "rows": results,
        "not_a_success_rate_experiment": True, "not_paper_end_to_end_latency": True})
    (output / "report_ko.md").write_text("\n".join(lines) + "\n")
    return verdict


def run_all(c, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = preflight(c)
        locked = output / "config.json"
        if locked.exists() and read_json(locked) != c:
            raise RuntimeError("Existing run uses a different config; choose a new output")
        write_json(locked, c)
        provenance.update({"hostname": socket.gethostname(), "command": sys.argv,
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"], text=True),
            "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "TORCHINDUCTOR_COMPILE_THREADS")},
            "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "config_sha256": sha(locked), "inputs": {key: sha(c[key]) for key in
                ["norm_stats", "condition_checkpoint", "generation_checkpoint", "bridge_checkpoint"]},
            "source_files": source_identity(c)})
        if (output / "provenance.json").exists():
            previous = read_json(output / "provenance.json")
            for key in ("source_files", "inputs", "config_sha256"):
                if previous[key] != provenance[key]:
                    raise RuntimeError(f"Existing run provenance changed: {key}; choose a new output")
        write_json(output / "provenance.json", provenance)
        prepare_inputs(c, output)
        for row in c["rows"]:
            for mode in ("eager", "compile"):
                directory = output / row / mode
                result = directory / "result.json"
                if result.exists() and read_json(result).get("verdict") == "RECORDED_INPUT_BENCHMARK_COMPLETE":
                    print(f"REUSE row={row} mode={mode}", flush=True)
                    continue
                if mode == "compile" and not (output / row / "eager/actions.pt").exists():
                    write_json(result, {"verdict": "EAGER_REFERENCE_MISSING"})
                    continue
                while True:
                    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                    if not busy:
                        break
                    print("GPU occupied; waiting 30 seconds without launching another GPU process", flush=True)
                    time.sleep(30)
                directory.mkdir(parents=True, exist_ok=True)
                write_json(output / "status.json", {"state": "running", "row": row, "mode": mode})
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                    "--config", str(locked), "--output", str(output), "--row", row, "--mode", mode]
                print(f"START row={row} mode={mode} log={directory/'run.log'}", flush=True)
                with (directory / "run.log").open("a") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(output/'compiler_cache')})
                    started = time.monotonic()
                    try:
                        while True:
                            try:
                                process.wait(timeout=30)
                                break
                            except subprocess.TimeoutExpired:
                                elapsed = time.monotonic() - started
                                progress_path = directory / "progress.json"
                                progress = read_json(progress_path) if progress_path.exists() else {"phase": "loading_or_first_compile"}
                                print(f"RUNNING row={row} mode={mode} elapsed={elapsed:.0f}s progress={progress}", flush=True)
                                if elapsed > c["worker_timeout_seconds"]:
                                    raise
                    except subprocess.TimeoutExpired:
                        stop_worker(process)
                    except KeyboardInterrupt:
                        stop_worker(process)
                        write_json(output / "status.json", {"state": "interrupted", "row": row, "mode": mode})
                        raise
                if process.returncode != 0:
                    write_json(result, {"verdict": "WORKER_FAILED", "exit_code": process.returncode,
                        "log": str(directory / "run.log")})
                print(f"FINISH row={row} mode={mode} rc={process.returncode}", flush=True)
                aggregate(c, output)
        verdict = aggregate(c, output)
        write_json(output / "status.json", {"state": "finished", "verdict": verdict})
        print(f"{verdict} report={output/'report_ko.md'}", flush=True)


def stop_worker(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["preflight", "prepare", "worker", "all", "summarize"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path)
    p.add_argument("--row")
    p.add_argument("--mode", choices=["eager", "compile"])
    args = p.parse_args()
    c = read_json(args.config)
    output = args.output or Path(c["storage"]) / "results/simvla/compile_benchmark/paired_long_inputs_v1"
    configure(c)
    try:
        if args.command == "preflight": print(json.dumps(preflight(c), indent=2))
        elif args.command == "prepare":
            output.mkdir(parents=True, exist_ok=True)
            prepare_inputs(c, output)
        elif args.command == "worker":
            if args.row not in c["rows"] or not args.mode: raise ValueError("Valid row and mode required")
            worker(c, output, args.row, args.mode)
        elif args.command == "summarize": print(aggregate(c, output))
        else: run_all(c, output)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
