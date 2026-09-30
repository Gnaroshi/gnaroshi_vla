"""Shared experiment contract for sd1 efficacy, not rb2 paper timing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from tools.simvla.compile_benchmark import read_json, write_json, sha

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "architectures/simvla/configs/error_compensation_sd1.json"
ARMS = ("same_condition", "true_condition", "true_condition_no_code")
ROWS = ("baseline", "condition_naive3", "condition_naive4", "condition_naive5", "parent", *ARMS)


def configure(c):
    upstream = Path(c["upstream"])
    os.environ["SIMVLA_UPSTREAM_ROOT"] = str(upstream)
    os.environ["HF_HOME"] = c["hf_home"]
    for p in (ROOT, upstream, upstream / "evaluation/libero/LIBERO"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def environment(c, gpu):
    if gpu not in (4, 5, 6, 7):
        raise ValueError("sd1 allows only physical GPUs 4,5,6,7")
    e = dict(os.environ)
    for key in ("GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE", "EGL_DEVICE_ID", "LP_NUM_THREADS"):
        e.pop(key, None)
    e.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", USE_TF="0", TOKENIZERS_PARALLELISM="false",
        HF_HOME=c["hf_home"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        SIMVLA_UPSTREAM_ROOT=c["upstream"], PYTHONHASHSEED=str(c["evaluation_seed"]),
        CUBLAS_WORKSPACE_CONFIG=":4096:8", CUDA_DEVICE_MAX_CONNECTIONS="1",
        NVIDIA_TF32_OVERRIDE="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        LIBERO_CONFIG_PATH=str(Path(c["output"]) / "libero_config"),
        NUMBA_CACHE_DIR=str(Path(c["output"]) / "runtime/numba"),
        MPLCONFIGDIR=str(Path(c["output"]) / "runtime/matplotlib"),
        PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    return e


def snapshots(c):
    hub = Path(c["hf_home"]) / "hub"
    repo = hub / "models--HuggingFaceTB--SmolVLM-500M-Instruct"
    return {"checkpoint_snapshot": str(hub / "models--YuankaiLuo--SimVLA-LIBERO/snapshots" / c["checkpoint_revision"]),
        "backbone_snapshot": str(repo / "snapshots" / (repo / "refs/main").read_text().strip())}


def identity(c):
    contract = read_json(Path(c["output"]) / "contract.json")
    if contract["config"] != c:
        raise RuntimeError("Config changed after preparation")
    for path, expected in contract["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise RuntimeError(f"Source changed: {path}")
    return contract["identity"]


def arm_inputs(arm, predicted, teacher, code):
    if arm not in ARMS:
        raise ValueError(arm)
    return (predicted if arm == "same_condition" else teacher,
        code * 0 if arm == "true_condition_no_code" else code)


def checkpoint_path(c, arm):
    return Path(c["output"]) / "train" / arm / "latest.pt"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
