"""External rollout wrapper for Seer continuity and synchronized R0 data."""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import BridgeTransition, StreamingBridgeTransitionWriter
from .hooks import SeerBoundaryCapture
from .layout import SeerTokenLayout
from .provenance import PUBLIC_SEER_33_SHA256
from .rendering import configured_renderer_backend


DEFAULT_LAYERS = (0, 3, 7, 11, 15, 19, 23)
DEFAULT_GROUPS = (
    "text",
    "state",
    "primary_resampled",
    "wrist_resampled",
    "primary_cls",
    "wrist_cls",
    "visual",
    "multimodal_context",
    "observation_prediction",
    "action",
    "all",
)


def _current_state_from_observation(obs: dict) -> np.ndarray:
    from utils.eval_utils_libero import quaternion_to_euler

    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(quaternion_to_euler(obs["robot0_eef_quat"]), dtype=np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    ).astype(np.float32)


class ContinuityModelWrapperMixin:
    """Mixin applied to upstream ``ModelWrapper`` without modifying upstream."""

    def _continuity_initialize(self) -> None:
        self._continuity_renderer = configured_renderer_backend()
        base = self.model.module if hasattr(self.model, "module") else self.model
        self._continuity_layout = SeerTokenLayout.from_model(base)
        requested = os.environ.get("LATENT_BRIDGE_CONTINUITY_LAYERS", "")
        self._continuity_layers = tuple(int(item) for item in requested.split(",") if item) or DEFAULT_LAYERS
        if any(index < 0 or index >= len(base.transformer_backbone.h) for index in self._continuity_layers):
            raise ValueError(f"invalid continuity layer list: {self._continuity_layers}")
        requested_groups = os.environ.get("LATENT_BRIDGE_CONTINUITY_GROUPS", "")
        self._continuity_groups = tuple(item for item in requested_groups.split(",") if item) or DEFAULT_GROUPS
        unknown = set(self._continuity_groups) - set(self._continuity_layout.per_timestep_slices)
        if unknown:
            raise ValueError(f"invalid continuity token groups: {sorted(unknown)}")
        self._continuity_max_offset = int(os.environ.get("LATENT_BRIDGE_CONTINUITY_MAX_OFFSET", "4"))
        self._continuity_metrics_enabled = bool(
            int(os.environ.get("LATENT_BRIDGE_CONTINUITY_METRICS", "1"))
        )
        self._continuity_history = {}
        self._continuity_pending_rows = []
        self._continuity_rows = []
        self._continuity_previous = None
        self._continuity_previous_action = None
        self._continuity_stable_layer = os.environ.get("LATENT_BRIDGE_STABLE_LAYER", "block_00")
        self._continuity_stable_group = os.environ.get("LATENT_BRIDGE_STABLE_GROUP", "action")
        if self._continuity_stable_layer == "ln_f":
            capture_layers = self._continuity_layers if self._continuity_metrics_enabled else ()
        else:
            stable_index = int(self._continuity_stable_layer.removeprefix("block_"))
            capture_layers = (
                tuple(sorted(set((*self._continuity_layers, stable_index))))
                if self._continuity_metrics_enabled
                else (stable_index,)
            )
        self._continuity_capture = SeerBoundaryCapture(
            self.model, layer_indices=tuple(capture_layers)
        )
        self._continuity_capture.__enter__()
        output = Path(os.environ["LATENT_BRIDGE_CONTINUITY_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        self._continuity_output = output
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self._continuity_rank = int(rank)
        self._continuity_collect_transitions = bool(
            int(os.environ.get("LATENT_BRIDGE_COLLECT_TRANSITIONS", "0"))
        )
        self._continuity_writer = None
        self._continuity_eval_seed = None
        if self._continuity_collect_transitions:
            self._continuity_writer = StreamingBridgeTransitionWriter(
                output / f"sync_transitions_rank{self._continuity_rank}.h5"
            )

    def reset(self):
        result = super().reset()
        self._continuity_history = {}
        self._continuity_pending_rows = []
        self._continuity_previous = None
        self._continuity_previous_action = None
        self._continuity_episode_transitions = []
        return result

    def _continuity_snapshot(self, selected_step: int) -> dict[str, torch.Tensor]:
        capture = self._continuity_capture
        capture.require_complete()
        snapshots = {}
        snapshot_indices = set(self._continuity_layers if self._continuity_metrics_enabled else ())
        if self._continuity_stable_layer != "ln_f":
            snapshot_indices.add(int(self._continuity_stable_layer.removeprefix("block_")))
        for index in sorted(snapshot_indices):
            name = f"block_{index:02d}"
            snapshots[name] = self._continuity_layout.as_timesteps(capture.layer_outputs[name])[
                :, selected_step
            ].detach().float().cpu()
        snapshots["ln_f"] = self._continuity_layout.as_timesteps(capture.final_output)[
            :, selected_step
        ].detach().float().cpu()
        return snapshots

    def _continuity_add_metrics(self, snapshots: dict[str, torch.Tensor], timestep: int) -> None:
        for layer, current_all in snapshots.items():
            history = self._continuity_history.setdefault(
                layer, deque(maxlen=self._continuity_max_offset)
            )
            for offset, previous_all in enumerate(reversed(history), start=1):
                for group in self._continuity_groups:
                    token_slice = self._continuity_layout.per_timestep_slices[group]
                    current = current_all[:, token_slice]
                    previous = previous_all[:, token_slice]
                    if current.shape[1] == 0:
                        continue
                    delta = current - previous
                    cosine = F.cosine_similarity(current, previous, dim=-1).mean()
                    self._continuity_pending_rows.append(
                        {
                            "timestep": int(timestep),
                            "layer": layer,
                            "token_group": group,
                            "offset": offset,
                            "cosine": float(cosine.item()),
                            "delta_l2": float(delta.norm(dim=-1).mean().item()),
                            "delta_mse": float(delta.square().mean().item()),
                        }
                    )
            # The stable-context decision uses a strict 0.999 threshold. Keep
            # measurement history in FP32 so storage quantization cannot change
            # which candidate passes that scientific criterion.
            history.append(current_all.clone())

    def step(self, obs, goal, timestep, frames=None, video_stride: int = 1):
        self._continuity_capture.clear()
        current_state = _current_state_from_observation(obs)
        action = super().step(obs, goal, timestep, frames=frames, video_stride=video_stride)
        selected_step = self._selected_step(len(self.img_queue))
        snapshots = self._continuity_snapshot(selected_step)
        if self._continuity_metrics_enabled:
            self._continuity_add_metrics(snapshots, timestep)

        current = {
            "condition": snapshots["ln_f"][
                :, self._continuity_layout.per_timestep_slices["action"]
            ][0].numpy(),
            "stable": snapshots[self._continuity_stable_layer][
                :, self._continuity_layout.per_timestep_slices[self._continuity_stable_group]
            ][0].numpy(),
            "state": current_state,
            "step": int(timestep),
        }
        if self._continuity_previous is not None and self._continuity_previous_action is not None:
            current["transition"] = BridgeTransition(
                previous_condition=self._continuity_previous["condition"],
                target_condition=current["condition"],
                stable_context=self._continuity_previous["stable"],
                current_state=current_state,
                previous_executed_action=self._continuity_previous_action,
                episode_id="pending",
                task_id=-1,
                step=int(timestep),
                success=-1,
                source="R0_sync",
            )
        self._continuity_previous = current
        self._continuity_previous_action = np.asarray(action, dtype=np.float32).copy()
        return action

    def finish_episode(self, task, env, success, steps, args):
        metrics = super().finish_episode(task, env, success, steps, args)
        self._continuity_eval_seed = int(args.seed)
        episode_id = f"task{int(env.task_id):02d}_trial{int(env.exp_id):03d}_seed{int(args.seed)}"
        for row in self._continuity_pending_rows:
            row.update(
                {
                    "episode_id": episode_id,
                    "task_id": int(env.task_id),
                    "task_name": str(env.task_name),
                    "success": int(success),
                    "seed": int(args.seed),
                }
            )
            self._continuity_rows.append(row)
        # Records are buffered until termination so success/failure and the
        # exact task/trial identity can be attached without rewriting HDF5.
        for transition in self._continuity_episode_transitions:
            transition.episode_id = episode_id
            transition.task_id = int(env.task_id)
            transition.success = int(success)
            if self._continuity_writer is not None:
                self._continuity_writer.append(transition)
        self._continuity_pending_rows = []
        self._continuity_episode_transitions = []
        return metrics

    def get_lrnode_stats(self):
        output_path = self._continuity_output / f"continuity_rank{self._continuity_rank}.jsonl"
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite continuity shard: {output_path}")
        with output_path.open("w", encoding="utf-8") as stream:
            for row in self._continuity_rows:
                stream.write(json.dumps(row) + "\n")
        if self._continuity_writer is not None:
            self._continuity_writer.close(
                {
                    "rank": self._continuity_rank,
                    "stable_layer": self._continuity_stable_layer,
                    "stable_token_group": self._continuity_stable_group,
                    "public_seer_checkpoint_sha256": PUBLIC_SEER_33_SHA256,
                    "eval_seed": self._continuity_eval_seed,
                    "renderer": self._continuity_renderer,
                    "action_protocol": "three-token prediction with temporal ensembling; one executed action",
                }
            )
        metadata = {
            "rank": self._continuity_rank,
            "layers": [f"block_{index:02d}" for index in self._continuity_layers] + ["ln_f"],
            "token_groups": list(self._continuity_groups),
            "max_offset": self._continuity_max_offset,
            "metrics_enabled": self._continuity_metrics_enabled,
            "num_rows": len(self._continuity_rows),
            "token_layout": self._continuity_layout.to_dict(),
        }
        (self._continuity_output / f"continuity_rank{self._continuity_rank}.meta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        stats = super().get_lrnode_stats()
        stats.update(
            {
                "method": "frozen_seer_continuity_collection",
                "renderer_backend": self._continuity_renderer,
                "base_checkpoint_sha256": PUBLIC_SEER_33_SHA256,
                "refresh_period": 1,
                "action_protocol": (
                    "three-token prediction with temporal ensembling; one executed action"
                ),
            }
        )
        return stats


def build_continuity_wrapper(base_wrapper):
    class ContinuityModelWrapper(ContinuityModelWrapperMixin, base_wrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._continuity_episode_transitions = []
            self._continuity_initialize()

        def step(self, *args, **kwargs):
            action = ContinuityModelWrapperMixin.step(self, *args, **kwargs)
            current = self._continuity_previous
            if current is not None and "transition" in current:
                self._continuity_episode_transitions.append(current["transition"])
            return action

    return ContinuityModelWrapper
