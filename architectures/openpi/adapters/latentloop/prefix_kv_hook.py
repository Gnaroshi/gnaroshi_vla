"""Source-audited extraction and reconstruction of pi0.5 prefix KV state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any

import torch
from torch import Tensor


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    payload = value.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope_to_keys(pre_rope_keys: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Match OpenPI's patched Gemma ``apply_rotary_pos_emb`` for K."""

    if pre_rope_keys.ndim != 4:
        raise ValueError("pre_rope_keys must be [B,H,S,D]")
    if cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("cos and sin must be aligned [B,S,D]")
    return pre_rope_keys * cos.unsqueeze(1) + rotate_half(pre_rope_keys) * sin.unsqueeze(1)


@dataclass(frozen=True)
class PrefixEmbeddingState:
    embeddings: Tensor
    pad_mask: Tensor
    attention_pattern: Tensor
    position_ids: Tensor

    def validate(self) -> None:
        if self.embeddings.ndim != 3:
            raise ValueError("prefix embeddings must be [B,S,E]")
        batch, tokens = self.embeddings.shape[:2]
        for name, value in (
            ("pad_mask", self.pad_mask),
            ("attention_pattern", self.attention_pattern),
            ("position_ids", self.position_ids),
        ):
            if value.shape != (batch, tokens):
                raise ValueError(f"{name} must be [B,S]")

    def to(self, *args: Any, **kwargs: Any) -> "PrefixEmbeddingState":
        embeddings = self.embeddings.to(*args, **kwargs)
        return replace(
            self,
            embeddings=embeddings,
            pad_mask=self.pad_mask.to(device=embeddings.device),
            attention_pattern=self.attention_pattern.to(device=embeddings.device),
            position_ids=self.position_ids.to(device=embeddings.device),
        )


@dataclass(frozen=True)
class PrefixKVState(PrefixEmbeddingState):
    """Pre-RoPE K and unmodified V for every PaliGemma prefix layer."""

    pre_rope_keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]

    def validate(self) -> None:
        super().validate()
        if not self.pre_rope_keys or len(self.pre_rope_keys) != len(self.values):
            raise ValueError("K/V layer lists must be nonempty and aligned")
        batch, tokens = self.embeddings.shape[:2]
        for layer, (key, value) in enumerate(zip(self.pre_rope_keys, self.values, strict=True)):
            if key.shape != value.shape or key.ndim != 4:
                raise ValueError(f"layer {layer} K/V must be aligned [B,H,S,D]")
            if key.shape[0] != batch or key.shape[2] != tokens:
                raise ValueError(f"layer {layer} K/V does not match prefix layout")

    @property
    def num_layers(self) -> int:
        return len(self.pre_rope_keys)

    @property
    def num_tokens(self) -> int:
        return self.embeddings.shape[1]

    def detach(self) -> "PrefixKVState":
        return replace(
            self,
            embeddings=self.embeddings.detach(),
            pad_mask=self.pad_mask.detach(),
            attention_pattern=self.attention_pattern.detach(),
            position_ids=self.position_ids.detach(),
            pre_rope_keys=tuple(value.detach() for value in self.pre_rope_keys),
            values=tuple(value.detach() for value in self.values),
        )

    def to(self, *args: Any, **kwargs: Any) -> "PrefixKVState":
        embedding_state = super().to(*args, **kwargs)
        return replace(
            self,
            embeddings=embedding_state.embeddings,
            pad_mask=embedding_state.pad_mask,
            attention_pattern=embedding_state.attention_pattern,
            position_ids=embedding_state.position_ids,
            pre_rope_keys=tuple(value.to(*args, **kwargs) for value in self.pre_rope_keys),
            values=tuple(value.to(*args, **kwargs) for value in self.values),
        )


@dataclass(frozen=True)
class PrefixExtraction:
    state: PrefixKVState
    source_cache: Any
    robot_state: Tensor
    prefix_embedding_ms: float
    full_prefix_ms: float


class _ProjectionCapture:
    def __init__(self, layers: list[Any]) -> None:
        self.layers = layers
        self.keys: list[Tensor | None] = [None] * len(layers)
        self.values: list[Tensor | None] = [None] * len(layers)
        self.handles: list[Any] = []

    def _hook(self, collection: list[Tensor | None], index: int):
        def capture(_module: Any, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            collection[index] = output

        return capture

    def __enter__(self) -> "_ProjectionCapture":
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(self._hook(self.keys, index)))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._hook(self.values, index)))
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self.handles:
            handle.remove()

    def tensors(self) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        if any(value is None for value in self.keys + self.values):
            raise RuntimeError("not every prefix projection hook fired")
        pre_keys: list[Tensor] = []
        values: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            key = self.keys[index]
            value = self.values[index]
            assert key is not None and value is not None
            batch, tokens = key.shape[:2]
            heads = layer.self_attn.config.num_key_value_heads
            head_dim = layer.self_attn.head_dim
            pre_keys.append(key.view(batch, tokens, heads, head_dim).transpose(1, 2).contiguous())
            values.append(value.view(batch, tokens, heads, head_dim).transpose(1, 2).contiguous())
        return tuple(pre_keys), tuple(values)


class PrefixKVHook:
    """External hook around the unmodified PR-854 ``PI0Pytorch`` model."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @property
    def language_model(self) -> Any:
        return self.model.paligemma_with_expert.paligemma.language_model

    @torch.no_grad()
    def embed(self, observation: Any) -> tuple[PrefixEmbeddingState, Tensor, float]:
        images, image_masks, language_tokens, language_masks, robot_state = self.model._preprocess_observation(  # noqa: SLF001
            observation, train=False
        )
        device = robot_state.device
        _sync(device)
        started = time.perf_counter()
        embeddings, pad_mask, attention_pattern = self.model.embed_prefix(
            images, image_masks, language_tokens, language_masks
        )
        _sync(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        position_ids = torch.cumsum(pad_mask, dim=1) - 1
        state = PrefixEmbeddingState(
            embeddings=embeddings,
            pad_mask=pad_mask,
            attention_pattern=attention_pattern,
            position_ids=position_ids,
        )
        state.validate()
        return state, robot_state, elapsed_ms

    @torch.no_grad()
    def extract(self, observation: Any) -> PrefixExtraction:
        embedding_state, robot_state, embedding_ms = self.embed(observation)
        return self.extract_from_embedding(embedding_state, robot_state, embedding_ms=embedding_ms)

    @torch.no_grad()
    def extract_from_embedding(
        self,
        embedding_state: PrefixEmbeddingState,
        robot_state: Tensor,
        *,
        embedding_ms: float = 0.0,
    ) -> PrefixExtraction:
        """Run only the PaliGemma prefix transformer from an existing embedding."""

        embedding_state.validate()
        model = self.language_model
        model.config._attn_implementation = "eager"  # noqa: SLF001
        attention_2d = self.model.make_att_2d_masks(  # type: ignore[attr-defined]
            embedding_state.pad_mask, embedding_state.attention_pattern
        ) if hasattr(self.model, "make_att_2d_masks") else None
        if attention_2d is None:
            # ``make_att_2d_masks`` is a module-level function in PR-854.
            cumsum = torch.cumsum(embedding_state.attention_pattern, dim=1)
            attention_2d = (cumsum[:, None, :] <= cumsum[:, :, None]) & (
                embedding_state.pad_mask[:, None, :] * embedding_state.pad_mask[:, :, None]
            )
        attention_4d = self.model._prepare_attention_masks_4d(attention_2d)  # noqa: SLF001
        _sync(embedding_state.embeddings.device)
        started = time.perf_counter()
        with _ProjectionCapture(list(model.layers)) as capture:
            _, source_cache = self.model.paligemma_with_expert.forward(
                inputs_embeds=[embedding_state.embeddings, None],
                attention_mask=attention_4d,
                position_ids=embedding_state.position_ids,
                past_key_values=None,
                use_cache=True,
            )
        _sync(embedding_state.embeddings.device)
        prefix_ms = (time.perf_counter() - started) * 1000.0
        keys, values = capture.tensors()
        state = PrefixKVState(
            **embedding_state.__dict__,
            pre_rope_keys=keys,
            values=values,
        )
        state.validate()
        return PrefixExtraction(
            state=state,
            source_cache=source_cache,
            robot_state=robot_state,
            prefix_embedding_ms=embedding_ms,
            full_prefix_ms=prefix_ms,
        )

    def rebuild_cache(self, state: PrefixKVState) -> Any:
        """Rebuild the exact post-RoPE cache while preserving adapter gradients."""
        state.validate()
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as exc:  # pragma: no cover - environment-specific failure
            raise RuntimeError("transformers DynamicCache is required") from exc

        language_model = self.language_model
        cos, sin = language_model.rotary_emb(state.embeddings, state.position_ids)
        cache = DynamicCache()
        cache_position = torch.arange(state.num_tokens, device=state.embeddings.device)
        for layer_index, (key, value) in enumerate(zip(state.pre_rope_keys, state.values, strict=True)):
            post_rope_key = apply_rope_to_keys(key, cos, sin)
            cache.update(
                post_rope_key,
                value,
                layer_index,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
        return cache

    def sample_actions_from_state(
        self,
        state: PrefixKVState,
        robot_state: Tensor,
        noise: Tensor,
        *,
        num_steps: int = 10,
    ) -> tuple[Tensor, dict[str, float]]:
        """Run the frozen action expert from an externally supplied prefix state.

        This method intentionally does not use ``torch.no_grad``. During adapter
        training, action-consistency losses must differentiate through the frozen
        expert into the predicted KV tensors. Frozen model parameters still have
        ``requires_grad=False`` and are never included in the optimizer.
        """
        if noise.ndim != 3:
            raise ValueError("noise must be [B,H,D]")
        if noise.shape[1] != self.model.config.action_horizon:
            raise ValueError("noise horizon does not match the frozen model")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        device = noise.device
        _sync(device)
        rebuild_started = time.perf_counter()
        cache = self.rebuild_cache(state)
        _sync(device)
        rebuild_ms = (time.perf_counter() - rebuild_started) * 1000.0

        x_t = noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        time_value = torch.tensor(1.0, dtype=torch.float32, device=device)
        _sync(device)
        action_started = time.perf_counter()
        while time_value >= -dt / 2:
            timestep = time_value.expand(noise.shape[0])
            velocity = self.model.denoise_step(
                robot_state,
                state.pad_mask,
                cache,
                x_t,
                timestep,
            )
            x_t = x_t + dt * velocity
            time_value = time_value + dt
        _sync(device)
        action_ms = (time.perf_counter() - action_started) * 1000.0
        return x_t, {"cache_rebuild_ms": rebuild_ms, "action_expert_ms": action_ms}

    def hashes(self, state: PrefixKVState) -> dict[str, object]:
        return {
            "prefix_embedding": tensor_sha256(state.embeddings),
            "prefix_pad_mask": tensor_sha256(state.pad_mask),
            "prefix_attention_pattern": tensor_sha256(state.attention_pattern),
            "prefix_position_ids": tensor_sha256(state.position_ids),
            "pre_rope_k": [tensor_sha256(value) for value in state.pre_rope_keys],
            "value": [tensor_sha256(value) for value in state.values],
        }

    def cache_allclose(
        self,
        state: PrefixKVState,
        source_cache: Any,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> dict[str, object]:
        rebuilt = self.rebuild_cache(state)
        layers: list[dict[str, object]] = []
        passed = True
        for index in range(state.num_layers):
            source_key, source_value = source_cache[index]
            rebuilt_key, rebuilt_value = rebuilt[index]
            key_equal = torch.allclose(source_key, rebuilt_key, atol=atol, rtol=rtol)
            value_equal = torch.allclose(source_value, rebuilt_value, atol=atol, rtol=rtol)
            passed &= key_equal and value_equal
            layers.append(
                {
                    "layer": index,
                    "key_allclose": key_equal,
                    "value_allclose": value_equal,
                    "key_max_abs": float((source_key - rebuilt_key).abs().max().item()),
                    "key_mean_abs": float((source_key - rebuilt_key).abs().mean().item()),
                    "value_max_abs": float((source_value - rebuilt_value).abs().max().item()),
                    "value_mean_abs": float((source_value - rebuilt_value).abs().mean().item()),
                }
            )
        return {"passed": passed, "atol": atol, "rtol": rtol, "layers": layers}


@contextmanager
def deterministic_torch(seed: int):
    """Temporarily set deterministic flags without hiding RNG-state changes."""

    previous = torch.are_deterministic_algorithms_enabled()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(previous, warn_only=True)
