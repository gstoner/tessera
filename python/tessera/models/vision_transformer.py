"""Reference vision tower + projector for multimodal model contracts.

This is a small numpy implementation used by importer/runtime tests and
compiler contract bring-up. It intentionally favors explicit shape contracts
over backend cleverness: native lowering can target the same patch/project/splice
ops later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .multimodal import MediaProcessorConfig, MultimodalContractError


class VisionTransformerError(ValueError):
    """Raised when vision tower shapes or media inputs are inconsistent."""


@dataclass(frozen=True)
class VisionTransformerConfig:
    image_size: int
    patch_size: int
    num_channels: int
    hidden_size: int
    output_hidden_size: int
    image_seq_length: int
    num_layers: int = 1
    num_heads: int = 4
    mlp_hidden_size: int | None = None
    spatial_merge_size: int = 1
    temporal_patch_size: int = 1
    max_frames: int = 1

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.num_channels

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def max_patch_tokens(self) -> int:
        grid = max(1, self.image_size // self.patch_size)
        return grid * grid

    @property
    def merged_tokens(self) -> int:
        merge = max(1, self.spatial_merge_size * self.spatial_merge_size)
        return max(1, self.max_patch_tokens // merge)


@dataclass(frozen=True)
class VisionLayerWeights:
    norm1: np.ndarray
    qkv: np.ndarray
    o: np.ndarray
    norm2: np.ndarray
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray


@dataclass(frozen=True)
class VisionTowerWeights:
    patch_weight: np.ndarray
    patch_bias: np.ndarray
    pos_embed: np.ndarray
    layers: tuple[VisionLayerWeights, ...]
    norm: np.ndarray


@dataclass(frozen=True)
class VisionProjectorWeights:
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class VisionRuntimeWeights:
    tower: VisionTowerWeights
    projector: VisionProjectorWeights


def config_from_processor(
    processor: MediaProcessorConfig,
    *,
    output_hidden_size: int,
    hidden_size: int | None = None,
    num_layers: int = 1,
    num_heads: int = 4,
    mlp_hidden_size: int | None = None,
) -> VisionTransformerConfig:
    cfg = VisionTransformerConfig(
        image_size=processor.image_size,
        patch_size=processor.patch_size,
        num_channels=processor.num_channels,
        hidden_size=hidden_size or output_hidden_size,
        output_hidden_size=output_hidden_size,
        image_seq_length=processor.image_seq_length,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_hidden_size=mlp_hidden_size,
        spatial_merge_size=processor.spatial_merge_size,
        temporal_patch_size=processor.temporal_patch_size,
        max_frames=processor.max_frames,
    )
    verify_config(cfg)
    return cfg


def verify_config(cfg: VisionTransformerConfig) -> None:
    if cfg.image_size <= 0 or cfg.patch_size <= 0:
        raise VisionTransformerError("image_size and patch_size must be positive")
    if cfg.num_channels <= 0:
        raise VisionTransformerError("num_channels must be positive")
    if cfg.hidden_size <= 0 or cfg.output_hidden_size <= 0:
        raise VisionTransformerError("hidden sizes must be positive")
    if cfg.hidden_size % cfg.num_heads != 0:
        raise VisionTransformerError("hidden_size must be divisible by num_heads")
    if cfg.num_layers < 0:
        raise VisionTransformerError("num_layers must be non-negative")
    if cfg.image_seq_length <= 0:
        raise VisionTransformerError("image_seq_length must be positive")
    if cfg.max_frames <= 0:
        raise VisionTransformerError("max_frames must be positive")


def synthetic_weights(cfg: VisionTransformerConfig, *, seed: int = 0) -> VisionRuntimeWeights:
    verify_config(cfg)
    rng = np.random.default_rng(seed)
    H = cfg.hidden_size
    F = cfg.mlp_hidden_size or 4 * H
    s = 1.0 / np.sqrt(H)

    def n(*shape, sc=s):
        return (rng.standard_normal(shape) * sc).astype(np.float64)

    layers = tuple(
        VisionLayerWeights(
            norm1=np.ones(H, dtype=np.float64),
            qkv=n(H, 3 * H),
            o=n(H, H),
            norm2=np.ones(H, dtype=np.float64),
            gate=n(H, F),
            up=n(H, F),
            down=n(F, H, sc=1.0 / np.sqrt(F)),
        )
        for _ in range(cfg.num_layers)
    )
    tower = VisionTowerWeights(
        patch_weight=n(cfg.patch_dim, H, sc=1.0 / np.sqrt(cfg.patch_dim)),
        patch_bias=np.zeros(H, dtype=np.float64),
        pos_embed=n(max(cfg.image_seq_length, cfg.merged_tokens), H),
        layers=layers,
        norm=np.ones(H, dtype=np.float64),
    )
    projector = VisionProjectorWeights(
        weight=n(H, cfg.output_hidden_size, sc=1.0 / np.sqrt(H)),
        bias=np.zeros(cfg.output_hidden_size, dtype=np.float64),
    )
    return VisionRuntimeWeights(tower=tower, projector=projector)


def encode_image(
    image,
    cfg: VisionTransformerConfig,
    weights: VisionRuntimeWeights,
    *,
    processor: MediaProcessorConfig,
) -> np.ndarray:
    """Encode one image into ``(image_seq_length, output_hidden_size)``."""
    frame = preprocess_image(image, processor)
    tokens = patch_embed_2d(frame, cfg, weights.tower)
    tokens = patch_merge(tokens, spatial_merge_size=cfg.spatial_merge_size)
    tokens = _resample_tokens(tokens, cfg.image_seq_length)
    return project_media_tokens(_run_tower(tokens, cfg, weights.tower), weights.projector)


def encode_video(
    video,
    cfg: VisionTransformerConfig,
    weights: VisionRuntimeWeights,
    *,
    processor: MediaProcessorConfig,
    frames: int | None = None,
) -> np.ndarray:
    """Encode sampled video frames into ``(frames*image_seq_length, H_text)``."""
    sample = sample_video_frames(video, processor, frames=frames)
    outputs = [
        encode_image(sample[i], cfg, weights, processor=processor)
        for i in range(sample.shape[0])
    ]
    return np.concatenate(outputs, axis=0)


def preprocess_image(image, processor: MediaProcessorConfig) -> np.ndarray:
    processor.validate()
    x = _to_hwc(np.asarray(image), processor.num_channels)
    if processor.do_resize and (x.shape[0] != processor.image_size or x.shape[1] != processor.image_size):
        x = _resize_nearest(x, processor.image_size, processor.image_size)
    x = x.astype(np.float64) * processor.rescale_factor
    mean = _channel_vector(processor.image_mean, processor.num_channels)
    std = _channel_vector(processor.image_std, processor.num_channels)
    return (x - mean) / std


def sample_video_frames(video, processor: MediaProcessorConfig, *, frames: int | None = None) -> np.ndarray:
    processor.validate()
    x = np.asarray(video)
    if x.ndim != 4:
        raise VisionTransformerError(f"video must be rank-4 (T,H,W,C or T,C,H,W), got shape {x.shape}")
    want = frames if frames is not None else min(x.shape[0], processor.max_frames)
    if want < 1 or want > processor.max_frames:
        raise VisionTransformerError(f"frames={want} outside [1, {processor.max_frames}]")
    if x.shape[0] < want:
        pad = np.repeat(x[-1:], want - x.shape[0], axis=0)
        x = np.concatenate([x, pad], axis=0)
    elif x.shape[0] > want:
        idx = np.linspace(0, x.shape[0] - 1, want).round().astype(np.int64)
        x = x[idx]
    return np.stack([_to_hwc(frame, processor.num_channels) for frame in x], axis=0)


def patch_embed_2d(image_hwc: np.ndarray, cfg: VisionTransformerConfig, weights: VisionTowerWeights) -> np.ndarray:
    if image_hwc.ndim != 3:
        raise VisionTransformerError(f"image_hwc must be rank-3, got {image_hwc.shape}")
    h, w, c = image_hwc.shape
    if c != cfg.num_channels:
        raise VisionTransformerError(f"image channels {c} != config {cfg.num_channels}")
    ph = h // cfg.patch_size
    pw = w // cfg.patch_size
    if ph < 1 or pw < 1:
        raise VisionTransformerError("patch_size is larger than image dimensions")
    cropped = image_hwc[:ph * cfg.patch_size, :pw * cfg.patch_size]
    patches = cropped.reshape(ph, cfg.patch_size, pw, cfg.patch_size, c)
    flat = patches.transpose(0, 2, 1, 3, 4).reshape(ph * pw, cfg.patch_dim)
    return flat @ weights.patch_weight + weights.patch_bias


def patch_merge(tokens: np.ndarray, *, spatial_merge_size: int) -> np.ndarray:
    x = np.asarray(tokens, dtype=np.float64)
    merge = max(1, int(spatial_merge_size) * int(spatial_merge_size))
    if merge == 1:
        return x
    usable = (x.shape[0] // merge) * merge
    if usable == 0:
        return x
    merged = x[:usable].reshape(usable // merge, merge, x.shape[-1]).mean(axis=1)
    if usable == x.shape[0]:
        return merged
    return np.concatenate([merged, x[usable:]], axis=0)


def project_media_tokens(tokens: np.ndarray, projector: VisionProjectorWeights) -> np.ndarray:
    return np.asarray(tokens, dtype=np.float64) @ projector.weight + projector.bias


def _run_tower(tokens: np.ndarray, cfg: VisionTransformerConfig, weights: VisionTowerWeights) -> np.ndarray:
    x = np.asarray(tokens, dtype=np.float64)
    if weights.pos_embed.shape[0] < x.shape[0]:
        raise VisionTransformerError(
            f"pos_embed has {weights.pos_embed.shape[0]} rows but needs {x.shape[0]}")
    x = x + weights.pos_embed[:x.shape[0]]
    for layer in weights.layers:
        a = _self_attention(_rmsnorm(x, layer.norm1), layer, cfg)
        x = x + a
        x = x + _swiglu(_rmsnorm(x, layer.norm2), layer.gate, layer.up, layer.down)
    return _rmsnorm(x, weights.norm)


def _self_attention(x: np.ndarray, layer: VisionLayerWeights, cfg: VisionTransformerConfig) -> np.ndarray:
    S, H = x.shape
    qkv = (x @ layer.qkv).reshape(S, 3, cfg.num_heads, cfg.head_dim)
    q = qkv[:, 0].transpose(1, 0, 2)
    k = qkv[:, 1].transpose(1, 0, 2)
    v = qkv[:, 2].transpose(1, 0, 2)
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(cfg.head_dim)
    probs = _softmax(scores)
    out = (probs @ v).transpose(1, 0, 2).reshape(S, H)
    return out @ layer.o


def _rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * w


def _swiglu(x: np.ndarray, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray) -> np.ndarray:
    g = x @ wg
    return ((g * (1.0 / (1.0 + np.exp(-g)))) * (x @ wu)) @ wd


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _resample_tokens(tokens: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(tokens, dtype=np.float64)
    if x.shape[0] == length:
        return x
    if x.shape[0] > length:
        idx = np.linspace(0, x.shape[0] - 1, length).round().astype(np.int64)
        return x[idx]
    pad = np.repeat(x[-1:], length - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


def _to_hwc(x: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 3:
        raise VisionTransformerError(f"image must be rank-3, got shape {arr.shape}")
    if arr.shape[-1] == channels:
        return arr
    if arr.shape[0] == channels:
        return arr.transpose(1, 2, 0)
    raise VisionTransformerError(f"cannot infer channel axis for image shape {arr.shape}")


def _resize_nearest(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    ys = np.linspace(0, x.shape[0] - 1, out_h).round().astype(np.int64)
    xs = np.linspace(0, x.shape[1] - 1, out_w).round().astype(np.int64)
    return x[ys][:, xs]


def _channel_vector(values: Sequence[float], channels: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        arr = np.repeat(arr, channels)
    if arr.size != channels:
        raise MultimodalContractError(f"channel vector length {arr.size} != {channels}")
    return arr.reshape(1, 1, channels)


__all__ = [
    "VisionLayerWeights",
    "VisionProjectorWeights",
    "VisionRuntimeWeights",
    "VisionTowerWeights",
    "VisionTransformerConfig",
    "VisionTransformerError",
    "config_from_processor",
    "encode_image",
    "encode_video",
    "patch_embed_2d",
    "patch_merge",
    "preprocess_image",
    "project_media_tokens",
    "sample_video_frames",
    "synthetic_weights",
    "verify_config",
]
