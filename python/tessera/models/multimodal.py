"""Reusable multimodal contracts for model importers and reference runtimes.

These dataclasses keep media processing facts explicit: processor metadata,
reserved prompt spans, patch-grid geometry, and projected embeddings are all
shape-checked objects instead of ad hoc dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class MultimodalContractError(ValueError):
    """Raised when multimodal processor/media metadata is inconsistent."""


@dataclass(frozen=True)
class MediaSegment:
    kind: str
    data: Any
    frames: int = 1
    mime_type: str | None = None


@dataclass(frozen=True)
class MediaSpan:
    kind: str
    start: int
    end: int
    frames: int = 1

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PatchGrid:
    frames: int
    grid_h: int
    grid_w: int
    patch_size: int
    temporal_patch_size: int = 1
    spatial_merge_size: int = 1

    @property
    def patch_count(self) -> int:
        temporal = max(1, (self.frames + self.temporal_patch_size - 1) // self.temporal_patch_size)
        return temporal * self.grid_h * self.grid_w

    @property
    def merged_count(self) -> int:
        merge = max(1, self.spatial_merge_size * self.spatial_merge_size)
        return max(1, self.patch_count // merge)


@dataclass(frozen=True)
class MediaBatch:
    segments: tuple[MediaSegment, ...]
    spans: tuple[MediaSpan, ...]
    processor: "MediaProcessorConfig"

    def validate(self) -> None:
        if len(self.segments) != len(self.spans):
            raise MultimodalContractError("media segments and spans length mismatch")
        for idx, (segment, span) in enumerate(zip(self.segments, self.spans)):
            if segment.kind != span.kind:
                raise MultimodalContractError(f"segment/span kind mismatch at {idx}: {segment.kind!r} != {span.kind!r}")
            if segment.frames != span.frames:
                raise MultimodalContractError(f"segment/span frame mismatch at {idx}: {segment.frames} != {span.frames}")
            if span.length <= 0:
                raise MultimodalContractError(f"media span {idx} must have positive length")


@dataclass(frozen=True)
class ProjectedMediaEmbeddings:
    embeddings: tuple[np.ndarray, ...]
    spans: tuple[MediaSpan, ...]
    hidden_size: int

    def validate(self) -> None:
        if len(self.embeddings) != len(self.spans):
            raise MultimodalContractError("media embedding/span length mismatch")
        for idx, (emb, span) in enumerate(zip(self.embeddings, self.spans)):
            arr = np.asarray(emb)
            expected = (span.length, self.hidden_size)
            if arr.shape != expected:
                raise MultimodalContractError(
                    f"media embedding {idx} shape {arr.shape} != expected {expected}")


@dataclass(frozen=True)
class MediaProcessorConfig:
    image_size: int
    patch_size: int
    image_seq_length: int
    spatial_merge_size: int = 1
    temporal_patch_size: int = 1
    max_frames: int = 1
    num_channels: int = 3
    do_resize: bool = True
    resize_mode: str = "nearest"
    rescale_factor: float = 1.0 / 255.0
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)

    def validate(self) -> None:
        if self.image_size <= 0:
            raise MultimodalContractError("image_size must be positive")
        if self.patch_size <= 0:
            raise MultimodalContractError("patch_size must be positive")
        if self.image_seq_length <= 0:
            raise MultimodalContractError("image_seq_length must be positive")
        if self.spatial_merge_size <= 0:
            raise MultimodalContractError("spatial_merge_size must be positive")
        if self.temporal_patch_size <= 0:
            raise MultimodalContractError("temporal_patch_size must be positive")
        if self.max_frames <= 0:
            raise MultimodalContractError("max_frames must be positive")
        if self.num_channels <= 0:
            raise MultimodalContractError("num_channels must be positive")
        if self.resize_mode not in {"nearest"}:
            raise MultimodalContractError(f"unsupported resize_mode {self.resize_mode!r}")
        if len(self.image_mean) not in {1, self.num_channels}:
            raise MultimodalContractError("image_mean length must be 1 or num_channels")
        if len(self.image_std) not in {1, self.num_channels}:
            raise MultimodalContractError("image_std length must be 1 or num_channels")
        if any(float(v) == 0.0 for v in self.image_std):
            raise MultimodalContractError("image_std values must be non-zero")

    def patch_grid(self, *, frames: int = 1) -> PatchGrid:
        if frames < 1 or frames > self.max_frames:
            raise MultimodalContractError(f"frames={frames} outside [1, {self.max_frames}]")
        grid = max(1, self.image_size // self.patch_size)
        return PatchGrid(
            frames=frames,
            grid_h=grid,
            grid_w=grid,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size,
        )


def processor_config_from_hf(
    cfg: Mapping[str, Any],
    *,
    default: MediaProcessorConfig,
) -> MediaProcessorConfig:
    """Read common HF processor/image-processor keys into a stable contract."""
    raw_nested = cfg.get("image_processor")
    nested = dict(raw_nested) if isinstance(raw_nested, Mapping) else {}
    merged: dict[str, Any] = {**nested, **dict(cfg)}

    def pick(*names: str, fallback: Any) -> Any:
        for name in names:
            if name in merged:
                return merged[name]
        return fallback

    size = pick("image_size", "size", fallback=default.image_size)
    if isinstance(size, Mapping):
        size = size.get("height", size.get("shortest_edge", default.image_size))
    mean = pick("image_mean", "mean", fallback=default.image_mean)
    std = pick("image_std", "std", fallback=default.image_std)
    out = MediaProcessorConfig(
        image_size=int(size),
        patch_size=int(pick("patch_size", fallback=default.patch_size)),
        image_seq_length=int(pick("image_seq_length", "image_feature_size", fallback=default.image_seq_length)),
        spatial_merge_size=int(pick("spatial_merge_size", "patch_merge_spatial", fallback=default.spatial_merge_size)),
        temporal_patch_size=int(pick("temporal_patch_size", "patch_merge_temporal", fallback=default.temporal_patch_size)),
        max_frames=int(pick("max_frames", "vision_segment_max_frames", fallback=default.max_frames)),
        num_channels=int(pick("num_channels", "image_channels", fallback=default.num_channels)),
        do_resize=bool(pick("do_resize", fallback=default.do_resize)),
        resize_mode=str(pick("resize_mode", "resample", fallback=default.resize_mode)).lower(),
        rescale_factor=float(pick("rescale_factor", fallback=default.rescale_factor)),
        image_mean=tuple(float(v) for v in _as_sequence(mean)),
        image_std=tuple(float(v) for v in _as_sequence(std)),
    )
    out.validate()
    return out


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, list | tuple):
        return value
    return (value,)


def processor_config_from_metadata(
    *,
    image_size: int,
    patch_size: int,
    image_seq_length: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
    max_frames: int,
) -> MediaProcessorConfig:
    out = MediaProcessorConfig(
        image_size=image_size,
        patch_size=patch_size,
        image_seq_length=image_seq_length,
        spatial_merge_size=spatial_merge_size,
        temporal_patch_size=temporal_patch_size,
        max_frames=max_frames,
    )
    out.validate()
    return out


__all__ = [
    "MediaSegment",
    "MediaSpan",
    "MediaBatch",
    "MediaProcessorConfig",
    "MultimodalContractError",
    "PatchGrid",
    "ProjectedMediaEmbeddings",
    "processor_config_from_hf",
    "processor_config_from_metadata",
]
