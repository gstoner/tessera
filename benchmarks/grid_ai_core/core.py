"""Generic gridded-AI benchmark core.

The benchmark is deliberately small and domain-neutral.  It is meant to prove
that a library can be built above Tessera's generic primitives for gridded
regional AI workloads without making the compiler know about any specific
application.

Forward path:

  tiled NHWC field input
    -> local 5-point stencil feature
    -> conv2d + fused epilogue
    -> attn_local_window_2d over H/W
    -> deterministic RNG/noise step
    -> output conv2d

The matching IR fixture in ``tests/tessera-ir/phase7`` pins the compiler side:
the stencil and local-window attention both lower through the halo pipeline, and
the conv/RNG ops stay visible as ordinary Graph IR operations.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import tessera as ts
from tessera.rng import RNGKey, normal


@dataclass(frozen=True)
class GridAICoreConfig:
    """Configuration for one tiny gridded-AI core run."""

    B: int = 1
    H: int = 16
    W: int = 16
    C_in: int = 3
    C_hid: int = 8
    C_out: int = 2
    heads: int = 2
    window: tuple[int, int] = (1, 1)
    tile: tuple[int, int] = (8, 8)
    stencil_alpha: float = 0.125
    sigma: float = 0.05
    seed: int = 0


def tile_field(
    arr: np.ndarray,
    tile: tuple[int, int],
    *,
    spatial_axes: tuple[int, int] = (1, 2),
) -> list[np.ndarray]:
    """Split an array into row-major spatial tiles.

    Defaults to NHWC ``(B, H, W, C)`` spatial axes.  The helper returns views so
    tests can verify the benchmark really models tiled input without copying.
    """

    x = np.asarray(arr)
    if x.ndim < 2:
        raise ValueError(f"tile_field expects rank >= 2; got {x.ndim}")
    ax_h, ax_w = tuple(a if a >= 0 else x.ndim + a for a in spatial_axes)
    if ax_h == ax_w:
        raise ValueError("spatial_axes must name two different axes")
    H, W = x.shape[ax_h], x.shape[ax_w]
    th, tw = map(int, tile)
    if th <= 0 or tw <= 0:
        raise ValueError(f"tile sizes must be positive; got {tile!r}")
    if H % th or W % tw:
        raise ValueError(
            f"tile shape {tile} does not divide spatial shape ({H}, {W})"
        )

    out: list[np.ndarray] = []
    for i in range(0, H, th):
        for j in range(0, W, tw):
            slc: list[object] = [slice(None)] * x.ndim
            slc[ax_h] = slice(i, i + th)
            slc[ax_w] = slice(j, j + tw)
            out.append(x[tuple(slc)])
    return out


def local_stencil_feature(
    x: np.ndarray,
    *,
    alpha: float = 0.125,
    boundary: str = "periodic",
) -> np.ndarray:
    """Apply a simple 5-point local feature stencil to an NHWC field.

    ``periodic`` uses wrap-around neighbors.  ``reflect`` follows the compiler
    pass's current clamp-style boundary convention.
    """

    field = np.asarray(x, dtype=np.float32)
    if field.ndim != 4:
        raise ValueError(f"local_stencil_feature expects NHWC rank 4; got {field.ndim}")
    if boundary == "periodic":
        north = np.roll(field, shift=1, axis=1)
        south = np.roll(field, shift=-1, axis=1)
        west = np.roll(field, shift=1, axis=2)
        east = np.roll(field, shift=-1, axis=2)
    elif boundary == "reflect":
        padded = np.pad(field, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        north = padded[:, 0:-2, 1:-1, :]
        south = padded[:, 2:, 1:-1, :]
        west = padded[:, 1:-1, 0:-2, :]
        east = padded[:, 1:-1, 2:, :]
    else:
        raise ValueError(f"unsupported boundary {boundary!r}")
    lap = north + south + west + east - 4.0 * field
    return field + float(alpha) * lap


def deterministic_noise_step(x: np.ndarray, key: RNGKey, sigma: float) -> np.ndarray:
    """Add deterministic normal noise using Tessera's typed RNG key."""

    arr = np.asarray(x, dtype=np.float32)
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative; got {sigma}")
    if sigma == 0:
        return arr.copy()
    return arr + float(sigma) * normal(key, arr.shape, dtype="fp32")


def periodic_halo_oracle(
    fields: Sequence[np.ndarray],
    axes_widths: Sequence[tuple[int, int]],
) -> list[np.ndarray]:
    """Independent numpy oracle for periodic ring halo exchange."""

    nranks = len(fields)
    out = [np.asarray(f).copy() for f in fields]
    for axis, width in axes_widths:
        if width <= 0:
            continue
        snapshot = [f.copy() for f in out]
        for rank in range(nranks):
            lo_dst: list[object] = [slice(None)] * out[rank].ndim
            lo_dst[axis] = slice(0, width)
            hi_src: list[object] = [slice(None)] * out[rank].ndim
            hi_src[axis] = slice(snapshot[rank - 1].shape[axis] - width, None)
            out[rank][tuple(lo_dst)] = snapshot[(rank - 1) % nranks][tuple(hi_src)]

            hi_dst: list[object] = [slice(None)] * out[rank].ndim
            hi_dst[axis] = slice(out[rank].shape[axis] - width, None)
            lo_src: list[object] = [slice(None)] * out[rank].ndim
            lo_src[axis] = slice(0, width)
            out[rank][tuple(hi_dst)] = snapshot[(rank + 1) % nranks][tuple(lo_src)]
    return out


class GridAICoreModel:
    """Tiny domain-neutral model composed only from generic Tessera primitives."""

    def __init__(self, cfg: GridAICoreConfig):
        if cfg.C_hid % cfg.heads:
            raise ValueError("C_hid must be divisible by heads")
        self.cfg = cfg
        root = RNGKey.from_seed(cfg.seed, name="grid_ai_core")
        self.W_stem = self._init(root.fold_in("W_stem"), (3, 3, cfg.C_in, cfg.C_hid))
        self.b_stem = normal(root.fold_in("b_stem"), (cfg.C_hid,), dtype="fp32") * 0.01
        self.W_out = self._init(root.fold_in("W_out"), (1, 1, cfg.C_hid, cfg.C_out))
        self.b_out = normal(root.fold_in("b_out"), (cfg.C_out,), dtype="fp32") * 0.01
        self.noise_key = root.fold_in("noise")

    @staticmethod
    def _init(key: RNGKey, shape: tuple[int, ...]) -> np.ndarray:
        fan_in = float(np.prod(shape[:-1]))
        scale = np.sqrt(2.0 / max(fan_in, 1.0))
        return normal(key, shape, dtype="fp32") * scale

    def forward(self, x: np.ndarray, *, step: int = 0) -> np.ndarray:
        cfg = self.cfg
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape != (cfg.B, cfg.H, cfg.W, cfg.C_in):
            raise ValueError(
                f"input shape {arr.shape} does not match "
                f"(B={cfg.B}, H={cfg.H}, W={cfg.W}, C_in={cfg.C_in})"
            )

        _ = tile_field(arr, cfg.tile)
        local = local_stencil_feature(arr, alpha=cfg.stencil_alpha)
        h = ts.ops.conv2d(local, self.W_stem, padding=1, layout="nhwc")
        h = ts.ops.fused_epilogue(h, bias=self.b_stem, activation="relu")

        head_dim = cfg.C_hid // cfg.heads
        tokens = h.reshape(cfg.B, cfg.H, cfg.W, cfg.heads, head_dim)
        tokens = tokens.transpose(0, 3, 1, 2, 4)
        attended = ts.ops.attn_local_window_2d(
            tokens, tokens, tokens, window=cfg.window
        )
        flat = attended.transpose(0, 2, 3, 1, 4).reshape(
            cfg.B, cfg.H, cfg.W, cfg.C_hid
        )

        noised = deterministic_noise_step(
            flat, self.noise_key.fold_in(int(step)), cfg.sigma
        )
        return ts.ops.conv2d(noised, self.W_out, self.b_out, padding=0, layout="nhwc")

    __call__ = forward


@dataclass(frozen=True)
class GridAICoreResult:
    backend: str
    op: str
    shape: dict[str, object]
    dtype: str
    latency_ms: float
    throughput_msps: float
    memory_bw_gb_s: float
    device: str
    tessera_version: str
    determinism_ok: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GridAICoreBenchmark:
    """Small CPU/reference benchmark harness for the generic core."""

    def __init__(self, *, warmup: int = 1, reps: int = 3):
        self.warmup = int(warmup)
        self.reps = int(reps)

    @staticmethod
    def make_input(cfg: GridAICoreConfig) -> np.ndarray:
        rng = np.random.default_rng(cfg.seed ^ 0xA11CE)
        return rng.standard_normal((cfg.B, cfg.H, cfg.W, cfg.C_in)).astype(np.float32)

    def run_one(self, cfg: GridAICoreConfig) -> GridAICoreResult:
        model = GridAICoreModel(cfg)
        x = self.make_input(cfg)
        for _ in range(self.warmup):
            model(x, step=0)

        start = time.perf_counter()
        for step in range(self.reps):
            model(x, step=step)
        elapsed = (time.perf_counter() - start) / max(self.reps, 1)

        ref = model(x, step=0)
        det = model(x, step=0)
        samples = cfg.B * cfg.H * cfg.W
        bytes_per_step = samples * (cfg.C_in + 2 * cfg.C_hid + cfg.C_out) * 4
        return GridAICoreResult(
            backend="tessera-reference",
            op="grid_ai_core_forward",
            shape={
                "B": cfg.B,
                "H": cfg.H,
                "W": cfg.W,
                "C_in": cfg.C_in,
                "C_hid": cfg.C_hid,
                "C_out": cfg.C_out,
                "heads": cfg.heads,
                "window": list(cfg.window),
                "tile": list(cfg.tile),
            },
            dtype="fp32",
            latency_ms=elapsed * 1000.0,
            throughput_msps=(samples / 1e6) / max(elapsed, 1e-12),
            memory_bw_gb_s=(bytes_per_step / 1e9) / max(elapsed, 1e-12),
            device="cpu",
            tessera_version="pre-alpha",
            determinism_ok=bool(np.array_equal(ref, det)),
        )

    def to_json(self, results: Sequence[GridAICoreResult], path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
