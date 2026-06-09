"""Grouped-GEMM + scale-layout contracts (DeepGEMM-inspired design rung).

DeepGEMM treats grouped GEMM as a *first-class op family* and makes FP8/FP4
scale *layout* (not just the dtype name) a compiler-visible contract.  This
module ports those ideas as Tessera contract metadata — pure data, no kernel
import.  It is a leaf module (depends only on the stdlib) so the audit registry
(`primitive_coverage`), `op_catalog`, and runtime can all import it without a
cycle.

Two contracts:

* :class:`GroupedLayout` — how a grouped GEMM lays its groups out: the family
  (dense / M-grouped contiguous / M-grouped masked / K-grouped), the grouped
  axis, per-group alignment, and which problem dims are JIT-specialized
  (``compiled_dims``) vs left runtime-dynamic (``dynamic_dims``).

* :class:`ScaleLayout` — how an FP8/FP4 scale tensor is shaped + packed:
  granularity (per-tensor / per-row / per-channel / block), block shape, the
  packed scale element format, the vector size sharing one scale, tensor
  alignment, and whether the scale tensor is TMA-ready transposed (MN-major).

Both flatten to plain JSON-style dicts via ``as_metadata_dict`` so they ride on
``PrimitiveCoverage.metadata`` and (later) Graph/Schedule/Tile/Target IR attrs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Grouped-GEMM families (DeepGEMM's four) ─────────────────────────────────
GROUPED_KINDS = ("dense", "contiguous", "masked", "k_grouped")

# group_axis implied by the family — contiguous/masked group along M (the token
# axis); k_grouped groups along the K contraction axis; dense has no grouping.
_GROUP_AXIS_BY_KIND = {
    "dense": None,
    "contiguous": "M",
    "masked": "M",
    "k_grouped": "K",
}

SCALE_GRANULARITIES = ("per_tensor", "per_row", "per_channel", "block")
# Packed scale element formats.  "none" = unscaled / scale is a plain float;
# e4m3/e5m2 = FP8-encoded scale; e8m0/ue8m0 = (unsigned) 8-bit exponent-only
# scale (DeepGEMM's Blackwell UE8M0 packed scale).
SCALE_PACKINGS = ("none", "e4m3", "e5m2", "e8m0", "ue8m0")


@dataclass(frozen=True)
class GroupedLayout:
    """Compiler-visible contract for one grouped-GEMM op.

    ``compiled_dims`` are the problem dims baked into a specialized kernel (so
    the JIT/autotuner can pick tile shapes for them); ``dynamic_dims`` stay
    runtime values (the group count + the grouped axis extent are not known at
    compile time).  Mirrors DeepGEMM compiling N/K while leaving M + num_groups
    dynamic.
    """

    kind: str
    alignment: int = 128
    compiled_dims: tuple[str, ...] = ("N", "K")
    dynamic_dims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in GROUPED_KINDS:
            raise ValueError(
                f"grouped kind must be one of {GROUPED_KINDS}; got {self.kind!r}")
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError(
                f"grouped alignment must be a positive power of two; got {self.alignment}")
        # Default dynamic_dims from the family when the caller didn't set them.
        if not self.dynamic_dims:
            if self.kind == "dense":
                object.__setattr__(self, "dynamic_dims", ("M",))
            elif self.kind == "k_grouped":
                object.__setattr__(self, "dynamic_dims", ("K", "num_groups"))
            else:  # contiguous / masked
                object.__setattr__(self, "dynamic_dims", ("M", "num_groups"))

    @property
    def group_axis(self) -> str | None:
        return _GROUP_AXIS_BY_KIND[self.kind]

    @property
    def is_grouped(self) -> bool:
        return self.kind != "dense"

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "group_axis": self.group_axis,
            "alignment": self.alignment,
            "compiled_dims": list(self.compiled_dims),
            "dynamic_dims": list(self.dynamic_dims),
        }


@dataclass(frozen=True)
class ScaleLayout:
    """Compiler-visible contract for an FP8/FP4 (de)quantization scale tensor."""

    granularity: str
    block: tuple[int, int] | None = None
    packing: str = "none"
    vector_size: int = 1
    alignment: int = 1
    transposed: bool = False

    def __post_init__(self) -> None:
        if self.granularity not in SCALE_GRANULARITIES:
            raise ValueError(
                f"scale granularity must be one of {SCALE_GRANULARITIES}; "
                f"got {self.granularity!r}")
        if self.packing not in SCALE_PACKINGS:
            raise ValueError(
                f"scale packing must be one of {SCALE_PACKINGS}; got {self.packing!r}")
        if self.granularity == "block":
            if self.block is None:
                raise ValueError("block granularity requires a block=(rows, cols) shape")
            if (len(self.block) != 2 or self.block[0] <= 0 or self.block[1] <= 0):
                raise ValueError(f"block must be a 2-tuple of positive ints; got {self.block}")
        elif self.block is not None:
            raise ValueError(
                f"block shape is only valid for granularity='block'; got {self.granularity!r}")
        if self.vector_size <= 0:
            raise ValueError(f"vector_size must be positive; got {self.vector_size}")
        if self.alignment <= 0:
            raise ValueError(f"alignment must be positive; got {self.alignment}")

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "block": list(self.block) if self.block is not None else None,
            "packing": self.packing,
            "vector_size": self.vector_size,
            "alignment": self.alignment,
            "transposed": self.transposed,
        }


# ── Family factory helpers ──────────────────────────────────────────────────
def dense_layout(**kw: Any) -> GroupedLayout:
    return GroupedLayout(kind="dense", **kw)


def contiguous_layout(alignment: int = 128, **kw: Any) -> GroupedLayout:
    return GroupedLayout(kind="contiguous", alignment=alignment, **kw)


def masked_layout(alignment: int = 128, **kw: Any) -> GroupedLayout:
    return GroupedLayout(kind="masked", alignment=alignment, **kw)


def k_grouped_layout(alignment: int = 128, **kw: Any) -> GroupedLayout:
    return GroupedLayout(kind="k_grouped", alignment=alignment, **kw)


# Canonical scale layout per low-precision dtype.  These describe the *default*
# production scale layout for each dtype (the dashboard records the contract).
def scale_layout_for(dtype: str) -> ScaleLayout | None:
    d = str(dtype).lower()
    if d in ("nvfp4",):
        # NVFP4: micro-block scaling, 16 elements share one FP8 (e4m3) scale.
        return ScaleLayout(granularity="block", block=(1, 16), packing="e4m3",
                           vector_size=16, alignment=16)
    if d in ("fp4_e2m1", "fp4"):
        # 1x128 block scaling with a UE8M0 (exponent-only) packed scale.
        return ScaleLayout(granularity="block", block=(1, 128), packing="ue8m0",
                           vector_size=128, alignment=128)
    if d in ("fp8_e4m3", "fp8_e5m2", "fp8"):
        # 1x128 block-scaled FP8 (DeepGEMM's default), UE8M0 packed scale.
        return ScaleLayout(granularity="block", block=(1, 128), packing="ue8m0",
                           vector_size=128, alignment=128)
    if d in ("fp6_e2m3", "fp6_e3m2", "fp6"):
        return ScaleLayout(granularity="block", block=(1, 128), packing="ue8m0",
                           vector_size=128, alignment=128)
    if d in ("int8",):
        return ScaleLayout(granularity="per_tensor", packing="none")
    return None


# ── Oracle reference (step 4) ───────────────────────────────────────────────
def reference_grouped_gemm(x: Any, w: Any, group_sizes: Any) -> "np.ndarray":
    """fp32 reference for an M-grouped contiguous GEMM.

    x: (T, K) tokens (groups laid out contiguously along T); w: (E, K, N) per-
    expert weights; group_sizes: (E,) token counts per expert.  Returns (T, N).
    This is the oracle the backend-selected (and FP8/FP4-dequantized) grouped
    GEMMs are compared against.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    gs = np.asarray(group_sizes).astype(np.int64)
    out = np.zeros((x.shape[0], w.shape[2]), dtype=np.float64)
    off = 0
    for e in range(w.shape[0]):
        n = int(gs[e])
        if n:
            out[off:off + n] = x[off:off + n] @ w[e]
        off += n
    return out
