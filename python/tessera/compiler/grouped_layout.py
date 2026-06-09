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


# ── Runtime enforcement (Rung A) ────────────────────────────────────────────
# Grouped-GEMM families the Apple GPU backend can actually express today.  The
# fused MSL grouped_gemm kernel folds per-token routing over a contiguous token
# layout (dense = the degenerate single-group case).  masked / k_grouped need a
# per-group *tiled* kernel that does not exist on Apple — they are rejected with
# a stable diagnostic (Decision #21) rather than silently computing contiguous.
APPLE_SUPPORTED_GROUPED_KINDS = ("dense", "contiguous")


def grouped_kind_unsupported_message(kind: str, *, target: str,
                                     op: str = "grouped_gemm") -> str:
    """Decision #21 message: names the op, the target, the rejected kind, and
    why — never a silent fall-through to a different family."""
    return (
        f"{op}: grouped_layout kind={kind!r} is not supported on target "
        f"{target!r}. The {target} backend implements only "
        f"{', '.join(APPLE_SUPPORTED_GROUPED_KINDS)} (masked / k_grouped require a "
        f"per-group tiled kernel — tracked for Phase G). Refusing to silently "
        f"compute a contiguous result for a different grouped family.")


def validate_grouped_alignment(group_sizes: Any, alignment: int,
                               *, op: str = "grouped_gemm") -> None:
    """Raise if any group's row count is not a multiple of ``alignment``.

    The contiguous/tiled grouped GEMM requires each group's M extent to be
    aligned to the kernel tile boundary so groups tile cleanly.  Only enforced
    when a caller explicitly declares ``alignment`` (the default per-token fused
    kernel is alignment-agnostic), so a plain ``grouped_gemm`` with ragged group
    sizes is unaffected.
    """
    if alignment is None or alignment <= 1:
        return
    gs = np.asarray(group_sizes).astype(np.int64).reshape(-1)
    bad = [(int(i), int(n)) for i, n in enumerate(gs) if int(n) % alignment != 0]
    if bad:
        raise ValueError(
            f"{op}: grouped_layout alignment={alignment} requires every group's "
            f"row count to be a multiple of {alignment}; offending (group, size) "
            f"pairs: {bad}. Pad the groups to the alignment, or omit alignment "
            f"to use the alignment-agnostic per-token path.")


# ── Quantized grouped GEMM (Rung B) ─────────────────────────────────────────
# Low-precision dtypes that have a runtime dequant-on-host grouped-GEMM path.
QUANT_GROUPED_DTYPES = ("fp8_e4m3", "fp8_e5m2", "fp8", "nvfp4")


def apply_quant_for_grouped(xa: Any, w: Any, quant: str, *,
                            quantize_fp8: Any, quantize_nvfp4: Any,
                            dequantize_nvfp4: Any) -> "tuple[np.ndarray, np.ndarray]":
    """Quantize-then-dequantize x ``(T,K)`` and per-expert w ``(E,K,N)`` per the
    canonical scale layout for ``quant``, returning fp32 arrays ready for the
    f32 grouped GEMM (the dequant-on-host quantized grouped-GEMM capability).

    The quantizers are dependency-injected so this stays a leaf module (no
    import of the ``tessera.ops`` namespace).
    """
    d = str(quant).lower()
    if d not in QUANT_GROUPED_DTYPES:
        raise ValueError(
            f"grouped_gemm: quant dtype {quant!r} has no dequant-on-host grouped "
            f"path; expected one of {QUANT_GROUPED_DTYPES}")
    xa = np.asarray(xa, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    E = int(w.shape[0])
    if d in ("fp8_e4m3", "fp8", "fp8_e5m2"):
        fmt = "e5m2" if d == "fp8_e5m2" else "e4m3"
        xa = np.asarray(quantize_fp8(xa, format=fmt)[0], dtype=np.float32)
        w = np.stack([np.asarray(quantize_fp8(w[e], format=fmt)[0], dtype=np.float32)
                      for e in range(E)])
    elif d == "nvfp4":
        bs = 16
        xa = np.asarray(dequantize_nvfp4(*quantize_nvfp4(xa, block_size=bs), block_size=bs),
                        dtype=np.float32)
        w = np.stack([
            np.asarray(dequantize_nvfp4(*quantize_nvfp4(w[e], block_size=bs), block_size=bs),
                       dtype=np.float32) for e in range(E)])
    else:  # pragma: no cover — guarded by scale_layout_for above
        raise ValueError(f"grouped_gemm: quant={quant!r} has no runtime dequant path")
    return xa, w


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
