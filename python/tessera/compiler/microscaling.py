"""Low-precision (FP8 / FP4 / MX) quantization contract — the compiler-side
"scale layout as a first-class IR operand" surface (M3, hardware-free).

Apple7 / Metal 4 expose FP8 (E4M3/E5M2), FP4 (E2M1) and MX block-scale (UE8M0)
as MTLTensor formats (see the apple7-m1max-gpu-feature-set memo), but the macOS
26.5 SDK on this machine does not yet expose those tensor formats through the
public Metal API — so real-silicon execution is toolchain-gated. What is *not*
gated is the compiler contract: how Tessera represents these dtypes, their
quantization semantics, and crucially the **scale layout** (the per-block shared
exponent for microscaled formats) as a first-class typed operand alongside the
data — the DeepGEMM extraction (see the deepgemm_compiler_extraction memo).

This module is the bit-accurate numpy reference + the typed contract. When a
Metal SDK exposes the FP8/FP4 MTLTensorDataType cases, the runtime lowering
plugs in beneath this contract; the Evaluator's metamorphic oracle
(``mx_matmul`` ≈ fp32 matmul within the quantization error bound) is the proof
that survives the transition.

All quantization math is faithful to the OCP Microscaling (MX) spec and NVIDIA's
NVFP4: a tensor is partitioned into contiguous blocks along an axis; each block
carries one shared scale; elements are stored in the low-precision element type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import ml_dtypes as _ml
except Exception as _exc:  # pragma: no cover - ml_dtypes is a declared dep
    raise ImportError("microscaling requires ml_dtypes") from _exc


# ── element + scale dtypes (bit-accurate via ml_dtypes) ─────────────────────
_FP8_E4M3 = _ml.float8_e4m3fn      # OCP/MX fp8 e4m3 (finite, no inf)
_FP8_E5M2 = _ml.float8_e5m2
_FP4_E2M1 = _ml.float4_e2m1fn      # OCP fp4 e2m1
_E8M0 = _ml.float8_e8m0fnu         # MX shared-exponent scale (power-of-2)


@dataclass(frozen=True)
class ScaleLayout:
    """The layout of the *scale* operand relative to the *data* operand — the
    first-class contract a microscaled tensor carries. ``block_size`` elements
    along ``axis`` share one scale of dtype ``scale_dtype``; the scale tensor's
    shape is the data shape with ``axis`` divided by ``block_size`` (ceil).

    ``block_size == 0`` denotes a single per-tensor scale (plain FP8, no MX)."""

    block_size: int
    axis: int = -1
    scale_dtype: str = "e8m0"          # "e8m0" (MX) | "fp8_e4m3" (NVFP4) | "fp32"

    def scale_shape(self, data_shape: tuple[int, ...]) -> tuple[int, ...]:
        if self.block_size == 0:
            return ()                  # per-tensor scalar scale
        ax = self.axis if self.axis >= 0 else len(data_shape) + self.axis
        n = data_shape[ax]
        nblocks = (n + self.block_size - 1) // self.block_size
        return tuple(nblocks if i == ax else d
                     for i, d in enumerate(data_shape))


@dataclass(frozen=True)
class LowPrecisionFormat:
    """A named low-precision storage format: element dtype + scale layout."""

    name: str
    element: str                       # "fp8_e4m3" | "fp8_e5m2" | "fp4_e2m1"
    layout: ScaleLayout


_ELEMENT_DTYPE = {
    "fp8_e4m3": _FP8_E4M3,
    "fp8_e5m2": _FP8_E5M2,
    "fp4_e2m1": _FP4_E2M1,
    "int8": np.int8,                   # integer element (round+clip, not a float grid)
}
#: Integer element types and their symmetric clip magnitude.
_INTEGER_ELEMENTS = {"int8": 127.0}
_SCALE_DTYPE = {"e8m0": _E8M0, "fp8_e4m3": _FP8_E4M3, "fp32": np.float32}

#: Named format variants (OCP-MX block-32, NVFP4 block-16 E4M3, per-tensor int8).
#: The *bare* registry dtype names (``fp8_e4m3``/``fp8_e5m2``/``fp4_e2m1``)
#: resolve via :func:`format_for_dtype` — the registry's declared contract — so
#: there is one source of truth for those; the ``mx*`` names are the OCP-MX
#: variants kept distinct here.
FORMATS: dict[str, LowPrecisionFormat] = {
    "mxfp8_e4m3": LowPrecisionFormat(
        "mxfp8_e4m3", "fp8_e4m3", ScaleLayout(32, -1, "e8m0")),
    "mxfp4_e2m1": LowPrecisionFormat(
        "mxfp4_e2m1", "fp4_e2m1", ScaleLayout(32, -1, "e8m0")),
    "nvfp4": LowPrecisionFormat(
        "nvfp4", "fp4_e2m1", ScaleLayout(16, -1, "fp8_e4m3")),
    # int8 — per-tensor symmetric (integer element, fp32 scale).
    "int8": LowPrecisionFormat("int8", "int8", ScaleLayout(0, -1, "fp32")),
}

# Registry-declared element dtype per low-precision dtype name (the data type;
# the scale layout is derived from grouped_layout.scale_layout_for).
_REGISTRY_ELEMENT = {
    "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2", "fp8": "fp8_e4m3",
    "fp4_e2m1": "fp4_e2m1", "fp4": "fp4_e2m1", "nvfp4": "fp4_e2m1",
    "int8": "int8",
}
_PACKING_TO_SCALE_DTYPE = {
    "ue8m0": "e8m0", "e8m0": "e8m0", "e4m3": "fp8_e4m3",
    "e5m2": "fp8_e5m2", "none": "fp32",
}


def format_for_dtype(dtype: str) -> LowPrecisionFormat | None:
    """Derive the executable low-precision format **from the registry's
    declared scale-layout contract** (`grouped_layout.scale_layout_for`) — so
    this bit-accurate reference and the audited compiler contract share one
    source of truth (block size, packing, granularity). Returns ``None`` for a
    dtype with no declared scale layout."""
    from .grouped_layout import scale_layout_for
    sl = scale_layout_for(dtype)
    element = _REGISTRY_ELEMENT.get(str(dtype).lower())
    if sl is None or element is None:
        return None
    if sl.granularity == "per_tensor":
        layout = ScaleLayout(0, -1, _PACKING_TO_SCALE_DTYPE.get(sl.packing, "fp32"))
    elif sl.granularity == "block" and sl.block is not None:
        # grouped_layout block is (rows, cols); the scale is shared along the
        # contraction (cols) axis → block_size = cols, axis = -1.
        layout = ScaleLayout(int(sl.block[1]), -1,
                             _PACKING_TO_SCALE_DTYPE.get(sl.packing, "fp32"))
    else:                              # per_row / per_channel — not modeled here
        return None
    return LowPrecisionFormat(str(dtype), element, layout)


# ── Metal bridge: the macOS 27.0 MTLTensorDataType + auxiliary-plane mapping ──
#
# Grounded in the macOS 27.0 SDK doc dump (Decision #27, authority tier 2). The
# low-precision element types (float8e4m3/float8e5m2/float4e2m1) and the MX
# shared-exponent scale type (float8ue8m0) land in macOS 27.0, alongside the
# multi-plane tensor machinery (MTLTensorAuxiliaryPlaneDescriptor.blockFactors /
# auxiliaryPlanes / MTLTensorBufferAttachments) — which is exactly the runtime
# representation of a ScaleLayout: one *data* plane in the element dtype plus one
# *auxiliary* scale plane whose ``blockFactors`` say how many data elements share
# one scale. This bridge validates that Tessera's hardware-free contract maps 1:1
# onto the concrete API and gives the future runtime its construction recipe.
#
# This machine is macOS 26.5 (SDK tops out at Int4/UInt4 @ 26.4 — no float8/float4
# /e8m0, no auxiliaryPlanes), so the *execution* path stays gated on a 27.0 SDK;
# the mapping itself is hardware-free and unit-testable now.
#
# The doc dump gives the Swift case names (``.float8e4m3`` …); ``mtl_symbol`` is
# the conventional ObjC enum spelling derived from them (not separately verified).

@dataclass(frozen=True)
class MetalTensorType:
    """A Tessera dtype's image under ``MTLTensorDataType``. ``min_macos`` is the
    SDK version that first exposes it (``"27.0"`` for fp8/fp4/e8m0; ``"26.4"`` for
    int4/uint4; ``"26.0"`` for the long-standing int8/float types)."""

    tessera_dtype: str
    swift_case: str           # authoritative from the 27.0 doc dump
    mtl_symbol: str           # conventional ObjC MTLTensorDataType* spelling
    min_macos: str


#: Tessera element/scale dtype name → its MTLTensorDataType image + availability.
_MTL_TENSOR_DATA_TYPE: dict[str, MetalTensorType] = {
    "fp8_e4m3": MetalTensorType("fp8_e4m3", "float8e4m3", "MTLTensorDataTypeFloat8E4M3", "27.0"),
    "fp8_e5m2": MetalTensorType("fp8_e5m2", "float8e5m2", "MTLTensorDataTypeFloat8E5M2", "27.0"),
    "fp4_e2m1": MetalTensorType("fp4_e2m1", "float4e2m1", "MTLTensorDataTypeFloat4E2M1", "27.0"),
    "e8m0":     MetalTensorType("e8m0", "float8ue8m0", "MTLTensorDataTypeFloat8UE8M0", "27.0"),
    "int8":     MetalTensorType("int8", "int8", "MTLTensorDataTypeInt8", "26.0"),
    "int4":     MetalTensorType("int4", "int4", "MTLTensorDataTypeInt4", "26.4"),
    "uint4":    MetalTensorType("uint4", "uint4", "MTLTensorDataTypeUInt4", "26.4"),
    "int2":     MetalTensorType("int2", "int2", "MTLTensorDataTypeInt2", "27.0"),
    "uint2":    MetalTensorType("uint2", "uint2", "MTLTensorDataTypeUInt2", "27.0"),
    "fp32":     MetalTensorType("fp32", "float32", "MTLTensorDataTypeFloat32", "26.0"),
}


@dataclass(frozen=True)
class MetalAuxiliaryPlane:
    """An MTLTensorAuxiliaryPlaneDescriptor for a scale plane: ``data_type`` is the
    scale's MTLTensorDataType symbol, ``block_factors`` is the per-axis
    MTLTensorExtents (data-plane elements per scale element — ``block_size`` on
    the scale axis, ``1`` elsewhere), ``scale_shape`` is the plane's logical
    shape."""

    data_type: str
    block_factors: tuple[int, ...]
    scale_shape: tuple[int, ...]


@dataclass(frozen=True)
class MetalPlanePlan:
    """The multi-plane MTLTensor construction recipe for a microscaled format:
    one data plane (element dtype) + zero-or-one auxiliary scale plane.
    ``min_macos`` is the highest SDK version any required type/feature needs."""

    element: MetalTensorType
    aux_planes: tuple[MetalAuxiliaryPlane, ...]
    min_macos: str


def _max_macos(*versions: str) -> str:
    return max(versions, key=lambda v: tuple(int(p) for p in v.split(".")))


def mtl_tensor_data_type(dtype: str) -> MetalTensorType | None:
    """The MTLTensorDataType image of a Tessera element/scale dtype, or None if
    the dtype has no Metal tensor type."""
    return _MTL_TENSOR_DATA_TYPE.get(str(dtype).lower())


def metal_plane_plan(fmt: LowPrecisionFormat,
                     data_shape: tuple[int, ...]) -> MetalPlanePlan | None:
    """Map a :class:`LowPrecisionFormat` onto its macOS 27.0 multi-plane MTLTensor
    layout. A per-tensor format (``block_size == 0``) needs no auxiliary plane
    (the scalar scale is applied in-kernel); a block-scaled (MX/NVFP4) format
    gets one scale plane whose ``blockFactors`` encode the block size on the scale
    axis. Returns None if any dtype lacks a Metal image."""
    elem = mtl_tensor_data_type(fmt.element)
    if elem is None:
        return None
    layout = fmt.layout
    if layout.block_size == 0:
        return MetalPlanePlan(elem, (), elem.min_macos)
    scale = mtl_tensor_data_type(layout.scale_dtype)
    if scale is None:
        return None
    rank = len(data_shape)
    ax = layout.axis if layout.axis >= 0 else rank + layout.axis
    block_factors = tuple(layout.block_size if i == ax else 1 for i in range(rank))
    plane = MetalAuxiliaryPlane(scale.mtl_symbol, block_factors,
                                layout.scale_shape(data_shape))
    # an auxiliary scale plane itself requires the 27.0 multi-plane machinery
    return MetalPlanePlan(
        elem, (plane,),
        _max_macos(elem.min_macos, scale.min_macos, "27.0"))


def metal_microscaling_available() -> bool:
    """True iff the Apple GPU runtime can actually drive microscaled (FP8/FP4/MX)
    tensors — i.e. it was built against a macOS >=27.0 SDK *and* is running on
    macOS 27.0+. False on the 26.5 toolchain (the contract + Metal bridge are
    hardware-free; only the multi-plane MTLTensor execution is version-gated).
    Probes the runtime's ``tessera_apple_gpu_supports_microscaling`` symbol."""
    try:
        import ctypes

        from .._apple_gpu_dispatch import apple_gpu_runtime
        lib = apple_gpu_runtime()
        if lib is None:
            return False
        lib.tessera_apple_gpu_supports_microscaling.restype = ctypes.c_int32
        return bool(lib.tessera_apple_gpu_supports_microscaling())
    except Exception:
        return False


def _elem_amax(fmt: LowPrecisionFormat) -> float:
    """Largest finite magnitude representable by the element dtype."""
    if fmt.element in _INTEGER_ELEMENTS:
        return _INTEGER_ELEMENTS[fmt.element]
    dt = _ELEMENT_DTYPE[fmt.element]
    return float(_ml.finfo(dt).max)


def _encode(scaled: np.ndarray, fmt: LowPrecisionFormat) -> np.ndarray:
    """Cast scaled values onto the element grid — round+clip for integer
    elements, the dtype's native rounding for float (fp8/fp4) elements."""
    dt = _ELEMENT_DTYPE[fmt.element]
    if fmt.element in _INTEGER_ELEMENTS:
        amax = _INTEGER_ELEMENTS[fmt.element]
        return np.clip(np.round(scaled), -amax, amax).astype(dt)
    return scaled.astype(dt)


@dataclass(frozen=True)
class MicroscaledArray:
    """A quantized tensor as a first-class (codes, scales, format) triple — the
    'scale layout as IR operand' representation. ``codes`` holds the element
    dtype; ``scales`` is the separate scale operand whose shape is dictated by
    ``format.layout.scale_shape(shape)``."""

    codes: np.ndarray                  # element dtype (fp8/fp4)
    scales: np.ndarray                 # scale dtype, per the layout
    format: LowPrecisionFormat
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        want = self.format.layout.scale_shape(self.shape)
        if tuple(self.scales.shape) != want:
            raise ValueError(
                f"scale layout violation: scales {self.scales.shape} != "
                f"required {want} for {self.format.name} over shape {self.shape}")


def _round_to_e8m0(scale: np.ndarray) -> np.ndarray:
    """Round a positive scale up to the nearest power of two (the E8M0 grid)."""
    scale = np.maximum(np.asarray(scale, np.float32), np.float32(2.0) ** -127)
    return (np.float32(2.0) ** np.ceil(np.log2(scale))).astype(np.float32)


def resolve_format(fmt: str | LowPrecisionFormat) -> LowPrecisionFormat:
    """A named variant from ``FORMATS``, else the registry's declared contract
    (``format_for_dtype``) for a bare dtype name."""
    if not isinstance(fmt, str):
        return fmt
    f = FORMATS.get(fmt) or format_for_dtype(fmt)
    if f is None:
        raise ValueError(f"unknown low-precision format {fmt!r}")
    return f


def quantize(x: Any, fmt: str | LowPrecisionFormat) -> MicroscaledArray:
    """Quantize ``x`` to a low-precision format, returning the (codes, scales)
    pair with the scale operand laid out per the format. Per-block (MX/NVFP4/
    DeepGEMM) or per-tensor (int8). Faithful to the declared scale contract."""
    f = resolve_format(fmt)
    x = np.asarray(x, np.float32)
    elem_max = _elem_amax(f)
    lay = f.layout

    if lay.block_size == 0:            # per-tensor scale
        amax = float(np.abs(x).max()) or 1.0
        scale = np.asarray(amax / elem_max, np.float32)
        codes = _encode(x / scale, f)
        return MicroscaledArray(codes, scale.reshape(()), f, tuple(x.shape))

    ax = lay.axis if lay.axis >= 0 else x.ndim + lay.axis
    n = x.shape[ax]
    bs = lay.block_size
    nblocks = (n + bs - 1) // bs
    pad = nblocks * bs - n
    xp = np.moveaxis(x, ax, -1)
    lead = xp.shape[:-1]
    if pad:
        xp = np.pad(xp, [(0, 0)] * (xp.ndim - 1) + [(0, pad)])
    blocks = xp.reshape(*lead, nblocks, bs)
    block_amax = np.abs(blocks).max(axis=-1)               # (*lead, nblocks)
    raw_scale = np.maximum(block_amax / elem_max, np.float32(1e-30))
    scale_dt = _SCALE_DTYPE[lay.scale_dtype]
    if lay.scale_dtype == "e8m0":
        scale = _round_to_e8m0(raw_scale).astype(scale_dt)
    else:                              # fp8/fp32 scale: round through the dtype
        scale = raw_scale.astype(scale_dt)
    scale_f = scale.astype(np.float32)[..., None]
    codes_b = _encode(blocks / scale_f, f)
    codes = codes_b.reshape(*lead, nblocks * bs)[..., :n]
    codes = np.moveaxis(codes, -1, ax)
    scales = np.moveaxis(scale, -1, ax) if scale.ndim else scale
    return MicroscaledArray(np.ascontiguousarray(codes),
                            np.ascontiguousarray(scales), f, tuple(x.shape))


def dequantize(mx: MicroscaledArray) -> np.ndarray:
    """Reconstruct the fp32 tensor from (codes, scales)."""
    f = mx.format
    lay = f.layout
    codes = np.asarray(mx.codes).astype(np.float32)
    if lay.block_size == 0:
        return codes * np.asarray(mx.scales, np.float32)
    ax = lay.axis if lay.axis >= 0 else codes.ndim + lay.axis
    n = codes.shape[ax]
    bs = lay.block_size
    cp = np.moveaxis(codes, ax, -1)
    lead = cp.shape[:-1]
    nblocks = (n + bs - 1) // bs
    pad = nblocks * bs - n
    if pad:
        cp = np.pad(cp, [(0, 0)] * (cp.ndim - 1) + [(0, pad)])
    blocks = cp.reshape(*lead, nblocks, bs)
    scale = np.moveaxis(np.asarray(mx.scales, np.float32), ax, -1)[..., None]
    out = (blocks * scale).reshape(*lead, nblocks * bs)[..., :n]
    return np.moveaxis(out, -1, ax)


def fake_quantize(x: Any, fmt: str | LowPrecisionFormat) -> np.ndarray:
    """Quantize then dequantize — the value ``x`` rounded onto the format's
    grid (QAT / error-analysis helper)."""
    return dequantize(quantize(x, fmt))


def mx_matmul(a: Any, b: Any, fmt: str = "mxfp8_e4m3") -> np.ndarray:
    """A microscaled GEMM reference: quantize both operands to ``fmt``,
    dequantize, and contract in fp32 (the numerics a real FP8/FP4 tensor-core
    GEMM with fp32 accumulation produces). The Evaluator's metamorphic oracle
    compares this to the full-precision matmul within the format's error bound."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return fake_quantize(a, fmt) @ fake_quantize(b, fmt)


def numeric_policy_for(fmt: str | LowPrecisionFormat) -> dict[str, Any]:
    """The numeric-policy contract for a low-precision format — storage element,
    fp32 accumulate, and the scale layout (block size / axis / scale dtype) as a
    first-class field rather than a fused dtype string."""
    f = resolve_format(fmt)
    return {
        "storage": f.element,
        "accum": "fp32",
        "scale_block_size": f.layout.block_size,
        "scale_axis": f.layout.axis,
        "scale_dtype": f.layout.scale_dtype,
        "quant_axis": f.layout.axis,
    }
