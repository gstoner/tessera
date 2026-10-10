"""Canonical Graph -> Schedule -> launch-Tile handoff for E2E-REAL-3."""

from __future__ import annotations

from functools import cache
import copy
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from pathlib import Path

from .graph_ir import GraphIRModule

if TYPE_CHECKING:  # deferred at runtime -- rocm_target imports late
    from .rocm_target import ROCmTargetProfile


_HASH_RE = re.compile(r'tessera\.schedule_hash = "([0-9a-f]{64})"')


# Mirrors kScheduledSm120MatmulPrefix in
# src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp.
# The runtime selects the scheduled-matmul launcher by this prefix, so it is
# ABI, not cosmetics.
_SM120_SCHEDULED_MATMUL_PREFIX = "nvidia_sm120_scheduled_matmul_"


def with_bounded_dynamic_m(matmul_module: GraphIRModule, bound: int) -> GraphIRModule:
    """Return a Graph matmul module whose M extent has a checked maximum bound."""
    from .graph_ir import tensor_ir_type

    if bound <= 0:
        raise ValueError("dynamic M bound must be positive")
    module = copy.deepcopy(matmul_module)
    if len(module.functions) != 1:
        raise ValueError("bounded dynamic M requires one Graph function")
    function = module.functions[0]
    if len(function.args) != 2 or len(function.result_types) != 1:
        raise ValueError("bounded dynamic M requires a two-input, one-result matmul")
    matmuls = [op for op in function.body if op.op_name == "tessera.matmul"]
    if len(matmuls) != 1:
        raise ValueError("bounded dynamic M requires one Graph matmul operation")
    lhs_type, rhs_type = function.args[0].ir_type, function.args[1].ir_type
    output_type = function.result_types[0]
    try:
        traced_m, k = (int(str(dim)) for dim in lhs_type.shape)
        rhs_k, n = (int(str(dim)) for dim in rhs_type.shape)
        out_m, out_n = (int(str(dim)) for dim in output_type.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("bounded dynamic M requires static traced tensor shapes") from exc
    if bound != traced_m or out_m != bound:
        raise ValueError("dynamic M bound must match the traced LHS capacity")
    if rhs_k != k or out_n != n:
        raise ValueError("bounded dynamic M requires matching static K and N")
    dynamic_lhs = tensor_ir_type(("?", str(k)), lhs_type.dtype, layout=lhs_type.layout)
    dynamic_output = tensor_ir_type(("?", str(n)), output_type.dtype, layout=output_type.layout)
    function.args[0].ir_type = dynamic_lhs
    function.result_types[0] = dynamic_output
    op = matmuls[0]
    op.operand_types[0] = str(dynamic_lhs)
    op.result_type = str(dynamic_output)
    op.inferred_type = dynamic_output
    op.inferred_types = (dynamic_output,)
    op.kwargs["shape_bounds"] = [bound, n, k]
    return module


def with_bounded_dynamic_mk(
    matmul_module: GraphIRModule, m_bound: int, k_bound: int,
) -> GraphIRModule:
    """Return a Graph matmul with independently bounded dynamic M and K axes."""
    from .graph_ir import tensor_ir_type

    if m_bound <= 0 or k_bound <= 0:
        raise ValueError("dynamic M and K bounds must be positive")
    module = copy.deepcopy(matmul_module)
    if len(module.functions) != 1:
        raise ValueError("bounded dynamic M/K requires one Graph function")
    function = module.functions[0]
    if len(function.args) != 2 or len(function.result_types) != 1:
        raise ValueError("bounded dynamic M/K requires a two-input, one-result matmul")
    matmuls = [op for op in function.body if op.op_name == "tessera.matmul"]
    if len(matmuls) != 1:
        raise ValueError("bounded dynamic M/K requires one Graph matmul operation")
    lhs_type, rhs_type = function.args[0].ir_type, function.args[1].ir_type
    output_type = function.result_types[0]
    try:
        traced_m, traced_k = (int(str(dim)) for dim in lhs_type.shape)
        rhs_k, n = (int(str(dim)) for dim in rhs_type.shape)
        out_m, out_n = (int(str(dim)) for dim in output_type.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("bounded dynamic M/K requires static traced tensor shapes") from exc
    if (traced_m, out_m) != (m_bound, m_bound):
        raise ValueError("dynamic M bound must match both traced row capacities")
    if traced_k != k_bound or rhs_k != k_bound:
        raise ValueError("dynamic K bound must match both traced contraction capacities")
    if out_n != n:
        raise ValueError("bounded dynamic M/K requires a matching static N extent")

    dynamic_lhs = tensor_ir_type(("?", "?"), lhs_type.dtype, layout=lhs_type.layout)
    dynamic_rhs = tensor_ir_type(("?", str(n)), rhs_type.dtype, layout=rhs_type.layout)
    dynamic_output = tensor_ir_type(("?", str(n)), output_type.dtype, layout=output_type.layout)
    function.args[0].ir_type = dynamic_lhs
    function.args[1].ir_type = dynamic_rhs
    function.result_types[0] = dynamic_output
    op = matmuls[0]
    op.operand_types = [str(dynamic_lhs), str(dynamic_rhs)]
    op.result_type = str(dynamic_output)
    op.inferred_type = dynamic_output
    op.inferred_types = (dynamic_output,)
    op.kwargs["shape_bounds"] = [m_bound, n, k_bound]
    return module


def with_bounded_dynamic_axes(
    matmul_module: GraphIRModule, axes: tuple[str, ...],
) -> GraphIRModule:
    """Project selected static Graph matmul axes to bounded runtime extents."""
    from .graph_ir import tensor_ir_type

    dynamic_axes = frozenset(axes)
    if not dynamic_axes or not dynamic_axes <= {"M", "N", "K"}:
        raise ValueError("bounded dynamic matmul axes must be a nonempty subset of M/N/K")
    module = copy.deepcopy(matmul_module)
    if len(module.functions) != 1:
        raise ValueError("bounded dynamic axes require one Graph function")
    function = module.functions[0]
    if not 2 <= len(function.args) <= 4 or len(function.result_types) != 1:
        raise ValueError("bounded dynamic axes require A/B with optional bias/residual and one result")
    matmuls = [op for op in function.body if op.op_name == "tessera.matmul"]
    if len(matmuls) != 1:
        raise ValueError("bounded dynamic axes require one Graph matmul operation")
    lhs_type, rhs_type = function.args[0].ir_type, function.args[1].ir_type
    output_type = function.result_types[0]
    try:
        m, k = (int(str(dim)) for dim in lhs_type.shape)
        rhs_k, n = (int(str(dim)) for dim in rhs_type.shape)
        out_m, out_n = (int(str(dim)) for dim in output_type.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("bounded dynamic axes require static traced tensor shapes") from exc
    if min(m, n, k) <= 0 or (rhs_k, out_m, out_n) != (k, m, n):
        raise ValueError("bounded dynamic axes require matching positive M/N/K capacities")

    dynamic_lhs = tensor_ir_type(
        ("?" if "M" in dynamic_axes else str(m),
         "?" if "K" in dynamic_axes else str(k)),
        lhs_type.dtype, layout=lhs_type.layout,
    )
    dynamic_rhs = tensor_ir_type(
        ("?" if "K" in dynamic_axes else str(k),
         "?" if "N" in dynamic_axes else str(n)),
        rhs_type.dtype, layout=rhs_type.layout,
    )
    dynamic_output = tensor_ir_type(
        ("?" if "M" in dynamic_axes else str(m),
         "?" if "N" in dynamic_axes else str(n)),
        output_type.dtype, layout=output_type.layout,
    )
    function.args[0].ir_type = dynamic_lhs
    function.args[1].ir_type = dynamic_rhs
    function.result_types[0] = dynamic_output
    op = matmuls[0]
    args = {arg.name:arg for arg in function.args}
    roles = []
    for role in ("bias","residual"):
        value = op.kwargs.get(role)
        if value in (None,False):
            continue
        if not isinstance(value,str) or value.removeprefix("%") not in args:
            raise ValueError("bounded dynamic epilogue must name a Graph argument")
        arg = args[value.removeprefix("%")]
        expected = (str(n),) if role == "bias" else (str(m),str(n))
        if tuple(str(dim) for dim in arg.ir_type.shape) != expected or arg.ir_type.dtype != "fp32":
            raise ValueError("bounded dynamic epilogue must match static fp32 capacities")
        shape = (("?" if "N" in dynamic_axes else str(n)),) if role == "bias" else (
            "?" if "M" in dynamic_axes else str(m),
            "?" if "N" in dynamic_axes else str(n))
        arg.ir_type = tensor_ir_type(shape,arg.ir_type.dtype,layout=arg.ir_type.layout)
        roles.append(arg.name)
    expected_names = [function.args[0].name,function.args[1].name,*roles]
    if [name.removeprefix("%") for name in op.operands] != expected_names or (
        [arg.name for arg in function.args] != expected_names
    ):
        raise ValueError("bounded dynamic matmul requires ordered A/B/bias/residual inputs")
    op.operand_types = [str(args[name].ir_type) for name in expected_names]
    op.result_type = str(dynamic_output)
    op.inferred_type = dynamic_output
    op.inferred_types = (dynamic_output,)
    op.kwargs["shape_bounds"] = [m, n, k]
    return module


@dataclass(frozen=True)
class ScheduledMatmulArtifact:
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target: str
    architecture: str
    function_name: str
    a_name: str
    b_name: str
    output_name: str
    m: int
    n: int
    k: int
    a_dtype: str
    b_dtype: str
    output_dtype: str
    storage: str
    accum: str
    macro_tile_m: int
    macro_tile_n: int
    schedule_digest: str
    bias_name: str | None = None
    residual_name: str | None = None
    activation: str = "none"
    dynamic_m: bool = False
    dynamic_n: bool = False
    dynamic_k: bool = False
    #: ROCM-SPLIT-K-1: cross-workgroup K slices (1 = unsplit) and the order
    #: their partials are summed in ("" when unsplit, else "ordered"). Both
    #: semantic: `verify_matmul_projection` requires them to equal the native
    #: Schedule's decision, and `validate` requires the Tile launch op to carry
    #: exactly this pair.
    split_k: int = 1
    split_k_reduction: str = ""
    b_layout: str = "col_major"

    @property
    def graph_digest(self) -> str:
        return digest_text(self.graph_ir)

    @property
    def schedule_ir_digest(self) -> str:
        return digest_text(self.schedule_ir)

    @property
    def tile_digest(self) -> str:
        return digest_text(self.tile_ir)

    def validate(self) -> None:
        if len(re.findall(r"(?m)^\s*%[^=]+ = schedule\.matmul\b", self.schedule_ir)) != 1:
            raise ValueError("scheduled matmul artifact requires one scheduled SSA operation")
        if len(re.findall(r"(?m)^\s*schedule\.artifact\b", self.schedule_ir)) != 1:
            raise ValueError("scheduled matmul artifact requires one durable schedule record")
        if self.schedule_ir.count(self.schedule_digest) != 3:
            raise ValueError("scheduled matmul artifact has incomplete Schedule digest identity")
        matmul_kernels = self.tile_ir.count("tile.matmul_kernel")
        macro_cta_kernels = self.tile_ir.count("tessera_nvidia.macro_cta_matmul")
        typed_mmas = len(re.findall(r"(?m)^\s*%[^=]+ = tile\.mma\b", self.tile_ir))
        if self.target == "nvidia_sm120":
            producers = int(bool(matmul_kernels)) + int(bool(typed_mmas)) + int(bool(macro_cta_kernels))
            if producers != 1 or matmul_kernels > 1 or typed_mmas > 1 or macro_cta_kernels > 1:
                raise ValueError(
                    "SM120 scheduled artifact requires exactly one typed MMA, "
                    "macro-CTA, or deferred Tile matmul producer"
                )
        elif matmul_kernels != 1:
            raise ValueError("scheduled matmul artifact requires exactly one Tile launch op")
        if "tessera.matmul" in self.tile_ir or "schedule." in self.tile_ir:
            raise ValueError("scheduled matmul Tile artifact must not retain Graph or Schedule ops")
        hashes = _HASH_RE.findall(self.tile_ir)
        if hashes != [self.schedule_digest]:
            raise ValueError("scheduled matmul Tile artifact has a stale schedule digest")
        for name, value in (
            ("tessera.macro_tile_m", self.macro_tile_m),
            ("tessera.macro_tile_n", self.macro_tile_n),
        ):
            if not re.search(rf"{re.escape(name)} = {value} : i64", self.tile_ir):
                raise ValueError(f"scheduled matmul Tile artifact has stale {name}")
        if self.activation not in {"none", "relu", "gelu", "silu"}:
            raise ValueError("scheduled matmul artifact has an unsupported activation")
        # ROCM-SPLIT-K-1: the Tile launch op carries the split as a pair, and
        # an unsplit artifact carries neither (CHECK-NOT in Python form).
        if type(self.split_k) is not int or self.split_k < 1:
            raise ValueError("scheduled matmul split_k must be a positive int")
        if (self.split_k > 1) != (self.split_k_reduction == "ordered") or (
                self.split_k == 1 and self.split_k_reduction):
            raise ValueError(
                "scheduled matmul split_k > 1 requires split_k_reduction='ordered' "
                "and an unsplit artifact names no reduction")
        split_attrs = re.findall(r"tessera\.split_k = (\d+) : i64", self.tile_ir)
        reduction_attrs = re.findall(r'tessera\.split_k_reduction = "(\w+)"', self.tile_ir)
        if self.split_k > 1:
            if split_attrs != [str(self.split_k)] or reduction_attrs != [self.split_k_reduction]:
                raise ValueError("scheduled matmul Tile artifact dropped or altered its split-K contract")
        elif split_attrs or reduction_attrs:
            raise ValueError("scheduled matmul Tile artifact carries a split-K the artifact does not state")
        if self.bias_name is not None and 'bias = true' not in self.tile_ir:
            raise ValueError("scheduled matmul artifact dropped its bias epilogue")
        if self.residual_name is not None and 'residual = true' not in self.tile_ir:
            raise ValueError("scheduled matmul artifact dropped its residual epilogue")
        if self.activation != "none" and f'activation = "{self.activation}"' not in self.tile_ir:
            raise ValueError("scheduled matmul artifact dropped its activation epilogue")
        if digest_text(self.schedule_ir) == self.tile_digest:
            raise ValueError("Schedule and Tile artifacts must be distinct boundary outputs")


@cache
def _rocm_profile(arch: str) -> "ROCmTargetProfile":
    """The target profile the ranking needs, built once per arch."""
    from .rocm_target import AMDArch, ROCmTargetProfile, rocm_arch_string
    for member in AMDArch:
        if rocm_arch_string(member) == arch:
            return ROCmTargetProfile(arch=member)
    raise ValueError(f"no AMDArch spells {arch!r}")


def _select_macro_tile(*args, **kwargs):
    """Deferred import: rocm_tiling pulls rocm_target, which imports late."""
    from .rocm_tiling import select_macro_tile
    return select_macro_tile(*args, **kwargs)


def _band_4x4(m: int, n: int, *, dynamic: bool) -> bool:
    """The fully tiled [1024, 2048) band where the 4x4 panel wins on both chips."""
    return not dynamic and 1024 <= m < 2048 and 1024 <= n < 2048 and m % 64 == 0 and n % 64 == 0


def rocm_gfx1201_panel(m: int, n: int, *, dynamic: bool) -> tuple[int, int]:
    """The gfx1201 macro tile, mirroring `getInferredMatmulSchedule` in
    PMPasses.cpp: the 4x4 panel for every static, fully tiled problem at 1024
    and above, the 1x1 otherwise.

    **This is the tile for every storage the chip admits, not just f16/bf16**
    (2026-09-19). It was derived on f16 and applied to f16 alone, leaving the
    fp8 and integer branches on a hardcoded 1x1 -- the storages with the
    *higher* ceilings (fp8 383 TFLOP/s and int4 766 TOP/s against f16's 191)
    sitting on the smallest tile. Measured on Tajasarus, 1x1 -> 4x4:

        fp8_e4m3  1024^3  15.80 -> 58.33   2048^3  20.23 -> 78.01
        int8      1024^3  15.96 -> 58.36   2048^3  19.70 -> 76.55
        int4      1024^3  14.95 -> 49.85   2048^3  16.96 -> 76.44

    3.3x to 4.5x, from a panel the generator already emitted correctly: the
    integer rows are *exact* against the i32 reference at every panel, so this
    was a selection omission and never a codegen limit.

    The panel axis stops at 4x4 for all of them, and the reason is
    **architectural**: RDNA4 ISA 3.3.2.1 -- "VGPRs are allocated in blocks of
    16 for wave32 or 8 for wave64, and a shader may have up to 256 VGPRs" --
    and dynamic VGPR mode (3.3.3) caps at the same 256 with a 32-VGPR block
    size. The 768 KiB VGPR file is per CU and shared across wave slots; no
    occupancy request hands one wave more. So 4x8 and 8x8 compile and spill
    (1433 and 4247 VGPRs) with no ceiling left to raise, and CDNA's
    one-wave-per-SIMD/512-VGPR recipe does not transfer here.

    What remains actionable is that the **shipped** 4x4 panel already spills
    126 VGPRs. That is a real cost, and the only lever on it is needing fewer
    live registers -- not a larger tile and not an occupancy attribute. Above
    the panel, latency hiding is the other lever -- see `rocm_k_unroll`."""
    return _select_macro_tile(
        m, n, profile=_rocm_profile("gfx1201"), dynamic=dynamic,
        measured_large_panel=(64, 64), measured_small_panel=(16, 16))


#: Bits one lane's 8-element fragment load actually moves, by storage. The
#: load is 8 elements whatever the storage, so a narrower operand leaves the
#: 128-bit interface idle -- 16-bit fills it, 8-bit uses half, 4-bit a quarter
#: (AMD's RDNA4 WMMA guide, part 2). This is the mechanism behind
#: `rocm_k_unroll`'s storage split, not a curve fit.
#: Keyed by BOTH spellings, because the Graph dtype name ("fp16", "fp8_e4m3")
#: and the artifact's storage name ("f16", "e4m3") both reach this rule and an
#: unlisted key must not quietly pick someone else's answer.
_STORAGE_LOAD_BITS = {
    "fp16": 128, "f16": 128, "bf16": 128,
    "fp8_e4m3": 64, "fp8_e5m2": 64, "e4m3": 64, "e5m2": 64, "int8": 64,
    "int4": 32,
}


def rocm_k_unroll(m: int, n: int, k: int, *, arch: str, dynamic: bool,
                  storage: str = "fp16") -> int:
    """Full 16-wide K slabs the typed matmul body issues per loop iteration.

    A physical (performance) knob, not a Schedule-IR decision, and it depends
    on the **storage** as well as the chip. A fragment load is 8 elements per
    lane whatever the storage, so the bits it moves shrink as the storage
    does: fp16 saturates the 128-bit interface, fp8 and int8 use 64 bits, int4
    uses 32. Issuing more slabs is how a narrow operand keeps that path busy,
    which is why the narrower the storage the deeper the unroll it wants.
    Measured 2026-09-19 on the panel each chip selects, TFLOP/s at k = 1/2/4
    (`benchmarks/baselines/`):

        gfx1201, 4x4 panel
          fp16      1024^3  46.4 / 58.7 / 57.5   2048^3  51.6 / 88.2 / 73.7
          fp8/int8  1024^3  64.8 / 76.1 / 55.0   2048^3  76.6 / 86.6 / 115.1
                                                 4096^3  68.5 / 77.3 / 128.5
          int4      1024^3  54.1 / 53.5 / 51.2   2048^3  77.5 / 65.4 /  96.0
        gfx1151, the panel its integer branch ships (2x4)
          int8      1024^3   9.6 / 17.0 / 15.2   2048^3  21.0 / 22.6 / 14.3
          int4      1024^3   9.6 / 15.0 / 20.9   2048^3  14.7 / 17.8 / 27.3

    fp8 and int8 are the same rule because they are the same load width; that
    agreement across two unrelated storages is the check on the mechanism.
    Margins inside run-to-run spread are not taken: gfx1151 int8 at 2048^3
    gains 7% from k=2 and keeps k=1, and gfx1201 int4 at 1024^3 spans 6%
    across all three and keeps k=1.

    **The unroll was assumed to be a workaround for a missing instruction. It
    is not -- measured 2026-09-19 and the assumption is withdrawn.** The
    reasoning was that a deeper unroll reaches the 128-bit load width by
    issuing *more* loads, where RDNA4's native double-K int4
    (`V_WMMA_I32_16X16X32_IU4`) fetches 16 elements in one, so the instruction
    should win. The typed route can emit it now, it is exact against the i32
    reference at 64^3, 256^3 and 512^3, and `v_wmma_i32_16x16x32_iu4` appears
    in the disassembly -- and it **loses** to the unroll at every shape
    (TOP/s, k16-unroll-4 vs k32-unroll-1): 47.5 vs 42.8 at 1024^3, 98.5 vs
    91.5 at 2048^3, 106.9 vs 100.8 at 4096^3. Unrolling the double-K form on
    top of that collapses (34.0 and 20.8), which is the register pressure.

    The load width was the right mechanism and the wrong conclusion. Both
    forms move the same bytes per lane; what differs is that the unroll has
    two INDEPENDENT MMAs in flight while the double-K is one dependent
    instruction. On a memory-latency-bound body the instruction-level
    parallelism is worth more than the instruction density. So the unroll
    stays the rule, and the double-K shape is a supported capability rather
    than a selection.

    Neither chip's number is evidence for the other.
    """
    if dynamic or k < 64:
        return 1
    bits = _STORAGE_LOAD_BITS.get(storage)
    if bits is None:
        # An unmeasured storage takes the established single-slab loop rather
        # than inheriting the rule of whichever width it resembles.
        return 1
    wide = min(m, n) >= 2048
    if arch.startswith("gfx1201"):
        if min(m, n) < 1024 or m % 64 or n % 64:
            return 1
        if bits >= 128:
            return 2                      # f16/bf16: already saturating
        if bits == 64:
            return 4 if wide else 2       # fp8, int8
        return 4 if wide else 1           # int4: nothing wins below 2048
    if arch.startswith("gfx1151"):
        # Enumerated, not width-derived: this chip has no fp8 WMMA at all, so
        # an 8-bit float here is a storage it cannot execute and certainly has
        # not swept. Falling through on width would hand it int8's rule.
        if storage == "int4":
            if not dynamic and min(m, n) >= 1024 and m % 64 == 0 and n % 64 == 0:
                return 4
            return 1
        if storage in ("fp16", "f16", "bf16", "int8"):
            return 2 if _band_4x4(m, n, dynamic=dynamic) else 1
        return 1
    return 1


def rocm_gfx1151_panel(m: int, n: int, *, dynamic: bool) -> tuple[int, int]:
    """The gfx1151 f16/bf16 macro tile: the committed 2x4 panel, except the
    typed 4x4 in the fully tiled [1024, 2048) band (10.8 vs 9.9 TFLOP/s at
    1024^3, losing again at 2048^3: 18.7 vs 21.2)."""
    return _select_macro_tile(
        m, n, profile=_rocm_profile("gfx1151"), dynamic=dynamic,
        measured_large_panel=(64, 64), measured_small_panel=(32, 64),
        measured_band=(1024, 2048))


#: The macro K block gfx1201 schedules select, mirroring
#: `getInferredMatmulSchedule` (ROCM-MACRO-K-TILE-1): 32 for a static K of at
#: least 64, none otherwise.
def rocm_gfx1201_block_k(k: int, *, dynamic_k: bool) -> int:
    return 32 if not dynamic_k and k >= 64 else 0


def rocm_split_k(m: int, n: int, k: int, *, target: str, storage: str,
                 macro_tile: tuple[int, int], dynamic: bool) -> tuple[int, str]:
    """ROCM-SPLIT-K-1 oracle: the (split_k, reduction) the native Schedule
    must have selected. The C++ `selectGfx1201SplitK` is the authority; this is
    its declared differential check (Decision #31), applied on every package by
    `verify_matmul_projection`. gfx1201 f16/bf16 only -- the scope the C++ rule
    admits and the scope with device evidence; gfx1151 and the fp8/integer
    storages are never split."""
    if target != "rocm_gfx1201" or storage not in {"f16", "bf16"}:
        return 1, ""
    from .rocm_tiling import select_split_k
    slices, _reason = select_split_k(
        m, n, k, macro_tile=macro_tile,
        block_k=rocm_gfx1201_block_k(k, dynamic_k=dynamic),
        profile=_rocm_profile("gfx1201"),
        dtype="bf16" if storage == "bf16" else "fp16", dynamic=dynamic)
    return (slices, "ordered") if slices > 1 else (1, "")


def schedule_split_k(schedule_ir: str) -> tuple[int, str]:
    """The (split_k, split_k_reduction) the native Schedule stated on its one
    `schedule.matmul`; absence means unsplit. Malformed or half-stated pairs
    are refused rather than read as unsplit (Decision #21a)."""
    records = re.findall(r"(?m)^\s*%[^=]+ = schedule\.matmul %\w+ \{([^{}]*)\}", schedule_ir)
    if len(records) != 1:
        raise ValueError("scheduled matmul requires one native schedule.matmul record")
    attrs = records[0]
    splits = re.findall(r"(?:^|, )split_k = (\d+) : i64(?:,|$)", attrs)
    reductions = re.findall(r'(?:^|, )split_k_reduction = "(\w*)"(?:,|$)', attrs)
    if len(splits) > 1 or len(reductions) > 1 or bool(splits) != bool(reductions):
        raise ValueError("native schedule.matmul split-K contract is malformed")
    if not splits:
        return 1, ""
    return int(splits[0]), reductions[0]


def _canonical_matmul_graph_ir(module: GraphIRModule, target: str, contract: tuple) -> str:
    """Normalize frontend target/name spelling without lowering physical IR."""
    targeted = copy.deepcopy(module)
    targeted.module_attrs["tessera.target"] = f'"{contract[0]}"'
    targeted.module_attrs["tessera.arch"] = f'"{contract[1]}"'
    # The Tile kernel symbol is derived by the C++ passes from this function's
    # name, and the runtime dispatches scheduled sm_120 matmuls by name prefix,
    # so the prefix has to be applied HERE -- renaming only the Python-side
    # descriptor entry desynchronises it from the symbol actually present in the
    # compiled PTX ("native PTX is missing entry ...").
    if targeted.functions and contract[0] == "nvidia_sm120":
        fn0 = targeted.functions[0]
        if not fn0.name.startswith(_SM120_SCHEDULED_MATMUL_PREFIX):
            fn0.name = f"{_SM120_SCHEDULED_MATMUL_PREFIX}{fn0.name}"
    if contract[12] == "int4":
        # Historical callers carry the unregistered !tessera.int4 spelling.
        # Normalize the frontend tensor description to canonical builtin i4.
        from .graph_ir import tensor_ir_type
        fn = targeted.functions[0]
        fn.body[0].op_name = "tessera.matmul"
        for arg in fn.args:
            arg.ir_type = tensor_ir_type(tuple(arg.ir_type.shape), arg.ir_type.dtype)
        fn.body[0].operand_types = [str(arg.ir_type) for arg in fn.args]
    return targeted.to_mlir(target=target, canonical=True)


def lower_scheduled_matmul(
    module: GraphIRModule,
    *,
    target: str,
) -> ScheduledMatmulArtifact:
    """Lower one bounded Graph matmul through the production C++ boundaries."""

    contract = _graph_contract(module, target)
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("scheduled matmul lowering requires production tessera-opt")

    graph_ir = _canonical_matmul_graph_ir(module, target, contract)
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    hashes = _HASH_RE.findall(tile_ir)
    if len(hashes) != 1:
        raise RuntimeError("scheduled matmul lowering did not preserve one schedule digest")

    (
        compiler_target,
        architecture,
        function_name,
        a_name,
        b_name,
        output_name,
        m,
        n,
        k,
        a_dtype,
        b_dtype,
        output_dtype,
        storage,
        accum,
        macro_tile_m,
        macro_tile_n,
        bias_name,
        residual_name,
        activation,
        dynamic_m,
        dynamic_n,
        dynamic_k,
    ) = contract
    if compiler_target == "nvidia_sm120":
        # Schedule-to-Tile owns the physical producer and its launch symbol.
        # Project that native product instead of duplicating the macro/typed
        # dispatch in Python (which drifts when a new tail route is admitted).
        entries = re.findall(r'\bllvm.func @(\w+)\(', tile_ir)
        if len(entries) != 1:
            raise RuntimeError("scheduled SM120 matmul requires one native Tile entry")
        function_name = entries[0]
    # ROCM-SPLIT-K-1: the artifact states what the AUTHORITY decided -- the C++
    # Schedule -- never what the Python oracle predicts. The oracle is compared
    # against it in `verify_matmul_projection`, so a disagreement reports as
    # oracle-vs-authority instead of as a Tile artifact that "dropped" a
    # contract it never carried.
    split_k, split_k_reduction = schedule_split_k(schedule_ir)
    layout_match = re.search(r'\bb_layout = "(row_major|col_major)"', schedule_ir)
    if layout_match is None:
        raise RuntimeError("scheduled matmul lowering lost its RHS layout contract")
    artifact = ScheduledMatmulArtifact(
        graph_ir=graph_ir,
        schedule_ir=schedule_ir,
        tile_ir=tile_ir,
        target=compiler_target,
        architecture=architecture,
        function_name=function_name,
        a_name=a_name,
        b_name=b_name,
        output_name=output_name,
        m=m,
        n=n,
        k=k,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        output_dtype=output_dtype,
        storage=storage,
        accum=accum,
        macro_tile_m=macro_tile_m,
        macro_tile_n=macro_tile_n,
        schedule_digest=hashes[0],
        b_layout=layout_match[1],
        bias_name=bias_name,
        residual_name=residual_name,
        activation=activation,
        dynamic_m=dynamic_m,
        dynamic_n=dynamic_n,
        dynamic_k=dynamic_k,
        split_k=split_k,
        split_k_reduction=split_k_reduction,
    )
    artifact.validate()
    return artifact


def supports_scheduled_matmul(module: GraphIRModule, *, target: str) -> bool:
    try:
        _graph_contract(module, target)
    except ValueError:
        return False
    return True


def _graph_contract(module: GraphIRModule, target: str) -> tuple:
    if len(module.functions) != 1:
        raise ValueError("scheduled matmul requires one Graph function")
    function = module.functions[0]
    if len(function.body) != 1 or len(function.result_types) != 1:
        raise ValueError("scheduled matmul requires one Graph operation and result")
    op = function.body[0]
    int4_gemm = (op.op_name == "tessera.gemm" and target == "nvidia_sm120"
                 and len(function.args) == 2 and all(a.ir_type.dtype == "int4" for a in function.args))
    nvfp4_scaled = op.op_name == "tessera.scaled_matmul" and target == "nvidia_sm120"
    if ((op.op_name != "tessera.matmul" and not int4_gemm and not nvfp4_scaled)
            or not 2 <= len(op.operands) <= 4):
        raise ValueError("scheduled matmul requires one supported Graph matmul")
    transpose_a, transpose_b = (op.kwargs.get(name, False) for name in ("transposeA", "transposeB"))
    if type(transpose_a) is not bool or type(transpose_b) is not bool:
        raise ValueError("scheduled matmul transpose flags must be boolean")
    if (transpose_a or transpose_b) and not nvfp4_scaled:
        raise ValueError("scheduled matmul does not support transpose")
    args = {arg.name: arg for arg in function.args}
    a_name, b_name = (value.removeprefix("%") for value in op.operands[:2])
    if a_name not in args or b_name not in args:
        raise ValueError("scheduled matmul operands must be function arguments")

    def extent(value: object) -> int | None:
        if isinstance(value, (int, str)):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    a_shape = tuple(extent(value) for value in args[a_name].ir_type.shape)
    b_shape = tuple(extent(value) for value in args[b_name].ir_type.shape)
    out_shape = tuple(extent(value) for value in function.result_types[0].shape)
    if transpose_a: a_shape = (*a_shape[:-2], a_shape[-1], a_shape[-2])
    if transpose_b: b_shape = (*b_shape[:-2], b_shape[-1], b_shape[-2])
    batch_shape: tuple[int, ...] | None = None
    independent_rhs = nvfp4_scaled and op.kwargs.get("batching") == "independent_rhs"
    shared_lhs = nvfp4_scaled and op.kwargs.get("batching") == "shared_lhs"
    if nvfp4_scaled and op.kwargs.get("batching") in {"shared_rhs_rows", "independent_rhs", "shared_lhs"}:
        rhs_batched = independent_rhs or shared_lhs
        if (len(out_shape) < 3 or len(a_shape) != (2 if shared_lhs else len(out_shape))
                or len(b_shape) != (len(out_shape) if rhs_batched else 2)
                or any(value is None or value <= 0 for value in (*a_shape, *b_shape, *out_shape))):
            raise ValueError("NVFP4 Schedule requires matching static batch/matrix dimensions")
        prefix = cast(tuple[int, ...], b_shape[:-2] if shared_lhs else a_shape[:-2])
        import math
        batch = math.prod(prefix)
        rows, logical_k = a_shape[-2:]
        assert batch is not None and rows is not None and logical_k is not None
        if out_shape[:-2] != prefix or out_shape[-2] != rows:
            raise ValueError("NVFP4 output batch/row dimensions differ")
        if batch * rows > 2**63 - 1:
            raise ValueError("shared-RHS NVFP4 row product overflows int64")
        if rhs_batched:
            if b_shape[:-2] != prefix:
                raise ValueError("independent RHS batch count differs")
            b_shape = b_shape[-2:]
        batch_shape = (*prefix, rows)
        a_shape = (batch * rows, logical_k)
        out_shape = (batch * rows, out_shape[-1])
    if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
        raise ValueError("scheduled matmul requires rank-2 tensors or named shared-RHS rows")
    m_value, k_value = a_shape
    kb_value, n_value = b_shape
    # M, N, and K are equality-linked across the three tensor types. Preserve
    # dynamicity if *any* occurrence of a logical dimension is dynamic; the
    # packaging and target guards must not silently treat the same dimension
    # as static merely because its first occurrence is static.
    dynamic_m = m_value is None or out_shape[0] is None
    dynamic_n = n_value is None or out_shape[1] is None
    dynamic_k = k_value is None or kb_value is None
    dynamic = dynamic_m or dynamic_n or dynamic_k
    raw_bounds = op.kwargs.get("shape_bounds")
    if dynamic:
        if target not in {"nvidia_sm120", "rocm_gfx1151", "rocm_gfx1201"} or not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 3:
            raise ValueError(
                "dynamic scheduled matmul requires an SM120 or gfx1151 "
                "shape_bounds=[M,N,K] contract"
            )
        try:
            m, n, k = (int(value) for value in raw_bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError("scheduled matmul shape_bounds must be positive integers") from exc
    else:
        assert m_value is not None and n_value is not None and k_value is not None
        m, n, k = m_value, n_value, k_value
    compatible = lambda value, expected: value is None or value == expected
    if (
        min(m, n, k) <= 0
        or not compatible(m_value, m)
        or not compatible(k_value, k)
        or not compatible(kb_value, k)
        or not compatible(n_value, n)
        or len(out_shape) != 2
        or not compatible(out_shape[0], m)
        or not compatible(out_shape[1], n)
    ):
        raise ValueError("scheduled matmul shapes do not form MxK @ KxN -> MxN")
    a_dtype = args[a_name].ir_type.dtype
    b_dtype = args[b_name].ir_type.dtype
    output_dtype = function.result_types[0].dtype
    activation = str(op.kwargs.get("activation", "none"))
    if activation not in {"none", "relu", "gelu", "silu"}:
        raise ValueError("scheduled matmul has an unsupported activation")
    bias_value = op.kwargs.get("bias")
    residual_value = op.kwargs.get("residual")
    bias_name = bias_value.removeprefix("%") if isinstance(bias_value, str) else None
    residual_name = (
        residual_value.removeprefix("%") if isinstance(residual_value, str) else None
    )
    # The canonical tracer records optional operand role markers, with the
    # actual SSA edge in operands. Preserve explicitly named argument intent;
    # only an unbound role marker resolves through the declared ABI position.
    if bias_value == "bias" and "bias" not in args and len(op.operands) > 2:
        bias_name = op.operands[2].removeprefix("%")
    residual_position = 2 + int(bias_name is not None)
    if (residual_value == "residual" and "residual" not in args
            and len(op.operands) > residual_position):
        residual_name = op.operands[residual_position].removeprefix("%")
    if bias_value not in {None, False} and bias_name is None:
        raise ValueError("scheduled matmul bias must name a Graph argument")
    if residual_value not in {None, False} and residual_name is None:
        raise ValueError("scheduled matmul residual must name a Graph argument")
    expected_operands = [f"%{a_name}", f"%{b_name}"]
    if nvfp4_scaled:
        if (len(op.operands) != 4 or len(function.args) != 4
                or op.kwargs.get("physical_contract") != "nvidia_sm120_nvfp4_blockscale_v1"
                or op.kwargs.get("activation", "none") != "none"
                or bias_value not in {None, False} or residual_value not in {None, False}
                or (a_dtype, b_dtype, output_dtype) != ("nvfp4", "nvfp4", "fp32")):
            raise ValueError("NVFP4 Schedule requires static scaled NVFP4 Graph inputs and f32 output")
        scale_a_name, scale_b_name = (v.removeprefix("%") for v in op.operands[2:4])
        scale_k = (k + 15) // 16
        expected_sa = ((batch_shape[-1], scale_k) if shared_lhs and batch_shape else
                       (*batch_shape, scale_k) if batch_shape else (m, scale_k))
        expected_sb = (*batch_shape[:-1], scale_k, n) if (independent_rhs or shared_lhs) and batch_shape else (scale_k, n)
        if transpose_a: expected_sa = (*expected_sa[:-2], expected_sa[-1], expected_sa[-2])
        if transpose_b: expected_sb = (*expected_sb[:-2], expected_sb[-1], expected_sb[-2])
        if (scale_a_name not in args or scale_b_name not in args
                or tuple(args[scale_a_name].ir_type.shape) != tuple(map(str, expected_sa))
                or tuple(args[scale_b_name].ir_type.shape) != tuple(map(str, expected_sb))
                or args[scale_a_name].ir_type.dtype != "uint8"
                or args[scale_b_name].ir_type.dtype != "uint8"
                or op.kwargs.get("scale_layout") != {"granularity": "block", "block": [1, 16], "format": "ue4m3"}
                or op.kwargs.get("numeric_policy") != {"accum": "fp32", "execution_mode": "exact_per_block"}):
            raise ValueError("NVFP4 Schedule scale operands/layout/numeric policy disagree")
        expected_operands.extend((f"%{scale_a_name}", f"%{scale_b_name}"))
        bias_name = residual_name = None
        activation = "none"
    if bias_name is not None:
        expected_operands.append(f"%{bias_name}")
    if residual_name is not None:
        expected_operands.append(f"%{residual_name}")
    if op.operands != expected_operands:
        raise ValueError(
            "scheduled matmul epilogue operands must follow A/B/bias/residual ABI order"
        )
    if bias_name is not None:
        bias = args.get(bias_name)
        if (bias is None or bias.ir_type.dtype != "fp32"
                or len(bias.ir_type.shape) != 1
                or extent(bias.ir_type.shape[0]) not in ((None,n) if dynamic_n else (n,))):
            raise ValueError("scheduled matmul bias must be an fp32 [N] argument")
    if residual_name is not None:
        residual = args.get(residual_name)
        if (
            residual is None
            or residual.ir_type.dtype != "fp32"
            or len(residual.ir_type.shape) != 2
            or extent(residual.ir_type.shape[0]) not in ((None,m) if dynamic_m else (m,))
            or extent(residual.ir_type.shape[1]) not in ((None,n) if dynamic_n else (n,))
        ):
            raise ValueError("scheduled matmul residual must be an fp32 [M,N] argument")
    # The fused bias/activation epilogue is a launch-contract field on NVIDIA
    # and, since 2026-09-17, on both ROCm chips: the ROCm Schedule->Tile branch
    # carries it onto tile.matmul_kernel and the typed Tile->ROCm consumer
    # applies it per element at the fragment store, so gfx1151 and gfx1201 get
    # one implementation across their two accumulator layouts. The residual
    # add is still NVIDIA-owned.
    fused_targets = {"nvidia_sm120", "rocm_gfx1151", "rocm_gfx1201"}
    if target not in fused_targets and (
        bias_name is not None or residual_name is not None or activation != "none"
    ):
        raise ValueError("scheduled fused matmul is currently NVIDIA/ROCm-owned")
    if target != "nvidia_sm120" and residual_name is not None:
        raise ValueError("scheduled matmul residual epilogue is currently NVIDIA-owned")
    output_name = op.result
    if not output_name:
        if len(function.return_values) != 1:
            raise ValueError("scheduled matmul requires one named Graph result")
        output_name = function.return_values[0].removeprefix("%")
    function_name = function.name
    if target == "x86" and (a_dtype, b_dtype, output_dtype) in (("fp32", "fp32", "fp32"), ("bf16", "bf16", "fp32"), ("fp64", "fp64", "fp64")):
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "x86",
            "zen5-avx512",
            "bf16" if a_dtype == "bf16" else "f64" if a_dtype == "fp64" else "f32",
            "f64" if output_dtype == "fp64" else "f32",
            16,
            16,
        )
    elif target == "x86" and (a_dtype, b_dtype, output_dtype) == ("uint8", "int8", "int32"):
        if max(m, n, k) > 2**31 - 1:
            raise ValueError("x86 mixed matmul extents exceed the i32 runtime ABI")
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "x86", "zen5-avx512", "u8", "i32", 16, 16,
        )
    elif target == "rocm_gfx1201" and a_dtype == b_dtype and a_dtype in {"fp16", "bf16"} and output_dtype == "fp32":
        panel_m, panel_n = rocm_gfx1201_panel(m, n, dynamic=dynamic_m or dynamic_n or dynamic_k)
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "rocm", "gfx1201", "bf16" if a_dtype == "bf16" else "f16", "f32", panel_m, panel_n,
        )
    elif (
        target in {"rocm_gfx1151", "rocm_gfx1201"}
        and a_dtype == b_dtype
        and a_dtype in {"int8", "int4"}
        and output_dtype == "int32"
    ):
        # Integer WMMA storage on both RDNA chips (IU8/IU4, i32 accumulate;
        # GFX1201-PARITY slice 1b). int4 values ride int8 containers, one
        # logical value per byte. The fused epilogue is float-only.
        if bias_name is not None or activation != "none":
            raise ValueError(
                "ROCm integer scheduled matmul carries no fused epilogue (the epilogue is float-only)")
        rdna4 = target == "rocm_gfx1201"
        # gfx1201 takes the same shape-selected panel as f16/bf16 (2026-09-19):
        # 1x1 was a placeholder from slice 1b, and it costs 3.7x-4.5x. gfx1151
        # keeps its committed 2x4 until its own sweep says otherwise -- proof
        # never transfers between the two chips.
        panel = (rocm_gfx1201_panel(m, n, dynamic=dynamic_m or dynamic_n or dynamic_k)
                 if rdna4 else (32, 64))
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "rocm", target.removeprefix("rocm_"), a_dtype, "i32", panel[0], panel[1],
        )
    elif (
        target == "rocm_gfx1201"
        and a_dtype in {"fp8_e4m3", "fp8_e5m2"}
        and b_dtype in {"fp8_e4m3", "fp8_e5m2"}
        and output_dtype == "fp32"
    ):
        # OCP FP8 on RDNA4 (device-audited WMMA forms, GFX1201-PARITY slice 5):
        # f32 accumulate, no fused epilogue yet. A and B may name DIFFERENT fp8
        # storages -- the hardware has V_WMMA_F32_16X16X16_FP8_BF8 and its
        # mirror, and every layer below selects the pair from the descriptor's
        # two types (ROCM-MIXED-FP8-1).
        if bias_name is not None or activation != "none":
            raise ValueError(
                "rocm_gfx1201 FP8 scheduled matmul carries no fused epilogue yet")
        panel = rocm_gfx1201_panel(m, n, dynamic=dynamic_m or dynamic_n or dynamic_k)
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "rocm", "gfx1201", "e4m3" if a_dtype == "fp8_e4m3" else "e5m2", "f32",
            panel[0], panel[1],
        )
    elif target == "rocm_gfx1151" and a_dtype == b_dtype and a_dtype in {"fp16", "bf16"} and output_dtype == "fp32":
        panel_m, panel_n = rocm_gfx1151_panel(m, n, dynamic=dynamic_m or dynamic_n or dynamic_k)
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "rocm", "gfx1151", "bf16" if a_dtype == "bf16" else "f16", "f32", panel_m, panel_n,
        )
    elif target == "nvidia_sm120" and (a_dtype, b_dtype, output_dtype) == ("nvfp4", "nvfp4", "fp32"):
        if (not nvfp4_scaled or dynamic_m or dynamic_n or dynamic_k
                or bias_name or residual_name or activation != "none"):
            raise ValueError("NVFP4 scheduled matmul requires static block-scaled inputs")
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "nvidia_sm120", "sm_120", "nvfp4", "f32", 16, 8)
        function_name = "tessera_tile_matmul_nvfp4"
    elif target == "nvidia_sm120" and (a_dtype, b_dtype, output_dtype) == ("int4", "int4", "int32"):
        if not op.result or function.return_values != ["%" + op.result] or len(function.args) != 2:
            raise ValueError("INT4 native entry must return its matmul result with two inputs")
        if any(a.layout is not None or a.ir_type.layout is not None for a in function.args) or function.result_types[0].layout is not None:
            raise ValueError("INT4 native packing does not accept layout overrides")
        if any(any(part in key for part in ("layout", "pack", "stride")) for key in op.kwargs):
            raise ValueError("INT4 native packing does not accept physical overrides")
        if bias_name or residual_name or activation != "none" or dynamic_m or dynamic_n or dynamic_k:
            raise ValueError("INT4 scheduled matmul requires static plain signed storage")
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "nvidia_sm120", "sm_120", "int4", "int32", 16, 8)
        function_name = "tessera_tile_matmul_int4"
    elif (
        target == "nvidia_sm120"
        and a_dtype == b_dtype
        and a_dtype in {"fp16", "bf16"}
        and output_dtype in {"fp32", "fp16"}
    ):
        # SM120 consumes the same canonical Schedule/Tile boundary as the
        # other native backends.  The NVIDIA package owns the physical
        # m16n8k16 MMA lowering; this contract records only the shared launch
        # envelope and keeps the schedule decision content-addressed.
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "nvidia_sm120",
            "sm_120",
            "f16" if a_dtype == "fp16" else "bf16",
            "f32",
            128,
            128,
        )
        # The eventual native entry is projected from Schedule-to-Tile
        # after lowering; the Graph contract has no physical symbol authority.
    elif target == "apple_gpu" and (a_dtype, b_dtype, output_dtype) == (
        "fp32",
        "fp32",
        "fp32",
    ):
        # Apple GPU has no rank-2 f32 cooperative-matrix GEMM; the shared launch
        # contract is consumed as a batch-1 MPS BMM.  The 16x16 macro-tile is a
        # logical default that the Apple package records as an explicit drop,
        # matching the C++ getMatmulSchedule default for this target.
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "apple_gpu",
            "apple7",
            "f32",
            "f32",
            16,
            16,
        )
    elif target == "apple_gpu" and (a_dtype, b_dtype, output_dtype) == (
        "fp16",
        "fp16",
        "fp32",
    ):
        # Apple7+ simdgroup_matrix GEMM: f16 storage, f32 accumulation.  This is
        # a compiler-emitted MSL route (not delegated MPS), so the 32x32 macro
        # tile is honored by the emitter.  Matches the C++ getMatmulSchedule
        # apple7 f16 branch.
        compiler_target, architecture, storage, accum, macro_tile_m, macro_tile_n = (
            "apple_gpu",
            "apple7",
            "f16",
            "f32",
            32,
            32,
        )
    else:
        # Name what was rejected (Decision #21): this branch is reached by a
        # reduced-precision accumulator, an unsupported storage, and a target
        # that has no GEMM for the pair -- and the old message, identical for
        # all three, told a caller nothing about which. It is also the marker
        # `test_rocm_wmma_form_reachability` matches a declared-unreachable
        # form against, so a form that starts failing for a DIFFERENT reason
        # can no longer read as the same known gap.
        raise ValueError(
            "SCHEDULED_MATMUL_DTYPE_CONTRACT_UNSUPPORTED: target "
            f"{target!r} has no scheduled-matmul contract for "
            f"a={a_dtype} b={b_dtype} -> out={output_dtype}. Storage and "
            "accumulator are separate (Decision #15a): an accumulator narrower "
            "than fp32/int32 is an opt-in accuracy class, not a contract the "
            "default route selects."
        )
    return (
        compiler_target,
        architecture,
        function_name,
        a_name,
        b_name,
        output_name,
        m,
        n,
        k,
        a_dtype,
        b_dtype,
        output_dtype,
        storage,
        accum,
        macro_tile_m,
        macro_tile_n,
        bias_name,
        residual_name,
        activation,
        dynamic_m,
        dynamic_n,
        dynamic_k,
    )


def find_tessera_opt() -> Path | None:
    for name in ("TESSERA_OPT", "TESSERA_OPT_BIN"):
        if configured := os.environ.get(name):
            path = Path(configured).expanduser()
            return path if path.is_file() else None
    root = Path(__file__).resolve().parents[3]
    if selected_build := os.environ.get("TESSERA_BUILD_DIR"):
        build = Path(selected_build).expanduser()
        if not build.is_absolute():
            build = root / build
        path = build / "tools/tessera-opt/tessera-opt"
        return path if path.is_file() else None
    for path in (
        root / "build/tools/tessera-opt/tessera-opt",
        root / "build-rocm-ci-local/tools/tessera-opt/tessera-opt",
    ):
        if path.is_file():
            return path
    found = shutil.which("tessera-opt")
    return Path(found) if found else None


def run_tessera_opt(tool: Path, source: str, option: str) -> str:
    result = subprocess.run(
        [str(tool), "-", option],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"scheduled compiler boundary {option} failed: "
            + (result.stderr.strip() or str(result.returncode))
        )
    return result.stdout


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def verify_matmul_projection(artifact: ScheduledMatmulArtifact) -> None:
    """Project matmul ABI fields from its native Schedule parent.

    The Graph text is historical provenance. Schedule replay proves the supplied
    Tile program; host binding labels remain aliases, not semantic authorities.
    """
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError('matmul projection requires native Schedule replay')
    runner = run_tessera_opt
    if artifact.target == "rocm":
        from .rocm_pass_cache import run as cached_replay

        def runner(tool, source, option):
            return cached_replay(tool, source, option, execute=run_tessera_opt)
    if runner(tool, artifact.schedule_ir, '--tessera-schedule-to-tile') != artifact.tile_ir:
        raise ValueError('matmul Tile product disagrees with native Schedule replay')
    parent = runner(tool, artifact.schedule_ir, '--canonicalize')
    header = re.match(r'\s*module attributes \{([^{}]*)\}', parent)
    if header is None:
        raise ValueError('matmul native target header is missing')
    for key, value in [('tessera.target', artifact.target), ('tessera.arch', artifact.architecture)]:
        if not re.search(r'(?:^|, )'+re.escape(key)+' = "'+re.escape(value)+'"(?:,|$)', header[1]):
            raise ValueError('matmul native target identity disagrees')
    functions = re.findall(
        r'func.func @(\w+)\(([^\n]*)\) -> (tensor<[^>]+>)'
        r'(?: attributes \{[^{}]*\})? \{',
        parent,
    )
    records = re.findall(r' = schedule.matmul %\w+ \{([^{}]*)\}', parent)
    keys = re.findall(r'shape_key = "M=(\d+);N=(\d+);K=(\d+);dtype=(\w+)"', parent)
    if len(functions) != 1 or len(records) != 1 or len(keys) != 1 or parent.count('func.func ') != 1:
        raise ValueError('matmul projection requires one native scheduled product')
    entry, args, output = functions[0]
    attrs = records[0]
    def string(key):
        values = re.findall(r'(?:^|, )'+key+r' = "(\w+)"(?:,|$)', attrs)
        if len(values) != 1:
            raise ValueError('matmul native string field disagrees: '+key)
        return values[0]
    def boolean(key):
        values = re.findall(r'(?:^|, )'+key+r' = (true|false)(?:,|$)', attrs)
        if len(values) != 1:
            raise ValueError('matmul native boolean field disagrees: '+key)
        return values[0] == 'true'
    def tensor(text):
        match = re.fullmatch(r'tensor<((?:(?:\?|[1-9][0-9]*)x)+)(f16|bf16|f32|f64|ui8|si8|i8|si4|i4|i32|f8E4M3FN|f8E5M2)>', text)
        if match is None:
            raise ValueError('matmul native tensor contract is unsupported')
        return tuple(None if d == '?' else int(d) for d in match[1].split('x')[:-1]), match[2]
    inputs = [tensor(t) for t in re.findall(r'tensor<[^>]+>', args)]
    out_shape, out_storage = tensor(output)
    bias, residual = boolean('bias'), boolean('residual')
    if len(inputs) != 2 + bias + residual or len(out_shape) != 2:
        raise ValueError('matmul native input arity/rank disagrees')
    m, n, k = map(int, keys[0][:3])
    storage = keys[0][3]
    if artifact.target == "nvidia_sm120":
        # MLIR replay above verifies the retained Graph and its canonical Tile
        # ABI. Project types by semantic operand role, not frontend arg order.
        # This reads the contract; it does not reconstruct/lower a Graph.
        argument_records = re.findall(r"(%[A-Za-z0-9_]+): (tensor<[^>]+>)", args)
        operands = re.findall(
            r"^\s*%[A-Za-z0-9_]+ = tessera\.matmul "
            r"((?:%[A-Za-z0-9_]+(?:,\s*)?)+)", parent, re.MULTILINE)
        if (len(argument_records) != len(inputs) or len(operands) != 1):
            raise ValueError("matmul native frontend role binding is ambiguous")
        by_name = {name: tensor(type_text) for name, type_text in argument_records}
        roles = re.findall(r"%[A-Za-z0-9_]+", operands[0])
        if (len(by_name) != len(inputs) or len(roles) != len(inputs)
                or len(set(roles)) != len(roles) or set(roles) != set(by_name)):
            raise ValueError("matmul native frontend role binding disagrees")
        inputs = [by_name[name] for name in roles]
    a, b = inputs[:2]
    if len(a[0]) != 2 or len(b[0]) != 2:
        raise ValueError("matmul native input arity/rank disagrees")
    for actual, bound in zip((*a[0], *b[0], *out_shape), (m, k, k, n, m, n)):
        if actual is not None and actual != bound:
            raise ValueError('matmul native shape bound disagrees with its signature')
    if artifact.target == "nvidia_sm120":
        entries = re.findall(r'llvm.func @(\w+)\(', artifact.tile_ir)
        if len(entries) != 1:
            raise ValueError("matmul native Tile entry is ambiguous")
        entry = entries[0]
    expected = dict(function_name=entry, m=m, n=n, k=k, storage=storage,
        a_dtype={'f16':'fp16','bf16':'bf16','f32':'fp32','f64':'fp64','ui8':'uint8','si8':'int8','i8':'int8','si4':'int4','i4':'int4','i32':'int32','f8E4M3FN':'fp8_e4m3','f8E5M2':'fp8_e5m2'}[a[1]],
        b_dtype={'f16':'fp16','bf16':'bf16','f32':'fp32','f64':'fp64','ui8':'uint8','si8':'int8','i8':'int8','si4':'int4','i4':'int4','i32':'int32','f8E4M3FN':'fp8_e4m3','f8E5M2':'fp8_e5m2'}[b[1]],
        output_dtype={'f16':'fp16','bf16':'bf16','f32':'fp32','f64':'fp64','ui8':'uint8','si8':'int8','i8':'int8','i32':'int32'}[out_storage],
        accum=string('accum'), activation=string('activation'),
        dynamic_m=a[0][0] is None or out_shape[0] is None,
        dynamic_n=b[0][1] is None or out_shape[1] is None,
        dynamic_k=a[0][1] is None or b[0][0] is None)
    expected['b_layout'] = string('b_layout')
    if (string('storage') != storage or string('output') != out_storage or string('a_layout') != 'row_major'
            or expected['b_layout'] not in {'row_major', 'col_major'}
            or (expected['b_layout'] == 'row_major' and artifact.target != 'nvidia_sm120')):
        raise ValueError('matmul native storage/layout disagrees')
    for key in ('macro_tile_m', 'macro_tile_n'):
        matches = re.findall(r'(?:^|, )'+key+r' = (\d+) : i64(?:,|$)', attrs)
        if len(matches) != 1:
            raise ValueError('matmul native tile decision is missing')
        expected[key] = int(matches[0])
    # ROCM-SPLIT-K-1: the native Schedule is the split-K authority. Absence
    # means unsplit (the C++ side states the pair only when it splits).
    split_matches = re.findall(r'(?:^|, )split_k = (\d+) : i64(?:,|$)', attrs)
    reduction_matches = re.findall(r'(?:^|, )split_k_reduction = "(\w*)"(?:,|$)', attrs)
    if len(split_matches) > 1 or len(reduction_matches) > 1 or (
            bool(split_matches) != bool(reduction_matches)):
        raise ValueError('matmul native split-K contract is malformed')
    expected['split_k'] = int(split_matches[0]) if split_matches else 1
    expected['split_k_reduction'] = reduction_matches[0] if reduction_matches else ''
    # ...and the Python predicate is its declared oracle: recompute the
    # decision from the projected shape/tile and refuse a disagreement, so the
    # two deciders cannot drift apart silently (Decision #31).
    if artifact.target == 'rocm':
        oracle = rocm_split_k(
            m, n, k, target=f'rocm_{artifact.architecture}', storage=storage,
            macro_tile=(expected['macro_tile_m'], expected['macro_tile_n']),
            dynamic=bool(expected['dynamic_m'] or expected['dynamic_n'] or expected['dynamic_k']))
        if oracle != (expected['split_k'], expected['split_k_reduction']):
            raise ValueError(
                'ROCM-SPLIT-K-1 oracle disagrees with the native Schedule: '
                f'rocm_tiling.select_split_k says {oracle}, the Schedule says '
                f"{(expected['split_k'], expected['split_k_reduction'])}")
    for key, value in expected.items():
        actual = getattr(artifact, key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError('matmul descriptor field disagrees with native IR: '+key)
    if bias != (artifact.bias_name is not None) or residual != (artifact.residual_name is not None):
        raise ValueError('matmul epilogue bindings disagree with native IR')
    extra = 2
    if bias:
        if inputs[extra][1] != 'f32' or len(inputs[extra][0]) != 1 or inputs[extra][0][0] not in (None, n):
            raise ValueError('matmul native bias shape/storage disagrees')
        extra += 1
    if residual and (inputs[extra][1] != 'f32' or len(inputs[extra][0]) != 2 or any(d not in (None, bound) for d, bound in zip(inputs[extra][0], (m, n)))):
        raise ValueError('matmul native residual shape/storage disagrees')
    names = [artifact.a_name, artifact.b_name, artifact.output_name]
    names += [x for x in (artifact.bias_name, artifact.residual_name) if x is not None]
    if any(type(x) is not str or not x.isidentifier() for x in names) or len(set(names)) != len(names):
        raise ValueError('matmul host aliases must be distinct identifiers')
