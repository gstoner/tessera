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
from typing import TYPE_CHECKING
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
    graph_ir = targeted.to_mlir(target=target, canonical=True)
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
        bias_name=bias_name,
        residual_name=residual_name,
        activation=activation,
        dynamic_m=dynamic_m,
        dynamic_n=dynamic_n,
        dynamic_k=dynamic_k,
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
    if (op.op_name != "tessera.matmul" and not int4_gemm) or not 2 <= len(op.operands) <= 4:
        raise ValueError("scheduled matmul requires one tessera.matmul")
    if op.kwargs.get("transposeA", False) or op.kwargs.get("transposeB", False):
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
    if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2:
        raise ValueError("scheduled matmul requires rank-2 tensors")
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
    if bias_value not in {None, False} and bias_name is None:
        raise ValueError("scheduled matmul bias must name a Graph argument")
    if residual_value not in {None, False} and residual_name is None:
        raise ValueError("scheduled matmul residual must name a Graph argument")
    expected_operands = [f"%{a_name}", f"%{b_name}"]
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
        if bias is None or tuple(bias.ir_type.shape) != (str(n),) or bias.ir_type.dtype != "fp32":
            raise ValueError("scheduled matmul bias must be an fp32 [N] argument")
    if residual_name is not None:
        residual = args.get(residual_name)
        if (
            residual is None
            or tuple(residual.ir_type.shape) != (str(m), str(n))
            or residual.ir_type.dtype != "fp32"
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
        # Schedule->Tile replaces the Graph tensor wrapper with this explicit
        # raw-pointer launch entry.  Keeping the suffix here makes the
        # package descriptor name the same canonical kernel that Tile IR
        # carries, rather than a Python-side substitute.
        fused = bias_name is not None or residual_name is not None or activation != "none"
        reduced = output_dtype == "fp16"
        suffix = (
            f"_fused_{storage}_{activation}_b{int(bias_name is not None)}"
            f"_r{int(residual_name is not None)}"
            if fused or reduced
            else ""
        )
        if reduced:
            suffix += "_outf16"
        # The runtime dispatches scheduled sm_120 matmuls by NAME PREFIX
        # (kScheduledSm120MatmulPrefix in tessera_nvidia_ptx_launch.cpp). Naming
        # the kernel after the caller's Graph function made that dispatch depend
        # on what the user happened to call their function: every name without
        # the prefix fell through the runtime's strcmp chain and the launch
        # returned rc=5. Exactly one place in the tree — a benchmark — named a
        # function to satisfy it; every other caller silently could not launch.
        # The prefix is part of the ABI, so the compiler emits it rather than
        # asking the frontend to spell it.
        base_name = function.name
        if not base_name.startswith(_SM120_SCHEDULED_MATMUL_PREFIX):
            base_name = f"{_SM120_SCHEDULED_MATMUL_PREFIX}{base_name}"
        function_name = f"{base_name}{suffix}" + (
            "_macro_kernel"
            if _uses_sm120_macro_cta(m, n, k, storage, accum)
            else "_kernel"
        )
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


def _uses_sm120_macro_cta(
    m: int, n: int, k: int, storage: str, accum: str
) -> bool:
    """Mirror the measured target-owned 32x32/four-warp admission contract.

    SuperBear's retained WSL pruning packet found staging/barrier overhead at
    small sizes and timing variance through 33.6M FLOPs.  Every measured
    67.1M+ case is both low-variance and materially faster. WSL evidence is
    deliberately ineligible for the global performance registry, but it is
    sufficient to keep the explicit scheduled route from selecting an
    unproven crossover while bare-metal selector evidence remains open.
    """
    work = 2 * m * n * k
    return (
        m >= 32
        and n >= 32
        and k >= 16
        and storage in {"f16", "bf16"}
        and accum == "f32"
        and work >= 67_108_864
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
    if run_tessera_opt(tool, artifact.schedule_ir, '--tessera-schedule-to-tile') != artifact.tile_ir:
        raise ValueError('matmul Tile product disagrees with native Schedule replay')
    parent = run_tessera_opt(tool, artifact.schedule_ir, '--canonicalize')
    header = re.match(r'\s*module attributes \{([^{}]*)\}', parent)
    if header is None:
        raise ValueError('matmul native target header is missing')
    for key, value in [('tessera.target', artifact.target), ('tessera.arch', artifact.architecture)]:
        if not re.search(r'(?:^|, )'+re.escape(key)+' = "'+re.escape(value)+'"(?:,|$)', header[1]):
            raise ValueError('matmul native target identity disagrees')
    functions = re.findall(r'func.func @(\w+)\(([^\n]*)\) -> (tensor<[^>]+>) \{', parent)
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
    if len(inputs) != 2 + bias + residual or any(len(shape) != 2 for shape, _ in inputs[:2]) or len(out_shape) != 2:
        raise ValueError('matmul native input arity/rank disagrees')
    m, n, k = map(int, keys[0][:3])
    storage = keys[0][3]
    a, b = inputs[:2]
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
    if string('storage') != storage or string('output') != out_storage or string('a_layout') != 'row_major' or string('b_layout') != 'col_major':
        raise ValueError('matmul native storage/layout disagrees')
    for key in ('macro_tile_m', 'macro_tile_n'):
        matches = re.findall(r'(?:^|, )'+key+r' = (\d+) : i64(?:,|$)', attrs)
        if len(matches) != 1:
            raise ValueError('matmul native tile decision is missing')
        expected[key] = int(matches[0])
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
