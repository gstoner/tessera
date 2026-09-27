"""gfx1201 block-scaled FP8 W8A8 on the typed route (ROCM-FP8-BLOCKSCALE-1).

``tessera.scaled_matmul`` over e4m3 A ``[M, K]`` and e4m3 B ``[K, N]`` with
fp32 scales and ``scale_layout.block = [scale_n, scale_k]`` is a *logical*
contract. Graph->Schedule derives ``rocm_fp8_w8a8_blockscale_v1`` from it (and
refuses a nonconforming fp32-scale form by name), Schedule->Tile carries it on
``tile.scaled_matmul_kernel``, and the C++ ``generate-wmma-gemm-kernel`` pass
emits the kernel: operands stay e4m3 into ``V_WMMA_F32_16X16X16_FP8_FP8``,
every ``scale_k`` group accumulates into an fp32 partial that starts from zero,
and only that partial is multiplied by ``lhs_scale[m, g] * rhs_scale[g, n //
scale_n]`` before it joins the running accumulator.

This module authors the Graph op, runs the compiler, and binds the resulting
HSACO to a launch descriptor. It never emits target code: the one lowering
authority is the C++ pipeline (Decision #31), and the Target IR directive it
produces is checked here field by field before anything is launched.

Scale layouts (both row-major, fp32):

* ``lhs_scale``: ``[M, K / scale_k]`` -- one scale per token per K group.
* ``rhs_scale``: ``[K / scale_k, ceil(N / scale_n)]`` -- one scale per weight
  block. AITER's ``b_scale`` is ``[ceil(N/128), K/128]``; its transpose is this.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
from typing import TypedDict

import numpy as np

from .native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
)
from .rocm_native import ROCMNativePackage, _compile_native_tile_ir
from .scheduled_matmul import find_tessera_opt, run_tessera_opt

FP8_W8A8_BLOCKSCALE_CONTRACT = "rocm_fp8_w8a8_blockscale_v1"
FP8_W8A8_BLOCKSCALE_NK_CONTRACT = "rocm_fp8_w8a8_blockscale_nk_v1"
GFX_FP8_W8A8_BLOCKSCALE_ABI = (
    "tessera.rocm.fp8_w8a8_blockscale.a_b_sa_sb_o_m_n_k."
    "e4m3_e4m3_f32_f32.wmma_exact.v1"
)
#: The same math with the weight stored [N, K] (K contiguous): the layout W8A8
#: checkpoints ship and the one AITER's kernel reads.
GFX_FP8_W8A8_BLOCKSCALE_NK_ABI = (
    "tessera.rocm.fp8_w8a8_blockscale.a_bnk_sa_sb_o_m_n_k."
    "e4m3_e4m3_f32_f32.wmma_exact.v1"
)
WEIGHT_LAYOUTS = {
    "kn": (FP8_W8A8_BLOCKSCALE_CONTRACT, GFX_FP8_W8A8_BLOCKSCALE_ABI,
           "a_b_lhs_scale_rhs_scale_d_m_n_k"),
    "nk": (FP8_W8A8_BLOCKSCALE_NK_CONTRACT, GFX_FP8_W8A8_BLOCKSCALE_NK_ABI,
           "a_bnk_lhs_scale_rhs_scale_d_m_n_k"),
}
_DIRECTIVE = "tessera_rocm.scaled_wmma_gemm"


@dataclass(frozen=True)
class BlockScaleShape:
    """A static W8A8 problem and its scale blocking."""

    m: int
    n: int
    k: int
    scale_k: int = 128
    scale_n: int = 128
    #: "kn": B is [K, N] row-major. "nk": the weight is [N, K] row-major
    #: (``transposeB``), K contiguous. A semantic key: it decides which
    #: matrix the bytes are, so it selects a distinct named contract.
    weight_layout: str = "kn"

    def __post_init__(self) -> None:
        if self.weight_layout not in WEIGHT_LAYOUTS:
            raise ValueError(f"weight_layout must be one of {sorted(WEIGHT_LAYOUTS)}")
        if min(self.m, self.n, self.k, self.scale_k, self.scale_n) <= 0:
            raise ValueError("W8A8 block-scale extents must be positive")
        if self.scale_k % 16:
            raise ValueError("scale_k must be a whole number of 16-wide WMMA K steps")
        if self.k % self.scale_k:
            raise ValueError(
                f"K={self.k} is not a whole number of scale groups of {self.scale_k}")

    @property
    def groups(self) -> int:
        return self.k // self.scale_k

    @property
    def n_groups(self) -> int:
        return (self.n + self.scale_n - 1) // self.scale_n


def author_blockscale_graph(shape: BlockScaleShape, *, entry: str = "w8a8_blockscale") -> str:
    """Author the logical Graph op; the physical contract is derived, not stated."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry):
        raise ValueError(f"invalid entry symbol {entry!r}")
    m, n, k, g, ng = shape.m, shape.n, shape.k, shape.groups, shape.n_groups
    nk = shape.weight_layout == "nk"
    b_type = f"tensor<{n}x{k}xf8E4M3FN>" if nk else f"tensor<{k}x{n}xf8E4M3FN>"
    transpose = ",\n      transposeB = true" if nk else ""
    return f'''module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
  func.func @{entry}(%a: tensor<{m}x{k}xf8E4M3FN>, %b: {b_type},
                     %sa: tensor<{m}x{g}xf32>, %sb: tensor<{g}x{ng}xf32>) -> tensor<{m}x{n}xf32> {{
    %0 = tessera.scaled_matmul %a, %b scales(%sa, %sb) {{
      numeric_policy = {{accum = "fp32", execution_mode = "exact_per_block"}},
      scale_layout = {{granularity = "block", block = [{shape.scale_n}, {shape.scale_k}], format = "fp32"}}{transpose}
    }} : (tensor<{m}x{k}xf8E4M3FN>, {b_type}, tensor<{m}x{g}xf32>,
         tensor<{g}x{ng}xf32>) -> tensor<{m}x{n}xf32>
    return %0 : tensor<{m}x{n}xf32>
  }}
}}
'''


@dataclass(frozen=True)
class BlockScaleProgram:
    """Graph/Schedule/Tile text for one W8A8 problem, before packaging."""

    shape: BlockScaleShape
    entry: str
    graph_ir: str
    schedule_ir: str
    tile_ir: str


def lower_blockscale(
    shape: BlockScaleShape, *, entry: str = "w8a8_blockscale", tessera_opt: Path | None = None,
) -> BlockScaleProgram:
    """Run Graph->Schedule->Tile through the compiler (no Python lowering)."""
    tool = tessera_opt or find_tessera_opt()
    if tool is None:
        raise RuntimeError("tessera-opt is required to lower a W8A8 block-scale matmul")
    graph_ir = author_blockscale_graph(shape, entry=entry)
    schedule_ir = run_tessera_opt(tool, graph_ir, "--tessera-graph-to-schedule")
    tile_ir = run_tessera_opt(tool, schedule_ir, "--tessera-schedule-to-tile")
    return BlockScaleProgram(shape, entry, graph_ir, schedule_ir, tile_ir)


def _one_line(text: str, needle: str, what: str) -> str:
    lines = [line.strip() for line in text.splitlines() if needle in line]
    if len(lines) != 1:
        raise ValueError(f"W8A8 block-scale packaging requires exactly one {what}")
    return lines[0]


def _string_attr(operation: str, name: str) -> str:
    match = re.search(rf"(?<![\w.]){re.escape(name)}\s*=\s*\"([^\"]*)\"", operation)
    if match is None:
        raise ValueError(f"W8A8 block-scale IR is missing {name}")
    return match.group(1)


def _int_attr(operation: str, name: str) -> int:
    match = re.search(rf"(?<![\w.]){re.escape(name)}\s*=\s*(-?\d+)(?:\s*:\s*i64)?", operation)
    if match is None:
        raise ValueError(f"W8A8 block-scale IR is missing {name}")
    return int(match.group(1))


class CheckedDirective(TypedDict):
    """What a validated Target directive tells the launch descriptor."""

    block_m: int
    block_n: int
    macro_k: int
    schedule_hash: str


def check_blockscale_target_ir(shape: BlockScaleShape, tile_ir: str, target_ir: str) -> CheckedDirective:
    """Validate the Target directive against the requested contract.

    Every semantic field is compared; nothing is defaulted. Returns the
    directive's panel and K blocking for the launch descriptor.
    """
    directive = _one_line(target_ir, _DIRECTIVE, _DIRECTIVE + " directive")
    carrier = _one_line(tile_ir, "tile.scaled_matmul_kernel", "tile.scaled_matmul_kernel carrier")
    contract, package_abi, pointer_abi = WEIGHT_LAYOUTS[shape.weight_layout]
    expected_strings = {
        "abi": pointer_abi,
        "physical_contract": contract,
        "package_abi": package_abi,
        "scale_format": "fp32",
        "partial_combine": "scale_outer_product_then_add",
        "k_step_schedule": "isolated_scale_group",
        "output": "f32",
    }
    for name, expected in expected_strings.items():
        if _string_attr(directive, name) != expected:
            raise ValueError(f"W8A8 Target IR requires {name}={expected!r}")
    expected_ints = {
        "m": shape.m, "n": shape.n, "k": shape.k, "instruction_k": 16,
        "scale_k": shape.scale_k, "scale_n": shape.scale_n,
    }
    for name, expected_int in expected_ints.items():
        if _int_attr(directive, name) != expected_int:
            raise ValueError(f"W8A8 Target IR requires {name}={expected_int}")
    macro_k = _int_attr(directive, "macro_k")
    if macro_k <= 0 or macro_k % shape.scale_k:
        raise ValueError("W8A8 Target IR macro_k must hold whole scale groups")
    block_m, block_n = _int_attr(directive, "block_m"), _int_attr(directive, "block_n")
    if block_m <= 0 or block_n <= 0 or block_m % 16 or block_n % 16:
        raise ValueError("W8A8 Target IR register panel must be 16-aligned")
    policy = re.search(r"\bnumeric_policy\s*=\s*\{([^}]*)\}", directive)
    if policy is None:
        raise ValueError("W8A8 Target IR is missing numeric_policy")
    for name, expected in {"accum": "f32", "storage": "e4m3", "execution_mode": "exact_per_block"}.items():
        if _string_attr(policy.group(1), name) != expected:
            raise ValueError(f"W8A8 Target IR requires numeric_policy.{name}={expected!r}")
    for name, expected in {
        "physical_contract": contract,
        "combine": "scale_outer_product_then_add",
        "init": "zero",
        "scope": "scale_group",
        "cross_step_motion": "forbid",
    }.items():
        if _string_attr(carrier, name) != expected:
            raise ValueError(f"W8A8 Tile IR requires {name}={expected!r}")
    if _int_attr(carrier, "tessera.scale_block_n") != shape.scale_n:
        raise ValueError("W8A8 Tile IR tessera.scale_block_n disagrees with the request")
    tile_hash = _string_attr(carrier, "tessera.schedule_hash")
    if _string_attr(directive, "tessera.schedule_hash") != tile_hash:
        raise ValueError("W8A8 Tile/Target tessera.schedule_hash mismatch")
    return CheckedDirective(block_m=block_m, block_n=block_n, macro_k=macro_k,
                            schedule_hash=tile_hash)


def package_blockscale(
    program: BlockScaleProgram, *, pipeline_name: str = "tessera-lower-to-rocm", k_unroll: int = 1,
    scale_group_panels: int = -1,
) -> ROCMNativePackage:
    """Compile the Tile program to a gfx1201 HSACO and bind its launch ABI.

    ``k_unroll`` (whole scale groups per loop iteration) and
    ``scale_group_panels`` (panels per inner step of one group) are
    performance keys; neither changes what a group computes. -1 keeps the
    generator's measured default."""
    shape = program.shape
    contract, package_abi, _ = WEIGHT_LAYOUTS[shape.weight_layout]
    target_ir, backend_ir, payload, compiler_fp, toolchain_fp, libraries, compile_state = (
        _compile_native_tile_ir(
            program.tile_ir, directive=_DIRECTIVE, family="matmul", architecture="gfx1201",
            staging="register", k_unroll=int(k_unroll),
            scale_group_panels=int(scale_group_panels),
        )
    )
    checked = check_blockscale_target_ir(shape, program.tile_ir, target_ir)
    image = NativeImageArtifact(
        target="rocm_gfx1201",
        architecture="gfx1201",
        pipeline_name=pipeline_name,
        compiler_fingerprint=compiler_fp,
        toolchain_fingerprint=toolchain_fp,
        target_ir_digest=hashlib.sha256(target_ir.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(program.entry, package_abi),),
        compile_state=compile_state,
        device_libraries=libraries,
    )
    m, n, k = shape.m, shape.n, shape.k
    nk = shape.weight_layout == "nk"
    bindings = (
        BufferBinding(0, "a", "input", "fp8_e4m3", 2, "row_major", 1),
        BufferBinding(1, "b", "input", "fp8_e4m3", 2, "row_major", 1),
        BufferBinding(2, "a_scale", "input", "fp32", 2, "row_major", 4),
        BufferBinding(3, "b_scale", "input", "fp32", 2, "row_major", 4),
        BufferBinding(4, "o", "output", "fp32", 2, "row_major", 4),
    )
    descriptor = LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=program.entry,
        abi_id=package_abi,
        buffers=bindings,
        scalars=(ScalarArgument(5, "M", "int64"), ScalarArgument(6, "N", "int64"),
                 ScalarArgument(7, "K", "int64")),
        shape_guards=(
            ShapeGuard("a", 0, "eq", m), ShapeGuard("a", 1, "eq", k),
            ShapeGuard("b", 0, "eq", n if nk else k), ShapeGuard("b", 1, "eq", k if nk else n),
            ShapeGuard("a_scale", 0, "eq", m), ShapeGuard("a_scale", 1, "eq", shape.groups),
            ShapeGuard("b_scale", 0, "eq", shape.groups),
            ShapeGuard("b_scale", 1, "eq", shape.n_groups),
            ShapeGuard("o", 0, "eq", m), ShapeGuard("o", 1, "eq", n),
        ),
        geometry=LaunchGeometry(policy="rocm_wmma_macro_tile_grid"),
        ordering=OrderingSemantics(ordered_submission=True, residency="none",
                                   synchronization=("completion",)),
        provenance={
            "work_item": "ROCM-FP8-BLOCKSCALE-1",
            "sync_key": "GFX1201-LANES-2026-09-27",
            "route": "canonical_scheduled_tile_consumer",
            "physical_contract": contract,
            "materializer": "generate-wmma-gemm-kernel",
            "b_layout": shape.weight_layout,
            "scale_group_panels": int(scale_group_panels),
            "physical_route": (f"gfx1201_register_wmma_blockscale_{shape.weight_layout}_"
                               f"{checked['block_m'] // 16}x"
                               f"{checked['block_n'] // 16}_k{checked['macro_k']}"
                               + (f"_u{k_unroll}" if k_unroll > 1 else "")),
            "shape": [m, n, k],
            "scale_k": shape.scale_k,
            "scale_n": shape.scale_n,
            "macro_k": checked["macro_k"],
            "k_unroll": int(k_unroll),
            "macro_tile": [checked["block_m"], checked["block_n"]],
            "workgroup": [32, 1, 1],
            "a_storage": "e4m3",
            "b_storage": "e4m3",
            "accum": "f32",
            "output_storage": "f32",
            "numeric_policy": {"storage": "e4m3", "accum": "f32",
                               "execution_mode": "exact_per_block"},
            "schedule_hash": checked["schedule_hash"],
            "graph_ir_sha256": hashlib.sha256(program.graph_ir.encode()).hexdigest(),
            "tile_ir_sha256": hashlib.sha256(program.tile_ir.encode()).hexdigest(),
            "target_ir_sha256": image.target_ir_digest,
        },
    )
    return ROCMNativePackage(program.tile_ir, target_ir, backend_ir, image, descriptor)


def compile_blockscale(
    shape: BlockScaleShape, *, entry: str = "w8a8_blockscale", k_unroll: int = 1,
    scale_group_panels: int = -1, tessera_opt: Path | None = None,
) -> ROCMNativePackage:
    """Graph -> Schedule -> Tile -> Target -> HSACO for one static problem."""
    return package_blockscale(lower_blockscale(shape, entry=entry, tessera_opt=tessera_opt),
                              k_unroll=k_unroll, scale_group_panels=scale_group_panels)


def blockscale_reference(
    a: np.ndarray, b: np.ndarray, a_scale: np.ndarray, b_scale: np.ndarray,
    *, scale_k: int, scale_n: int,
) -> np.ndarray:
    """fp64 oracle of the block-scaled math: sum_g (A_g @ B_g) * sa[:, g] * sb[g, n // scale_n]."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    m, k = a64.shape
    n = b64.shape[1]
    groups = k // scale_k
    if k % scale_k or a_scale.shape != (m, groups) or b_scale.shape[0] != groups:
        raise ValueError("block-scale reference shapes disagree with scale_k")
    column_block = np.arange(n) // scale_n
    out = np.zeros((m, n), dtype=np.float64)
    for g in range(groups):
        span = slice(g * scale_k, (g + 1) * scale_k)
        partial = a64[:, span] @ b64[span, :]
        out += (partial * np.asarray(a_scale[:, g], np.float64)[:, None]
                * np.asarray(b_scale[g, column_block], np.float64)[None, :])
    return out


__all__ = [
    "BlockScaleProgram",
    "BlockScaleShape",
    "CheckedDirective",
    "FP8_W8A8_BLOCKSCALE_CONTRACT",
    "FP8_W8A8_BLOCKSCALE_NK_CONTRACT",
    "GFX_FP8_W8A8_BLOCKSCALE_ABI",
    "GFX_FP8_W8A8_BLOCKSCALE_NK_ABI",
    "WEIGHT_LAYOUTS",
    "author_blockscale_graph",
    "blockscale_reference",
    "check_blockscale_target_ir",
    "compile_blockscale",
    "lower_blockscale",
    "package_blockscale",
]
