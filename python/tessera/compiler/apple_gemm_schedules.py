"""Apple GEMM schedule candidates — autotune seeds mined from MLX (production).

The user's MLX deep-dive (2026-06-17) flagged that MLX's GEMM tile selection is
"exactly the kind of knobs Tessera should encode as schedule candidates rather than
hard-coded one-offs." This module encodes MLX's heuristic — grounded by reading
``mlx/backend/metal/matmul.cpp`` (the ``GEMM_TPARAM_MACRO`` tile table + the
``align_M/N/K`` / ``do_axpby`` / ``swizzle_log`` function constants) — as Tessera
**schedule candidates**:

  * a **seed default** per (device class × dtype × transpose × size) — MLX's
    heuristic pick, the autotuner's warm start;
  * a **sweep set** of the distinct tiles MLX uses — the candidates the autotuner /
    ``flywheel`` should measure;
  * the orthogonal schedule **axes** (alignment, fused αAB+βC epilogue, threadblock
    swizzle) MLX expresses as Metal function constants.

These are *seeds and knobs*, NOT copied kernels (Decision #23 — MLX is a reference,
never a runtime dep). Every tile's ``bm/bn/bk`` is a whole 8×8 simdgroup-fragment
multiple, so it feeds :func:`tessera.compiler.msl_gemm_emit.emit_steel_gemm_msl`
directly. ``wm``/``wn`` are the simdgroup grid (how many simdgroups tile M/N).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .msl_gemm_emit import SIMDGROUP_FRAG


class DeviceClass(Enum):
    """MLX keys tiles on the GPU's size class (the ``devc`` char in matmul.cpp):
    ``'g'``/``'p'`` → small (e.g. base M-series), ``'d'`` → large (Max/Ultra),
    else → medium."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass(frozen=True)
class AppleGemmTile:
    """A GEMM threadgroup tile: ``BM×BN`` output over ``BK`` steps, computed by a
    ``wm×wn`` grid of simdgroups (8×8 ``simdgroup_matrix`` fragments)."""

    bm: int
    bn: int
    bk: int
    wm: int   # simdgroups along M
    wn: int   # simdgroups along N

    def is_steel_compatible(self) -> bool:
        """True when ``bm/bn/bk`` are positive 8×8-fragment multiples — i.e. this
        tile is directly emittable by ``emit_steel_gemm_msl``."""
        f = SIMDGROUP_FRAG
        return all(d > 0 and d % f == 0 for d in (self.bm, self.bn, self.bk))


# The distinct tiles MLX's GEMM_TPARAM_MACRO selects across device/dtype/transpose
# — the autotuner sweep set (grounded from mlx/backend/metal/matmul.cpp).
MLX_SEED_TILES: tuple[AppleGemmTile, ...] = (
    AppleGemmTile(64, 64, 16, 1, 2),   # half/bf16 nn; medium default; large reasonable-K
    AppleGemmTile(64, 32, 32, 2, 2),   # half/bf16 nt; float32 nn (small)
    AppleGemmTile(32, 64, 16, 1, 2),   # float32 nt; half/bf16 large small-K
    AppleGemmTile(64, 64, 16, 2, 2),   # medium-device default
    AppleGemmTile(64, 32, 8, 4, 1),    # complex64
)


def select_seed_tile(
    device_class: DeviceClass,
    dtype: str = "bf16",
    *,
    transpose_b: bool = True,
    large: bool = False,
) -> AppleGemmTile:
    """The MLX heuristic seed tile for a problem (the autotuner's warm start),
    grounded from ``matmul.cpp``'s ``GEMM_TPARAM_MACRO``. ``transpose_b`` selects
    the ``nt`` path (``!transpose_a && transpose_b``); ``large`` is MLX's
    ``batch·M·N ≥ 2^20`` large-matmul branch (large device only)."""
    is_float32 = dtype in ("f32", "fp32", "float32")
    is_complex = dtype in ("complex64", "c64")
    if is_complex:
        return AppleGemmTile(64, 32, 8, 4, 1)

    if device_class is DeviceClass.MEDIUM:
        return AppleGemmTile(64, 64, 16, 2, 2)

    if device_class is DeviceClass.SMALL:
        if transpose_b and not is_float32:                 # nt half/bf16
            return AppleGemmTile(64, 32, 32, 2, 2)
        return AppleGemmTile(64, 64, 16, 1, 2)             # nn / general

    # LARGE device.
    if large:
        if not is_float32:                                 # half/bf16, large matmul
            return AppleGemmTile(64, 64, 16, 1, 2)
        return AppleGemmTile(32, 64, 16, 1, 2)
    # smaller matmul on a large device.
    if not is_float32:
        return (AppleGemmTile(64, 32, 32, 2, 2) if transpose_b
                else AppleGemmTile(64, 64, 16, 1, 2))
    return (AppleGemmTile(32, 64, 16, 1, 2) if transpose_b
            else AppleGemmTile(64, 32, 32, 2, 2))


@dataclass(frozen=True)
class GemmScheduleAxes:
    """The orthogonal schedule knobs MLX expresses as Metal function constants
    (``matmul.cpp``) — the axes Tessera's scheduler/autotuner should carry per
    problem, on top of the tile choice."""

    align_m: bool   # M % bm == 0  (fc 200) — skip ragged-M masking
    align_n: bool   # N % bn == 0  (fc 201)
    align_k: bool   # K % bk == 0  (fc 202)
    do_axpby: bool  # alpha != 1 or beta != 1 (fc 110) — fused α·AB + β·C epilogue
    swizzle_log: int  # threadblock swizzle: tile = 1 << swizzle_log


def schedule_axes_for(
    m: int, n: int, k: int, tile: AppleGemmTile,
    *, alpha: float = 1.0, beta: float = 0.0,
) -> GemmScheduleAxes:
    """Derive MLX's per-problem schedule axes for ``(M,N,K)`` under ``tile`` —
    alignment booleans, the αAB+βC epilogue flag, and the threadblock swizzle
    (``swizzle_log = tm<=3 ? 0 : 1`` where ``tm = ceil(M/bm)``)."""
    tm = (m + tile.bm - 1) // tile.bm
    swizzle_log = 0 if tm <= 3 else 1
    return GemmScheduleAxes(
        align_m=(m % tile.bm == 0),
        align_n=(n % tile.bn == 0),
        align_k=(k % tile.bk == 0),
        do_axpby=(alpha != 1.0 or beta != 0.0),
        swizzle_log=swizzle_log,
    )


__all__ = [
    "DeviceClass",
    "AppleGemmTile",
    "MLX_SEED_TILES",
    "GemmScheduleAxes",
    "select_seed_tile",
    "schedule_axes_for",
]
