"""NVIDIA / AMD matmul optimization ladder (cloudrift GEMM blog → Tessera).

The cloudrift "GPU Matmul Optimization" article builds a naive matmul up to
cuBLAS parity on an **RTX 5090 (Blackwell, sm_120)** — the *same* architecture as
the NVIDIA bring-up box — through a fixed sequence of techniques, each with a
measured speedup. This module encodes that sequence as a Tessera optimization
**ladder**: every technique names the target(s) it applies to, the Tessera pass
or autotuner axis that owns it, its measured blog speedup, and whether it is
verifiable on hardware-free infrastructure *here* or gated on real silicon.

Why a ladder and not just a doc: it doubles as the NVIDIA/AMD Evaluator bring-up
sequence — each rung is a measurable step toward cuBLAS/rocBLAS parity with a
known expected delta, and every rung must preserve numerics (metamorphic) while
moving latency. When the Blackwell / Strix Halo boxes come online the port is a
checklist with oracles, not a blank page.

One technique — **split-K** — is a semantics-preserving *rewrite*, so it ships
with a real executable oracle (:func:`verify_split_k_equivalence`): the partial-K
products, reduced, must equal the dense product (up to float reassociation). That
correctness is provable on CPU / Apple GPU **now**, before any NVIDIA/AMD run.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (matmul opt ladder).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


class OptTarget(enum.Enum):
    """Which backend(s) a technique applies to."""

    NVIDIA = "nvidia"
    AMD = "amd"
    BOTH = "both"


@dataclass(frozen=True)
class OptTechnique:
    """One rung of the matmul optimization ladder.

    ``owner`` is the Tessera pass or autotuner axis that already exists (or must
    exist) to express the technique. ``blog_speedup`` is the article's measured
    lever in isolation. ``verifiable_now`` is True when the technique can be
    proven on hardware-free infra here (numerics / structural emission) — False
    when only real silicon can measure its benefit.
    """

    key: str
    title: str
    targets: OptTarget
    owner: str
    blog_speedup: str
    verifiable_now: bool
    mechanism: str
    notes: str = ""


# The ladder, in the blog's naive → cuBLAS-parity order. Owners reference real
# Tessera surfaces: autotune_v2.TuningConfig (tile_m/n/k, num_warps, num_stages),
# WarpSpecializationPass, NVTMADescriptorPass, AsyncCopyLoweringPass,
# ptx_emit (WGMMA), rocdl_emit (WMMA).
MATMUL_OPT_LADDER: tuple[OptTechnique, ...] = (
    OptTechnique(
        "register_tile", "Register tiling (outer-product, FM×FN per thread)",
        OptTarget.BOTH, "autotune_v2.TuningConfig (thread reg-tile axis — new)",
        "5.2× (dominant single lever)", True,
        "Each thread owns an FM×FN grid of accumulators; operands reused across "
        "MACs via register outer products. Wins despite 67%→17% occupancy — "
        "arithmetic intensity beats occupancy.",
        "Verifiable now as an arithmetic-intensity model (see "
        "register_tile_intensity); the perf needs hardware. Make this the "
        "autotuner's PRIMARY axis.",
    ),
    OptTechnique(
        "smem_staging", "Shared-memory / LDS staging",
        OptTarget.BOTH, "TilingPass + AsyncCopyLoweringPass",
        "foundation (enables all data-movement opts)", False,
        "Cooperatively stage BM×BK and BK×BN slabs into shared/LDS; all threads "
        "read operands from on-chip memory.",
        "Already structural in Tessera's tile lowering.",
    ),
    OptTechnique(
        "tensor_core", "Tensor-core / MFMA / WMMA matmul",
        OptTarget.BOTH, "ptx_emit (WGMMA) / rocdl_emit (WMMA)",
        "~3× over scalar FMA", True,
        "One matrix instruction per warp/wavefront (mma.sync / wgmma on NVIDIA, "
        "v_wmma / v_mfma on AMD); fp16/bf16 inputs, fp32 accumulate.",
        "Emission is verifiable now (WGMMA PTX text rung 3; WMMA assembles to "
        "real v_wmma at rung 4 on-host via llc). Execution gated on silicon.",
    ),
    OptTechnique(
        "double_buffer_pipeline", "Double-buffering + software pipelining",
        OptTarget.BOTH, "autotune_v2.num_stages (pipeline depth) + AsyncCopyLoweringPass",
        "~30% of cumulative staging-path gains", False,
        "Prologue/main/epilogue K-loop: issue slab k+1 while computing slab k; "
        "pipeline depth = buffer count hides memory latency under compute.",
        "num_stages already an autotuner axis — sweep it deeper.",
    ),
    OptTechnique(
        "async_copy_tma", "Async copy (cp.async sm_80+) / TMA (sm_90+)",
        OptTarget.NVIDIA, "AsyncCopyLoweringPass + NVTMADescriptorPass",
        "part of the staging pipeline", False,
        "cp.async bypasses registers global→shared; TMA copies a whole 2-D tile "
        "atomically via a descriptor + mbarrier with hardware swizzle.",
        "AMD has no TMA/cp.async — the cp.async path's optimizations (esp. LDS "
        "padding) carry to AMD's ds_read/ds_write + s_waitcnt pipeline instead.",
    ),
    OptTechnique(
        "warp_specialization", "Warp specialization (producer/consumer + mbarrier ring)",
        OptTarget.NVIDIA, "WarpSpecializationPass",
        "beats cuBLAS fp16 (105%)", False,
        "One producer warp issues TMA copies; consumer warps compute only, "
        "synchronized via an mbarrier ring buffer (depth ≥2). Decouples loads "
        "from compute entirely.",
        "Validates Tessera's structural warp-role design (Decision #8). Does NOT "
        "transfer to AMD (no mbarrier) — use the manual double-buffered pipeline.",
    ),
    OptTechnique(
        "smem_pad_swizzle", "Shared-memory / LDS bank-conflict padding (+1 stride / swizzle)",
        OptTarget.BOTH, "smem/LDS layout pass (new) — free under TMA hardware swizzle",
        "3.7× on the cp.async transport", False,
        "Pad the inner stride so successive rows land in different banks, breaking "
        "XOR collisions on vector reads.",
        "Highest-value cheap lever on the NON-TMA path — i.e. sm_80 cp.async AND "
        "the AMD LDS path. Zero benefit under TMA (hardware swizzles).",
    ),
    OptTechnique(
        "cta_swizzle", "CTA swizzle (GROUP_M block remap)",
        OptTarget.BOTH, "persistent-CTA grid mapping",
        "5% at 8192³ (L2-bound)", False,
        "Remap blockIdx so consecutive blocks walk GROUP_M tiles down M before "
        "stepping N, raising L2 hit rate.",
        "Cheap grid-remap preamble; only helps L2-bound (low-intensity) tiles.",
    ),
    OptTechnique(
        "split_k", "Split-K reduction",
        OptTarget.BOTH, "matmul_opt_ladder.split_k_matmul + reduction/collective insertion",
        "7.1× on skinny large-K (128×128×16384)", True,
        "Partition K across blocks; each computes a partial sum, results reduced "
        "(atomicAdd, or a 2-kernel tree reduction). Fills idle SMs when the output "
        "tile grid is smaller than the SM count.",
        "Semantics-preserving rewrite — proven HERE by verify_split_k_equivalence "
        "(partials reduced == dense product up to fp reassociation).",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Split-K — the executable, semantics-preserving rung
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SplitKConfig:
    """How the K (contraction) dimension is partitioned across blocks."""

    splits: int
    reduce: str = "tree"   # "tree" = pairwise (deterministic) | "atomic" = sequential (models atomicAdd order)

    def __post_init__(self) -> None:
        if self.splits < 1:
            raise ValueError("splits must be >= 1")
        if self.reduce not in {"tree", "atomic"}:
            raise ValueError("reduce must be 'tree' or 'atomic'")


def _split_bounds(k: int, splits: int) -> list[tuple[int, int]]:
    """Partition ``[0, k)`` into ``splits`` near-equal contiguous ranges."""
    if splits > k:
        splits = k  # never more splits than there are K elements
    base, rem = divmod(k, splits)
    bounds, lo = [], 0
    for i in range(splits):
        hi = lo + base + (1 if i < rem else 0)
        bounds.append((lo, hi))
        lo = hi
    return bounds


def split_k_matmul(
    A: np.ndarray, B: np.ndarray, config: SplitKConfig,
    *, matmul: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Compute ``A @ B`` by partitioning the K dimension and reducing partials.

    ``A`` is ``(M, K)``, ``B`` is ``(K, N)``. Each of ``config.splits`` blocks
    computes ``A[:, lo:hi] @ B[lo:hi, :]``; the partials are summed. ``matmul``
    lets a caller route each partial through a real backend (e.g. ``ops.matmul``
    on Apple GPU) to prove the rewrite end-to-end; it defaults to numpy.

    Numerically equal to ``A @ B`` up to floating-point reassociation — that is
    the metamorphic invariant the oracle checks.
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shape mismatch for matmul: A {A.shape} @ B {B.shape}")
    mm = matmul or (lambda a, b: np.asarray(a, np.float64) @ np.asarray(b, np.float64))
    partials = [np.asarray(mm(A[:, lo:hi], B[lo:hi, :]), np.float64)
                for lo, hi in _split_bounds(A.shape[1], config.splits)]
    if config.reduce == "tree":
        # Pairwise reduction (lower error growth) — the two-kernel tree variant.
        while len(partials) > 1:
            partials = [partials[i] + partials[i + 1] if i + 1 < len(partials)
                        else partials[i] for i in range(0, len(partials), 2)]
        return partials[0]
    # "atomic": sequential accumulation, the order an atomicAdd path would take.
    acc = partials[0]
    for p in partials[1:]:
        acc = acc + p
    return acc


@dataclass(frozen=True)
class SplitKVerdict:
    relation: str            # "equivalent" | "divergent"
    max_rel_err: float
    splits: int
    reduce: str
    detail: str = ""

    @property
    def is_equivalent(self) -> bool:
        return self.relation == "equivalent"


def verify_split_k_equivalence(
    A: np.ndarray, B: np.ndarray, config: SplitKConfig,
    *, matmul: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    rtol: float = 1e-9,
) -> SplitKVerdict:
    """Oracle: split-K ``A @ B`` equals the dense product up to fp reassociation.

    A miscompiled split (a dropped K range, an overlapping partition, a wrong
    reduction) diverges here. Executable on CPU now and, via ``matmul``, on Apple
    GPU — the rewrite is proven correct before any NVIDIA/AMD run.
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    ref = A @ B
    got = split_k_matmul(A, B, config, matmul=matmul)
    scale = float(np.max(np.abs(ref))) or 1.0
    max_rel_err = float(np.max(np.abs(got - ref)) / scale) if got.shape == ref.shape else float("inf")
    rel = "equivalent" if max_rel_err <= rtol else "divergent"
    detail = (f"split-K({config.splits},{config.reduce}) rel_err={max_rel_err:.2e} "
              f"(≤ {rtol:.0e})" if rel == "equivalent" else
              f"split-K diverges: rel_err={max_rel_err:.2e} > {rtol:.0e}")
    return SplitKVerdict(rel, max_rel_err, config.splits, config.reduce, detail)


# ─────────────────────────────────────────────────────────────────────────────
# Planning models (verifiable now; feed the autotuner's ranking)
# ─────────────────────────────────────────────────────────────────────────────


def split_k_profitable(M: int, N: int, K: int, *, tile_m: int, tile_n: int,
                        sm_count: int, splits: int) -> bool:
    """Split-K helps when the output-tile grid leaves SMs idle and K is large
    enough to amortize the extra reduction. Mirrors the blog's skinny-matrix case
    (few output tiles vs many SMs)."""
    if splits < 2:
        return False
    output_tiles = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
    if output_tiles >= sm_count:
        return False  # grid already saturates the device; split-K only adds cost
    # The split must not over-fragment K below a useful per-block depth.
    return (K // splits) >= tile_m  # heuristic: keep each K-slice at least tile-deep


def register_tile_intensity(tile_m: int, tile_n: int, *, elt_bytes: int = 4) -> float:
    """Arithmetic intensity (FLOPs per byte of operand load) of an FM×FN register
    tile's inner step — the lever register tiling pulls. A 1×1 tile loads 2 elems
    per 2 FLOPs (intensity ∝ 1); an FM×FN tile loads FM+FN elems for 2·FM·FN
    FLOPs, so intensity grows ~ (FM·FN)/(FM+FN). This is why the blog's coarse
    tile wins 5× despite far lower occupancy."""
    if tile_m < 1 or tile_n < 1:
        raise ValueError("tile dims must be >= 1")
    loads_bytes = (tile_m + tile_n) * elt_bytes
    flops = 2.0 * tile_m * tile_n
    return flops / loads_bytes


def ladder_for_target(target: OptTarget | str) -> tuple[OptTechnique, ...]:
    """The ladder rungs that apply to ``target`` (BOTH always included)."""
    if isinstance(target, str):
        target = OptTarget(target)
    return tuple(t for t in MATMUL_OPT_LADDER
                 if t.targets is target or t.targets is OptTarget.BOTH)


def render_ladder_markdown() -> str:
    """Render the ladder as a Markdown table (for the roadmap doc)."""
    head = ("| # | Technique | Targets | Owner (Tessera) | Blog speedup | "
            "Verifiable now |\n|---|---|---|---|---|---|\n")
    rows = []
    for i, t in enumerate(MATMUL_OPT_LADDER, 1):
        rows.append(f"| {i} | {t.title} | {t.targets.value} | {t.owner} | "
                    f"{t.blog_speedup} | {'yes' if t.verifiable_now else 'silicon-gated'} |")
    return head + "\n".join(rows) + "\n"


__all__ = [
    "OptTarget",
    "OptTechnique",
    "MATMUL_OPT_LADDER",
    "SplitKConfig",
    "split_k_matmul",
    "SplitKVerdict",
    "verify_split_k_equivalence",
    "split_k_profitable",
    "register_tile_intensity",
    "ladder_for_target",
    "render_ladder_markdown",
]
