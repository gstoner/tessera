"""Apple rung-2.5 — MSL ``simdgroup_matrix`` GEMM emission (host-free Stage-A spike).

The Apple analog of :mod:`tessera.compiler.ptx_emit` (NVIDIA WGMMA PTX, rung 2.5)
and the AMD ``rocdl.wmma`` path. See ``docs/audit/compiler/EVALUATOR_PLAN.md`` §2
(rung ladder) and ``docs/audit/backend/apple/APPLE_AUDIT.md`` (the MLX "steel"
blueprint that motivates this lane).

**The gap this addresses.** Tessera's Apple GPU matmul today runs through MPS /
MPSGraph — a black box that gives no fusion control. MLX (the production Apple-
Silicon ML framework) does *not* use MPS for GEMM: it ships its own templated MSL
library ("steel") built on ``metal::simdgroup_matrix`` + ``simdgroup_multiply_
accumulate`` (the native Apple7+ SIMD-scoped matrix-multiply, confirmed available
on this M1 Max in the Metal Feature Set Tables). This module is the first step of
a **native ``simdgroup_matrix`` GEMM lane** — the "clear-MPS" direction — emitting
the documented MSL ``simdgroup_matrix`` GEMM and validating its structure host-free.

**Honesty ceiling (read this).** Like ``ptx_emit``'s WGMMA skeleton, this emits the
documented ``simdgroup_matrix`` MMA sequence (the steel structure: 8×8 fragment
load → multiply-accumulate over the K-loop → store) inside a structurally-valid
MSL kernel. It is **not** a perf-optimal or fully boundary-correct kernel: a
production GEMM also needs threadgroup-memory staging, cooperative loads across the
simdgroup, multi-fragment M/N tiling, and ragged-edge masking — deliberately
omitted here. The ``simdgroup_matrix`` API is grounded from MLX ``steel/gemm/mma.h``
(production source) + the MSL spec ch.6; it is **not** compile-verified on this host
(the offline ``metal`` compiler is absent under CommandLineTools — the same
situation as ``ptxas`` on the arm64 dev Mac). So:

  * :func:`validate_msl_gemm_structure` checks the MSL scaffolding + that the
    emitted ``simdgroup_matrix`` fragment dtypes/shape match the request and the
    documented load→mma→store sequence is present — verifiable *here*, no toolchain.
    This is what earns rung 2.5.
  * :func:`metal_compile` invokes the real offline ``metal`` compiler (rung 3) —
    Darwin + Metal-toolchain only; it skip-cleans when ``metal`` is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# Apple GPU SIMD-scoped matrix multiply uses 8×8 fragments (the simdgroup_matrix
# fragment size). GEMM tile dims must be whole multiples of the fragment.
SIMDGROUP_FRAG = 8

# Input element dtype → MSL scalar type. bfloat needs MSL 3.1 (memory:
# apple7-m1max-gpu-feature-set); half/float are long-standing.
_MSL_SCALAR: dict[str, str] = {
    "f16": "half",
    "fp16": "half",
    "bf16": "bfloat",
    "f32": "float",
    "fp32": "float",
}

# Input dtype → minimum -std=metal language version the kernel must be compiled at.
_MIN_METAL_STD: dict[str, str] = {
    "half": "metal3.0",
    "bfloat": "metal3.1",   # bfloat is an MSL 3.1 type
    "float": "metal3.0",
}


@dataclass(frozen=True)
class MslGemmShape:
    """A ``simdgroup_matrix`` GEMM tile. ``M``/``N``/``K`` are the per-threadgroup
    tile dims; each must be a positive multiple of the 8×8 fragment."""

    m: int
    n: int
    k: int

    def is_valid(self) -> bool:
        f = SIMDGROUP_FRAG
        return all(d > 0 and d % f == 0 for d in (self.m, self.n, self.k))


def _scalar(dtype: str) -> str:
    s = _MSL_SCALAR.get(dtype)
    if s is None:
        raise ValueError(
            f"unsupported simdgroup_matrix input dtype {dtype!r} "
            f"(supported: {sorted(set(_MSL_SCALAR))})")
    return s


def emit_simdgroup_gemm_msl(
    dtype: str = "bf16",
    m: int = 8,
    n: int = 8,
    k: int = 8,
    *,
    accum: str = "f32",
    entry: str | None = None,
) -> str:
    """Emit a structurally-valid MSL ``simdgroup_matrix`` GEMM ``C = A·B`` for an
    ``(m, n, k)`` per-threadgroup tile and ``dtype`` inputs.

    Accumulation is in ``accum`` (fp32 by default — the production pattern that
    matches Tessera's ``numeric_policy{storage, accum}`` and MLX steel's fp32 acc).
    Skeleton only (see module docstring): the fragment load/store strides are the
    documented row-major form; threadgroup staging / ragged-edge masking are omitted.
    """
    shape = MslGemmShape(m, n, k)
    if not shape.is_valid():
        raise ValueError(
            f"({m},{n},{k}) is not a valid simdgroup_matrix GEMM tile "
            f"(each dim must be a positive multiple of {SIMDGROUP_FRAG}) — refusing to emit")
    T = _scalar(dtype)
    ACC = _scalar(accum)
    f = SIMDGROUP_FRAG
    name = entry or f"tessera_simdgroup_gemm_{dtype}"
    # One 8×8 output fragment per simdgroup; m/n/k > 8 loop over fragments. This
    # slice emits the canonical single-output-fragment K-loop (the steel inner
    # loop); multi-fragment M/N tiling is the documented next sub-step.
    return f"""//
// Tessera rung-2.5 emission — Apple simdgroup_matrix {dtype} GEMM (steel-style skeleton).
// NOT a complete/optimal kernel: a production GEMM also needs threadgroup staging,
// cooperative simdgroup loads, multi-fragment M/N tiling, and ragged-edge masking,
// omitted here. metal-compile is the rung-3 gate; this asserts the documented
// simdgroup_matrix<{T},{f},{f}> load -> multiply_accumulate -> store sequence with an
// {ACC} accumulator. API grounded from MLX steel/gemm/mma.h + MSL spec ch.6.
//
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void {name}(
    device const {T}*  A [[buffer(0)]],
    device const {T}*  B [[buffer(1)]],
    device {ACC}*      C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{{
  // 8x8 simdgroup_matrix fragments (Apple7+ SIMD-scoped matrix multiply).
  const uint m0 = tgid.y * {m}u;   // tile origin row
  const uint n0 = tgid.x * {n}u;   // tile origin col
  simdgroup_matrix<{ACC}, {f}, {f}> acc =
      make_filled_simdgroup_matrix<{ACC}, {f}, {f}>({ACC}(0));
  for (uint k0 = 0; k0 < K; k0 += {f}u) {{
    simdgroup_matrix<{T}, {f}, {f}> a, b;
    simdgroup_load(a, A + m0 * K + k0, K);      // row-major A tile, stride K
    simdgroup_load(b, B + k0 * N + n0, N);      // row-major B tile, stride N
    simdgroup_multiply_accumulate(acc, a, b, acc);  // acc = a * b + acc
  }}
  simdgroup_store(acc, C + m0 * N + n0, N);     // row-major C tile, stride N
}}
"""


# Structural tokens every emitted simdgroup_matrix GEMM must carry (host-free check).
_REQUIRED_TOKENS: tuple[str, ...] = (
    "#include <metal_simdgroup_matrix>",
    "make_filled_simdgroup_matrix",
    "simdgroup_load",
    "simdgroup_multiply_accumulate",
    "simdgroup_store",
)


@dataclass(frozen=True)
class MslValidation:
    ok: bool
    reasons: tuple[str, ...]


def validate_msl_gemm_structure(
    msl: str, *, dtype: str = "bf16", accum: str = "f32",
    shape: MslGemmShape | None = None,
) -> MslValidation:
    """Host-free rung-2.5 check: the emitted MSL carries the documented
    ``simdgroup_matrix`` GEMM structure for the requested dtype/accumulator/shape.
    No Metal toolchain required."""
    reasons: list[str] = []
    for tok in _REQUIRED_TOKENS:
        if tok not in msl:
            reasons.append(f"missing required token: {tok!r}")
    T, ACC, f = _scalar(dtype), _scalar(accum), SIMDGROUP_FRAG
    if f"simdgroup_matrix<{T}, {f}, {f}>" not in msl:
        reasons.append(f"missing {T} input fragment simdgroup_matrix<{T}, {f}, {f}>")
    if f"simdgroup_matrix<{ACC}, {f}, {f}>" not in msl:
        reasons.append(f"missing {ACC} accumulator fragment")
    # Exactly two operand loads (A and B) feed the multiply-accumulate.
    if msl.count("simdgroup_load") < 2:
        reasons.append("expected >=2 simdgroup_load (A and B operands)")
    if "for (uint k0 = 0; k0 < K;" not in msl:
        reasons.append("missing the K-reduction loop over 8-wide fragments")
    if shape is not None and not shape.is_valid():
        reasons.append(f"invalid tile shape {(shape.m, shape.n, shape.k)}")
    return MslValidation(ok=not reasons, reasons=tuple(reasons))


def min_metal_std(dtype: str) -> str:
    """The minimum ``-std=metal*`` the emitted kernel must be compiled at for
    ``dtype`` (bf16 → metal3.1, since ``bfloat`` is an MSL 3.1 type)."""
    return _MIN_METAL_STD[_scalar(dtype)]


@dataclass(frozen=True)
class MetalCompileResult:
    status: str            # "ok" | "failed" | "skipped"
    detail: str


def metal_compile(msl: str, *, dtype: str = "bf16") -> MetalCompileResult:
    """Rung-3: compile the emitted MSL with the offline ``metal`` compiler.

    Darwin + Metal-toolchain only. **Skip-cleans** (``status="skipped"``) when the
    ``metal`` tool is absent — the case on this CommandLineTools-only arm64 Mac,
    exactly like ``ptxas`` skip-cleaning. Assemblability is NOT claimed here; this
    reports what the real compiler says when a toolchain is present (e.g. in CI on
    a Metal-capable runner)."""
    metal = shutil.which("metal")
    if metal is None:
        try:
            metal = subprocess.run(
                ["xcrun", "-f", "metal"], capture_output=True, text=True, timeout=20
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            metal = None
    if not metal:
        return MetalCompileResult("skipped", "offline 'metal' compiler not available")
    import tempfile
    from pathlib import Path
    std = min_metal_std(dtype)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "gemm.metal"
        out = Path(td) / "gemm.air"
        src.write_text(msl)
        try:
            proc = subprocess.run(
                [metal, f"-std={std}", "-c", str(src), "-o", str(out)],
                capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            return MetalCompileResult("failed", f"metal invocation error: {e}")
        if proc.returncode == 0 and out.exists():
            return MetalCompileResult("ok", f"compiled to AIR with -std={std}")
        return MetalCompileResult("failed", proc.stderr.strip() or "metal returned nonzero")


__all__ = [
    "SIMDGROUP_FRAG",
    "MslGemmShape",
    "MslValidation",
    "MetalCompileResult",
    "emit_simdgroup_gemm_msl",
    "validate_msl_gemm_structure",
    "min_metal_std",
    "metal_compile",
]
