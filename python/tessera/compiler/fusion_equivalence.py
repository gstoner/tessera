"""M5 — Evaluator-gated synthesizer-displacement oracle.

As the Apple GPU lane displaces the name→MPS/MSL dispatcher with fused MSL
codegen (M2/M4), each displacement must be provably safe: the synthesized Metal
kernel must equal the unfused library/numpy reference on *hidden* inputs (fresh
RNG the codegen never saw), and it must genuinely run on Metal — a silent numpy
fallback can never earn a "displaced" verdict (the Evaluator's provenance
invariant). This module is the reusable gate; a future migration calls
``displacement_verdict`` and only ships the lane when it returns ``equivalent``.

See docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md (M5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fusion as F


@dataclass(frozen=True)
class DisplacementVerdict:
    """The result of gating one synthesizer codegen path against its reference.

    ``relation``: ``equivalent`` (ran on Metal AND matched the reference),
    ``divergent`` (ran on Metal but mismatched — a real codegen bug), or
    ``not_displaced`` (fell back to the reference — no Metal, so no displacement
    to credit). Only ``equivalent`` is a green light to ship the lane."""

    kind: str
    relation: str
    executed: str
    max_rel_err: float | None
    detail: str


def _verdict(kind: str, executed: str, got: np.ndarray, ref: np.ndarray,
             rtol: float, atol: float) -> DisplacementVerdict:
    if executed != "metal_runtime":
        return DisplacementVerdict(
            kind, "not_displaced", executed, None,
            "fell back to the reference — no Metal execution to displace into")
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    if got.shape != ref.shape:
        return DisplacementVerdict(kind, "divergent", executed, None,
                                   f"shape mismatch {got.shape} != {ref.shape}")
    scale = float(np.max(np.abs(ref))) or 1.0
    max_err = float(np.max(np.abs(got - ref)))
    tol = atol + rtol * scale
    rel = "equivalent" if max_err <= tol else "divergent"
    return DisplacementVerdict(kind, rel, executed, max_err / scale,
                               f"max_abs_err={max_err:.3e} tol={tol:.3e}")


def displacement_verdict(kind: str, shape: tuple[int, ...], *, seed: int = 0,
                         rtol: float = 1e-3, atol: float = 1e-4
                         ) -> DisplacementVerdict:
    """Gate one synthesizer codegen path on **hidden** inputs (fresh RNG keyed by
    ``seed``). ``kind`` ∈ {matmul_epilogue, norm_chain, attention, pointwise}.

    ``shape`` is interpreted per kind: matmul_epilogue/attention use (M, K, N)
    (and a head_dim for attention's V); norm_chain/pointwise use (rows, cols)."""
    rng = np.random.default_rng(seed)

    if kind == "matmul_epilogue":
        M, K, N = shape
        region = F.FusedRegion(epilogue=("gelu",))
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        out, ex = F.run_fused_region(region, A, B)
        return _verdict(kind, ex, out, region.reference(A, B), rtol, atol)

    if kind == "norm_chain":
        rows, cols = shape
        region = F.NormChainRegion("rmsnorm", add_residual=True, weight=True)
        X = rng.standard_normal((rows, cols)).astype(np.float32)
        R = rng.standard_normal((rows, cols)).astype(np.float32)
        G = rng.standard_normal((cols,)).astype(np.float32)
        out, ex = F.run_norm_chain_region(region, X, residual=R, gamma=G)
        return _verdict(kind, ex, out, region.reference(X, R, G), rtol, atol)

    if kind == "attention":
        M, D, Nk = shape
        region = F.AttentionRegion()
        Q = rng.standard_normal((M, D)).astype(np.float32)
        Kt = rng.standard_normal((Nk, D)).astype(np.float32)
        V = rng.standard_normal((Nk, D)).astype(np.float32)
        out, ex = F.run_fused_attention(region, Q, Kt, V)
        return _verdict(kind, ex, out, region.reference(Q, Kt, V), rtol, atol)

    if kind == "pointwise":
        rows, cols = shape
        region = F.PointwiseGraphRegion(
            ops=(("mul", ("x", "a"), "m"), ("add", ("m", "b"), "s"),
                 ("gelu", ("s",), "o")),
            inputs=("x", "a", "b"), output="o")
        arrs = [rng.standard_normal((rows, cols)).astype(np.float32)
                for _ in range(3)]
        out, ex = F.run_pointwise_graph(region, arrs)
        return _verdict(kind, ex, out, region.reference(*arrs), rtol, atol)

    raise ValueError(f"unknown displacement kind {kind!r}")


#: The synthesizer codegen lanes that displace the per-op dispatcher today.
DISPLACED_LANES = ("matmul_epilogue", "norm_chain", "attention", "pointwise")


def gate_all(shape_by_kind: dict[str, tuple[int, ...]] | None = None, *,
             seed: int = 0) -> dict[str, DisplacementVerdict]:
    """Gate every displaced lane on hidden inputs — the M5 sweep. Returns a
    per-lane verdict; a lane is safe to ship iff its relation is ``equivalent``
    (or ``not_displaced`` where Metal is unavailable — never ``divergent``)."""
    shapes = shape_by_kind or {
        "matmul_epilogue": (16, 64, 128),
        "norm_chain": (8, 64),
        "attention": (8, 32, 16),
        "pointwise": (8, 64),
    }
    return {k: displacement_verdict(k, shapes[k], seed=seed)
            for k in DISPLACED_LANES}
