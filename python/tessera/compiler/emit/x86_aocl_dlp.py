"""Workstream C1b — x86 Tier-3 candidate: AMD AOCL-DLP (opt-in, separated from C1).

[amd/aocl-dlp](https://github.com/amd/aocl-dlp) is AMD's BLIS-family deep-learning
primitive library: low-precision GEMM / batched GEMM (incl. INT4 / FP16) with
pre/post-ops that map onto ``fused_epilogue`` (per-column bias + a pointwise
activation), symmetric quantization, and OpenMP threading — AVX-512 based, so it
fits the Zen 5 box (no AMX) and fills the x86 backend's OpenMP-threading + INT4 /
FP16 gaps the generic scalar-C lane lacks.

Per the plan this is a **hand-tuned Tier-3 candidate the D1 arbiter measures, NOT
part of the core x86 plugin** — opt-in behind an env/library probe (the AVX-512
BLAS-family concession is Decision #23-clean: it sits behind the hardware-free
Target IR, like Accelerate on Apple). The arbiter selects it only where it
measures faster than the generic kernels on Zen *and* passes the same universal
F4 oracle.

**State (this box has no aocl-dlp installed):** the candidate is registered and
arbiter-visible, but :meth:`available` returns ``False`` until the library is
present, so it silently drops out of enumeration and never mis-selects here — the
whole point of the availability probe. The concrete GEMM call is deliberately
routed through :func:`_aocl_dlp_gemm`, which **declines to the reference unless a
verified library + symbol are wired** (via ``TESSERA_AOCL_DLP_LIB`` /
``TESSERA_AOCL_DLP_SGEMM``): the exact post-op ABI must be bound against real
aocl-dlp headers on a licensed install before this becomes a linked lane
(**license review is a gate** — plan C1b), not guessed from absence of evidence.
Once wired, the F4 oracle proves correctness before the arbiter ever trusts it.
"""
from __future__ import annotations

import ctypes
import os
from typing import Any

from tessera.compiler.emit.candidate import (
    OP_FUSED_REGION,
    Candidate,
    Tier,
    register_candidate,
)
from tessera.compiler.fusion_core import FusedRegion

_TARGET = "x86"
_REAL_TAG = "x86_aocl_dlp"
#: The concrete aocl-dlp GEMM post-op ctypes ABI is not yet bound (pending real
#: headers + license review on a licensed install). Until an implementer wires
#: :func:`_aocl_dlp_gemm` and flips this ``True``, the candidate stays UNAVAILABLE
#: even when ``$TESSERA_AOCL_DLP_LIB``/``$TESSERA_AOCL_DLP_SGEMM`` resolve a real
#: symbol — an opt-in install must never silently demote a supported fused GEMM to
#: the numpy reference by winning arbitration and then declining (PR #289 review).
_ABI_WIRED = False
#: Activations AOCL-DLP fuses as a GEMM post-op, applied after the bias add — the
#: same bias-then-activation envelope the region must match (mirrors the ROCm WMMA
#: candidate's fusable set so both crown-jewel GEMM lanes share one applicability).
_ACTS = ("relu", "gelu", "silu")

_lib_cache: dict[str, Any] = {}


def _aocl_epilogue(region: Any) -> tuple[bool, str] | None:
    """Map ``region`` to AOCL-DLP's ``(has_bias, activation)`` post-op, or ``None``
    when it is not representable (a reduction / residual / prologue, or an epilogue
    that is not a bias-before-{relu,gelu,silu} subsequence)."""
    if not isinstance(region, FusedRegion):
        return None
    if region.reduction is not None or region.residual or region.prologue:
        return None
    epi = list(region.epilogue)
    has_bias = False
    if epi and epi[0] == "bias":
        has_bias, epi = True, epi[1:]
    if not epi:
        return has_bias, "none"
    if len(epi) == 1 and epi[0] in _ACTS:
        return has_bias, epi[0]
    return None


def _aocl_dlp_lib() -> Any | None:
    """Load the aocl-dlp shared object named by ``$TESSERA_AOCL_DLP_LIB`` (cached),
    or ``None`` when the var is unset / the lib is unloadable. No path is guessed:
    the operator points this at their licensed install — absence ⇒ candidate off."""
    path = os.environ.get("TESSERA_AOCL_DLP_LIB")
    if not path:
        return None
    lib = _lib_cache.get(path)
    if lib is None:
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            return None
        _lib_cache[path] = lib
    return lib


def _aocl_dlp_gemm(a: Any, b: Any, bias: Any, activation: str) -> Any | None:
    """Run one f32 GEMM + fused bias/activation post-op on aocl-dlp, or ``None`` if
    the lane is not wired here. **The concrete post-op ABI must be bound against
    real aocl-dlp headers on a licensed install** — until the symbol named by
    ``$TESSERA_AOCL_DLP_SGEMM`` resolves, this returns ``None`` so the candidate
    declines honestly rather than call an unverified signature (Decision #21/#27)."""
    if not _ABI_WIRED:
        # ABI not bound yet — decline unconditionally (the candidate is also
        # unavailable, so this is only reachable via a forced, unverified run).
        return None
    lib = _aocl_dlp_lib()
    if lib is None:
        return None
    sym = os.environ.get("TESSERA_AOCL_DLP_SGEMM")
    fn = getattr(lib, sym, None) if sym else None
    if fn is None:
        # Library present but the verified entry point is not configured — decline
        # rather than guess the ABI. This is the seam the Zen-box bring-up fills.
        return None
    raise NotImplementedError(  # pragma: no cover - reached only once wired
        "aocl-dlp GEMM post-op ABI binding is pending header verification on a "
        "licensed install; wire the ctypes signature here, then F4-gate it")


class X86AoclDlpCandidate(Candidate):
    """Tier-3 hand-tuned x86 GEMM candidate backed by AMD AOCL-DLP (opt-in). Serves
    the same bias/relu/gelu/silu matmul-bound middle ground as the generic C lane
    but on AOCL's AVX-512 BLIS kernels; drops out of arbitration wherever the
    library is absent or the region is not representable."""

    name = "x86_aocl_dlp"
    tier = Tier.HAND_TUNED
    target = _TARGET
    op = OP_FUSED_REGION

    def available(self) -> bool:
        # Present iff the post-op ABI is actually bound (`_ABI_WIRED`) AND the
        # library + verified GEMM entry point resolve. The `_ABI_WIRED` gate is
        # what stops a real symbol from making this "available" while the run path
        # still declines — which would let it win by tier and demote a supported
        # fused GEMM to the reference (PR #289 review).
        if not _ABI_WIRED or _aocl_dlp_lib() is None:
            return False
        sym = os.environ.get("TESSERA_AOCL_DLP_SGEMM")
        return bool(sym and getattr(_aocl_dlp_lib(), sym, None))

    def applies_to(self, region: Any) -> bool:
        return _aocl_epilogue(region) is not None

    def run(self, region: Any, A: Any, B: Any, bias: Any = None,
            *a: Any, **k: Any) -> tuple[Any, str]:
        import numpy as np
        epi = _aocl_epilogue(region)
        if epi is None:
            return region.reference(A, B, bias), "reference"
        has_bias, activation = epi
        if has_bias and bias is None:
            return region.reference(A, B, bias), "reference"
        try:
            out = _aocl_dlp_gemm(np.ascontiguousarray(A, np.float32),
                                 np.ascontiguousarray(B, np.float32),
                                 bias, activation)
            if out is not None:
                return np.asarray(out, np.float32), _REAL_TAG
        except Exception:
            pass
        return region.reference(A, B, bias), "reference"


# Opt-in registration (import side effect; imported by emit.x86_llvm). Registered
# even when absent so it is arbiter-visible; available() gates actual selection.
register_candidate(X86AoclDlpCandidate())
