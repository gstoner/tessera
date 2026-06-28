"""Spectral (FFT) plan contract + planner — Spectral PR1.

The *hardware-free* plan metadata every Tessera FFT execution is built from. The
x86 (AVX-512 radix-2 / Bluestein) and ROCm FFT kernels (PR2-PR4) consume a
``SpectralPlan`` rather than re-deriving the factorization / normalization /
strategy each call; the planner here is the single place those decisions live.

The contract mirrors the existing ``tessera_spectral.plan`` op
(``src/solvers/spectral/lib/Dialect/Spectral/SpectralOps.td``: ``axes``,
``radix_seq``, ``elem_precision``, ``acc_precision``, ``scaling``, ``inplace``,
``is_real_input``, ``norm_policy``) — :meth:`SpectralPlan.to_plan_attrs` emits
exactly those attributes so the runtime plan and the IR plan op stay in lock-step.

Locked decisions (the "plan metadata" PR1 fixes):

* **radix sequence** — power-of-two lengths factor into a radix-2 stage list;
  non-power-of-two uses Bluestein (a power-of-two convolution) or, for tiny
  lengths, a naive DFT. The *strategy is chosen by the plan*, never hidden in a
  kernel.
* **normalization** — numpy's conventions: ``backward`` (default — forward
  unscaled, inverse 1/N), ``ortho`` (1/√N both), ``forward`` (forward 1/N).
* **twiddle layout** — interleaved (re, im) f32 pairs, one table per transform
  length; deterministic (precomputed, not re-derived per stage).
* **workspace** — complex scratch element count (0 for in-place radix-2; the
  Bluestein padded length otherwise).
* **real/complex mode** — ``c2c`` / ``r2c`` (rfft) / ``c2r`` (irfft).
* **dtype policy** — complex64 storage, f32 accumulate (the only shipped tier).
* **deterministic** — twiddle tables + a fixed stage order ⇒ bit-reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Non-power-of-two lengths at or below this use the naive O(n²) DFT directly
#: (cheaper than a Bluestein power-of-two convolution for tiny n).
TINY_DFT_MAX = 8

_VALID_MODES = ("c2c", "r2c", "c2r")
_VALID_NORMS = ("backward", "ortho", "forward")
_VALID_STRATEGIES = ("radix2", "bluestein", "dft")


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x - 1).bit_length())


def radix2_sequence(n: int) -> tuple:
    """Radix-2 stage list for a power-of-two ``n`` (log2(n) twos)."""
    if not is_power_of_two(n):
        raise ValueError(f"radix2_sequence requires a power of two; got {n}")
    return (2,) * (n.bit_length() - 1)


def _norm_scale(n: int, inverse: bool, norm: str) -> float:
    """The output scale applied for ``norm`` (matching numpy.fft)."""
    if norm == "ortho":
        return 1.0 / math.sqrt(n)
    if norm == "forward":
        return (1.0 if inverse else 1.0 / n)
    # backward (numpy default): forward unscaled, inverse 1/N
    return (1.0 / n if inverse else 1.0)


@dataclass(frozen=True)
class SpectralPlan:
    """The locked, hardware-free execution plan for one 1-D FFT along an axis."""

    n: int                       # transform length along the axis
    axis: int                    # transform axis (may be negative)
    mode: str                    # c2c | r2c | c2r
    inverse: bool                # forward vs inverse transform
    strategy: str                # radix2 | bluestein | dft
    radix_seq: tuple             # radix-2 stage list (() for bluestein/dft)
    norm: str                    # backward | ortho | forward
    scale: float                 # applied output scale
    dtype: str = "complex64"     # element storage dtype
    accum: str = "float32"       # accumulate precision
    deterministic: bool = True
    bluestein_m: int = 0         # padded conv length (0 unless strategy=bluestein)
    twiddle_layout: str = "interleaved"
    workspace_elems: int = 0     # complex scratch elements
    metadata: dict = field(default_factory=dict)

    def to_plan_attrs(self) -> dict:
        """The ``tessera_spectral.plan`` op attributes for this plan — keeps the
        runtime plan and the IR plan op in lock-step."""
        return {
            "axes": [int(self.axis)],
            "radix_seq": list(self.radix_seq) or None,
            "elem_precision": self.dtype,
            "acc_precision": self.accum,
            "scaling": f"{self.scale:.17g}",
            "inplace": self.strategy == "radix2" and self.workspace_elems == 0,
            "is_real_input": self.mode in ("r2c",),
            "norm_policy": self.norm,
        }


def plan_fft(n: int, *, axis: int = -1, mode: str = "c2c",
             inverse: bool = False, norm: str = "backward",
             dtype: str = "complex64", deterministic: bool = True
             ) -> SpectralPlan:
    """Build the :class:`SpectralPlan` for a length-``n`` 1-D FFT.

    Strategy selection (plan-owned, not hidden in a kernel):
    power-of-two → ``radix2``; tiny non-power-of-two (``n <= TINY_DFT_MAX``) →
    ``dft``; otherwise → ``bluestein`` over a power-of-two ``bluestein_m =
    next_pow2(2n-1)``.
    """
    if n <= 0:
        raise ValueError(f"FFT length must be positive; got {n}")
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}; got {mode!r}")
    if norm not in _VALID_NORMS:
        raise ValueError(f"norm must be one of {_VALID_NORMS}; got {norm!r}")

    if is_power_of_two(n):
        strategy, radix, bm, ws = "radix2", radix2_sequence(n), 0, 0
    elif n <= TINY_DFT_MAX:
        strategy, radix, bm, ws = "dft", (), 0, 0
    else:
        strategy, radix = "bluestein", ()
        bm = next_power_of_two(2 * n - 1)
        ws = bm  # complex scratch for the padded chirp convolution

    return SpectralPlan(
        n=int(n), axis=int(axis), mode=mode, inverse=bool(inverse),
        strategy=strategy, radix_seq=radix, norm=norm,
        scale=_norm_scale(n, inverse, norm), dtype=dtype, accum="float32",
        deterministic=bool(deterministic), bluestein_m=bm,
        twiddle_layout="interleaved", workspace_elems=ws)
