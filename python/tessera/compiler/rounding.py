"""tessera.compiler.rounding — canonical floating-point rounding-mode vocabulary.

The numeric policy (Decision #15a) carries a ``rounding`` field separate from the
storage dtype.  Until now that field's spelling drifted across modules
(``"nearest_even"`` vs ``"round_to_nearest_even"`` vs ``"trunc"`` vs
``"toward_zero"``).  This leaf module is the single source of truth: a canonical
mode set + an alias table that normalizes every spelling, the IEEE sweep the
moonmath CDNA3 attention writeup benchmarks (RTNE / RTNA / RTZ), and the short
MLIR token each mode lowers to.

Why a *swept* knob: the article shows the best rounding mode is shape-dependent
(RTNE/RTNA/RTZ trade 1.08–1.18× across sequence lengths), so it belongs with the
other autotuner/evaluator axes — not hard-wired.  Pure data, stdlib only, no
``tessera`` imports (so the autotuner, the coverage registry, and the runtime
can all import it without a cycle).
"""

from __future__ import annotations

from typing import Iterable

# ── canonical modes ──────────────────────────────────────────────────────────

#: Round to nearest, ties to even (IEEE-754 default).  Article "RTNE".
RTNE = "round_to_nearest_even"
#: Round to nearest, ties away from zero.  Article "RTNA".
RTNA = "round_to_nearest_away"
#: Round toward zero (truncate).  Article "RTZ".
RTZ = "round_toward_zero"
#: Stochastic rounding (not an IEEE mode; used by low-precision training paths).
STOCHASTIC = "stochastic"

_CANONICAL: frozenset[str] = frozenset({RTNE, RTNA, RTZ, STOCHASTIC})

#: The three IEEE rounding modes the article benchmarks, in its table order.
IEEE_ROUNDING_SWEEP: tuple[str, ...] = (RTNE, RTNA, RTZ)

# ── alias normalization ──────────────────────────────────────────────────────
# Every legacy / shorthand spelling maps to one canonical mode.  Canonical names
# map to themselves so ``normalize_rounding_mode`` is idempotent.
_ALIASES: dict[str, str] = {
    # RTNE family
    RTNE: RTNE,
    "rtne": RTNE,
    "rne": RTNE,
    "nearest_even": RTNE,
    "round_nearest_even": RTNE,
    "ties_to_even": RTNE,
    "even": RTNE,
    # RTNA family
    RTNA: RTNA,
    "rtna": RTNA,
    "rna": RTNA,
    "nearest_away": RTNA,
    "ties_away": RTNA,
    "away": RTNA,
    # RTZ family
    RTZ: RTZ,
    "rtz": RTZ,
    "trunc": RTZ,
    "truncate": RTZ,
    "toward_zero": RTZ,
    "tz": RTZ,
    "zero": RTZ,
    # stochastic family
    STOCHASTIC: STOCHASTIC,
    "sr": STOCHASTIC,
    "stochastic_rounding": STOCHASTIC,
}

#: Short token each canonical mode lowers to in IR / hardware setup.
_MLIR_TOKEN: dict[str, str] = {
    RTNE: "rtne",
    RTNA: "rtna",
    RTZ: "rtz",
    STOCHASTIC: "sr",
}


class TesseraRoundingError(ValueError):
    """Raised for an unknown / unsupported rounding-mode spelling."""


def canonical_rounding_modes() -> frozenset[str]:
    """The canonical rounding-mode name set."""
    return _CANONICAL


def rounding_aliases() -> dict[str, str]:
    """A copy of the alias→canonical table (every accepted spelling)."""
    return dict(_ALIASES)


def is_canonical_rounding_mode(s: str) -> bool:
    """True iff ``s`` is already a canonical rounding-mode name."""
    return s in _CANONICAL


def normalize_rounding_mode(s: str) -> str:
    """Canonicalize a rounding-mode spelling (idempotent on canonical names).

    Accepts the article's RTNE/RTNA/RTZ shorthands and every legacy spelling in
    the codebase; raises :class:`TesseraRoundingError` with the known set for an
    unrecognized name rather than silently passing it through.
    """
    if not isinstance(s, str):
        raise TesseraRoundingError(
            f"rounding mode must be a string, got {type(s).__name__}")
    key = s.strip().lower()
    try:
        return _ALIASES[key]
    except KeyError as e:
        raise TesseraRoundingError(
            f"unknown rounding mode {s!r}; known: {sorted(_ALIASES)}") from e


def rounding_to_mlir(s: str) -> str:
    """Short IR/hardware token for a rounding mode (normalizes first)."""
    return _MLIR_TOKEN[normalize_rounding_mode(s)]


def rounding_sweep(modes: Iterable[str] | None = None) -> tuple[str, ...]:
    """Canonical, de-duplicated rounding sweep.

    Defaults to :data:`IEEE_ROUNDING_SWEEP` (the article's three modes).  Pass an
    explicit ``modes`` iterable to sweep a custom subset; each is normalized and
    duplicates are dropped while preserving first-seen order.
    """
    src = IEEE_ROUNDING_SWEEP if modes is None else modes
    out: list[str] = []
    for m in src:
        c = normalize_rounding_mode(m)
        if c not in out:
            out.append(c)
    return tuple(out)


__all__ = [
    "RTNE",
    "RTNA",
    "RTZ",
    "STOCHASTIC",
    "IEEE_ROUNDING_SWEEP",
    "TesseraRoundingError",
    "canonical_rounding_modes",
    "rounding_aliases",
    "is_canonical_rounding_mode",
    "normalize_rounding_mode",
    "rounding_to_mlir",
    "rounding_sweep",
]
