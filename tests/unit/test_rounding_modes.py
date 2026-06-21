"""Unit tests for the canonical rounding-mode vocabulary (rounding.py) and its
use as a swept numeric-policy axis.

Ported from the moonmath CDNA3 attention writeup: RTNE/RTNA/RTZ are benchmarked
as a single execution-derived knob (best mode is shape-dependent), so the
rounding field is canonicalized and sweepable like the other autotuner axes.
"""

from __future__ import annotations

import pytest

from tessera.compiler.primitive_coverage import NumericPolicy
from tessera.compiler.rounding import (
    IEEE_ROUNDING_SWEEP,
    RTNA,
    RTNE,
    RTZ,
    STOCHASTIC,
    TesseraRoundingError,
    canonical_rounding_modes,
    is_canonical_rounding_mode,
    normalize_rounding_mode,
    rounding_sweep,
    rounding_to_mlir,
)


# ── canonical set + normalization ────────────────────────────────────────────


def test_canonical_set_membership() -> None:
    assert canonical_rounding_modes() == frozenset({RTNE, RTNA, RTZ, STOCHASTIC})
    for m in (RTNE, RTNA, RTZ, STOCHASTIC):
        assert is_canonical_rounding_mode(m)


@pytest.mark.parametrize(
    "alias,canon",
    [
        ("rtne", RTNE),
        ("RNE", RTNE),
        ("nearest_even", RTNE),
        ("ties_to_even", RTNE),
        ("rtna", RTNA),
        ("nearest_away", RTNA),
        ("away", RTNA),
        ("rtz", RTZ),
        ("trunc", RTZ),
        ("truncate", RTZ),
        ("toward_zero", RTZ),
        ("sr", STOCHASTIC),
        ("  RTZ  ", RTZ),  # whitespace + case tolerated
    ],
)
def test_alias_normalization(alias: str, canon: str) -> None:
    assert normalize_rounding_mode(alias) == canon


def test_normalize_is_idempotent() -> None:
    for m in canonical_rounding_modes():
        assert normalize_rounding_mode(m) == m


def test_normalize_rejects_unknown() -> None:
    with pytest.raises(TesseraRoundingError, match="unknown rounding mode"):
        normalize_rounding_mode("banker_special")


def test_normalize_rejects_non_string() -> None:
    with pytest.raises(TesseraRoundingError, match="must be a string"):
        normalize_rounding_mode(3)  # type: ignore[arg-type]


# ── MLIR token + sweep ───────────────────────────────────────────────────────


def test_mlir_tokens() -> None:
    assert rounding_to_mlir(RTNE) == "rtne"
    assert rounding_to_mlir("nearest_even") == "rtne"
    assert rounding_to_mlir("trunc") == "rtz"
    assert rounding_to_mlir(RTNA) == "rtna"


def test_default_sweep_is_the_article_triplet() -> None:
    assert rounding_sweep() == IEEE_ROUNDING_SWEEP == (RTNE, RTNA, RTZ)


def test_sweep_normalizes_and_dedups() -> None:
    # Mixed spellings of the same mode collapse; order is first-seen.
    assert rounding_sweep(["rtz", "truncate", "rne", "rtne"]) == (RTZ, RTNE)


# ── NumericPolicy integration ────────────────────────────────────────────────


def test_policy_canonicalizes_rounding_at_construction() -> None:
    p = NumericPolicy("bf16", accum="fp32", rounding="nearest_even")
    assert p.rounding == RTNE
    # Legacy "trunc" spelling is accepted and canonicalized.
    assert NumericPolicy("bf16", rounding="trunc").rounding == RTZ


def test_policy_rejects_bad_rounding() -> None:
    with pytest.raises(TesseraRoundingError):
        NumericPolicy("bf16", rounding="nonsense")


def test_policy_mlir_rounding_token() -> None:
    assert NumericPolicy("bf16", rounding="rtna").mlir_rounding_token == "rtna"


def test_policy_with_rounding_replaces_only_rounding() -> None:
    base = NumericPolicy("bf16", accum="fp32", rounding=RTNE)
    rtz = base.with_rounding("rtz")
    assert rtz.rounding == RTZ
    assert rtz.storage == base.storage and rtz.accum == base.accum
    assert base.rounding == RTNE  # original unchanged (frozen)


def test_policy_rounding_sweep_default() -> None:
    base = NumericPolicy("bf16", accum="fp32")
    variants = base.rounding_sweep()
    assert tuple(v.rounding for v in variants) == (RTNE, RTNA, RTZ)
    # Only rounding differs across the sweep.
    assert all(v.storage == "bf16" and v.accum == "fp32" for v in variants)


def test_policy_rounding_sweep_custom_modes() -> None:
    base = NumericPolicy("fp16")
    variants = base.rounding_sweep(["rtne", "rtz"])
    assert tuple(v.rounding for v in variants) == (RTNE, RTZ)
