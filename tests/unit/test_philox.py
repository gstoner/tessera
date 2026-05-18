"""M6 Step 4 — Philox-4x32-10 reference + MSL template.

Coverage:

  - **Reference vectors** from the Random123 paper / Philox
    canonical test set.  These pin the algorithm against any
    drift.
  - **Determinism** — same ``(counter, key)`` always produces
    the same output.
  - **Counter independence** — adjacent counters produce
    uncorrelated outputs (sanity, not crypto-grade).
  - **Box-Muller pair** is finite and standard-normal-shaped
    (mean ≈ 0, std ≈ 1 over a sample).
  - **MSL template** contains the same constants + round
    structure as the Python reference.  This is the contract
    the future C++ runtime side will satisfy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler import philox


# ---------------------------------------------------------------------------
# Constants — pinned
# ---------------------------------------------------------------------------

def test_constants_are_canonical() -> None:
    """The Philox-4x32 constants from Salmon et al. (2011).
    Any drift breaks cross-platform determinism.

    Ordering: PHILOX_M0 multiplies ctr[0], PHILOX_M1 multiplies
    ctr[2] — see the Random123 source.  Swapping them silently
    breaks every reference vector."""
    assert int(philox.PHILOX_M0) == 0xD2511F53
    assert int(philox.PHILOX_M1) == 0xCD9E8D57
    assert int(philox.PHILOX_W0) == 0x9E3779B9
    assert int(philox.PHILOX_W1) == 0xBB67AE85
    assert philox.PHILOX_ROUNDS == 10
    assert philox.PHILOX_OUTPUT_WORDS == 4


# ---------------------------------------------------------------------------
# Reference vectors — the canonical Philox-4x32-10 test set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("counter, key, expected", [
    # Zero counter + zero key.
    (
        (0, 0, 0, 0),
        (0, 0),
        (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
    ),
    # All-ones counter + all-ones key.
    (
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
    ),
    # Mathematical constants — π digits as counter, e digits as key.
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
])
def test_known_reference_vectors(counter, key, expected) -> None:
    """Locks the algorithm against the canonical Philox test set.
    These values are from the Random123 reference implementation."""
    out = philox.philox_4x32_10(
        np.array(counter, dtype=np.uint32),
        np.array(key, dtype=np.uint32),
    )
    assert tuple(int(v) for v in out) == expected


# ---------------------------------------------------------------------------
# Determinism — same input ⇒ same output, always
# ---------------------------------------------------------------------------

def test_same_counter_and_key_yield_same_output_twice() -> None:
    ctr = np.array([1, 2, 3, 4], dtype=np.uint32)
    key = np.array([5, 6], dtype=np.uint32)
    a = philox.philox_4x32_10(ctr, key)
    b = philox.philox_4x32_10(ctr, key)
    np.testing.assert_array_equal(a, b)


def test_philox_does_not_mutate_inputs() -> None:
    """Functional: the caller's counter / key arrays survive
    intact across the call."""
    ctr = np.array([1, 2, 3, 4], dtype=np.uint32)
    key = np.array([5, 6], dtype=np.uint32)
    ctr_before = ctr.copy()
    key_before = key.copy()
    philox.philox_4x32_10(ctr, key)
    np.testing.assert_array_equal(ctr, ctr_before)
    np.testing.assert_array_equal(key, key_before)


# ---------------------------------------------------------------------------
# Counter independence — adjacent counters give different outputs
# ---------------------------------------------------------------------------

def test_adjacent_counters_produce_uncorrelated_outputs() -> None:
    """Crypto-grade collision-resistance isn't the goal here; just
    that incrementing the counter actually changes the output.
    This catches a regression that accidentally drops a round."""
    key = np.array([42, 99], dtype=np.uint32)
    out0 = philox.philox_4x32_10(np.array([0, 0, 0, 0], np.uint32), key)
    out1 = philox.philox_4x32_10(np.array([1, 0, 0, 0], np.uint32), key)
    # Every word must differ.
    assert not (out0 == out1).any(), (
        f"counter increment did not change every output word: "
        f"out0={out0}  out1={out1}"
    )


def test_different_keys_produce_uncorrelated_outputs() -> None:
    ctr = np.array([0, 0, 0, 0], dtype=np.uint32)
    out0 = philox.philox_4x32_10(ctr, np.array([0, 0], np.uint32))
    out1 = philox.philox_4x32_10(ctr, np.array([1, 0], np.uint32))
    assert not (out0 == out1).any()


# ---------------------------------------------------------------------------
# Uniform + normal distribution shape (statistical, not exact)
# ---------------------------------------------------------------------------

def test_philox_uniform_stays_in_open_unit_interval() -> None:
    """Output of philox_uniform is in (0, 1).  Specifically we
    add 0.5 + scale by 2⁻³² so neither 0 nor 1 are reachable."""
    samples = []
    for i in range(64):
        ctr = np.array([i, 0, 0, 0], dtype=np.uint32)
        samples.append(philox.philox_uniform(ctr, np.array([0, 0], np.uint32)))
    arr = np.concatenate(samples)
    assert (arr > 0.0).all() and (arr < 1.0).all()


def test_philox_normal_pair_is_finite_and_well_distributed() -> None:
    """Box-Muller from Philox uniforms produces a (z₁, z₂)
    pair that's standard-normal-shaped across a sample."""
    samples = []
    for i in range(1024):
        ctr = np.array([i, 0, 0, 0], dtype=np.uint32)
        z = philox.philox_normal_pair(ctr, np.array([0, 0], np.uint32))
        samples.extend(z)
    arr = np.asarray(samples, dtype=np.float64)
    assert np.isfinite(arr).all()
    # Mean ≈ 0, std ≈ 1.  Loose tolerance — N=2048 samples.
    assert abs(float(arr.mean())) < 0.1
    assert abs(float(arr.std()) - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_philox_rejects_wrong_counter_shape() -> None:
    with pytest.raises(ValueError, match="counter must be shape"):
        philox.philox_4x32_10(
            np.array([0, 0, 0], dtype=np.uint32),
            np.array([0, 0], dtype=np.uint32),
        )


def test_philox_rejects_wrong_key_shape() -> None:
    with pytest.raises(ValueError, match="key must be shape"):
        philox.philox_4x32_10(
            np.array([0, 0, 0, 0], dtype=np.uint32),
            np.array([0], dtype=np.uint32),
        )


# ---------------------------------------------------------------------------
# MSL template — cross-platform invariants
# ---------------------------------------------------------------------------

def test_msl_template_contains_canonical_constants() -> None:
    """The MSL source must declare the same constants as the
    Python reference.  Drift here would silently break
    cross-platform determinism."""
    src = philox.philox_msl_source()
    assert "PHILOX_M0 = 0xD2511F53u" in src
    assert "PHILOX_M1 = 0xCD9E8D57u" in src
    assert "PHILOX_W0 = 0x9E3779B9u" in src
    assert "PHILOX_W1 = 0xBB67AE85u" in src


def test_msl_template_has_10_round_loop() -> None:
    """The 10-round structure is part of the algorithm name —
    locking it prevents accidental 8/12 drift."""
    src = philox.philox_msl_source()
    assert "for (int r = 0; r < 10; ++r)" in src


def test_msl_template_declares_required_helpers() -> None:
    """The C++ runtime side reads the template and expects
    these four functions.  Locking the names so the integration
    doesn't break silently."""
    src = philox.philox_msl_source()
    for helper in (
        "philox_mulhilo", "philox_round", "philox_bump_key",
        "philox_4x32_10", "philox_to_uniform",
    ):
        assert helper in src, f"MSL template missing {helper!r}"


def test_msl_template_uses_unsigned_32bit_arithmetic() -> None:
    """All Philox arithmetic must be uint32 / uint64; if anyone
    accidentally writes ``int`` instead of ``uint`` the algorithm
    silently corrupts on negative-overflow paths."""
    src = philox.philox_msl_source()
    # ulong (uint64) for the mulhilo product.
    assert "ulong p = (ulong)a * (ulong)b" in src
    # The constants are declared ``uint`` (the 'u' suffix is the lock).
    assert "0xD2511F53u" in src
    assert "0xCD9E8D57u" in src


def test_msl_template_uniform_conversion_uses_2_minus_32() -> None:
    """``(x + 0.5) * 2⁻³²`` matches the Python ``philox_uniform``.
    A different scale would put samples in a wrong range."""
    src = philox.philox_msl_source()
    assert "0x1.0p-32f" in src


# ---------------------------------------------------------------------------
# Cross-platform invariant — Python output structure is fixed
# ---------------------------------------------------------------------------

def test_output_is_4_uint32() -> None:
    out = philox.philox_4x32_10(
        np.zeros(4, dtype=np.uint32),
        np.zeros(2, dtype=np.uint32),
    )
    assert out.shape == (4,)
    assert out.dtype == np.uint32
