"""Spectral plan contract (PR1) — the FFT planner's locked decisions: radix
sequence, strategy selection, normalization scale, real/complex mode, workspace,
and the IR-plan-op attribute mapping. Pure Python (no kernel / device)."""

from __future__ import annotations

import math

import pytest

from tessera.compiler import spectral_plan as sp


def test_pow2_uses_radix2():
    for n in (2, 4, 8, 16, 1024, 4096):
        p = sp.plan_fft(n)
        assert p.strategy == "radix2"
        assert p.radix_seq == (2,) * (n.bit_length() - 1)
        assert len(p.radix_seq) == int(math.log2(n))
        assert p.bluestein_m == 0 and p.workspace_elems == 0


def test_tiny_non_pow2_uses_dft():
    for n in (3, 5, 6, 7):
        p = sp.plan_fft(n)
        assert p.strategy == "dft"
        assert p.radix_seq == () and p.bluestein_m == 0


def test_large_non_pow2_uses_bluestein():
    p = sp.plan_fft(100)
    assert p.strategy == "bluestein"
    assert p.bluestein_m == sp.next_power_of_two(2 * 100 - 1) == 256
    assert p.workspace_elems == p.bluestein_m
    # 12289 is prime → next_pow2(2n-1) padded conv
    assert sp.plan_fft(12289).bluestein_m == sp.next_power_of_two(2 * 12289 - 1)


@pytest.mark.parametrize("n", [8, 100])
def test_normalization_scales(n):
    # backward (numpy default): fwd unscaled, inv 1/N
    assert sp.plan_fft(n, inverse=False, norm="backward").scale == 1.0
    assert sp.plan_fft(n, inverse=True, norm="backward").scale == pytest.approx(
        1.0 / n)
    # ortho: 1/sqrt(N) both directions
    s = 1.0 / math.sqrt(n)
    assert sp.plan_fft(n, norm="ortho").scale == pytest.approx(s)
    assert sp.plan_fft(n, inverse=True, norm="ortho").scale == pytest.approx(s)
    # forward: fwd 1/N, inv unscaled
    assert sp.plan_fft(n, norm="forward").scale == pytest.approx(1.0 / n)
    assert sp.plan_fft(n, inverse=True, norm="forward").scale == 1.0


def test_modes_and_attrs():
    p = sp.plan_fft(16, mode="r2c")
    attrs = p.to_plan_attrs()
    assert attrs["is_real_input"] is True
    assert attrs["radix_seq"] == [2, 2, 2, 2]
    assert attrs["norm_policy"] == "backward"
    assert attrs["elem_precision"] == "complex64"
    assert attrs["acc_precision"] == "float32"
    assert attrs["inplace"] is True            # radix2, no workspace
    # bluestein plan is not in-place (has workspace)
    assert sp.plan_fft(100).to_plan_attrs()["inplace"] is False


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="positive"):
        sp.plan_fft(0)
    with pytest.raises(ValueError, match="mode must be"):
        sp.plan_fft(8, mode="bogus")
    with pytest.raises(ValueError, match="norm must be"):
        sp.plan_fft(8, norm="bogus")


def test_helpers():
    assert sp.is_power_of_two(1024) and not sp.is_power_of_two(1000)
    assert sp.next_power_of_two(199) == 256
    assert sp.next_power_of_two(256) == 256
    assert sp.radix2_sequence(8) == (2, 2, 2)
    assert sp.plan_fft(8).deterministic is True
