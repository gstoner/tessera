"""Host-free tests for the x86 FFT benchmark HARNESS (not for the x86 FFT).

Read the distinction before changing anything here, because this file used to
get it wrong. The harness under test runs
``rt._x86_fft_c2c_rows`` and compares the result against ``scipy.fft``; these
tests substitute that entry point, so **nothing x86 executes** and nothing here
is evidence about the kernel. That is fine -- row emission, JSON schema and the
numerical gate are all worth covering on any host -- but it means a numerical
assertion over the substituted path is a tautology.

The 2026-09-19 version asserted ``max_abs_error == 0.0`` while the substitute
WAS ``scipy.fft``, i.e. ``|scipy - scipy| == 0``. It would have passed had the
real kernel been arbitrarily wrong, because the real kernel never ran. The tell
was the exactness: the harness itself allows ``2.0e-4`` relative, so a genuine
cross-library comparison could never come back bit-identical.

What replaces it is the test that assertion was pretending to be: feed the
harness a deliberately WRONG stand-in and prove its numerical gate FIRES. A
gate that has never been seen to reject anything is not known to work.

Exact-device coverage of the kernel lives in ``test_x86_fft_compiled.py``,
which gates on ``libtessera_x86_elementwise.so`` and skips on a host without
it -- including every arm64 Mac, where x86 does not execute at any macOS
version.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

scipy_fft = pytest.importorskip(
    "scipy.fft",
    reason="the benchmark harness compares against scipy; it is a test "
    "dependency, not a runtime one (Decision #23)",
)

from benchmarks.spectral import benchmark_x86_fft as bench  # noqa: E402


def _argv(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_x86_fft.py", "--sizes", "64", "--warmup", "0", "--repeats", "1"],
    )
    monkeypatch.setattr(bench.rt, "_x86_elementwise_available", lambda: True)
    monkeypatch.setattr(bench, "_measure", lambda call, warmup, repeats: 1.0)


def _scipy_stand_in(values, inverse, np_mod):
    """What the harness's own reference computes, so the comparison is exact."""
    out = (
        scipy_fft.ifft(values, axis=-1, workers=1) * values.shape[-1]
        if inverse
        else scipy_fft.fft(values, axis=-1, workers=1)
    )
    return out.astype(np.complex64)


def test_benchmark_emits_forward_and_inverse_rows(monkeypatch, capsys) -> None:
    """Row emission and JSON schema. NOT a numerical claim about x86."""
    _argv(monkeypatch)
    monkeypatch.setattr(bench.rt, "_x86_fft_c2c_rows", _scipy_stand_in)

    assert bench.main() == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["transform"] for row in rows] == ["c2c_forward", "c2c_inverse"]
    for row in rows:
        # Schema only. `max_abs_error` is 0.0 here BY CONSTRUCTION -- the
        # stand-in is the reference -- so asserting a value would restate the
        # substitution, not measure anything.
        assert {"transform", "max_abs_error", "baseline"} <= row.keys()
        assert np.isfinite(row["max_abs_error"])


@pytest.mark.parametrize("scale", [1.5, -1.0])
def test_benchmark_numerical_gate_rejects_a_wrong_kernel(monkeypatch, scale) -> None:
    """The gate FIRES. This is what the old `== 0.0` assertion only looked like.

    A stand-in that is wrong by a fixed factor must be refused, with the
    transform and the measured error named -- an unlabelled failure would send
    the next reader to the wrong place.
    """
    _argv(monkeypatch)
    monkeypatch.setattr(
        bench.rt,
        "_x86_fft_c2c_rows",
        lambda values, inverse, np_mod: _scipy_stand_in(values, inverse, np_mod) * scale,
    )

    with pytest.raises(RuntimeError, match=r"numerical comparison failed"):
        bench.main()
