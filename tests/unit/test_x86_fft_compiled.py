"""x86 spectral FFT lane (Spectral PR2) — fft / ifft / rfft / irfft over a
power-of-two axis, on the AVX-512 radix-2 C2C kernel + r2c/c2r pack-unpack.
Reachable via `compiler_path="x86_fft_compiled"`. Validated vs np.fft.

Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op_name, kwargs, x):
    from tessera.compiler.scheduled_fft import lower_scheduled_fft

    scheduled = lower_scheduled_fft(
        target="x86",
        op_name=op_name,
        input_shape=tuple(int(dim) for dim in x.shape),
        axis=int(kwargs.get("axis", -1)),
        n=kwargs.get("n"),
    ).to_metadata()
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_fft_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "scheduled_fft": scheduled,
    })


_TOL = dict(atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("n", [2, 8, 64, 1024])
@pytest.mark.parametrize("shape_pre", [(), (3,), (2, 5)])
def test_fft_ifft(n, shape_pre):
    rt = _x86_or_skip()
    rng = np.random.default_rng(1 + n + len(shape_pre))
    x = (rng.standard_normal(shape_pre + (n,))
         + 1j * rng.standard_normal(shape_pre + (n,))).astype(np.complex64)
    rf = rt.launch(_art(rt, "tessera.fft", {"axis": -1}, x), (x,))
    assert rf["ok"] is True, rf.get("reason")
    assert rf["compiler_path"] == "x86_fft_compiled"
    np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=-1).astype(np.complex64),
                               **_TOL)
    ri = rt.launch(_art(rt, "tessera.ifft", {"axis": -1}, x), (x,))
    assert ri["ok"] is True, ri.get("reason")
    np.testing.assert_allclose(np.asarray(ri["output"]).astype(np.complex64),
                               np.fft.ifft(x, axis=-1).astype(np.complex64),
                               **_TOL)


def test_fft_inner_axis():
    rt = _x86_or_skip()
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((4, 16, 3))
         + 1j * rng.standard_normal((4, 16, 3))).astype(np.complex64)
    res = rt.launch(_art(rt, "tessera.fft", {"axis": 1}, x), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=1).astype(np.complex64), **_TOL)


@pytest.mark.parametrize("n", [8, 64, 256])
def test_rfft_irfft(n):
    rt = _x86_or_skip()
    rng = np.random.default_rng(3 + n)
    x = (rng.standard_normal((2, n))).astype(np.float32)
    rf = rt.launch(_art(rt, "tessera.rfft", {"axis": -1}, x), (x,))
    assert rf["ok"] is True, rf.get("reason")
    ref = np.fft.rfft(x, axis=-1).astype(np.complex64)
    got = np.asarray(rf["output"]).astype(np.complex64)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, **_TOL)
    # round-trip irfft(rfft(x)) == x
    ir = rt.launch(_art(rt, "tessera.irfft", {"axis": -1, "n": n}, ref), (ref,))
    assert ir["ok"] is True, ir.get("reason")
    np.testing.assert_allclose(np.asarray(ir["output"]).astype(np.float32), x,
                               **_TOL)


def test_even_real_fft_does_not_reenter_full_complex_lane(monkeypatch):
    rt = _x86_or_skip()
    x = np.arange(128, dtype=np.float32).reshape(2, 64)
    monkeypatch.setattr(
        rt,
        "_x86_transform_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("packed real FFT re-entered full C2C")
        ),
    )
    result = rt.launch(_art(rt, "tessera.rfft", {"axis": -1}, x), (x,))
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(result["output"], np.fft.rfft(x), **_TOL)


@pytest.mark.parametrize("n", [3, 5, 6, 7])      # tiny -> naive DFT (gemm)
def test_fft_tiny_dft(n):
    rt = _x86_or_skip()
    rng = np.random.default_rng(20 + n)
    x = (rng.standard_normal((2, n))
         + 1j * rng.standard_normal((2, n))).astype(np.complex64)
    rf = rt.launch(_art(rt, "tessera.fft", {"axis": -1}, x), (x,))
    assert rf["ok"] is True, rf.get("reason")
    np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=-1).astype(np.complex64),
                               atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("n", [9, 12, 100, 127, 384])   # non-pow2 -> Bluestein
def test_fft_bluestein(n):
    rt = _x86_or_skip()
    rng = np.random.default_rng(40 + n)
    x = (rng.standard_normal((2, n))
         + 1j * rng.standard_normal((2, n))).astype(np.complex64)
    rf = rt.launch(_art(rt, "tessera.fft", {"axis": -1}, x), (x,))
    assert rf["ok"] is True, rf.get("reason")
    np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                               np.fft.fft(x, axis=-1).astype(np.complex64),
                               atol=1e-2, rtol=1e-2)
    # inverse round-trips
    ri = rt.launch(
        _art(rt, "tessera.ifft", {"axis": -1}, rf["output"]), (rf["output"],)
    )
    assert ri["ok"] is True, ri.get("reason")
    np.testing.assert_allclose(np.asarray(ri["output"]).astype(np.complex64), x,
                               atol=1e-2, rtol=1e-2)


def test_rfft_irfft_non_pow2():
    rt = _x86_or_skip()
    rng = np.random.default_rng(99)
    for n in (6, 100):
        x = rng.standard_normal((2, n)).astype(np.float32)
        rf = rt.launch(_art(rt, "tessera.rfft", {"axis": -1}, x), (x,))
        assert rf["ok"] is True, rf.get("reason")
        np.testing.assert_allclose(np.asarray(rf["output"]).astype(np.complex64),
                                   np.fft.rfft(x, axis=-1).astype(np.complex64),
                                   atol=1e-2, rtol=1e-2)
        spectrum = np.fft.rfft(x, axis=-1).astype(np.complex64)
        ir = rt.launch(_art(rt, "tessera.irfft", {"axis": -1, "n": n}, spectrum),
                       (spectrum,))
        assert ir["ok"] is True, ir.get("reason")
        np.testing.assert_allclose(np.asarray(ir["output"]).astype(np.float32),
                                   x, atol=1e-2, rtol=1e-2)


def test_fft_unknown_op_rejected():
    from tessera import runtime as rt
    x = np.zeros((8,), np.complex64)
    with pytest.raises(ValueError, match="x86_fft_compiled executor"):
        rt._execute_x86_compiled_fft(
            rt.RuntimeArtifact(
                metadata={
                    "scheduled_fft": {"op_name": "tessera.dct"},
                    "arg_names": ["x"],
                }
            ),
            (x,),
        )


def test_native_fft_helper_preserves_contiguous_input():
    rt = _x86_or_skip()
    rng = np.random.default_rng(91)
    x = (rng.standard_normal((2, 64)) + 1j * rng.standard_normal((2, 64))).astype(
        np.complex64
    )
    before = x.copy()
    _ = rt._x86_fft_c2c_rows(x, False, np)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("n", [12, 15, 68, 289, 768, 3072])
def test_mixed_radix_avx512_candidate(n):
    rt = _x86_or_skip()
    rng = np.random.default_rng(100 + n)
    x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
        np.complex64
    )
    actual = rt._x86_mixed_radix_rows(x, False, np)
    np.testing.assert_allclose(actual, np.fft.fft(x, axis=-1), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("n", [127, 257, 509])
def test_rader_candidate(n):
    rt = _x86_or_skip()
    rng = np.random.default_rng(200 + n)
    x = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(
        np.complex64
    )
    actual = rt._x86_rader_rows(x, False, np)
    np.testing.assert_allclose(actual, np.fft.fft(x, axis=-1), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("n", [1024, 65536])
def test_bailey_six_step_candidate(n):
    rt = _x86_or_skip()
    rng = np.random.default_rng(300 + n)
    x = (rng.standard_normal((1, n)) + 1j * rng.standard_normal((1, n))).astype(
        np.complex64
    )
    actual = rt._x86_six_step_rows(x, False, np)
    np.testing.assert_allclose(actual, np.fft.fft(x, axis=-1), atol=2e-3, rtol=2e-3)


def test_bluestein_plan_cache_reuses_immutable_tables():
    rt = _x86_or_skip()
    rt._x86_bluestein_plan.cache_clear()
    first = rt._x86_bluestein_plan(257, False)
    second = rt._x86_bluestein_plan(257, False)
    assert first[1] is second[1]
    assert first[2] is second[2]
    assert first[1].flags.writeable is False
    assert first[2].flags.writeable is False
