"""Exact-host proof for native compiler forward products on AVX-512/gfx1151."""

from __future__ import annotations

import numpy as np
import pytest

import tessera


@tessera.jit(target="x86", autodiff="jvp", wrt=("x",))
def _x86_sum(x):
    return tessera.ops.sum(x, axis=-1)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x",))
def _rocm_sum(x):
    return tessera.ops.sum(x, axis=-1)


@tessera.jit(target="x86", autodiff="jvp", wrt=("x",))
def _x86_rmsnorm(x):
    return tessera.ops.rmsnorm(x, eps=1e-5)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x",))
def _rocm_rmsnorm(x):
    return tessera.ops.rmsnorm(x, eps=1e-5)


@tessera.jit(target="x86", autodiff="jvp", wrt=("x",))
def _x86_rfft(x):
    return tessera.ops.rfft(x, axis=-1)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x",))
def _rocm_rfft(x):
    return tessera.ops.rfft(x, axis=-1)


@tessera.jit(target="x86", autodiff="jvp", wrt=("x",))
def _x86_dct4(x):
    return tessera.ops.dct(x, type=4, axis=-1, norm="ortho")


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x",))
def _rocm_dct4(x):
    return tessera.ops.dct(x, type=4, axis=-1, norm="ortho")


@tessera.jit(target="x86", autodiff="jvp", wrt=("x", "gamma", "beta"))
def _x86_affine_layernorm(x, gamma, beta):
    return tessera.ops.layer_norm(x, gamma=gamma, beta=beta, eps=1e-5)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x", "gamma", "beta"))
def _rocm_affine_layernorm(x, gamma, beta):
    return tessera.ops.layer_norm(x, gamma=gamma, beta=beta, eps=1e-5)


@tessera.jit(target="x86", autodiff="jvp", wrt=("spectrum", "filter"))
def _x86_spectral_filter_product(spectrum, filter):
    return tessera.ops.spectral_filter(spectrum, filter)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("spectrum", "filter"))
def _rocm_spectral_filter_product(spectrum, filter):
    return tessera.ops.spectral_filter(spectrum, filter)


@tessera.jit(target="x86", autodiff="jvp", wrt=("spectrum", "window"))
def _x86_istft_product(spectrum, window):
    return tessera.ops.istft(spectrum, window, hop=4)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("spectrum", "window"))
def _rocm_istft_product(spectrum, window):
    return tessera.ops.istft(spectrum, window, hop=4)


def _affine_layernorm_oracle(x, gamma, beta, dx, dgamma, dbeta):
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    inv = 1.0 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)
    normalized = centered * inv
    dcentered = dx - dx.mean(axis=-1, keepdims=True)
    projected = (dcentered - normalized * np.mean(dcentered * normalized, axis=-1, keepdims=True)) * inv
    return normalized * gamma + beta, projected * gamma + normalized * dgamma + dbeta


@pytest.mark.compiler_avx512
def test_native_avx512_affine_normalization_product():
    from tessera import runtime
    if not runtime._x86_elementwise_available():
        pytest.skip("AVX-512 production image unavailable")
    rng = np.random.default_rng(181)
    x = rng.normal(size=(5, 64)).astype(np.float32)
    gamma = rng.normal(size=(64,)).astype(np.float32)
    beta = rng.normal(size=(64,)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    dgamma = rng.normal(size=gamma.shape).astype(np.float32)
    dbeta = rng.normal(size=beta.shape).astype(np.float32)
    actual = _x86_affine_layernorm.native_jvp(
        x, gamma, beta, tangents=(dx, dgamma, dbeta)
    )
    expected = _affine_layernorm_oracle(x, gamma, beta, dx, dgamma, dbeta)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=5e-5, atol=5e-5)


@pytest.mark.hardware_rocm
def test_native_gfx1151_affine_normalization_product():
    from tessera import runtime
    if runtime._tessera_opt_path() is None or runtime._rocm_chip() != "gfx1151":
        pytest.skip("exact gfx1151 compiler/device required")
    rng = np.random.default_rng(191)
    x = rng.normal(size=(3, 64)).astype(np.float32)
    gamma = rng.normal(size=(64,)).astype(np.float32)
    beta = rng.normal(size=(64,)).astype(np.float32)
    tangents = (rng.normal(size=x.shape).astype(np.float32),
                rng.normal(size=gamma.shape).astype(np.float32),
                rng.normal(size=beta.shape).astype(np.float32))
    actual = _rocm_affine_layernorm.native_jvp(
        x, gamma, beta, tangents=tangents
    )
    expected = _affine_layernorm_oracle(x, gamma, beta, *tangents)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize(
    "compiled,marker", [
        pytest.param(_x86_spectral_filter_product, "x86", marks=pytest.mark.compiler_avx512),
        pytest.param(_rocm_spectral_filter_product, "rocm", marks=pytest.mark.hardware_rocm),
    ],
)
def test_native_compound_spectral_product_rule(compiled, marker):
    from tessera import runtime
    if marker == "x86" and not runtime._x86_elementwise_available():
        pytest.skip("AVX-512 spectral package unavailable")
    if marker == "rocm" and (
        runtime._tessera_opt_path() is None or runtime._rocm_chip() != "gfx1151"
    ):
        pytest.skip("exact gfx1151 compiler/device required")
    rng = np.random.default_rng(229)
    spectrum = (rng.normal(size=(4, 9)) + 1j * rng.normal(size=(4, 9))).astype(np.complex64)
    filter = (rng.normal(size=(4, 9)) + 1j * rng.normal(size=(4, 9))).astype(np.complex64)
    ds = (rng.normal(size=(4, 9)) + 1j * rng.normal(size=(4, 9))).astype(np.complex64)
    df = (rng.normal(size=(4, 9)) + 1j * rng.normal(size=(4, 9))).astype(np.complex64)
    primal, tangent = compiled.native_jvp(
        spectrum, filter, tangents=(ds, df)
    )
    np.testing.assert_allclose(primal, spectrum * filter, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(
        tangent, ds * filter + spectrum * df, rtol=8e-4, atol=8e-4
    )


@pytest.mark.compiler_avx512
def test_native_x86_istft_window_product_matches_centered_difference():
    from tessera import runtime
    if not runtime._x86_elementwise_available():
        pytest.skip("AVX-512 spectral package unavailable")
    lib = runtime._load_x86_elementwise()
    if lib is None or not hasattr(lib, "tessera_x86_istft_jvp_f32"):
        pytest.skip("AVX-512 ISTFT window-JVP image is stale")
    rng = np.random.default_rng(241)
    spectrum = (
        rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    ).astype(np.complex64)
    spectrum[:, (0, -1)] = spectrum[:, (0, -1)].real
    window = (np.hanning(8) + 0.2).astype(np.float32)
    dspectrum = (
        rng.normal(size=spectrum.shape) + 1j * rng.normal(size=spectrum.shape)
    ).astype(np.complex64)
    dspectrum[:, (0, -1)] = dspectrum[:, (0, -1)].real
    dwindow = rng.normal(size=window.shape).astype(np.float32)
    primal, tangent = _x86_istft_product.native_jvp(
        spectrum, window, tangents=(dspectrum, dwindow)
    )
    eps = np.float32(2.0e-3)
    plus = tessera.ops.istft(
        spectrum + eps * dspectrum, window + eps * dwindow, hop=4
    )
    minus = tessera.ops.istft(
        spectrum - eps * dspectrum, window - eps * dwindow, hop=4
    )
    oracle = (np.asarray(plus) - np.asarray(minus)) / (2.0 * eps)
    np.testing.assert_allclose(
        primal, tessera.ops.istft(spectrum, window, hop=4),
        rtol=3e-5, atol=3e-5,
    )
    np.testing.assert_allclose(tangent, oracle, rtol=2e-3, atol=2e-3)


@pytest.mark.hardware_rocm
def test_native_gfx1151_istft_window_product_matches_centered_difference():
    from tessera import runtime
    from tessera.compiler.emit.spectral_candidates import _amd_composite_lib

    if runtime._tessera_opt_path() is None or runtime._rocm_chip() != "gfx1151":
        pytest.skip("exact gfx1151 compiler/device required")
    lib = _amd_composite_lib()
    if lib is None or not hasattr(lib, "ts_istft_jvp_plan_hostptr_batch_amd"):
        pytest.skip("gfx1151 ISTFT window-JVP image is stale")
    rng = np.random.default_rng(251)
    spectrum = (
        rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
    ).astype(np.complex64)
    spectrum[:, (0, -1)] = spectrum[:, (0, -1)].real
    window = (np.hanning(8) + 0.2).astype(np.float32)
    dspectrum = (
        rng.normal(size=spectrum.shape) + 1j * rng.normal(size=spectrum.shape)
    ).astype(np.complex64)
    dspectrum[:, (0, -1)] = dspectrum[:, (0, -1)].real
    dwindow = rng.normal(size=window.shape).astype(np.float32)
    primal, tangent = _rocm_istft_product.native_jvp(
        spectrum, window, tangents=(dspectrum, dwindow)
    )
    eps = np.float32(2.0e-3)
    plus = tessera.ops.istft(
        spectrum + eps * dspectrum, window + eps * dwindow, hop=4
    )
    minus = tessera.ops.istft(
        spectrum - eps * dspectrum, window - eps * dwindow, hop=4
    )
    oracle = (np.asarray(plus) - np.asarray(minus)) / (2.0 * eps)
    np.testing.assert_allclose(
        primal, tessera.ops.istft(spectrum, window, hop=4),
        rtol=3e-4, atol=3e-4,
    )
    np.testing.assert_allclose(tangent, oracle, rtol=3e-3, atol=3e-3)


def _rmsnorm_jvp(x: np.ndarray, dx: np.ndarray, eps: float) -> np.ndarray:
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(mean_square + eps)
    projection = np.mean(x * dx, axis=-1, keepdims=True)
    return dx * inv - x * projection * inv**3


@pytest.mark.compiler_avx512
@pytest.mark.parametrize("compiled", [_x86_sum, _x86_rmsnorm, _x86_rfft, _x86_dct4])
def test_native_avx512_jvp_matches_oracle(compiled):
    from tessera import runtime
    if not runtime._x86_elementwise_available():
        pytest.skip("AVX-512 production image unavailable")
    rng = np.random.default_rng(17)
    x = rng.normal(size=(7, 64)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    primal, tangent = compiled.native_jvp(x, tangents=dx)
    if compiled is _x86_sum:
        expected_primal = np.sum(x, axis=-1)
        expected_tangent = np.sum(dx, axis=-1)
    elif compiled is _x86_rmsnorm:
        inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        expected_primal = x * inv
        expected_tangent = _rmsnorm_jvp(x, dx, 1e-5)
    elif compiled is _x86_rfft:
        expected_primal = np.fft.rfft(x, axis=-1).astype(np.complex64)
        expected_tangent = np.fft.rfft(dx, axis=-1).astype(np.complex64)
    else:
        expected_primal = tessera.ops.dct(x, type=4, norm="ortho")
        expected_tangent = tessera.ops.dct(dx, type=4, norm="ortho")
    np.testing.assert_allclose(primal, expected_primal, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-5, atol=3e-5)
    assert compiled.last_jvp_execution["execution_mode"] == "cpu_avx512"
    assert len(compiled.last_jvp_execution["artifact_hash"]) == 64


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("compiled", [_rocm_sum, _rocm_rmsnorm, _rocm_rfft, _rocm_dct4])
def test_native_gfx1151_jvp_matches_oracle(compiled):
    from tessera import runtime
    if runtime._tessera_opt_path() is None or not runtime._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler or device unavailable")
    if runtime._rocm_chip() != "gfx1151":
        pytest.skip("exact gfx1151 required")
    rng = np.random.default_rng(29)
    x = rng.normal(size=(5, 64)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    primal, tangent = compiled.native_jvp(x, tangents=dx)
    if compiled is _rocm_sum:
        expected_primal = np.sum(x, axis=-1)
        expected_tangent = np.sum(dx, axis=-1)
    elif compiled is _rocm_rmsnorm:
        inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        expected_primal = x * inv
        expected_tangent = _rmsnorm_jvp(x, dx, 1e-5)
    elif compiled is _rocm_rfft:
        expected_primal = np.fft.rfft(x, axis=-1).astype(np.complex64)
        expected_tangent = np.fft.rfft(dx, axis=-1).astype(np.complex64)
    else:
        expected_primal = tessera.ops.dct(x, type=4, norm="ortho")
        expected_tangent = tessera.ops.dct(dx, type=4, norm="ortho")
    np.testing.assert_allclose(primal, expected_primal, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-4, atol=3e-4)
    assert compiled.last_jvp_execution["execution_mode"] == "hip_runtime"
    assert compiled.last_jvp_execution["evidence_target"] == "rocm_gfx1151"
