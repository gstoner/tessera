"""Public compound-spectral native VJP ownership and x86 execution proof."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "filter"))
def _x86_spectral_filter(x, filter):
    return ts.ops.spectral_filter(x, filter)


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "kernel"))
def _x86_spectral_conv(x, kernel):
    return ts.ops.spectral_conv(x, kernel, axis=-1, norm="backward")


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "filter"))
def _rocm_spectral_filter(x, filter):
    return ts.ops.spectral_filter(x, filter)


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "kernel"))
def _rocm_spectral_conv(x, kernel):
    return ts.ops.spectral_conv(x, kernel, axis=-1, norm="backward")


def _require_x86_package() -> None:
    from tessera import runtime

    lib = runtime._load_x86_elementwise()
    if lib is None or not hasattr(
        lib, "tessera_x86_avx512_spectral_filter_bwd_c64"
    ):
        pytest.skip("AVX-512 compound spectral backward package unavailable")


def test_x86_public_spectral_filter_backward_uses_family_plugin() -> None:
    _require_x86_package()
    rng = np.random.default_rng(577)
    x = (rng.standard_normal(17) + 1j * rng.standard_normal(17)).astype(np.complex64)
    filt = (rng.standard_normal(17) + 1j * rng.standard_normal(17)).astype(np.complex64)
    dy = (rng.standard_normal(17) + 1j * rng.standard_normal(17)).astype(np.complex64)
    dx, df = _x86_spectral_filter.native_backward(
        x, filt, out_cotangents=dy
    )
    np.testing.assert_allclose(dx, dy * np.conj(filt), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(df, dy * np.conj(x), rtol=2e-6, atol=2e-6)
    proof = _x86_spectral_filter.last_backward_execution
    assert proof["compiler_path"] == "x86_spectral_backward_compiled"
    assert proof["frontend_authority"] == "tracer"
    assert proof["target_consumer"] == "x86.avx512_spectral_backward"
    assert len(proof["source_graph_ir_digest"]) == 64
    assert len(proof["schedule_artifact_hash"]) == 64
    assert len(proof["tile_program_digest"]) == 64


def test_x86_public_spectral_conv_backward_uses_native_full_convolution() -> None:
    _require_x86_package()
    rng = np.random.default_rng(578)
    x = rng.standard_normal(9).astype(np.float32)
    kernel = rng.standard_normal(5).astype(np.float32)
    dy = rng.standard_normal(13).astype(np.float32)
    dx, dk = _x86_spectral_conv.native_backward(
        x, kernel, out_cotangents=dy
    )
    expected_dx = np.array(
        [sum(dy[i + j] * kernel[j] for j in range(kernel.size)) for i in range(x.size)],
        dtype=np.float32,
    )
    expected_dk = np.array(
        [sum(dy[i + j] * x[i] for i in range(x.size)) for j in range(kernel.size)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(dx, expected_dx, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(dk, expected_dk, rtol=2e-6, atol=2e-6)
    assert _x86_spectral_conv.last_backward_execution["family"] == "spectral_backward"


def test_native_spectral_backward_fails_closed_outside_physical_envelope() -> None:
    x = np.ones((2, 4), dtype=np.float32)
    kernel = np.ones((1, 3), dtype=np.float32)
    dy = np.ones((2, 6), dtype=np.float32)
    with pytest.raises(Exception, match="batch broadcasting"):
        _x86_spectral_conv.native_backward(x, kernel, out_cotangents=dy)


@pytest.mark.hardware_rocm
@pytest.mark.parametrize(
    "compiled,kind",
    [(_rocm_spectral_filter, "filter"), (_rocm_spectral_conv, "conv")],
)
def test_rocm_public_compound_spectral_backward_uses_prebuilt_image(
    compiled, kind
) -> None:
    from tessera import runtime

    if runtime._tessera_opt_path() is None or not runtime._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    rng = np.random.default_rng(579 + int(kind == "conv"))
    if kind == "filter":
        x = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).astype(np.complex64)
        parameter = (
            rng.standard_normal(8) + 1j * rng.standard_normal(8)
        ).astype(np.complex64)
        dy = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).astype(np.complex64)
        expected = (dy * np.conj(parameter), dy * np.conj(x))
    else:
        x = rng.standard_normal(7).astype(np.float32)
        parameter = rng.standard_normal(4).astype(np.float32)
        dy = rng.standard_normal(10).astype(np.float32)
        expected = (
            np.array(
                [sum(dy[i + j] * parameter[j] for j in range(parameter.size))
                 for i in range(x.size)],
                dtype=np.float32,
            ),
            np.array(
                [sum(dy[i + j] * x[i] for i in range(x.size))
                 for j in range(parameter.size)],
                dtype=np.float32,
            ),
        )
    actual = compiled.native_backward(x, parameter, out_cotangents=dy)
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, rtol=2e-5, atol=2e-5)
    proof = compiled.last_backward_execution
    assert proof["compiler_path"] == "rocm_spectral_backward_compiled"
    assert proof["target_consumer"] == "rocm.gfx1151_spectral_backward"
    assert proof["frontend_authority"] == "tracer"
