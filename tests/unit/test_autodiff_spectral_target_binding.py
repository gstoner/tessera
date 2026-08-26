"""Public compound-spectral native VJP ownership and x86 execution proof."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tessera as ts


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "filter"))
def _x86_spectral_filter(x, filter):
    return ts.ops.spectral_filter(x, filter)


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "kernel"))
def _x86_spectral_conv(x, kernel):
    return ts.ops.spectral_conv(x, kernel, axis=-1, norm="backward")


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "window"))
def _x86_stft(x, window):
    return ts.ops.stft(
        x, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("spectrum", "window"))
def _x86_istft(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, length=56, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("spectrum", "window"))
def _x86_istft_odd(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=-1, n_fft=15, hop=5, center=False,
        onesided=True, length=35, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "window"))
def _x86_stft_ragged_batch(x, window):
    return ts.ops.stft(
        x, window, axis=-1, n_fft=18, hop=7, center=False,
        onesided=True, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("spectrum", "window"))
def _x86_istft_ragged_batch(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=-1, n_fft=18, hop=7, center=False,
        onesided=True, length=46, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "window"))
def _x86_stft_centered_reflect(x, window):
    return ts.ops.stft(
        x, window, axis=-1, n_fft=18, hop=7, center=True,
        pad_mode="reflect", onesided=True, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("spectrum", "window"))
def _x86_istft_centered_crop(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=-1, n_fft=18, hop=7, center=True,
        onesided=True, length=40, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "window"))
def _x86_stft_centered_axis1(x, window):
    return ts.ops.stft(
        x, window, axis=1, n_fft=18, hop=7, center=True,
        pad_mode="reflect", onesided=True, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("spectrum", "window"))
def _x86_istft_centered_axis2(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=2, n_fft=18, hop=7, center=True,
        onesided=True, length=40, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "window"))
def _x86_stft_full_n20_w15_axis1(x, window):
    return ts.ops.stft(
        x, window, axis=1, n_fft=20, hop=6, center=True,
        pad_mode="constant", onesided=False, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("spectrum", "window"))
def _x86_istft_full_n20_w15_axis2(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=2, n_fft=20, hop=6, center=True,
        onesided=False, length=38, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "window"))
def _x86_stft_broadcast_full_axis1(x, window):
    return ts.ops.stft(
        x, window, axis=1, n_fft=10, hop=4, center=False,
        onesided=False, norm="backward",
    )


@ts.jit(target="x86", autodiff="reverse", wrt=("spectrum", "window"))
def _x86_istft_broadcast_full_axis2(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=2, n_fft=10, hop=4, center=False,
        onesided=False, length=22, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "filter"))
def _rocm_spectral_filter(x, filter):
    return ts.ops.spectral_filter(x, filter)


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "kernel"))
def _rocm_spectral_conv(x, kernel):
    return ts.ops.spectral_conv(x, kernel, axis=-1, norm="backward")


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "window"))
def _rocm_stft(x, window):
    return ts.ops.stft(
        x, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("spectrum", "window"))
def _rocm_istft(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, length=56, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "window"))
def _rocm_stft_ragged_batch(x, window):
    return ts.ops.stft(
        x, window, axis=-1, n_fft=18, hop=7, center=False,
        onesided=True, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("spectrum", "window"))
def _rocm_istft_ragged_batch(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=-1, n_fft=18, hop=7, center=False,
        onesided=True, length=46, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "window"))
def _rocm_stft_centered_reflect(x, window):
    return ts.ops.stft(
        x, window, axis=-1, n_fft=18, hop=7, center=True,
        pad_mode="reflect", onesided=True, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("spectrum", "window"))
def _rocm_istft_centered_crop(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=-1, n_fft=18, hop=7, center=True,
        onesided=True, length=40, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "window"))
def _rocm_stft_centered_axis1(x, window):
    return ts.ops.stft(
        x, window, axis=1, n_fft=18, hop=7, center=True,
        pad_mode="reflect", onesided=True, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("spectrum", "window"))
def _rocm_istft_centered_axis2(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=2, n_fft=18, hop=7, center=True,
        onesided=True, length=40, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "window"))
def _rocm_stft_broadcast_full_axis1(x, window):
    return ts.ops.stft(
        x, window, axis=1, n_fft=10, hop=4, center=False,
        onesided=False, norm="backward",
    )


@ts.jit(target="rocm", autodiff="reverse", wrt=("spectrum", "window"))
def _rocm_istft_broadcast_full_axis2(spectrum, window):
    return ts.ops.istft(
        spectrum, window, axis=2, n_fft=10, hop=4, center=False,
        onesided=False, length=22, norm="backward",
    )


def _require_x86_package() -> None:
    from tessera import runtime

    lib = runtime._load_x86_elementwise()
    required = (
        "tessera_x86_avx512_spectral_filter_bwd_c64",
        "tessera_x86_avx512_stft_bwd_f32",
        "tessera_x86_avx512_istft_bwd_f32",
    )
    if lib is None or any(not hasattr(lib, symbol) for symbol in required):
        pytest.skip("AVX-512 compound spectral backward package unavailable")


def _require_rocm_package() -> None:
    from tessera import runtime
    from tessera.compiler.emit import spectral_candidates

    lib = spectral_candidates._amd_composite_lib()
    if (not runtime._rocm_wmma_runtime_available() or lib is None or
            not hasattr(lib, "ts_stft_backward_hostptr_broadcast_layout_storage_amd")):
        pytest.skip("gfx1151 compound spectral backward package unavailable")


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


@pytest.mark.parametrize("kind", ["stft", "istft"])
def test_x86_public_stft_istft_backward_matches_reference_vjp(kind: str) -> None:
    _require_x86_package()
    from tessera.autodiff import vjp

    rng = np.random.default_rng(581 if kind == "stft" else 582)
    window = np.hanning(16).astype(np.float32)
    if kind == "stft":
        primal = rng.standard_normal(56).astype(np.float32)
        dy = (rng.standard_normal((6, 9)) +
              1j * rng.standard_normal((6, 9))).astype(np.complex64)
        actual = _x86_stft.native_backward(primal, window, out_cotangents=dy)
        expected = vjp._VJPS["stft"](
            dy, primal, window, axis=-1, n_fft=16, hop=8,
            center=False, onesided=True, norm="backward",
        )
    else:
        primal = (rng.standard_normal((6, 9)) +
                  1j * rng.standard_normal((6, 9))).astype(np.complex64)
        dy = rng.standard_normal(56).astype(np.float32)
        actual = _x86_istft.native_backward(primal, window, out_cotangents=dy)
        expected = vjp._VJPS["istft"](
            dy, primal, window, axis=-1, n_fft=16, hop=8,
            center=False, onesided=True, length=56, norm="backward",
        )
    np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=3e-5, atol=3e-5)
    proof = (_x86_stft if kind == "stft" else _x86_istft).last_backward_execution
    assert proof["compiler_path"] == "x86_spectral_backward_compiled"
    assert proof["target_consumer"] == "x86.avx512_spectral_backward"
    expected_algorithm = (
        "packed_c2r_stored_bin_v1"
        if kind == "stft"
        else "normalized_overlap_add_r2c_v1"
    )
    assert proof["algorithm"] == expected_algorithm
    assert proof["execution_certificate"]["algorithm"] == expected_algorithm
    assert proof["execution_certificate"]["artifact_identities"][
        "tile_program_digest"
    ]


@pytest.mark.parametrize("kind", ["stft", "istft"])
def test_x86_generalized_spectral_backward_consumes_native_layout_and_full_spectrum(
    kind: str,
) -> None:
    _require_x86_package()
    from tessera.autodiff import vjp

    rng = np.random.default_rng(608 if kind == "stft" else 609)
    window = np.hanning(30).astype(np.float32)[::2]
    assert not window.flags.c_contiguous
    if kind == "stft":
        primal = rng.standard_normal((3, 2, 44)).astype(np.float32).transpose(0, 2, 1)
        dy_storage = np.empty((3, 8, 40, 2), dtype=np.complex64)
        dy = dy_storage[:, :, ::2, :]
        dy[...] = (rng.standard_normal(dy.shape) +
                   1j * rng.standard_normal(dy.shape)).astype(np.complex64)
        actual = _x86_stft_full_n20_w15_axis1.native_backward(
            primal, window, out_cotangents=dy
        )
        expected = vjp._VJPS["stft"](
            dy, primal, window, axis=1, n_fft=20, hop=6, center=True,
            pad_mode="constant", onesided=False, norm="backward",
        )
    else:
        primal_storage = np.empty((3, 8, 40, 2), dtype=np.complex64)
        primal = primal_storage[:, :, ::2, :]
        primal[...] = (rng.standard_normal(primal.shape) +
                       1j * rng.standard_normal(primal.shape)).astype(np.complex64)
        dy = rng.standard_normal((3, 2, 38)).astype(np.float32).transpose(0, 2, 1)
        actual = _x86_istft_full_n20_w15_axis2.native_backward(
            primal, window, out_cotangents=dy
        )
        expected = vjp._VJPS["istft"](
            dy, primal, window, axis=2, n_fft=20, hop=6, center=True,
            onesided=False, length=38, norm="backward",
        )
    np.testing.assert_allclose(actual[0], expected[0], rtol=7e-5, atol=7e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=7e-5, atol=7e-5)
    proof = (
        _x86_stft_full_n20_w15_axis1 if kind == "stft"
        else _x86_istft_full_n20_w15_axis2
    ).last_backward_execution
    assert proof["algorithm"] == "full_complex_direct_dft_v1"


def test_native_spectral_backward_fails_closed_outside_physical_envelope() -> None:
    x = np.ones((2, 4), dtype=np.float32)
    kernel = np.ones((1, 3), dtype=np.float32)
    dy = np.ones((2, 6), dtype=np.float32)
    with pytest.raises(Exception, match="batch broadcasting"):
        _x86_spectral_conv.native_backward(x, kernel, out_cotangents=dy)


def test_native_istft_backward_supports_odd_direct_dft_envelope() -> None:
    from tessera.autodiff import vjp

    spectrum = np.ones((5, 8), dtype=np.complex64)
    window = np.ones(15, dtype=np.float32)
    dy = np.ones(35, dtype=np.float32)
    actual = _x86_istft_odd.native_backward(
        spectrum, window, out_cotangents=dy
    )
    expected = vjp._VJPS["istft"](
        dy, spectrum, window, axis=-1, n_fft=15, hop=5, center=False,
        onesided=True, length=35, norm="backward",
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=4e-5, atol=4e-5)


def test_native_stft_istft_low_precision_storage_must_match() -> None:
    signal = np.ones(46, dtype=np.float16)
    window_f32 = np.ones(18, dtype=np.float32)
    stft_dy = np.ones((5, 10), dtype=np.complex64)
    with pytest.raises(Exception, match="matching f16/bf16/f32 signal/window"):
        _x86_stft_ragged_batch.native_backward(
            signal, window_f32, out_cotangents=stft_dy
        )

    spectrum = np.ones((5, 10), dtype=np.complex64)
    window_f16 = np.ones(18, dtype=np.float16)
    istft_dy = np.ones(46, dtype=np.float32)
    with pytest.raises(Exception, match="matching f16/bf16/f32 window/cotangent"):
        _x86_istft_ragged_batch.native_backward(
            spectrum, window_f16, out_cotangents=istft_dy
        )


def test_native_stft_algorithm_identity_fails_closed() -> None:
    from tessera.compiler.native_spectral_vjp import (
        build_native_spectral_vjp_package,
        validate_native_spectral_vjp_contract,
    )

    source = SimpleNamespace(
        op_name="tessera.stft",
        kwargs={
            "axis": -1,
            "n_fft": 16,
            "hop": 8,
            "center": False,
            "onesided": True,
            "norm": "backward",
        },
    )
    package = build_native_spectral_vjp_package(
        source_graph_ir="module {}",
        source=source,
        target="x86",
        ordered_inputs=(
            np.ones(56, dtype=np.float32),
            np.ones(16, dtype=np.float32),
        ),
        arg_names=("x", "window"),
        out_cotangent=np.ones((6, 9), dtype=np.complex64),
    )
    altered = package.contract()
    altered["algorithm"] = "direct_stored_bin_odd_tail_v1"
    with pytest.raises(ValueError, match="Tile artifact|algorithm identity"):
        validate_native_spectral_vjp_contract(altered)

    altered = package.contract()
    altered["numeric_policy"]["accum"] = "fp16"
    with pytest.raises(ValueError, match="Schedule artifact|fp32 accumulation"):
        validate_native_spectral_vjp_contract(altered)


def test_rocm_stft_noncontiguous_runtime_executes_layout_abi() -> None:
    from tessera.autodiff import vjp

    _require_rocm_package()
    signal = np.arange(112, dtype=np.float32)[::2]
    assert not signal.flags.c_contiguous
    window = np.ones(16, dtype=np.float32)
    cotangent = np.ones((6, 9), dtype=np.complex64)
    actual = _rocm_stft.native_backward(
        signal, window, out_cotangents=cotangent
    )
    expected = vjp._VJPS["stft"](
        cotangent, signal, window, axis=-1, n_fft=16, hop=8,
        center=False, onesided=True, norm="backward",
    )
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, atol=8e-4, rtol=8e-4)


def test_rocm_stft_package_binds_broader_explicit_length() -> None:
    from tessera.compiler.native_spectral_vjp import build_native_spectral_vjp_package

    source = SimpleNamespace(
        op_name="tessera.stft",
        kwargs={
            "axis": -1,
            "n_fft": 20,
            "hop": 10,
            "center": False,
            "onesided": True,
            "norm": "backward",
        },
    )
    package = build_native_spectral_vjp_package(
        source_graph_ir="module {}",
        source=source,
        target="rocm",
        ordered_inputs=(
            np.ones(60, dtype=np.float32),
            np.ones(15, dtype=np.float32),
        ),
        arg_names=("x", "window"),
        out_cotangent=np.ones((5, 11), dtype=np.complex64),
    )
    assert package.logical_length == 20
    assert package.output_types[-1] == "tensor<15xf32>"


@pytest.mark.parametrize(
    "op_name,axis,inputs,cotangent",
    [
        (
            "tessera.stft", 2,
            (np.ones((2, 46), np.float32), np.ones(18, np.float32)),
            np.ones((2, 7, 10), np.complex64),
        ),
        (
            "tessera.istft", 3,
            (np.ones((2, 7, 10), np.complex64), np.ones(18, np.float32)),
            np.ones((2, 40), np.float32),
        ),
    ],
)
def test_native_spectral_vjp_rejects_out_of_range_axis(
    op_name, axis, inputs, cotangent
) -> None:
    from tessera.compiler.native_spectral_vjp import build_native_spectral_vjp_package

    source = SimpleNamespace(
        op_name=op_name,
        kwargs={
            "axis": axis, "n_fft": 18, "hop": 7, "center": True,
            "onesided": True, "length": 40, "norm": "backward",
        },
    )
    with pytest.raises(ValueError, match="axis is out of range"):
        build_native_spectral_vjp_package(
            source_graph_ir="module {}", source=source, target="x86",
            ordered_inputs=inputs, arg_names=("x", "window"),
            out_cotangent=cotangent,
        )


def test_native_stft_istft_backward_status_symbols_return_int() -> None:
    import ctypes
    from tessera import runtime

    _require_x86_package()
    lib = runtime._load_x86_elementwise()
    assert lib.tessera_x86_avx512_stft_bwd_f32.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_istft_bwd_f32.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_stft_bwd_storage.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_istft_bwd_storage.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_stft_bwd_policy_storage.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_istft_bwd_policy_storage.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_stft_bwd_policy_strided_storage.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_istft_bwd_policy_strided_storage.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_stft_bwd_policy_layout_storage.restype is ctypes.c_int
    assert lib.tessera_x86_avx512_istft_bwd_policy_layout_storage.restype is ctypes.c_int
    assert lib.tessera_x86_stft_policy_strided_storage.restype is ctypes.c_int
    assert lib.tessera_x86_istft_policy_strided_storage.restype is ctypes.c_int

    one = np.ones(2, dtype=np.float32)
    f32 = one.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    raw = one.ctypes.data_as(ctypes.c_void_p)
    huge = np.iinfo(np.int64).max
    assert lib.tessera_x86_avx512_stft_bwd_storage(
        b"extent-overflow", f32, raw, raw, raw, raw,
        huge, 2, 1, 1, 1, 1, ctypes.c_float(1.0),
    ) == 10
    assert lib.tessera_x86_avx512_istft_bwd_storage(
        b"extent-overflow", raw, f32, raw, f32, raw,
        huge, 2, 1, 1, 1, ctypes.c_float(1.0),
    ) == 10
    assert lib.tessera_x86_avx512_stft_bwd_policy_storage(
        b"extent-overflow", f32, raw, raw, raw, raw,
        huge, 2, 2, 1, 1, 0, ctypes.c_float(1.0), 0, 0,
    ) == 22
    assert lib.tessera_x86_avx512_istft_bwd_policy_storage(
        b"extent-overflow", raw, f32, raw, f32, raw,
        huge, 2, 1, 1, 0, ctypes.c_float(1.0), 0, 2,
    ) == 32
    assert lib.tessera_x86_avx512_stft_bwd_policy_strided_storage(
        b"extent-overflow", f32, raw, raw, raw, raw,
        huge, 2, 2, 1, 1, 1, 0, ctypes.c_float(1.0), 0, 0,
    ) == 40
    assert lib.tessera_x86_avx512_istft_bwd_policy_strided_storage(
        b"extent-overflow", raw, f32, raw, f32, raw,
        huge, 1, 1, 2, 1, 1, 0, ctypes.c_float(1.0), 0, 1,
    ) == 50
    assert lib.tessera_x86_stft_policy_strided_storage(
        b"extent-overflow", raw, raw, f32,
        huge, 2, 1, 1, 1, 2, 0, ctypes.c_float(1.0), 0, 0,
    ) == 32
    assert lib.tessera_x86_istft_policy_strided_storage(
        b"extent-overflow", f32, raw, raw,
        huge, 2, 1, 1, 1, 1, 0, ctypes.c_float(1.0), 0, 2,
    ) == 41


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_x86_ragged_batched_stft_istft_backward_storage_envelopes(
    storage: str,
) -> None:
    _require_x86_package()
    from tessera.autodiff import vjp

    dtype = np.float32
    if storage == "f16":
        dtype = np.float16
    elif storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(20260827)
    signal = rng.normal(size=(3, 46)).astype(dtype)
    window = (np.hanning(18) + 0.25).astype(dtype)
    stft_dy = (
        rng.normal(size=(3, 5, 10)) + 1j * rng.normal(size=(3, 5, 10))
    ).astype(np.complex64)
    stft_actual = _x86_stft_ragged_batch.native_backward(
        signal, window, out_cotangents=stft_dy
    )
    stft_expected = vjp._VJPS["stft"](
        stft_dy, signal, window, axis=-1, n_fft=18, hop=7,
        center=False, onesided=True, norm="backward",
    )

    spectrum = (
        rng.normal(size=(3, 5, 10)) + 1j * rng.normal(size=(3, 5, 10))
    ).astype(np.complex64)
    istft_dy = rng.normal(size=(3, 46)).astype(dtype)
    istft_actual = _x86_istft_ragged_batch.native_backward(
        spectrum, window, out_cotangents=istft_dy
    )
    istft_expected = vjp._VJPS["istft"](
        istft_dy, spectrum, window, axis=-1, n_fft=18, hop=7,
        center=False, onesided=True, length=46, norm="backward",
    )

    tolerance = {"f32": 8e-5, "f16": 3e-2, "bf16": 2e-1}[storage]
    for actual_values, expected_values in (
        (stft_actual, stft_expected),
        (istft_actual, istft_expected),
    ):
        for value, reference in zip(actual_values, expected_values, strict=True):
            np.testing.assert_allclose(
                np.asarray(value).astype(np.complex64 if np.iscomplexobj(value) else np.float32),
                np.asarray(reference).astype(np.complex64 if np.iscomplexobj(reference) else np.float32),
                rtol=tolerance,
                atol=tolerance,
            )
    assert str(np.asarray(stft_actual[0]).dtype) == str(np.dtype(dtype))
    assert str(np.asarray(stft_actual[1]).dtype) == str(np.dtype(dtype))
    assert str(np.asarray(istft_actual[1]).dtype) == str(np.dtype(dtype))
    contract = _x86_stft_ragged_batch.last_backward_execution
    assert contract["execution_certificate_schema"] == (
        "tessera.native_vjp_execution.v1"
    )
    assert contract["algorithm"] == "packed_c2r_stored_bin_v1"
    assert contract["execution_certificate"]["algorithm"] == (
        "packed_c2r_stored_bin_v1"
    )


def test_x86_per_batch_broadcast_reverse_reduces_window_cotangent() -> None:
    _require_x86_package()
    from tessera.autodiff import vjp

    rng = np.random.default_rng(20260908)
    signal = rng.normal(size=(2, 48, 3)).astype(np.float32)[:, ::2, :]
    window = np.stack((np.hanning(8), np.hamming(8)), axis=0).astype(
        np.float32
    )[:, None, :]
    stft_dy = (
        rng.normal(size=(2, 4, 10, 3))
        + 1j * rng.normal(size=(2, 4, 10, 3))
    ).astype(np.complex64)
    actual_stft = _x86_stft_broadcast_full_axis1.native_backward(
        signal, window, out_cotangents=stft_dy
    )
    expected_stft = vjp._VJPS["stft"](
        stft_dy, signal, window, axis=1, n_fft=10, hop=4,
        center=False, onesided=False, norm="backward",
    )

    spectrum = (
        rng.normal(size=(2, 4, 10, 3))
        + 1j * rng.normal(size=(2, 4, 10, 3))
    ).astype(np.complex64)
    istft_dy = rng.normal(size=(2, 22, 3)).astype(np.float32)
    actual_istft = _x86_istft_broadcast_full_axis2.native_backward(
        spectrum, window, out_cotangents=istft_dy
    )
    expected_istft = vjp._VJPS["istft"](
        istft_dy, spectrum, window, axis=2, n_fft=10, hop=4,
        center=False, onesided=False, length=22, norm="backward",
    )
    for actual, expected in ((actual_stft, expected_stft),
                             (actual_istft, expected_istft)):
        assert actual[1].shape == window.shape
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference, atol=3e-4, rtol=3e-4)


@pytest.mark.hardware_rocm
def test_gfx1151_per_batch_broadcast_short_window_full_spectrum_reverse() -> None:
    _require_rocm_package()
    from tessera.autodiff import vjp

    rng = np.random.default_rng(20260909)
    signal = rng.normal(size=(2, 48, 3)).astype(np.float32)[:, ::2, :]
    window = np.stack((np.hanning(8), np.hamming(8)), axis=0).astype(
        np.float32
    )[:, None, :]
    stft_dy = (
        rng.normal(size=(2, 4, 10, 3))
        + 1j * rng.normal(size=(2, 4, 10, 3))
    ).astype(np.complex64)
    actual_stft = _rocm_stft_broadcast_full_axis1.native_backward(
        signal, window, out_cotangents=stft_dy
    )
    expected_stft = vjp._VJPS["stft"](
        stft_dy, signal, window, axis=1, n_fft=10, hop=4,
        center=False, onesided=False, norm="backward",
    )

    spectrum = (
        rng.normal(size=(2, 4, 10, 3))
        + 1j * rng.normal(size=(2, 4, 10, 3))
    ).astype(np.complex64)
    istft_dy = rng.normal(size=(2, 22, 3)).astype(np.float32)
    actual_istft = _rocm_istft_broadcast_full_axis2.native_backward(
        spectrum, window, out_cotangents=istft_dy
    )
    expected_istft = vjp._VJPS["istft"](
        istft_dy, spectrum, window, axis=2, n_fft=10, hop=4,
        center=False, onesided=False, length=22, norm="backward",
    )
    for actual, expected in ((actual_stft, expected_stft),
                             (actual_istft, expected_istft)):
        assert actual[1].shape == window.shape
        for value, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, reference, atol=1.5e-3, rtol=1.5e-3)
    from tessera import runtime
    from tests.unit.test_rocm_spectral_compiled import _art

    forward = runtime.launch(
        _art(runtime, "tessera.stft", (signal, window), {
            "axis": 1, "n_fft": 10, "hop": 4, "onesided": False,
        }), (signal, window),
    )
    assert forward["ok"] is True, forward.get("reason")
    lhs = float(np.real(np.vdot(stft_dy, forward["output"])))
    rhs = float(np.vdot(signal, actual_stft[0]))
    assert abs(lhs - rhs) <= 1.5e-3 * max(abs(lhs), 1.0)
    assert _rocm_stft_broadcast_full_axis1.last_backward_execution[
        "physical_attestation"
    ]["device_arch"] == "gfx1151"


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_x86_centered_reflect_and_cropped_reverse_match_independent_vjp(storage):
    _require_x86_package()
    from tessera.autodiff import vjp

    dtype = np.float32 if storage == "f32" else np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(20260831)
    signal = rng.normal(size=(2, 46)).astype(dtype)
    window = (np.hanning(18) + 0.25).astype(dtype)
    stft_dy = (rng.normal(size=(2, 7, 10)) +
               1j * rng.normal(size=(2, 7, 10))).astype(np.complex64)
    actual_stft = _x86_stft_centered_reflect.native_backward(
        signal, window, out_cotangents=stft_dy
    )
    expected_stft = vjp._VJPS["stft"](
        stft_dy, signal, window, axis=-1, n_fft=18, hop=7, center=True,
        pad_mode="reflect", onesided=True, norm="backward",
    )
    spectrum = (rng.normal(size=(2, 7, 10)) +
                1j * rng.normal(size=(2, 7, 10))).astype(np.complex64)
    istft_dy = rng.normal(size=(2, 40)).astype(dtype)
    actual_istft = _x86_istft_centered_crop.native_backward(
        spectrum, window, out_cotangents=istft_dy
    )
    expected_istft = vjp._VJPS["istft"](
        istft_dy, spectrum, window, axis=-1, n_fft=18, hop=7, center=True,
        onesided=True, length=40, norm="backward",
    )
    tolerance = {"f32": 1e-4, "f16": 4e-2, "bf16": 2.5e-1}[storage]
    for actual, expected in ((actual_stft, expected_stft),
                             (actual_istft, expected_istft)):
        for value, reference in zip(actual, expected, strict=True):
            comparison_dtype = np.complex64 if np.iscomplexobj(value) else np.float32
            np.testing.assert_allclose(
                np.asarray(value, comparison_dtype),
                np.asarray(reference, comparison_dtype),
                atol=tolerance, rtol=tolerance,
            )
    if storage == "f32":
        from tests.unit.test_x86_spectral_compiled import _art
        from tessera import runtime

        forward = runtime.launch(
            _art(runtime, "tessera.stft", (signal, window), {
                "hop": 7, "center": True, "pad_mode": "reflect",
            }), (signal, window),
        )
        lhs = float(np.real(np.vdot(stft_dy, forward["output"])))
        rhs = float(np.vdot(signal, actual_stft[0]))
        assert abs(lhs - rhs) <= 1e-4 * max(abs(lhs), 1.0)


def _assert_centered_arbitrary_axis_reverse(
    compiled_stft, compiled_istft, *, target: str, storage: str
) -> None:
    from tessera.autodiff import vjp

    dtype = np.float32 if storage == "f32" else np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(20260902 + int(target == "rocm"))
    signal = rng.normal(size=(2, 46, 3)).astype(dtype)
    window = (np.hanning(18) + 0.25).astype(dtype)
    stft_dy = (rng.normal(size=(2, 7, 10, 3)) +
               1j * rng.normal(size=(2, 7, 10, 3))).astype(np.complex64)
    actual_stft = compiled_stft.native_backward(
        signal, window, out_cotangents=stft_dy
    )
    expected_stft = vjp._VJPS["stft"](
        stft_dy, signal, window, axis=1, n_fft=18, hop=7, center=True,
        pad_mode="reflect", onesided=True, norm="backward",
    )
    spectrum = (rng.normal(size=(2, 7, 10, 3)) +
                1j * rng.normal(size=(2, 7, 10, 3))).astype(np.complex64)
    istft_dy = rng.normal(size=(2, 40, 3)).astype(dtype)
    actual_istft = compiled_istft.native_backward(
        spectrum, window, out_cotangents=istft_dy
    )
    expected_istft = vjp._VJPS["istft"](
        istft_dy, spectrum, window, axis=2, n_fft=18, hop=7, center=True,
        onesided=True, length=40, norm="backward",
    )
    tolerance = {
        "x86": {"f32": 2e-4, "f16": 5e-2, "bf16": 3e-1},
        "rocm": {"f32": 8e-4, "f16": 6e-2, "bf16": 3.5e-1},
    }[target][storage]
    for actual, expected in ((actual_stft, expected_stft),
                             (actual_istft, expected_istft)):
        for value, reference in zip(actual, expected, strict=True):
            comparison_dtype = np.complex64 if np.iscomplexobj(value) else np.float32
            np.testing.assert_allclose(
                np.asarray(value, comparison_dtype),
                np.asarray(reference, comparison_dtype),
                atol=tolerance, rtol=tolerance,
            )
    if storage == "f32":
        from tessera import runtime

        if target == "x86":
            from tests.unit.test_x86_spectral_compiled import _art
        else:
            from tests.unit.test_rocm_spectral_compiled import _art
        forward = runtime.launch(
            _art(runtime, "tessera.stft", (signal, window), {
                "axis": 1, "hop": 7, "center": True,
                "pad_mode": "reflect",
            }), (signal, window),
        )
        assert forward["ok"] is True, forward.get("reason")
        lhs = float(np.real(np.vdot(stft_dy, forward["output"])))
        rhs = float(np.vdot(signal, actual_stft[0]))
        assert abs(lhs - rhs) <= tolerance * max(abs(lhs), 1.0)
        if target == "rocm":
            assert forward["physical_attestation"]["device_arch"] == "gfx1151"


@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_x86_centered_arbitrary_axis_reverse_matches_independent_vjp(storage):
    _require_x86_package()
    _assert_centered_arbitrary_axis_reverse(
        _x86_stft_centered_axis1, _x86_istft_centered_axis2,
        target="x86", storage=storage,
    )


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


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("kind", ["stft", "istft"])
def test_rocm_stft_istft_backward_matches_independent_vjp(kind: str) -> None:
    from tessera import runtime
    from tessera.autodiff import vjp
    from tessera.compiler.native_vjp_plugins import (
        validate_native_vjp_execution_certificate,
    )

    if runtime._tessera_opt_path() is None or not runtime._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    rng = np.random.default_rng(20260828 + int(kind == "istft"))
    window = (np.hanning(16) + 0.125).astype(np.float32)
    if kind == "stft":
        primal = rng.normal(size=56).astype(np.float32)
        dy = (rng.normal(size=(6, 9)) + 1j * rng.normal(size=(6, 9))).astype(
            np.complex64
        )
        actual = _rocm_stft.native_backward(primal, window, out_cotangents=dy)
        expected = vjp._VJPS["stft"](
            dy, primal, window, axis=-1, n_fft=16, hop=8,
            center=False, onesided=True, norm="backward",
        )
        compiled = _rocm_stft
        algorithm = "direct_stored_bin_gfx1151_v1"
    else:
        primal = (
            rng.normal(size=(6, 9)) + 1j * rng.normal(size=(6, 9))
        ).astype(np.complex64)
        dy = rng.normal(size=56).astype(np.float32)
        actual = _rocm_istft.native_backward(primal, window, out_cotangents=dy)
        expected = vjp._VJPS["istft"](
            dy, primal, window, axis=-1, n_fft=16, hop=8,
            center=False, onesided=True, length=56, norm="backward",
        )
        compiled = _rocm_istft
        algorithm = "normalized_overlap_add_direct_dft_gfx1151_v1"
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=2e-4, atol=2e-4)
    proof = compiled.last_backward_execution
    assert proof["algorithm"] == algorithm
    assert proof["target_consumer"] == "rocm.gfx1151_spectral_backward"
    certificate = proof["execution_certificate"]
    assert certificate["evidence_scope"] == "exact_device"
    assert certificate["physical_attestation"]["device_arch"] == "gfx1151"
    validate_native_vjp_execution_certificate(certificate)


@pytest.mark.hardware_rocm
def test_rocm_stft_forward_and_adjoint_satisfy_inner_product_identity() -> None:
    from tests.unit.test_rocm_spectral_compiled import _art
    from tessera import runtime

    if runtime._tessera_opt_path() is None or not runtime._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    rng = np.random.default_rng(20260829)
    signal = rng.normal(size=56).astype(np.float32)
    window = (np.hanning(16) + 0.125).astype(np.float32)
    cotangent = (
        rng.normal(size=(6, 9)) + 1j * rng.normal(size=(6, 9))
    ).astype(np.complex64)
    forward = runtime.launch(
        _art(runtime, "tessera.stft", (signal, window), {"hop": 8}),
        (signal, window),
    )
    assert forward["ok"] is True, forward.get("reason")
    assert forward["physical_attestation"]["device_arch"] == "gfx1151"
    signal_cotangent, _ = _rocm_stft.native_backward(
        signal, window, out_cotangents=cotangent
    )
    lhs = float(np.real(np.vdot(cotangent, np.asarray(forward["output"]))))
    rhs = float(np.dot(signal, np.asarray(signal_cotangent, np.float32)))
    assert abs(lhs - rhs) <= 3e-5 * max(abs(lhs), 1.0), (lhs, rhs)


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_rocm_ragged_batched_stft_istft_backward_storage_envelopes(
    storage: str,
) -> None:
    from tessera import runtime
    from tessera.autodiff import vjp

    if runtime._tessera_opt_path() is None or not runtime._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    dtype = np.float32
    if storage == "f16":
        dtype = np.float16
    elif storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(20260830)
    signal = rng.normal(size=(3, 46)).astype(dtype)
    window = (np.hanning(18) + 0.25).astype(dtype)
    stft_dy = (
        rng.normal(size=(3, 5, 10)) + 1j * rng.normal(size=(3, 5, 10))
    ).astype(np.complex64)
    stft_actual = _rocm_stft_ragged_batch.native_backward(
        signal, window, out_cotangents=stft_dy
    )
    stft_expected = vjp._VJPS["stft"](
        stft_dy, signal, window, axis=-1, n_fft=18, hop=7,
        center=False, onesided=True, norm="backward",
    )
    spectrum = (
        rng.normal(size=(3, 5, 10)) + 1j * rng.normal(size=(3, 5, 10))
    ).astype(np.complex64)
    istft_dy = rng.normal(size=(3, 46)).astype(dtype)
    istft_actual = _rocm_istft_ragged_batch.native_backward(
        spectrum, window, out_cotangents=istft_dy
    )
    istft_expected = vjp._VJPS["istft"](
        istft_dy, spectrum, window, axis=-1, n_fft=18, hop=7,
        center=False, onesided=True, length=46, norm="backward",
    )
    tolerance = {"f32": 4e-4, "f16": 4e-2, "bf16": 2.5e-1}[storage]
    for actual_values, expected_values in (
        (stft_actual, stft_expected), (istft_actual, istft_expected)
    ):
        for value, reference in zip(actual_values, expected_values, strict=True):
            target_dtype = np.complex64 if np.iscomplexobj(value) else np.float32
            np.testing.assert_allclose(
                np.asarray(value).astype(target_dtype),
                np.asarray(reference).astype(target_dtype),
                rtol=tolerance, atol=tolerance,
            )
    assert np.asarray(stft_actual[0]).dtype == np.asarray(signal).dtype
    assert np.asarray(stft_actual[1]).dtype == np.asarray(window).dtype
    assert np.asarray(istft_actual[1]).dtype == np.asarray(window).dtype


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_rocm_centered_reflect_and_cropped_reverse_match_independent_vjp(storage):
    from tessera import runtime
    from tessera.autodiff import vjp

    if runtime._tessera_opt_path() is None or not runtime._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    dtype = np.float32 if storage == "f32" else np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(20260901)
    signal = rng.normal(size=(2, 46)).astype(dtype)
    window = (np.hanning(18) + 0.25).astype(dtype)
    stft_dy = (rng.normal(size=(2, 7, 10)) +
               1j * rng.normal(size=(2, 7, 10))).astype(np.complex64)
    actual_stft = _rocm_stft_centered_reflect.native_backward(
        signal, window, out_cotangents=stft_dy
    )
    expected_stft = vjp._VJPS["stft"](
        stft_dy, signal, window, axis=-1, n_fft=18, hop=7, center=True,
        pad_mode="reflect", onesided=True, norm="backward",
    )
    spectrum = (rng.normal(size=(2, 7, 10)) +
                1j * rng.normal(size=(2, 7, 10))).astype(np.complex64)
    istft_dy = rng.normal(size=(2, 40)).astype(dtype)
    actual_istft = _rocm_istft_centered_crop.native_backward(
        spectrum, window, out_cotangents=istft_dy
    )
    expected_istft = vjp._VJPS["istft"](
        istft_dy, spectrum, window, axis=-1, n_fft=18, hop=7, center=True,
        onesided=True, length=40, norm="backward",
    )
    tolerance = {"f32": 5e-4, "f16": 5e-2, "bf16": 3e-1}[storage]
    for actual, expected in ((actual_stft, expected_stft),
                             (actual_istft, expected_istft)):
        for value, reference in zip(actual, expected, strict=True):
            comparison_dtype = np.complex64 if np.iscomplexobj(value) else np.float32
            np.testing.assert_allclose(
                np.asarray(value, comparison_dtype),
                np.asarray(reference, comparison_dtype),
                atol=tolerance, rtol=tolerance,
            )
    proof = _rocm_stft_centered_reflect.last_backward_execution
    assert proof["execution_certificate"]["evidence_scope"] == "exact_device"
    if storage == "f32":
        from tests.unit.test_rocm_spectral_compiled import _art

        forward = runtime.launch(
            _art(runtime, "tessera.stft", (signal, window), {
                "hop": 7, "center": True, "pad_mode": "reflect",
            }), (signal, window),
        )
        lhs = float(np.real(np.vdot(stft_dy, forward["output"])))
        rhs = float(np.vdot(signal, actual_stft[0]))
        assert abs(lhs - rhs) <= 5e-4 * max(abs(lhs), 1.0)


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_rocm_centered_arbitrary_axis_reverse_matches_independent_vjp(storage):
    from tessera import runtime

    if runtime._tessera_opt_path() is None or not runtime._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/gfx1151 runtime unavailable")
    _assert_centered_arbitrary_axis_reverse(
        _rocm_stft_centered_axis1, _rocm_istft_centered_axis2,
        target="rocm", storage=storage,
    )
