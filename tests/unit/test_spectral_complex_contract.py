"""Complex is a logical dtype carried in a real pair — declared where it is.

The four spectral ops were exempted as "returns complex64, which is
planned_gated per Decision #15a; declaring it conflicts with the dtype
capability contract". It does not conflict: the contract has an explicit path
for planned/gated dtypes (``allow_planned_gated=True`` plus
``metadata.dtype_status``), and naming a dtype in a shape rule is not the same
as claiming a backend implements it.

Meanwhile `complex_same` and `_shape_rfft` were already written and registered
while all four ops stayed exempt — two rules with no consumer sitting next to
the ops they were written for, which is what Decision #29 exists to prevent.

The moment the rules started reporting `fft(f32) -> complex64` honestly, the
legality gate rejected the whole FFT chain: no target's capability table had
ever listed a complex dtype, because the old rule returned the operand's dtype
and an `f32 -> f32` FFT passes any dtype check vacuously. The gate was right.

The answer is NOT complex-as-a-storage-format on four backends. No target ISA
here has a native complex type — CUDA's `cuComplex` is a `float2`, Metal has
`float2`, x86 SIMD interleaves, and Tessera's own ROCm ODS declares its FFT
over "a batch of interleaved-complex rows". The existing `complex_mul` /
`complex_exp` capabilities agree: they declare dtype ``("fp32",)``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tessera.compiler.capabilities import get_target_capability, supports_op
from tessera.compiler.graph_ir import (
    _canonicalize_spectral_attrs,
    _infer_result_type,
    tensor_ir_type,
)
from tessera.compiler.op_catalog import DELIBERATELY_UNDECLARED, shape_rule_for

_FOUR_BACKENDS = ("x86", "nvidia_sm120", "rocm_gfx1151", "apple_cpu")


def test_python_graph_producer_canonicalizes_spectral_identity():
    attrs = {"axis": -1, "n": 9, "norm": "ortho"}
    _canonicalize_spectral_attrs(
        "tessera.irfft", [tensor_ir_type(("5",), "complex64")], attrs
    )
    assert attrs == {
        "axis": -1,
        "logical_length": 9,
        "normalization": "ortho",
        "spectrum_layout": "half_spectrum_nyquist_explicit",
        "hermitian_weight": "none",
    }

    dct_attrs = {}
    _canonicalize_spectral_attrs(
        "tessera.dct", [tensor_ir_type(("8",), "fp32")], dct_attrs
    )
    assert dct_attrs == {
        "normalization": "backward",
        "logical_length": 8,
        "spectrum_layout": "full_real",
        "type": 2,
        "transpose_basis": False,
    }


def test_compound_spectral_identity_is_explicit_and_content_addressable():
    stft_attrs = {"axis": 0, "n_fft": 16, "onesided": False}
    _canonicalize_spectral_attrs(
        "tessera.stft",
        [tensor_ir_type(("31", "3"), "fp32"), tensor_ir_type(("8",), "fp32")],
        stft_attrs,
    )
    assert stft_attrs == {
        "axis": 0,
        "logical_length": 16,
        "onesided": False,
        "normalization": "backward",
        "spectrum_layout": "full_complex",
        "center": False,
        "pad_mode": "constant",
    }

    conv_attrs = {"axis": 1}
    _canonicalize_spectral_attrs(
        "tessera.spectral_conv",
        [tensor_ir_type(("2", "9"), "fp32"), tensor_ir_type(("1", "5"), "fp32")],
        conv_attrs,
    )
    assert conv_attrs["logical_length"] == 16
    assert conv_attrs["spectrum_layout"] == "half_spectrum_nyquist_explicit"


def test_stft_and_istft_shape_rules_preserve_batch_axes_and_length_policy():
    signal = tensor_ir_type(("31", "3"), "fp32")
    window = tensor_ir_type(("8",), "fp32")
    spectrum = _infer_result_type(
        "tessera.stft",
        [signal, window],
        {
            "axis": 0,
            "hop": 4,
            "logical_length": 8,
            "center": False,
            "onesided": True,
        },
    )
    assert spectrum.shape == ("6", "5", "3")
    restored = _infer_result_type(
        "tessera.istft",
        [spectrum, window],
        {
            "axis": 1,
            "hop": 4,
            "logical_length": 8,
            "output_length": 31,
            "center": False,
            "onesided": True,
        },
    )
    assert restored.shape == signal.shape


@pytest.mark.parametrize("op", ["fft", "ifft", "rfft", "irfft"])
def test_every_spectral_exemption_is_retired(op):
    assert f"tessera.{op}" not in DELIBERATELY_UNDECLARED
    assert shape_rule_for(f"tessera.{op}") != "unclassified"


@pytest.mark.parametrize(
    ("real", "expected"), [("fp32", "complex64"), ("fp64", "complex128")]
)
def test_complex_width_follows_the_input_width(real, expected):
    """`fft(f32) -> complex64`, not complex128.

    The numpy-backed reference returned complex128 for every input because
    `np.fft.*` always computes in double. That is the `cos(int32) -> float64`
    defect again — the host library's precision choice overriding the
    compiler's — and it contradicts the family's own declared numeric policy,
    `_spectral_policy`, which is `storage="fp32", accum="fp32"`.
    """
    t = tensor_ir_type(("8",), real)
    assert _infer_result_type("tessera.fft", [t]).dtype == expected


def test_irfft_returns_real_not_complex():
    """`irfft` is the member of the family that returns REAL values.

    All four were exempted under one sentence — "returns complex64" — true of
    its three siblings and simply wrong here. The same one-reason-for-N-ops
    error that hid `kv_cache.read` among the cache mutators.
    """
    t = tensor_ir_type(("5",), "complex64")
    result = _infer_result_type("tessera.irfft", [t])
    assert result.dtype == "fp32", result
    assert result.shape == ("8",), f"2*(n-1) inverse of rfft's n//2+1: {result}"


def test_rfft_and_irfft_round_trip_in_the_rules_and_the_reference():
    """The declared shapes must survive a round trip, and match the reference.

    A rule that is self-consistent but disagrees with the op is the failure
    this registry exists to catch.
    """
    from tessera import ops

    real = tensor_ir_type(("8",), "fp32")
    spectrum = _infer_result_type("tessera.rfft", [real])
    back = _infer_result_type("tessera.irfft", [spectrum])
    assert back.shape == real.shape and back.dtype == real.dtype, (real, spectrum, back)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        f = np.asarray(ops.rfft(x))
        r = np.asarray(ops.irfft(f))
    assert f.shape == (5,) and f.dtype == np.dtype("complex64"), (f.shape, f.dtype)
    assert r.shape == (8,) and r.dtype == np.dtype("float32"), (r.shape, r.dtype)
    np.testing.assert_allclose(r, x, atol=1e-6)


def test_complex_is_not_promoted_to_a_backend_STORAGE_dtype():
    """The per-backend storage contracts stay authoritative and untouched.

    A first cut appended complex to every target's `supported_dtypes` and broke
    five contract gates (`x86_dtype_contract`, `nvidia_dtype_contract`,
    `rocm_dtype_contract`, ...). They were right to fail: those tuples answer
    "what STORAGE has this backend proven", and complex is not a storage format
    any of these ISAs has.
    """
    for target in _FOUR_BACKENDS:
        dtypes = get_target_capability(target).supported_dtypes
        assert not [d for d in dtypes if "complex" in d], (
            f"{target} declares complex as a STORAGE dtype: {dtypes}"
        )


def test_complex_is_declared_only_on_ops_that_carry_it():
    """Widening `_ops()`'s fp32 default would declare complex `matmul`."""
    for target in _FOUR_BACKENDS:
        assert not supports_op(target, "tessera.matmul", dtype="complex64").supported, (
            f"{target} claims a complex matmul"
        )


def test_complex_support_follows_whether_the_target_declares_the_op_at_all():
    """A target with no `fft` entry is stating it has no `fft`.

    Synthesising entries for absent ops looked tidier and was a false claim: it
    made the NVIDIA dashboard assert `fp8_e4m3` and `int8` FFT kernels on
    sm_90 — nine new `artifact_only` rows for a backend where zero source files
    mention FFT. Rejecting a complex FFT there is the correct answer.
    """
    assert supports_op("x86", "tessera.fft", dtype="complex64").supported
    rocm_fft = supports_op("rocm_gfx1151", "tessera.fft", dtype="complex64")
    assert rocm_fft.supported
    assert rocm_fft.runtime_status == "ready"
    assert not supports_op("nvidia_sm120", "tessera.fft", dtype="complex64").supported


@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_rfft_halves_the_SELECTED_axis_not_always_the_last(axis):
    """`rfft(8x16, axis=0)` is `5x16`, not `8x9`.

    Both real-FFT rules resized the trailing dimension unconditionally. `fft` /
    `ifft` were unaffected because they preserve shape — which is why the bug
    lived only in the two rules that resize, and why a spot-check on the
    default `axis=-1` could not see it.
    """
    from tessera import ops

    x = np.linspace(0.0, 1.0, 8 * 16, dtype=np.float32).reshape(8, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = np.asarray(ops.rfft(x, axis=axis))
    predicted = _infer_result_type(
        "tessera.rfft", [tensor_ir_type(("8", "16"), "fp32")], {"axis": axis})
    assert tuple(predicted.shape) == tuple(str(d) for d in actual.shape), (
        f"axis={axis}: rule {predicted} vs actual {actual.shape}"
    )


def test_rfft_honours_the_axes_tuple_form():
    """`axes=(0,)` selects axis 0, mirroring the reference's `_axis_from_axes`
    (an explicit `axes` wins and its LAST entry is the transformed axis)."""
    from tessera import ops

    x = np.linspace(0.0, 1.0, 8 * 16, dtype=np.float32).reshape(8, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = np.asarray(ops.rfft(x, axes=(0,)))
    predicted = _infer_result_type(
        "tessera.rfft", [tensor_ir_type(("8", "16"), "fp32")], {"axes": (0,)})
    assert tuple(predicted.shape) == tuple(str(d) for d in actual.shape)


def test_irfft_restores_the_SELECTED_axis():
    from tessera import ops

    x = np.linspace(0.0, 1.0, 8 * 16, dtype=np.float32).reshape(8, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectrum = np.asarray(ops.rfft(x, axis=0))
        actual = np.asarray(ops.irfft(spectrum, axis=0))
    predicted = _infer_result_type(
        "tessera.irfft",
        [tensor_ir_type(tuple(str(d) for d in spectrum.shape), "complex64")],
        {"axis": 0})
    assert tuple(predicted.shape) == tuple(str(d) for d in actual.shape)


@pytest.mark.parametrize("width", ["int8", "int32", "int64"])
def test_integer_fft_agrees_between_reference_and_rule(width):
    """An integer input has no float storage width to inherit, so it takes the
    declared compute float — the same promotion every other integer-input op
    gets. Choosing the complex width from the INTEGER's byte width instead made
    `fft(int64)` return complex128 while the rule inferred complex64, which is
    the runtime-versus-compiler mismatch this work exists to remove.
    """
    from tessera import ops

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actual = np.asarray(ops.fft(np.arange(1, 9, dtype=width)))
    predicted = _infer_result_type(
        "tessera.fft", [tensor_ir_type(("8",), width)])
    assert predicted.dtype == actual.dtype.name == "complex64", (
        f"{width}: rule {predicted.dtype}, reference {actual.dtype.name}"
    )
