"""Low-precision storage for the Apple normalization VJP.

The policy under test is the project's reduced-precision rule: **accumulate in
f32, store back at the operand's own dtype**. So an f16 graph must come back with
f16 gradients rather than f32 ones the caller has to cast — and the error must be
dominated by that single final rounding, not by the accumulation.

Tolerances are therefore derived from each storage format's epsilon rather than
picked. A kernel that accumulated in f16 would blow through them; one that
returned widened f32 gradients would fail the dtype assertions.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.native_vjp_plugins import execute_native_vjp_family

# Mantissa bits -> relative epsilon. Correct round-to-nearest storage lands
# within ~0.5 ulp; the margin below allows a few ulp of accumulated slack.
_STORAGE_EPS = {"float32": 2.0 ** -23, "float16": 2.0 ** -10, "bfloat16": 2.0 ** -7}
_TOLERANCE_ULPS = 8.0


def _bf16():
    try:
        import ml_dtypes
    except ImportError:  # pragma: no cover - environment dependent
        return None
    return ml_dtypes.bfloat16


def _apple_available() -> bool:
    from tessera.compiler import apple_native

    return apple_native.tools_available()


class _Source:
    def __init__(self, op_name: str, eps: float = 1e-5) -> None:
        self.op_name = op_name
        self.result = "y"
        self.kwargs = {"eps": eps}


def _reference(op_name: str, x, gamma, dy, eps: float):
    """float64 analytic VJP, fed the SAME rounded values the kernel received.

    Rounding the inputs first matters: comparing a low-precision kernel against
    an oracle fed full-precision inputs would charge it for the caller's own
    input rounding and make any tolerance meaningless.
    """
    x64 = np.asarray(x, np.float64)
    dy64 = np.asarray(dy, np.float64)
    scale = np.asarray(gamma, np.float64)
    n = x64.shape[-1]
    layer_norm = op_name == "tessera.layer_norm"
    mean = x64.mean(-1, keepdims=True) if layer_norm else 0.0
    var = ((x64 - mean) ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    y_hat = (x64 - mean) * inv
    scaled = dy64 * scale
    centred = scaled.sum(-1, keepdims=True) if layer_norm else 0.0
    dx = (inv / n) * (n * scaled - centred - y_hat * (scaled * y_hat).sum(-1, keepdims=True))
    return (dx,
            (dy64 * y_hat).reshape(-1, n).sum(0),
            dy64.reshape(-1, n).sum(0))


def _storage_arrays(dtype, shape, cols, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    gamma = rng.standard_normal(cols).astype(np.float32)
    dy = rng.standard_normal(shape).astype(np.float32)
    cast = [np.ascontiguousarray(a.astype(dtype)) for a in (x, gamma, dy)]
    return cast


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(not _apple_available(), reason="Apple GPU runtime dylib unavailable")
@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
@pytest.mark.parametrize("op_name", ["tessera.rmsnorm", "tessera.layer_norm"])
def test_low_precision_norm_vjp_accumulates_in_f32(storage, op_name) -> None:
    dtype = np.float16 if storage == "float16" else _bf16()
    if dtype is None:
        pytest.skip("bf16 storage requires the ml_dtypes package")
    rows, cols, eps = 32, 64, 1e-5
    x, gamma, dy = _storage_arrays(dtype, (rows, cols), cols, seed=77)
    layer_norm = op_name == "tessera.layer_norm"
    names = ["x", "gamma", "beta"] if layer_norm else ["x", "gamma"]
    inputs = [x, gamma] + ([np.zeros(cols, dtype)] if layer_norm else [])

    result = execute_native_vjp_family(
        source=_Source(op_name, eps), target="apple_gpu", ordered_inputs=inputs,
        arg_names=names, out_cotangents=dy, wrt_names=names,
    )

    assert result is not None
    assert result.execution["execution_kind"] == "native_gpu"
    # Storage-preserving: an f16 graph keeps f16 gradients rather than acquiring
    # f32 ones the caller would have to cast back.
    for gradient in result.gradients:
        assert np.asarray(gradient).dtype == dtype, (
            f"gradient came back as {np.asarray(gradient).dtype}, not {dtype}"
        )

    expected = _reference(op_name, x.astype(np.float64), gamma.astype(np.float64),
                          dy.astype(np.float64), eps)
    tolerance = _TOLERANCE_ULPS * _STORAGE_EPS[storage]
    for got, want in zip(result.gradients, expected):
        got64 = np.asarray(got, np.float64)
        scale = max(float(np.abs(want).max()), 1e-30)
        relative = float(np.abs(got64 - want).max()) / scale
        assert relative < tolerance, (
            f"{op_name} {storage}: relative error {relative:.2e} exceeds "
            f"{_TOLERANCE_ULPS} ulp ({tolerance:.2e}) — accumulation is not f32"
        )


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(not _apple_available(), reason="Apple GPU runtime dylib unavailable")
@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
def test_low_precision_is_far_better_than_low_precision_accumulation(storage) -> None:
    """Guard the *claim*, not just the number.

    Accumulating a 64-wide row in the storage format would leave an error near
    that format's epsilon; f32 accumulation leaves roughly one rounding. This
    asserts the observed error sits well below the low-precision-accumulation
    level, so a future change that quietly dropped the f32 accumulator would be
    caught rather than absorbed by a loose bound.
    """
    dtype = np.float16 if storage == "float16" else _bf16()
    if dtype is None:
        pytest.skip("bf16 storage requires the ml_dtypes package")
    rows, cols, eps = 64, 128, 1e-5
    x, gamma, dy = _storage_arrays(dtype, (rows, cols), cols, seed=1234)
    result = execute_native_vjp_family(
        source=_Source("tessera.rmsnorm", eps), target="apple_gpu",
        ordered_inputs=[x, gamma], arg_names=["x", "gamma"],
        out_cotangents=dy, wrt_names=["x", "gamma"],
    )
    expected_dx = _reference("tessera.rmsnorm", x.astype(np.float64),
                             gamma.astype(np.float64), dy.astype(np.float64), eps)[0]
    got = np.asarray(result.gradients[0], np.float64)
    relative = float(np.abs(got - expected_dx).max()) / max(float(np.abs(expected_dx).max()), 1e-30)
    # sqrt(cols) ulps is roughly where storage-format accumulation would land.
    accumulation_level = np.sqrt(cols) * _STORAGE_EPS[storage]
    assert relative < 0.5 * accumulation_level, (
        f"{storage}: error {relative:.2e} is near the low-precision accumulation "
        f"level {accumulation_level:.2e}; the f32 accumulator may be gone"
    )
