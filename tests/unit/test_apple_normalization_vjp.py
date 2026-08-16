"""Apple GPU normalization VJP — the E2E-REAL-6 / WS-2 native reverse product.

Apple is the fourth target to register in the native-VJP boundary. The oracle
here is derived independently in float64 from the definitions in
``autodiff/vjp.py`` rather than compared against another kernel, because a
kernel checked against itself proves nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.native_vjp_plugins import (
    execute_native_vjp_family,
    native_vjp_plugin_declarations,
)


def _apple_available() -> bool:
    from tessera.compiler import apple_native

    return apple_native.tools_available()


class _Source:
    """The minimal Graph-op view the native-VJP plugin consumes."""

    def __init__(self, op_name: str, eps: float = 1e-5) -> None:
        self.op_name = op_name
        self.result = "y"
        self.kwargs = {"eps": eps}


def _reference(op_name: str, x, gamma, dy, eps: float):
    """Analytic VJP in float64, matching `autodiff/vjp.py`.

    RMSNorm does not centre, so its `d(mean)/dx` term is absent — that asymmetry
    is the thing most easily got wrong, so both forms are spelled out rather
    than sharing a parameterised expression.
    """
    x64, dy64 = np.asarray(x, np.float64), np.asarray(dy, np.float64)
    n = x64.shape[-1]
    scale = np.ones(n) if gamma is None else np.asarray(gamma, np.float64)
    layer_norm = op_name == "tessera.layer_norm"
    mean = x64.mean(-1, keepdims=True) if layer_norm else 0.0
    var = ((x64 - mean) ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    y_hat = (x64 - mean) * inv
    scaled = dy64 * scale
    centred = scaled.sum(-1, keepdims=True) if layer_norm else 0.0
    dx = (inv / n) * (n * scaled - centred - y_hat * (scaled * y_hat).sum(-1, keepdims=True))
    # The affine gradients reduce over EVERY leading axis, not just the first:
    # the kernel sees `rows = prod(shape[:-1])`. Writing `.sum(0)` here passes at
    # rank 2 and is silently wrong at rank 3, which is why a rank-3 shape is in
    # the parametrization.
    rows_first = (dy64 * y_hat).reshape(-1, n)
    return dx, rows_first.sum(0), dy64.reshape(-1, n).sum(0)


def test_apple_gpu_is_registered_for_every_normalization_op() -> None:
    """All three registry ops must name a concrete Apple Target consumer."""
    declarations = native_vjp_plugin_declarations()
    for op in ("rmsnorm", "rmsnorm_safe", "layer_norm"):
        consumers = declarations[op].target_consumers
        assert consumers.get("apple_gpu") == "apple.metal_normalization", (
            f"{op} has no Apple Target consumer"
        )


def test_apple_normalization_vjp_rows_are_registered_in_the_execution_matrix() -> None:
    from tessera.compiler.execution_matrix import KNOWN_EXECUTORS, _MATRIX

    for path in ("apple_gpu_rmsnorm_bwd_compiled", "apple_gpu_layer_norm_bwd_compiled"):
        row = _MATRIX[("apple_gpu", path)]
        assert row.executor_id == "apple_gpu_norm_bwd_compiled"
        assert row.executor_id in KNOWN_EXECUTORS
        assert row.execution_kind == "native_gpu"
        assert row.direction == "backward"
    # `rmsnorm_safe` differentiates identically and rides the rmsnorm row.
    assert "rmsnorm_safe" in _MATRIX[
        ("apple_gpu", "apple_gpu_rmsnorm_bwd_compiled")
    ].backward_aliases


def test_normalization_bwd_symbols_gate_dylib_freshness() -> None:
    """A prebuilt dylib without these exports must be rejected, not accepted.

    Without this, a runtime built before the VJP landed clears the older
    staleness sentinel, gets loaded, and then fails at the call site with a
    missing symbol — which reads as a broken consumer rather than the stale
    dylib it is. Both loaders gate on the symbols, so both are checked.
    """
    from pathlib import Path

    from tessera import _apple_gpu_dispatch as agd
    from tessera import runtime as R

    legacy = Path(R.__file__).read_text(encoding="utf-8")
    # EVERY storage variant must gate, not just f32. A dylib built between the
    # f32 landing and the low-precision one exports the f32 names, so probing
    # only those would accept it as current and then fail at the call site.
    for family in ("rmsnorm", "layer_norm"):
        for storage in ("f32", "f16", "bf16"):
            symbol = f"tessera_apple_gpu_{family}_bwd_{storage}"
            assert symbol in agd._SENTINEL_SYMBOLS, symbol
            assert symbol in agd.APPLE_ABI, f"{symbol} is not in the canonical ABI registry"
            assert f'getattr(lib, "{symbol}")' in legacy, (
                f"{symbol} does not gate runtime.py's legacy candidate scan")


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(not _apple_available(), reason="Apple GPU runtime dylib unavailable")
@pytest.mark.parametrize("op_name", ["tessera.rmsnorm", "tessera.rmsnorm_safe",
                                     "tessera.layer_norm"])
@pytest.mark.parametrize("shape", [(16, 32), (4, 8, 64), (5, 13)])
def test_apple_normalization_vjp_matches_the_analytic_reference(op_name, shape) -> None:
    """Every registered op, affine, against an independent float64 VJP.

    A ragged width (13) is included deliberately: the column reduction is
    one thread per channel, so a width that is not a multiple of the
    threadgroup size is the case a tidy power-of-two shape would hide.
    """
    eps = 1e-5
    cols = shape[-1]
    rng = np.random.default_rng(4242)
    x = np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
    gamma = np.ascontiguousarray(rng.standard_normal(cols), dtype=np.float32)
    dy = np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
    layer_norm = op_name == "tessera.layer_norm"
    names = ["x", "gamma", "beta"] if layer_norm else ["x", "gamma"]
    inputs = [x, gamma] + ([np.zeros(cols, np.float32)] if layer_norm else [])

    result = execute_native_vjp_family(
        source=_Source(op_name, eps), target="apple_gpu", ordered_inputs=inputs,
        arg_names=names, out_cotangents=dy, wrt_names=names,
    )

    assert result is not None, "Apple has no registered consumer for this op"
    # Placement is proven by a status-returning ABI, so a numerically correct
    # CPU fallback cannot pass this.
    assert result.execution["execution_kind"] == "native_gpu"
    assert result.execution["target_consumer"] == "apple.metal_normalization"
    assert result.execution["evidence_target"] == "apple7"

    expected_dx, expected_dg, expected_db = _reference(op_name, x, gamma, dy, eps)
    np.testing.assert_allclose(result.gradients[0], expected_dx, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(result.gradients[1], expected_dg, rtol=2e-4, atol=2e-5)
    if layer_norm:
        np.testing.assert_allclose(result.gradients[2], expected_db, rtol=2e-4, atol=2e-5)


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(not _apple_available(), reason="Apple GPU runtime dylib unavailable")
def test_apple_normalization_vjp_affine_gradients_are_deterministic() -> None:
    """dGamma/dBeta reduce over rows in fixed order, so repeats are bit-identical."""
    rng = np.random.default_rng(99)
    x = np.ascontiguousarray(rng.standard_normal((64, 128)), dtype=np.float32)
    gamma = np.ascontiguousarray(rng.standard_normal(128), dtype=np.float32)
    beta = np.zeros(128, np.float32)
    dy = np.ascontiguousarray(rng.standard_normal((64, 128)), dtype=np.float32)
    runs = []
    for _ in range(3):
        result = execute_native_vjp_family(
            source=_Source("tessera.layer_norm"), target="apple_gpu",
            ordered_inputs=[x, gamma, beta], arg_names=["x", "gamma", "beta"],
            out_cotangents=dy, wrt_names=["x", "gamma", "beta"],
        )
        runs.append(tuple(np.asarray(g).copy() for g in result.gradients))
    for later in runs[1:]:
        for first, repeat in zip(runs[0], later):
            assert np.array_equal(first, repeat)
