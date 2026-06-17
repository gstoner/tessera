"""Apple GPU concat — the third structural data-mover on a real MPSGraph kernel.

tessera.cat now runs on Metal via concatTensors:dimension: (was the numpy
fallback): two operands stacked along one axis — the KV-cache-append case. v1
envelope is a 2-operand concat flattened to an (outer, axis, inner) view (any
rank / axis); >2 operands or mixed dtypes fall back to np.concatenate inside the
dispatcher. Each data-mover added here widens what the general per_op_metal
residency gate accepts (a KV append mid-program now stays GPU-resident).

Landing this also required two frontend fixes shared with `stack`: the AST
GraphIRBuilder and the abstract-interp tracer both now expand a list/tuple of
tensor operands (``cat([a, b], axis)``) into flat operands instead of dropping
the op, and the op-catalog arity for cat/stack widened from fixed-1 to variadic.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import apple_gpu_envelope as env

DARWIN = sys.platform == "darwin"
_RNG = np.random.default_rng(20260617)


def test_concat_is_a_first_class_gpu_op():
    assert env.lane_for("tessera.cat") == "concat"
    assert "tessera.cat" in env.runtime_ops()


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_concat_handler_matches_numpy(axis):
    from tessera.runtime import _apple_gpu_dispatch_concat
    a = _RNG.standard_normal((3, 4)).astype(np.float32)
    b = _RNG.standard_normal((3, 4) if axis in (1, -1) else (2, 4)).astype(np.float32)
    out = _apple_gpu_dispatch_concat("tessera.cat", [a, b], {"axis": axis}, np)
    np.testing.assert_array_equal(np.asarray(out), np.concatenate([a, b], axis=axis))


def test_concat_rank3_seq_axis_matches_numpy():
    """KV-cache-append shape: (B, S, D) concatenated along the sequence axis."""
    from tessera.runtime import _apple_gpu_dispatch_concat
    cache = _RNG.standard_normal((2, 5, 8)).astype(np.float32)
    new = _RNG.standard_normal((2, 1, 8)).astype(np.float32)
    out = _apple_gpu_dispatch_concat("tessera.cat", [cache, new], {"axis": 1}, np)
    np.testing.assert_array_equal(np.asarray(out), np.concatenate([cache, new], axis=1))


def test_concat_f16_matches_numpy():
    from tessera.runtime import _apple_gpu_dispatch_concat
    a = _RNG.standard_normal((4, 5)).astype(np.float16)
    b = _RNG.standard_normal((3, 5)).astype(np.float16)
    out = _apple_gpu_dispatch_concat("tessera.cat", [a, b], {"axis": 0}, np)
    np.testing.assert_array_equal(np.asarray(out), np.concatenate([a, b], axis=0))


def test_concat_three_operands_falls_back_correctly():
    """>2 operands take the host path inside the dispatcher — still correct."""
    from tessera.runtime import _apple_gpu_dispatch_concat
    xs = [_RNG.standard_normal((2, 4)).astype(np.float32) for _ in range(3)]
    out = _apple_gpu_dispatch_concat("tessera.cat", xs, {"axis": 0}, np)
    np.testing.assert_array_equal(np.asarray(out), np.concatenate(xs, axis=0))


def test_jit_concat_matches_numpy():
    @ts.jit(target="apple_gpu")
    def f(a, b):
        return ts.ops.cat([a, b], axis=0)

    a = _RNG.standard_normal((2, 6)).astype(np.float32)
    b = _RNG.standard_normal((3, 6)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(f(a, b)), np.concatenate([a, b], 0),
                               rtol=1e-6, atol=1e-6)


def test_jit_concat_is_native_gpu():
    """The whole jitted cat program is GPU-resident (single in-envelope op)."""
    @ts.jit(target="apple_gpu")
    def f(a, b):
        return ts.ops.cat([a, b], axis=1)

    a = _RNG.standard_normal((4, 2)).astype(np.float32)
    b = _RNG.standard_normal((4, 5)).astype(np.float32)
    f(a, b)
    assert f.execution_kind == "native_gpu"


@pytest.mark.skipif(not DARWIN, reason="GPU-residency check requires Metal (Darwin)")
def test_concat_runs_on_metal_no_fallback():
    from tessera.compiler import apple_gpu_coverage as cov

    @ts.jit(target="apple_gpu")
    def f(a, b):
        return ts.ops.cat([a, b], axis=0)

    a = _RNG.standard_normal((2, 8)).astype(np.float32)
    b = _RNG.standard_normal((3, 8)).astype(np.float32)
    assert cov.fallback_histogram(lambda: f(a, b)) == {}


@pytest.mark.skipif(not DARWIN, reason="GPU-residency check requires Metal (Darwin)")
def test_concat_compounds_with_matmul_per_op_metal():
    """matmul -> cat: the data-mover mid-program keeps the chain GPU-resident."""
    from tessera.compiler import apple_gpu_coverage as cov

    @ts.jit(target="apple_gpu")
    def chain(x, w, tail):
        return ts.ops.cat([ts.ops.matmul(x, w), tail], axis=0)

    x = _RNG.standard_normal((3, 4)).astype(np.float32)
    w = _RNG.standard_normal((4, 5)).astype(np.float32)
    tail = _RNG.standard_normal((2, 5)).astype(np.float32)
    out = np.asarray(chain(x, w, tail))
    np.testing.assert_allclose(out, np.concatenate([x @ w, tail], 0), rtol=1e-4, atol=1e-4)
    assert chain.execution_kind == "native_gpu"
    assert cov.fallback_histogram(lambda: chain(x, w, tail)) == {}
