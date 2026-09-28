"""ODS triage WIRE slice 1 (GOV-ODS-CONSUMER-1): the frontend-emitted Graph
composites reach a consumer.

* ``tessera.target_verify(tokens, logits)`` is rewritten to
  ``tessera.softmax(logits)`` by ``CompositeDecomposition.h`` -- inside
  libtessera_jit here, so ``@jit`` programs calling ``ops.target_verify`` run
  through the MLIR -> LLVM CPU lane (a real native execution, proven by the
  invocation counter) and match the Python reference.
* ``tessera.ntk_rope`` is rewritten to ``tessera.rope(x, theta / scale)``; its
  Tile/Target rows stay un-borrowed (see test_compiler_audit.py) and the
  host-free lit fixtures own the rewrite proof.

The rewrite itself is C++; the lit fixtures
``tests/tessera-ir/phase8/composite_decomposition{,_invalid,_apple_gpu}.mlir``
prove it on every route that runs it. This file owns the executed CPU check and
the Python-side guards of the index-operand path.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import _jit_boundary as jb
from tessera import ops

_RNG = np.random.default_rng(20260927)


def _softmax_ref(logits: np.ndarray) -> np.ndarray:
    lg = logits.astype(np.float64)
    e = np.exp(lg - lg.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@ts.jit
def _target_verify(tokens, logits):
    return ops.target_verify(tokens, logits)


@ts.jit
def _target_verify_then_scale(tokens, logits, w):
    p = ops.target_verify(tokens, logits)
    return ops.mul(p, w)


@ts.jit
def _ntk_rope(x, theta):
    return ops.ntk_rope(x, theta, scale=2.0)


def test_frontend_emits_both_composites():
    """The producer half: @jit emits the Graph ops the rewrite consumes."""
    tv_ir = _target_verify.ir_text()
    nr_ir = _ntk_rope.ir_text()
    assert "tessera.target_verify" in tv_ir
    assert "tessera.ntk_rope" in nr_ir and "scale = 2.0" in nr_ir


@pytest.mark.skipif(not jb.is_available(), reason="libtessera_jit not built")
@pytest.mark.parametrize("S,V", [(1, 7), (3, 8), (5, 129)])
def test_target_verify_executes_through_cpu_jit(S, V):
    tokens = np.arange(S, dtype=np.int32)
    logits = _RNG.standard_normal((S, V)).astype(np.float32)
    n0 = jb.invocation_count()
    got = np.asarray(_target_verify(tokens, logits))
    assert jb.invocation_count() - n0 == 1, "target_verify fell back to numpy"
    assert _target_verify.last_fallback_reason is None
    np.testing.assert_allclose(got, _softmax_ref(logits), rtol=1e-5, atol=1e-6)
    ref = ops.target_verify(tokens, logits)  # eager call: the Python reference
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not jb.is_available(), reason="libtessera_jit not built")
def test_target_verify_composes_with_graph_ops_in_one_compile():
    tokens = np.arange(4, dtype=np.int32)
    logits = _RNG.standard_normal((4, 16)).astype(np.float32)
    w = _RNG.standard_normal((4, 16)).astype(np.float32)
    n0 = jb.invocation_count()
    got = np.asarray(_target_verify_then_scale(tokens, logits, w))
    assert jb.invocation_count() - n0 == 1
    np.testing.assert_allclose(got, _softmax_ref(logits) * w, rtol=1e-5, atol=1e-6)


def test_graphfn_refuses_index_operand_outside_its_declaring_op():
    """An i32 argument may only reach the operand an op declares as an index
    operand; anywhere else the graph is refused (so @jit falls back to numpy
    instead of computing on reinterpreted integer bits)."""
    g = jb.GraphFn()
    t = g.arg((3,), elem="i32")
    x = g.arg((3,))
    with pytest.raises(jb.TesseraJitError, match="declared index operand"):
        g.add(t, x)
    with pytest.raises(jb.TesseraJitError, match="rank-1 i32"):
        g.target_verify(x, g.arg((3, 4)))
    with pytest.raises(jb.TesseraJitError, match="S = len"):
        g.target_verify(t, g.arg((2, 4)))


def test_graphfn_index_arguments_are_cpu_lane_only():
    g = jb.GraphFn(target="apple_gpu")
    with pytest.raises(jb.TesseraJitError, match="CPU-lane only"):
        g.arg((3,), elem="i32")


@pytest.mark.skipif(not jb.is_available(), reason="libtessera_jit not built")
def test_int32_argument_to_a_non_index_op_falls_back_without_jit():
    @ts.jit
    def _bad(a, b):
        return ops.add(a, b)

    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    b = _RNG.standard_normal((2, 3)).astype(np.float32)
    n0 = jb.invocation_count()
    try:
        _bad(a, b)
    except Exception:  # noqa: BLE001 -- the reference may refuse mixed dtypes
        pass
    assert jb.invocation_count() == n0, "an i32 operand of add must not reach the JIT lane"
