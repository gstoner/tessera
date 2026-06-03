"""PK8a — Graph IR → ``.mtlpackage`` authoring hook.

Two layers:

* **recognition** (pure, no GPU): a real ``GraphIRModule`` maps to the right
  :class:`AuthorPlan` — single ops, matmul, and the fused chains.
* **author + dispatch** (gated on packaged ML): the recognized plan authors a
  package that loads + dispatches through PK1-PK7 and matches numpy.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from tessera.apple_mlpkg import (
    compile_mlpackage,
    first_function_name,
    packaged_ml_available,
    packaged_ml_skip_reason,
)
from tessera.compiler.apple_package_author import (
    AuthorPlan,
    author_package_from_graph_ir,
    recognize,
)
from tessera.compiler.graph_ir import (
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
)


def _t(rows, cols):
    return IRType(f"tensor<{rows}x{cols}xf32>", (str(rows), str(cols)), "fp32")


def _t1(n):
    return IRType(f"tensor<{n}xf32>", (str(n),), "fp32")


def _module(args, body, ret):
    fn = GraphIRFunction(
        name="f", args=args,
        result_types=[body[-1].result_type], body=body, return_values=ret,
    )
    return GraphIRModule(functions=[fn])


def _matmul(res, a_t, b_t, out_t, ops=("%a", "%b")):
    return IROp(result=res, op_name="tessera.matmul", operands=list(ops),
                operand_types=[a_t.mlir_str, b_t.mlir_str],
                result_type=out_t.mlir_str)


# ── recognition (pure) ───────────────────────────────────────────────────


def test_recognize_single_matmul():
    m = _module(
        [IRArg("a", _t(4, 6)), IRArg("b", _t(6, 5))],
        [_matmul("c", _t(4, 6), _t(6, 5), _t(4, 5))],
        ["%c"],
    )
    plan = recognize(m)
    assert plan == AuthorPlan(kind="matmul", name="matmul", dims=(4, 6, 5))


def test_recognize_single_softmax():
    m = _module(
        [IRArg("x", _t(8, 16))],
        [IROp(result="y", op_name="tessera.softmax", operands=["%x"],
              operand_types=[_t(8, 16).mlir_str], result_type=_t(8, 16).mlir_str)],
        ["%y"],
    )
    plan = recognize(m)
    assert plan == AuthorPlan(kind="op", name="softmax", dims=(8, 16))


def test_recognize_rmsnorm_weighted_from_operand_count():
    m = _module(
        [IRArg("x", _t(4, 8)), IRArg("g", _t1(8))],
        [IROp(result="y", op_name="tessera.rmsnorm", operands=["%x", "%g"],
              operand_types=[_t(4, 8).mlir_str, _t1(8).mlir_str],
              result_type=_t(4, 8).mlir_str, kwargs={"eps": 1e-6})],
        ["%y"],
    )
    plan = recognize(m)
    assert plan is not None
    assert plan.kind == "op" and plan.name == "rmsnorm"
    assert plan.dims == (4, 8) and plan.weighted is True
    assert plan.eps == pytest.approx(1e-6)


def test_recognize_matmul_softmax_chain():
    m = _module(
        [IRArg("a", _t(4, 6)), IRArg("b", _t(6, 5))],
        [
            _matmul("c", _t(4, 6), _t(6, 5), _t(4, 5)),
            IROp(result="y", op_name="tessera.softmax", operands=["%c"],
                 operand_types=[_t(4, 5).mlir_str], result_type=_t(4, 5).mlir_str),
        ],
        ["%y"],
    )
    plan = recognize(m)
    assert plan == AuthorPlan(kind="chain", name="matmul_softmax",
                              dims=(4, 6, 5))


def test_recognize_attention_block_chain():
    m = _module(
        [IRArg("a", _t(4, 6)), IRArg("b", _t(6, 5)), IRArg("c", _t(5, 3))],
        [
            _matmul("s", _t(4, 6), _t(6, 5), _t(4, 5)),
            IROp(result="p", op_name="tessera.softmax", operands=["%s"],
                 operand_types=[_t(4, 5).mlir_str], result_type=_t(4, 5).mlir_str),
            _matmul("y", _t(4, 5), _t(5, 3), _t(4, 3), ops=("%p", "%c")),
        ],
        ["%y"],
    )
    plan = recognize(m)
    assert plan == AuthorPlan(kind="chain", name="matmul_softmax_matmul",
                              dims=(4, 6, 5, 3))


def test_recognize_rmsnorm_matmul_chain():
    m = _module(
        [IRArg("x", _t(4, 6)), IRArg("g", _t1(6)), IRArg("w", _t(6, 5))],
        [
            IROp(result="n", op_name="tessera.rmsnorm", operands=["%x", "%g"],
                 operand_types=[_t(4, 6).mlir_str, _t1(6).mlir_str],
                 result_type=_t(4, 6).mlir_str),
            _matmul("y", _t(4, 6), _t(6, 5), _t(4, 5), ops=("%n", "%w")),
        ],
        ["%y"],
    )
    plan = recognize(m)
    assert plan == AuthorPlan(kind="chain", name="rmsnorm_matmul",
                              dims=(4, 6, 5))


def test_rmsnorm_matmul_rejects_mismatched_gamma():
    # Review glass-jaw R5: gamma must be [K] with K = x[:, K] = W[K, :].
    # Here x is 4x6 (K=6) and W is 6x5, but gamma is [5] — must be rejected,
    # not silently packaged with an unbindable norm argument.
    m = _module(
        [IRArg("x", _t(4, 6)), IRArg("g", _t1(5)), IRArg("w", _t(6, 5))],
        [
            IROp(result="n", op_name="tessera.rmsnorm", operands=["%x", "%g"],
                 operand_types=[_t(4, 6).mlir_str, _t1(5).mlir_str],
                 result_type=_t(4, 6).mlir_str),
            _matmul("y", _t(4, 6), _t(6, 5), _t(4, 5), ops=("%n", "%w")),
        ],
        ["%y"],
    )
    assert recognize(m) is None


def test_weighted_rmsnorm_rejects_mismatched_gamma():
    # Standalone weighted rmsnorm: x is 4x8 (K=8); gamma [5] must be rejected.
    m = _module(
        [IRArg("x", _t(4, 8)), IRArg("g", _t1(5))],
        [IROp(result="y", op_name="tessera.rmsnorm", operands=["%x", "%g"],
              operand_types=[_t(4, 8).mlir_str, _t1(5).mlir_str],
              result_type=_t(4, 8).mlir_str)],
        ["%y"],
    )
    assert recognize(m) is None


def test_recognize_rejects_dynamic_shapes():
    dyn = IRType("tensor<?x8xf32>", ("?", "8"), "fp32")
    m = _module(
        [IRArg("x", dyn)],
        [IROp(result="y", op_name="tessera.softmax", operands=["%x"],
              operand_types=[dyn.mlir_str], result_type=dyn.mlir_str)],
        ["%y"],
    )
    assert recognize(m) is None


def test_recognize_rejects_unknown_op():
    m = _module(
        [IRArg("x", _t(4, 4))],
        [IROp(result="y", op_name="tessera.mystery_op", operands=["%x"],
              operand_types=[_t(4, 4).mlir_str], result_type=_t(4, 4).mlir_str)],
        ["%y"],
    )
    assert recognize(m) is None


def test_recognize_rejects_non_fp32():
    bf = IRType("tensor<4x4xbf16>", ("4", "4"), "bf16")
    m = _module(
        [IRArg("x", bf)],
        [IROp(result="y", op_name="tessera.relu", operands=["%x"],
              operand_types=[bf.mlir_str], result_type=bf.mlir_str)],
        ["%y"],
    )
    assert recognize(m) is None


# ── author + dispatch from Graph IR (gated) ──────────────────────────────


def _require_packaged_ml():
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")


def _dispatch(pkg, inputs, out_shape):
    fn = first_function_name(pkg) or "main"
    pipe = compile_mlpackage(pkg, function_name=fn)
    assert pipe is not None
    try:
        assert pipe.prepare_tensors()
        for i, arr in enumerate(inputs):
            assert pipe.fill_input_at(i, arr.astype(np.float32).tobytes())
        assert pipe.dispatch(timeout_ms=30_000)
        r, c = out_shape
        raw = pipe.read_output_at(len(inputs), r * c * 4)
        return np.frombuffer(raw, dtype=np.float32).reshape(r, c)
    finally:
        pipe.destroy()


def test_author_single_op_from_graph_ir_dispatches():
    """A recognized single-op module authors + dispatches matching numpy."""
    _require_packaged_ml()
    m = _module(
        [IRArg("x", _t(4, 8))],
        [IROp(result="y", op_name="tessera.silu", operands=["%x"],
              operand_types=[_t(4, 8).mlir_str], result_type=_t(4, 8).mlir_str)],
        ["%y"],
    )
    d = tempfile.mkdtemp(prefix="pk8a_")
    pkg = os.path.join(d, "silu.mtlpackage")
    plan = author_package_from_graph_ir(m, pkg)
    assert plan is not None and plan.name == "silu"
    rng = np.random.default_rng(30)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    out = _dispatch(pkg, [x], (4, 8))
    ref = x * (1.0 / (1.0 + np.exp(-x)))
    assert np.allclose(out, ref, rtol=1e-4, atol=2e-4)


def test_author_matmul_softmax_chain_from_graph_ir_dispatches():
    """The flagship: a matmul→softmax Graph IR region becomes a fused
    packaged kernel and dispatches bitwise-close to numpy."""
    _require_packaged_ml()
    M, K, N = 4, 6, 5
    m = _module(
        [IRArg("a", _t(M, K)), IRArg("b", _t(K, N))],
        [
            _matmul("c", _t(M, K), _t(K, N), _t(M, N)),
            IROp(result="y", op_name="tessera.softmax", operands=["%c"],
                 operand_types=[_t(M, N).mlir_str], result_type=_t(M, N).mlir_str),
        ],
        ["%y"],
    )
    d = tempfile.mkdtemp(prefix="pk8a_")
    pkg = os.path.join(d, "ms.mtlpackage")
    plan = author_package_from_graph_ir(m, pkg)
    assert plan is not None and plan.name == "matmul_softmax"
    rng = np.random.default_rng(31)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    out = _dispatch(pkg, [a, b], (M, N))
    e = np.exp((a @ b) - (a @ b).max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(out, ref, rtol=1e-4, atol=2e-4)


def test_author_returns_none_for_unrecognized():
    """Unrecognized region → None (caller keeps the normal lowering path).
    Pure path, no device needed."""
    m = _module(
        [IRArg("x", _t(4, 4))],
        [IROp(result="y", op_name="tessera.mystery_op", operands=["%x"],
              operand_types=[_t(4, 4).mlir_str], result_type=_t(4, 4).mlir_str)],
        ["%y"],
    )
    d = tempfile.mkdtemp(prefix="pk8a_")
    assert author_package_from_graph_ir(m, os.path.join(d, "x.mtlpackage")) is None
