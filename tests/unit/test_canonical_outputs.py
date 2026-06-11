"""Graph outputs in canonical compile metadata (Next Work #1 remainder).

`CompileResult.outputs` / `canonical_outputs` surfaces the program's returned
values — each with its producer op + type/shape/dtype — derived from the now-
populated GraphIRFunction.return_values (the jit AST path previously emitted a
value-less `return`, so outputs were empty).
"""

from __future__ import annotations

import numpy as np

import tessera as ts
from tessera.compiler.graph_ir import GraphIRBuilder


def _outputs_meta(fn):
    return fn.runtime_artifact().metadata.get("canonical_outputs") or {}


def test_return_values_now_populated_in_graph_ir():
    builder = GraphIRBuilder()

    def f(a, b):
        return ts.ops.add(ts.ops.matmul(a, b), b)

    builder.lower(f)
    gfn = builder.module().functions[-1]
    assert gfn.return_values, "jit AST function should declare its return value(s)"
    assert len(gfn.return_values) == len(gfn.result_types)


def test_canonical_outputs_in_jit_runtime_artifact():
    @ts.jit(target="apple_cpu")
    def f(a, b):
        return ts.ops.add(ts.ops.matmul(a, b), b)

    A = np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)
    f(A, A.copy())
    co = _outputs_meta(f)
    assert co.get("schema") == "tessera.compile.outputs.v1"
    po = co.get("program_outputs", [])
    assert len(po) == 1
    out = po[0]
    # the program returns the `add` op's result
    assert out["producer"] == "add"
    assert set(out) >= {"index", "name", "producer", "dtype", "shape"}


def test_tuple_return_yields_multiple_outputs():
    builder = GraphIRBuilder()

    def f(a, b):
        return ts.ops.matmul(a, b), ts.ops.add(a, b)

    builder.lower(f)
    gfn = builder.module().functions[-1]
    assert len(gfn.return_values) == 2
    assert len(gfn.result_types) == 2


def test_statically_typed_outputs_carry_shape_dtype():
    @ts.jit(target="apple_cpu")
    def f(a: "tensor<4x8xf32>", b: "tensor<8x4xf32>"):
        return ts.ops.matmul(a, b)

    A = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32)
    B = np.random.default_rng(2).standard_normal((8, 4)).astype(np.float32)
    f(A, B)
    po = _outputs_meta(f).get("program_outputs", [])
    assert len(po) == 1
    # static annotations flow to a concrete output shape/dtype.
    assert po[0]["producer"] == "matmul"
    assert po[0]["shape"] == ["4", "4"]
    assert po[0]["dtype"] == "fp32"
    assert po[0]["type"] == "tensor<4x4xf32>"
