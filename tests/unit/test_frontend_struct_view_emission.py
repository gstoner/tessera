"""P1 frontend emission (S_SERIES_GAP_CLOSURE_PLAN §6.B) — a ``@jit`` body that
uses the 6 structural view ops (squeeze / unsqueeze / expand / broadcast /
permute / flatten) must EMIT the corresponding Graph IR op so it enters the
device_verified_jit pipeline, instead of dropping the call to a bogus ``%?`` operand.

Before this fix the structural ops resolved through the generic call mapper but
their positional axes/perm/shape argument was dropped to a ``%?`` operand —
which violates the P1a ODS contract (those ops take only ``TensorType:$x``).
The fix binds the trailing positional arg as a discardable attribute via
``_POSITIONAL_ATTR_PARAMS`` using the canonical lit-fixture attr vocabulary
(``axes`` / ``perm`` / ``shape`` / ``start``+``end``).
"""

from __future__ import annotations

import pytest

from tessera import ops
from tessera.compiler.graph_ir import GraphIRBuilder


# Module-level so inspect.getsource() can retrieve the body for AST lowering.
def _squeeze_fn(x):
    return ops.squeeze(x, (0, 2))


def _unsqueeze_fn(x):
    return ops.unsqueeze(x, 0)


def _permute_fn(x):
    return ops.permute(x, (2, 0, 1))


def _expand_fn(x):
    return ops.expand(x, (5, 3, 4))


def _broadcast_fn(x):
    return ops.broadcast(x, (20, 3))


def _flatten_fn(x):
    return ops.flatten(x, 0, 1)


def _reshape_fn(x):
    return ops.reshape(x, (6, 4))


def _view_fn(x):
    return ops.view(x, (24,))


def _chain_fn(x):
    a = ops.squeeze(x, (0, 2))
    b = ops.unsqueeze(a, 0)
    c = ops.permute(b, (2, 0, 1))
    d = ops.expand(c, (5, 3, 4))
    e = ops.flatten(d, 0, 1)
    return ops.broadcast(e, (20, 3))


def _emit(fn) -> str:
    b = GraphIRBuilder()
    b.lower(fn)
    module = b.module()
    # Internal verifier must accept the emitted module.
    assert module.verify().ok, "emitted structural-view IR failed verification"
    return module.to_mlir(verify=False)


def _op_line(mlir: str, op_name: str) -> str:
    for line in mlir.splitlines():
        if op_name in line:
            return line.strip()
    raise AssertionError(f"{op_name} not emitted:\n{mlir}")


def test_squeeze_emits_axes_attr():
    line = _op_line(_emit(_squeeze_fn), "tessera.squeeze")
    assert "axes = [0, 2]" in line, line
    # exactly one SSA operand, no dropped "%?"
    assert "%?" not in line
    assert line.count("%") == 2  # result + single operand


def test_unsqueeze_emits_axes_attr():
    line = _op_line(_emit(_unsqueeze_fn), "tessera.unsqueeze")
    assert "axes = 0" in line and "%?" not in line, line


def test_permute_emits_perm_attr():
    # `perm` is the spelling the C++ PermuteOp::fold reads.
    line = _op_line(_emit(_permute_fn), "tessera.permute")
    assert "perm = [2, 0, 1]" in line and "%?" not in line, line


def test_expand_emits_shape_attr():
    line = _op_line(_emit(_expand_fn), "tessera.expand")
    assert "shape = [5, 3, 4]" in line and "%?" not in line, line


def test_broadcast_emits_shape_attr():
    line = _op_line(_emit(_broadcast_fn), "tessera.broadcast")
    assert "shape = [20, 3]" in line and "%?" not in line, line


def test_flatten_emits_start_end_attrs():
    line = _op_line(_emit(_flatten_fn), "tessera.flatten")
    assert "start = 0" in line and "end = 1" in line and "%?" not in line, line


def test_reshape_emits_shape_attr():
    line = _op_line(_emit(_reshape_fn), "tessera.reshape")
    assert "shape = [6, 4]" in line and "%?" not in line, line


def test_view_emits_shape_attr():
    line = _op_line(_emit(_view_fn), "tessera.view")
    assert "shape = [24]" in line and "%?" not in line, line


def test_chain_emits_all_six_with_no_bogus_operand():
    mlir = _emit(_chain_fn)
    for op in ("squeeze", "unsqueeze", "permute", "expand", "flatten",
               "broadcast"):
        assert f"tessera.{op}" in mlir, f"missing tessera.{op}:\n{mlir}"
    # The whole module must be free of dropped operands.
    assert "%?" not in mlir, mlir


def _static_reshape_fn(x: "tensor<2x3x4xf32>"):
    a = ops.reshape(x, (6, 4))
    return ops.view(a, (24,))


def _static_squeeze_fn(x: "tensor<1x3x1x4xf32>"):
    return ops.squeeze(x, (0, 2))


def _bad_static_squeeze_fn(x: "tensor<2x3xf32>"):
    return ops.squeeze(x, (0,))


def _bad_static_permute_fn(x: "tensor<2x3xf32>"):
    return ops.permute(x, (2, 0))


def test_static_shape_result_types_derived_from_attr():
    """The result type must come from the shape/axes attr, NOT the input type —
    otherwise a static reshape emits `<in> -> <in>` and the identity folder
    erases it (silent miscompile). Regression for PR #202 review."""
    mlir = _emit(_static_reshape_fn)
    rl = _op_line(mlir, "tessera.reshape")
    assert "-> tensor<6x4xf32>" in rl, rl
    assert "tensor<2x3x4xf32> -> tensor<2x3x4xf32>" not in rl, rl
    vl = _op_line(mlir, "tessera.view")
    assert "-> tensor<24xf32>" in vl, vl
    # squeeze on static 1x3x1x4 with axes [0,2] -> 3x4 (not the input type)
    sl = _op_line(_emit(_static_squeeze_fn), "tessera.squeeze")
    assert "-> tensor<3x4xf32>" in sl, sl


def test_static_squeeze_rejects_non_unit_axis():
    with pytest.raises(ValueError, match="size-1 dimensions"):
        _emit(_bad_static_squeeze_fn)


def test_static_permute_rejects_non_permutation():
    with pytest.raises(ValueError, match="full permutation"):
        _emit(_bad_static_permute_fn)
