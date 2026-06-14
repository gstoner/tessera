"""M5.1 — the model-class quant pillars promoted to first-class catalog ops +
drift-gated coverage rows.

Pins that `dequant_matmul` / `dequant_grouped_gemm` are (a) in the op catalog,
(b) callable via `tessera.ops`, (c) carry a hardened coverage contract with
registered autodiff and an honest `backend_kernel=partial` (composed GPU, no
fused MSL kernel yet).
"""

from __future__ import annotations

import numpy as np

import tessera
from tessera.compiler import primitive_coverage as pc
from tessera.compiler.op_catalog import OP_SPECS
from tessera.stdlib import quant as q

NEW_OPS = ("dequant_matmul", "dequant_grouped_gemm")


def test_in_op_catalog():
    for name in NEW_OPS:
        assert name in OP_SPECS
        assert OP_SPECS[name].graph_name == f"tessera.{name}"
        assert OP_SPECS[name].lowering == "loop_nest"


def test_coverage_rows_hardened():
    allc = pc.all_primitive_coverages()
    for name in NEW_OPS:
        e = allc[name]
        assert e.category == "quantization"
        assert e.graph_name == f"tessera.{name}"
        cs = e.contract_status
        for axis in ("math_semantics", "shape_rule", "dtype_layout_rule",
                     "lowering_rule", "tests"):
            assert cs[axis] == "complete", f"{name}.{axis}={cs[axis]}"
        # honest: composed Apple GPU matmul lane, fused dequant MSL kernel pending
        assert cs["backend_kernel"] == "partial"
        # numeric_policy (fp32 accumulate) attached
        assert e.metadata["numeric_policy"]["accum"] == "fp32"
        assert e.metadata.get("contract_schema") == "explicit_semantic"


def test_dequant_matmul_autodiff_registered():
    from tessera.autodiff.vjp import _VJPS
    from tessera.autodiff.jvp import _JVPS
    assert "dequant_matmul" in _VJPS
    assert "dequant_matmul" in _JVPS
    allc = pc.all_primitive_coverages()
    assert allc["dequant_matmul"].contract_status["vjp"] == "complete"
    assert allc["dequant_matmul"].contract_status["jvp"] == "complete"


def test_dequant_grouped_gemm_grouped_layout_attached():
    e = pc.all_primitive_coverages()["dequant_grouped_gemm"]
    assert e.metadata["grouped_layout"]["kind"] == "contiguous"
    # structural grouped op — VJP/JVP are not a single-array rule
    assert e.contract_status["vjp"] == "not_applicable"


def test_ops_callable_matches_stdlib():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((48, 32)) / 8).astype(np.float32)
    x = rng.standard_normal((6, 48)).astype(np.float32)
    packed = q.quantize_weight(w, "int4", group_size=16)
    via_ops = tessera.ops.dequant_matmul(x, packed)
    via_lib = q.dequant_matmul(x, packed)
    np.testing.assert_allclose(via_ops, via_lib, rtol=1e-9, atol=1e-9)


def test_vjp_matches_dequantized_gemm():
    """The registered VJP is the GEMM rule on the dequantized weight."""
    from tessera.autodiff.vjp import _VJPS
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 8))
    w = rng.standard_normal((8, 5))
    dout = rng.standard_normal((4, 5))
    dx, dw = _VJPS["dequant_matmul"](dout, x, w)
    np.testing.assert_allclose(dx, dout @ w.T, rtol=1e-9)
    np.testing.assert_allclose(dw, x.T @ dout, rtol=1e-9)
