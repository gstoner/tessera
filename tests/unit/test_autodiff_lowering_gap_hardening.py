"""Focused coverage for autodiff and Graph IR gap hardening."""

from __future__ import annotations

import numpy as np

import tessera as ts
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp
from tessera.compiler.graph_ir import GraphIRBuilder
from tessera.compiler.primitive_coverage import coverage_for


def test_s11_loss_vjp_and_jvp_rules_are_registered_and_analytical():
    pred = np.array([1.0, 3.0], dtype=np.float64)
    target = np.array([0.0, 1.0], dtype=np.float64)

    grad_pred, grad_target = get_vjp("mse_loss")(1.0, pred, target, reduction="mean")
    np.testing.assert_allclose(grad_pred, np.array([1.0, 2.0]))
    np.testing.assert_allclose(grad_target, -grad_pred)

    primal, tangent = get_jvp("mse_loss")(
        (pred, target),
        (np.ones_like(pred), np.zeros_like(target)),
        reduction="mean",
    )
    np.testing.assert_allclose(primal, ts.losses.mse_loss(pred, target))
    np.testing.assert_allclose(tangent, 3.0)


def test_s10_sgd_and_s7_linear_general_have_transform_rules():
    params = np.array([1.0, 2.0], dtype=np.float64)
    grads = np.array([0.25, -0.5], dtype=np.float64)
    dout = np.ones_like(params)

    dparams, dgrads = get_vjp("sgd")(dout, params, grads, lr=0.1)
    np.testing.assert_allclose(dparams, dout)
    np.testing.assert_allclose(dgrads, -0.1 * dout)
    _, tangent = get_jvp("sgd")((params, grads), (dout, dout), lr=0.1)
    np.testing.assert_allclose(tangent, 0.9 * dout)

    x = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    w = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    y, dy = get_jvp("linear_general")((x, w), (np.ones_like(x), np.zeros_like(w)), axis=-1)
    np.testing.assert_allclose(y, ts.nn.linear_general(x, w))
    np.testing.assert_allclose(dy, np.ones_like(x) @ w)


def test_focused_python_reference_primitives_have_graph_ir_lowering_entries():
    for name in ("linear_general", "sgd", "mse_loss", "binary_cross_entropy_loss"):
        entry = coverage_for(name)
        assert entry.metadata["graph_ir_lowering"] == "registered"
        assert entry.contract_status["lowering_rule"] == "complete"
        assert entry.contract_status["vjp"] == "complete"
        assert entry.contract_status["jvp"] == "complete"


def test_graph_ir_builder_lowers_promoted_s7_s10_s11_calls():
    def train_leaf(w, x, target, grad):
        pred = ts.nn.linear_general(x, w)
        loss = ts.losses.mse_loss(pred, target)
        return ts.optim.sgd(w, grad, lr=0.1) + loss

    fn_ir = GraphIRBuilder().lower(train_leaf)
    op_names = [op.op_name for op in fn_ir.body]

    assert "tessera.linear_general" in op_names
    assert "tessera.loss.mse" in op_names
    assert "tessera.sgd" in op_names
