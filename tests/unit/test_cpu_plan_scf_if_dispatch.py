"""Followup 1 — real backend ``scf.if`` lowering (CPU plan execution).

A.2 renamed the diagnostic from ``JIT_EAGER_FALLBACK_UNSUPPORTED_OP``
to ``JIT_EAGER_FALLBACK_CONTROL_FLOW`` but didn't actually make
``tessera.scf.if.*`` executable through a compiled path. This follow-up
ships the smallest **real** backend pass: the CPU plan executor
(``CPUPlan.execute``) now walks ``tessera.scf.if.{begin,else,end}``
markers with bracket-matching, evaluates the SSA-operand condition
(or static-literal condition), and dispatches the live branch only —
the dead branch's ops are NOT executed.

Scope ladder:

* ``scf.if`` with SSA-operand condition  → executable through CPU plan
* ``scf.if`` with static-literal condition → executable
* ``scf.if`` with text-only condition (no operand, no static literal) → eager
* Static-trip-count ``scf.for`` → executable through CPU plan
* SSA-bound dynamic-trip-count ``scf.for`` → executable through CPU plan
* Text-only ``scf.for`` and ``scf.while`` → eager
* Nested supported ``scf.if`` / static ``scf.for`` markers dispatch
  recursively

Branch-dependent outputs (``y_then`` in one branch, ``y_else`` in the
other, then ``return y``) still need phi/yield semantics — a separate,
larger surface. v1 of the CF-aware executor supports both branches
binding the same SSA name (real backends can rely on this via
side-effect-free or in-place semantics).

These tests build synthetic Graph IR directly (bypassing the verifier
where needed) and execute it through ``CPUPlan.execute``. They prove
the dispatch is real — the dead branch's ops do NOT run.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.graph_ir import (
    GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
)
from tessera.compiler.matmul_pipeline import (
    CPUPlan, _find_scf_for_end, _find_scf_if_brackets,
    _scf_body_is_plannable, build_cpu_plan, explain_cpu_plan,
)


_T = IRType("tensor<*x?>", ("*",), "fp32")


def _scf_op(kind: str, *, cond: str | None = "%cond", kwargs=None) -> IROp:
    """Helper to construct an scf marker op."""
    operands = [cond] if cond else []
    op_types = ["tensor<*x?>"] if cond else []
    return IROp(
        result=None, op_name=f"tessera.scf.{kind}",
        operands=operands, operand_types=op_types,
        kwargs=kwargs or {"kind": "dynamic"},
    )


def _make_cpu_plan(body: list[IROp], output_name: str) -> CPUPlan:
    """Bypass the Graph IR verifier — synthetic IR with same-name
    SSA-rebind across branches is intentional. The plan executor only
    runs one branch per execution, so DUP_VALUE is moot at runtime."""
    return CPUPlan(
        function_name="cond_demo",
        ops=tuple(body),
        output_name=output_name,
        tile=(128, 128, 64),
        target_kind="cpu",
        graph_ir="", schedule_ir="", tile_ir="", target_ir="",
    )


def _scf_for_op(kind: str, *, trip_count=3, induction: str = "i") -> IROp:
    return IROp(
        result=None, op_name=f"tessera.scf.for.{kind}",
        operands=[], operand_types=[],
        kwargs={"induction": induction, "trip_count": trip_count},
    )


# ---- Bracket matcher correctness ----------------------------------------

def test_find_scf_if_brackets_basic():
    body = [
        _scf_op("if.begin"),
        IROp(result="y", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("else"),
        IROp(result="y", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("if.end"),
    ]
    else_idx, end_idx = _find_scf_if_brackets(body, 0)
    assert else_idx == 2
    assert end_idx == 4


def test_find_scf_if_brackets_no_else():
    body = [
        _scf_op("if.begin"),
        IROp(result="y", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("if.end"),
    ]
    else_idx, end_idx = _find_scf_if_brackets(body, 0)
    assert else_idx is None
    assert end_idx == 2


def test_find_scf_if_brackets_nested():
    """Nested scf.if must not confuse the matcher — the outer
    ``if.end`` is the FIRST one at depth 0, not the inner one."""
    body = [
        _scf_op("if.begin"),
        _scf_op("if.begin"),  # nested
        IROp(result="a", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("if.end"),  # inner end
        _scf_op("if.end"),  # outer end (the one we want)
    ]
    else_idx, end_idx = _find_scf_if_brackets(body, 0)
    assert else_idx is None
    assert end_idx == 4  # outer end


def test_find_scf_if_brackets_unbalanced_raises():
    body = [_scf_op("if.begin"),
            IROp(result="y", op_name="tessera.relu", operands=["%x"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>")]
    with pytest.raises(ValueError, match="unbalanced"):
        _find_scf_if_brackets(body, 0)


def test_find_scf_for_end_basic():
    body = [
        _scf_for_op("begin"),
        IROp(result="y", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_for_op("end"),
    ]
    assert _find_scf_for_end(body, 0) == 2


def test_find_scf_for_end_nested():
    body = [
        _scf_for_op("begin"),
        _scf_for_op("begin"),
        _scf_for_op("end"),
        _scf_for_op("end"),
    ]
    assert _find_scf_for_end(body, 0) == 3


# ---- Real branch dispatch (the v1 contract) -----------------------------

def test_scf_if_executes_then_branch_only():
    """Cond truthy → only the then-branch op runs; the else-branch op
    is NOT executed. We prove this by giving the else branch a result
    name that would crash if its body ran (an op with an unbound
    operand)."""
    body = [
        _scf_op("if.begin"),
        IROp(result="y", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("else"),
        IROp(result="y", op_name="tessera.relu", operands=["%not_bound"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("if.end"),
    ]
    plan = _make_cpu_plan(body, output_name="y")
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    out = plan.execute([x, 1.0], {}, ["x", "cond"])
    # If the else branch had run, it would have raised on the unbound
    # %not_bound operand. It didn't — proves the dead branch is skipped.
    np.testing.assert_array_equal(out, np.array([1.0, 0.0, 3.0]))


def test_scf_if_executes_else_branch_only():
    """Symmetric — cond falsy → only the else-branch op runs."""
    body = [
        _scf_op("if.begin"),
        IROp(result="y", op_name="tessera.relu", operands=["%not_bound"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("else"),
        IROp(result="y", op_name="tessera.mul", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>",
             kwargs={"scalar": -1.0, "scalar_side": "right"}),
        _scf_op("if.end"),
    ]
    plan = _make_cpu_plan(body, output_name="y")
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    out = plan.execute([x, 0.0], {}, ["x", "cond"])
    np.testing.assert_array_equal(out, np.array([-1.0, 2.0, -3.0]))


def test_scf_if_with_static_condition_true_takes_then():
    """Static-attr condition (D's "case 1"): no SSA operand, condition
    is on ``kwargs["condition"]``. The executor still dispatches."""
    body = [
        IROp(result=None, op_name="tessera.scf.if.begin",
             operands=[], operand_types=[],
             kwargs={"condition": True}),
        IROp(result="y", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        IROp(result=None, op_name="tessera.scf.else",
             operands=[], operand_types=[], kwargs={"condition": True}),
        IROp(result="y", op_name="tessera.relu", operands=["%not_bound"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        IROp(result=None, op_name="tessera.scf.if.end",
             operands=[], operand_types=[], kwargs={"condition": True}),
    ]
    plan = _make_cpu_plan(body, output_name="y")
    x = np.array([1.0, -2.0], dtype=np.float32)
    out = plan.execute([x], {}, ["x"])
    np.testing.assert_array_equal(out, np.array([1.0, 0.0]))


def test_scf_if_without_else_does_not_crash_on_falsy():
    """``if cond: y = ...`` (no else) — when cond is falsy, no branch
    runs. The output must already be bound by a prior op or the plan
    legitimately reports "no output." Here we bind y before the if."""
    body = [
        IROp(result="y", op_name="tessera.mul", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>",
             kwargs={"scalar": 2.0, "scalar_side": "right"}),
        _scf_op("if.begin"),
        # then-branch overwrites y; falsy cond skips this.
        IROp(result="y", op_name="tessera.relu", operands=["%not_bound"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("if.end"),
    ]
    plan = _make_cpu_plan(body, output_name="y")
    x = np.array([1.0, -2.0], dtype=np.float32)
    out = plan.execute([x, 0.0], {}, ["x", "cond"])
    np.testing.assert_array_equal(out, np.array([2.0, -4.0]))


# ---- Static scf.for dispatch (Sprint C) ---------------------------------

def test_scf_for_static_trip_count_executes_body_each_iteration():
    """Static trip-count loops execute through the CPU plan. Same-name
    SSA rebinding is the v1 loop-carried value contract."""
    body = [
        IROp(result="y", op_name="tessera.mul", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>",
             kwargs={"scalar": 1.0, "scalar_side": "right"}),
        _scf_for_op("begin", trip_count=3),
        IROp(result="y", op_name="tessera.mul", operands=["%y"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>",
             kwargs={"scalar": 2.0, "scalar_side": "right"}),
        _scf_for_op("end", trip_count=3),
    ]
    plan = _make_cpu_plan(body, output_name="y")
    out = plan.execute([np.array([1.0, -2.0], dtype=np.float32)], {}, ["x"])
    np.testing.assert_array_equal(out, np.array([8.0, -16.0], dtype=np.float32))


def test_scf_for_static_trip_count_binds_induction_temporarily():
    body = [
        IROp(result="y", op_name="tessera.mul", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>",
             kwargs={"scalar": 1.0, "scalar_side": "right"}),
        _scf_for_op("begin", trip_count=4, induction="i"),
        IROp(result="y", op_name="tessera.add", operands=["%y", "%i"],
             operand_types=["tensor<*x?>", "index"],
             result_type="tensor<*x?>"),
        _scf_for_op("end", trip_count=4, induction="i"),
    ]
    plan = _make_cpu_plan(body, output_name="y")
    out = plan.execute([np.array([10.0], dtype=np.float32)], {}, ["x"])
    np.testing.assert_array_equal(out, np.array([16.0], dtype=np.float32))


# ---- SSA-bound dynamic scf.for dispatch (Sprint D) ----------------------

def test_scf_for_dynamic_ssa_trip_count_executes_body_each_iteration():
    body = [
        IROp(result="y", op_name="tessera.mul", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>",
             kwargs={"scalar": 1.0, "scalar_side": "right"}),
        IROp(result=None, op_name="tessera.scf.for.begin",
             operands=["%n"], operand_types=["index"],
             kwargs={"kind": "dynamic", "induction": "i"}),
        IROp(result="y", op_name="tessera.mul", operands=["%y"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>",
             kwargs={"scalar": 2.0, "scalar_side": "right"}),
        IROp(result=None, op_name="tessera.scf.for.end",
             operands=["%n"], operand_types=["index"],
             kwargs={"kind": "dynamic", "induction": "i"}),
    ]
    plan = _make_cpu_plan(body, output_name="y")
    out = plan.execute(
        [np.array([1.0, -2.0], dtype=np.float32), np.array(4)],
        {}, ["x", "n"])
    np.testing.assert_array_equal(out, np.array([16.0, -32.0], dtype=np.float32))


def test_scf_for_dynamic_ssa_trip_count_is_plannable():
    body = [
        IROp(result=None, op_name="tessera.scf.for.begin",
             operands=["%n"], operand_types=["index"],
             kwargs={"kind": "dynamic", "induction": "i"}),
        IROp(result=None, op_name="tessera.scf.for.end",
             operands=["%n"], operand_types=["index"],
             kwargs={"kind": "dynamic", "induction": "i"}),
    ]
    assert _scf_body_is_plannable(body) is True


def test_build_cpu_plan_accepts_dynamic_ssa_scf_for():
    fn = GraphIRFunction(
        name="f", args=[IRArg("xs", _T), IRArg("n", _T)],
        result_types=[_T],
        body=[
            IROp(result=None, op_name="tessera.scf.for.begin",
                 operands=["%n"], operand_types=["tensor<*x?>"],
                 kwargs={"kind": "dynamic", "induction": "i"}),
            IROp(result="y", op_name="tessera.relu", operands=["%xs"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
            IROp(result=None, op_name="tessera.scf.for.end",
                 operands=["%n"], operand_types=["tensor<*x?>"],
                 kwargs={"kind": "dynamic", "induction": "i"}),
        ],
        return_values=["%y"],
    )
    assert build_cpu_plan(GraphIRModule(functions=[fn])) is not None


# ---- explain_cpu_plan reports the right status --------------------------

def test_plannable_scf_function_does_not_trigger_eager_fallback():
    """When the body is a plannable scf.if (SSA condition + supported
    branch ops), ``explain_cpu_plan`` must NOT emit the
    ``EAGER_FALLBACK_CONTROL_FLOW`` info note — the function compiles
    through the real plan."""
    fn = GraphIRFunction(
        name="f", args=[IRArg("x", _T), IRArg("cond", _T)],
        result_types=[_T],
        body=[
            _scf_op("if.begin"),
            IROp(result="y", op_name="tessera.relu", operands=["%x"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
            _scf_op("else"),
            IROp(result="y", op_name="tessera.mul", operands=["%x"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>",
                 kwargs={"scalar": 2.0, "scalar_side": "right"}),
            _scf_op("if.end"),
        ],
        return_values=["%y"],
    )
    mod = GraphIRModule(functions=[fn])
    diag = explain_cpu_plan(mod)
    assert diag.code != "JIT_EAGER_FALLBACK_CONTROL_FLOW", diag


def test_static_scf_for_function_does_not_trigger_eager_fallback():
    """Static scf.for is now in the CPU plan executor's accept set."""
    fn = GraphIRFunction(
        name="f", args=[IRArg("xs", _T)], result_types=[_T],
        body=[
            IROp(result=None, op_name="tessera.scf.for.begin",
                 operands=[], operand_types=[],
                 kwargs={"induction": "i", "trip_count": 3}),
            IROp(result="y", op_name="tessera.relu", operands=["%xs"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
            IROp(result=None, op_name="tessera.scf.for.end",
                 operands=[], operand_types=[],
                 kwargs={"induction": "i", "trip_count": 3}),
        ],
        return_values=["%y"],
    )
    mod = GraphIRModule(functions=[fn])
    diag = explain_cpu_plan(mod)
    assert diag.code != "JIT_EAGER_FALLBACK_CONTROL_FLOW", diag


def test_dynamic_ssa_scf_for_function_does_not_trigger_eager_fallback():
    """SSA-bound dynamic scf.for is now in the CPU plan executor's accept set."""
    fn = GraphIRFunction(
        name="f", args=[IRArg("xs", _T), IRArg("n", _T)],
        result_types=[_T],
        body=[
            IROp(result=None, op_name="tessera.scf.for.begin",
                 operands=["%n"], operand_types=["tensor<*x?>"],
                 kwargs={"kind": "dynamic", "induction": "i"}),
            IROp(result="y", op_name="tessera.relu", operands=["%xs"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
            IROp(result=None, op_name="tessera.scf.for.end",
                 operands=["%n"], operand_types=["tensor<*x?>"],
                 kwargs={"kind": "dynamic", "induction": "i"}),
        ],
        return_values=["%y"],
    )
    mod = GraphIRModule(functions=[fn])
    diag = explain_cpu_plan(mod)
    assert diag.code != "JIT_EAGER_FALLBACK_CONTROL_FLOW", diag


def test_text_only_scf_for_function_still_falls_back_to_eager():
    """Text-only scf.for remains outside the executor rung."""
    fn = GraphIRFunction(
        name="f", args=[IRArg("xs", _T)], result_types=[_T],
        body=[
            IROp(result=None, op_name="tessera.scf.for.begin",
                 operands=[], operand_types=[],
                 kwargs={"kind": "dynamic", "induction": "i",
                         "iter_text": "range(n)"}),
            IROp(result="y", op_name="tessera.relu", operands=["%xs"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
            IROp(result=None, op_name="tessera.scf.for.end",
                 operands=[], operand_types=[],
                 kwargs={"kind": "dynamic", "induction": "i",
                         "iter_text": "range(n)"}),
        ],
        return_values=["%y"],
    )
    mod = GraphIRModule(functions=[fn])
    diag = explain_cpu_plan(mod)
    assert diag.code == "JIT_EAGER_FALLBACK_CONTROL_FLOW"
    assert "text-only scf.for" in diag.message or "scf.for" in diag.message


# ---- _scf_body_is_plannable classifier ---------------------------------

def test_classifier_accepts_scf_if_with_ssa_condition():
    body = [
        _scf_op("if.begin", cond="%c"),
        IROp(result="y", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        _scf_op("if.end", cond="%c"),
    ]
    assert _scf_body_is_plannable(body) is True


def test_classifier_rejects_text_only_condition():
    """``condition_text="x.sum() > 0"`` only (no SSA operand, no static
    literal) — the executor can't evaluate this, must fall back."""
    body = [
        IROp(result=None, op_name="tessera.scf.if.begin",
             operands=[], operand_types=[],
             kwargs={"kind": "dynamic", "condition_text": "x.sum() > 0"}),
        IROp(result="y", op_name="tessera.relu", operands=["%x"],
             operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
        IROp(result=None, op_name="tessera.scf.if.end",
             operands=[], operand_types=[]),
    ]
    assert _scf_body_is_plannable(body) is False


def test_classifier_rejects_dynamic_scf_for():
    body = [
        IROp(result=None, op_name="tessera.scf.for.begin",
             operands=[], operand_types=[], kwargs={"kind": "dynamic"}),
        IROp(result=None, op_name="tessera.scf.for.end",
             operands=[], operand_types=[], kwargs={"kind": "dynamic"}),
    ]
    assert _scf_body_is_plannable(body) is False


def test_classifier_accepts_static_scf_for():
    body = [
        IROp(result=None, op_name="tessera.scf.for.begin",
             operands=[], operand_types=[],
             kwargs={"induction": "i", "trip_count": 3}),
        IROp(result=None, op_name="tessera.scf.for.end",
             operands=[], operand_types=[],
             kwargs={"induction": "i", "trip_count": 3}),
    ]
    assert _scf_body_is_plannable(body) is True


def test_build_cpu_plan_accepts_static_scf_for():
    fn = GraphIRFunction(
        name="f", args=[IRArg("xs", _T)], result_types=[_T],
        body=[
            IROp(result=None, op_name="tessera.scf.for.begin",
                 operands=[], operand_types=[],
                 kwargs={"induction": "i", "trip_count": 2}),
            IROp(result="y", op_name="tessera.relu", operands=["%xs"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
            IROp(result=None, op_name="tessera.scf.for.end",
                 operands=[], operand_types=[],
                 kwargs={"induction": "i", "trip_count": 2}),
        ],
        return_values=["%y"],
    )
    assert build_cpu_plan(GraphIRModule(functions=[fn])) is not None


# ---- build_cpu_plan rejects scf.if when condition is text-only --------

def test_build_cpu_plan_rejects_text_only_scf_if():
    """Catches the audit-flagged failure mode: the executor would
    crash if the planner accepted a text-only-condition scf.if.
    Confirm the planner returns None for that case."""
    fn = GraphIRFunction(
        name="f", args=[IRArg("x", _T)], result_types=[_T],
        body=[
            IROp(result=None, op_name="tessera.scf.if.begin",
                 operands=[], operand_types=[],
                 kwargs={"kind": "dynamic", "condition_text": "x.sum() > 0"}),
            IROp(result="y", op_name="tessera.relu", operands=["%x"],
                 operand_types=["tensor<*x?>"], result_type="tensor<*x?>"),
            IROp(result=None, op_name="tessera.scf.if.end",
                 operands=[], operand_types=[]),
        ],
        return_values=["%y"],
    )
    mod = GraphIRModule(functions=[fn])
    plan = build_cpu_plan(mod)
    assert plan is None
