"""The Python->MLIR frontier declares what it does not carry.

FRONTEND-IR-MEDIUM-1 down-payment item (iii), Decision #32. The tracer rebuilds
its function arguments from shape and dtype alone, so three facts the frontend
holds stop dead at the frontier: the region privileges of Decision #2, the
ConstraintSolver's registered facts (Decision #4), and the symbolic dimension
names that `Tensor['M','K']` carried before specialization.

That loss was total and undeclared -- nothing in the tree recorded a drop under
this plan item, so the debt had no owner and no gate.

**Why the existing W1.3 verifier rather than a new one** (Decisions #29/#31).
`--tessera-verify-metadata-obligation` compares a recorded `before` against the
IR's `after`. Its `before` normally comes from `--tessera-record-metadata`,
which can only see what is already in MLIR -- and these three facts never enter
MLIR at all, so that pass can never record them. The frontier is the one
boundary whose `before` lives in Python, so the frontend writes the snapshot
itself and the verifier is untouched.

A first attempt declared the drops WITHOUT recording them and was rejected as
`METADATA_OBLIGATION_STALE_DECLARATION`. That is the design working: a
declaration only means something beside the record of what was lost, or it
silently licenses a future drop nobody reviewed.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import Region, Tensor
from tessera.compiler.graph_ir import (
    FRONTIER_PLAN_ITEM,
    IRArg,
    IRType,
    declare_frontier_debt,
    frontier_facts,
)

ARGS = (np.zeros((8, 16), np.float32), np.zeros((16, 4), np.float32))


def _traced_mlir(fn):
    return fn._traced_autodiff_module(ARGS, {}).to_mlir(canonical=True)


# ── What the frontend knew ───────────────────────────────────────────────────


def test_region_privilege_is_recorded_and_declared():
    @tessera.jit
    def g(a: Region["read"], b: Region["read"]):
        return tessera.ops.matmul(a, b)

    text = _traced_mlir(g)
    assert 'region_privilege = ["read", 2]' in text, text
    assert f'region_privilege = "not_yet_carried:{FRONTIER_PLAN_ITEM}"' in text


def test_symbolic_dim_names_are_recorded_and_declared():
    """The elaboration boundary: `Tensor['M','K']` is specialized to a concrete
    `tensor<8x16xf32>` and the NAMES are what is lost. The type system cannot
    catch this one -- which is why the verifier's usual reason for not tracking
    shape does not cover it."""
    @tessera.jit
    def f(a: Tensor["M", "K"], b: Tensor["K", "N"]):
        return tessera.ops.matmul(a, b)

    text = _traced_mlir(f)
    assert "tensor<8x16xf32>" in text, "the trace should be specialized"
    assert '"[M, K]"' in text and '"[K, N]"' in text, text
    assert f'dim_names = "not_yet_carried:{FRONTIER_PLAN_ITEM}"' in text


def test_constraints_are_recorded_and_declared():
    @tessera.jit
    def h(a: Tensor["M", "K"], b: Tensor["K", "N"]):
        tessera.require(tessera.constraint.Divisible("K", 8))
        return tessera.ops.matmul(a, b)

    text = _traced_mlir(h)
    assert "Divisible" in text and "constraints = [" in text, text
    assert f'constraints = "not_yet_carried:{FRONTIER_PLAN_ITEM}"' in text


def test_all_three_facts_ride_together():
    """The three are independent; a function carrying all of them declares all
    of them, and each fact appears exactly once in the declaration."""
    @tessera.jit
    def h(a: Region["read"], b: Tensor["K", "N"]):
        tessera.require(tessera.constraint.Divisible("K", 8))
        return tessera.ops.matmul(a, b)

    text = _traced_mlir(h)
    for fact in ("region_privilege", "dim_names", "constraints"):
        assert text.count(f'{fact} = "not_yet_carried:') == 1, (fact, text)


# ── What it did NOT know ─────────────────────────────────────────────────────


def test_a_plain_function_declares_nothing():
    """No privileges, no constraints, no symbolic dims -- so nothing to declare.
    Stamping an empty snapshot would be Decision #29's unconsumed declaration,
    and would make every traced module carry a meaningless attribute."""
    @tessera.jit
    def plain(a, b):
        return tessera.ops.matmul(a, b)

    text = _traced_mlir(plain)
    assert "tessera.metadata_snapshot" not in text
    assert "tessera.lowering.dropped" not in text


def test_frontier_facts_omits_empty_groups():
    args = [IRArg("%a", IRType("tensor<8xf32>"))]
    assert frontier_facts(args) == {}
    assert frontier_facts(args, constraints=[]) == {}


def test_a_solver_is_read_not_iterated():
    """`@jit` hands a ConstraintSolver, not a list. Reading `_constraints` is
    deliberate: iterating an object nobody can enumerate would fabricate a
    snapshot, which is worse than declaring nothing."""
    from tessera.compiler.constraints import ConstraintSolver

    solver = ConstraintSolver()
    solver.add(tessera.constraint.Divisible("K", 8))
    args = [IRArg("%a", IRType("tensor<8xf32>"))]
    assert frontier_facts(args, solver) == {"constraints": {"Divisible('K', 8)": 1}}


def test_an_unenumerable_constraints_object_records_nothing():
    class Opaque:
        pass

    args = [IRArg("%a", IRType("tensor<8xf32>"), effect="read")]
    facts = frontier_facts(args, Opaque())
    assert "constraints" not in facts
    assert facts["region_privilege"] == {"read": 1}


def test_a_fact_the_ir_still_carries_is_not_declared():
    """Guards against the stale declaration directly: if a future change makes
    the tracer carry `tessera.dim_names`, the declaration must disappear on its
    own rather than becoming a licence for a real drop."""
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule

    args = [IRArg("%a", IRType("tensor<8xf32>"), dim_names=("M",))]
    function = GraphIRFunction(
        name="f",
        args=[IRArg("%a", IRType("tensor<8xf32>"), dim_names=("M",))],
        fn_attrs={},
    )
    module = GraphIRModule(functions=[function])
    assert declare_frontier_debt(module, args=args) is False
    assert "tessera.metadata_snapshot" not in module.module_attrs


# ── The gate itself ──────────────────────────────────────────────────────────


def test_the_declaration_and_the_record_always_agree():
    """Every declared name must appear in the snapshot and vice versa. A
    declaration without a record is METADATA_OBLIGATION_STALE_DECLARATION; a
    record without a declaration is METADATA_OBLIGATION_SILENT_DROP. Both are
    verifier errors, so the Python side must never emit either shape."""
    import re

    @tessera.jit
    def h(a: Region["read"], b: Tensor["K", "N"]):
        tessera.require(tessera.constraint.Divisible("K", 8))
        return tessera.ops.matmul(a, b)

    text = _traced_mlir(h)
    snapshot = re.search(r"tessera\.metadata_snapshot = \{h = \{(.*?)\}\}", text)
    dropped = re.search(r"tessera\.lowering\.dropped = \{(.*?)\}", text)
    assert snapshot and dropped
    recorded = set(re.findall(r"(\w+) = \[", snapshot.group(1)))
    declared = set(re.findall(r"(\w+) = \"not_yet_carried", dropped.group(1)))
    assert recorded == declared, (recorded, declared)


def test_every_declaration_names_the_plan_item():
    """`not_yet_carried` with no owner is refused by the verifier
    (METADATA_OBLIGATION_DEBT_UNATTRIBUTED), so the emitter may never write the
    bare form. This is the property that keeps the debt attributable."""
    @tessera.jit
    def h(a: Region["read"], b: Tensor["K", "N"]):
        return tessera.ops.matmul(a, b)

    text = _traced_mlir(h)
    assert 'not_yet_carried"' not in text
    assert text.count("not_yet_carried:") == text.count("not_yet_carried")
