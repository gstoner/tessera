"""GRAPHLESS-LOWERING-1 (2026-10-11): a row with no Graph IR op does not
report ``lowering_rule: complete`` off its category alone.

`_LOWERING_RULE_BY_CATEGORY` says each family "decomposes" to catalog ops.
Measured before this gate: 134 rows had ``graph_name is None`` yet
``lowering_rule == "complete"``, 132 of them purely from that table, and 101
of those had no Graph op by any route (e.g. ``wasserstein_distance``, an
``np.sort`` numpy reference that no tracer ever sees).  `pipeline_gates` legality read the stamp and admitted them.  Decision
#29: a declaration with no consumer is deleted or reopened, never left closed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import pipeline_gates
from tessera.compiler import primitive_coverage as pc
from tessera.compiler.op_catalog import graph_name_for
from tessera.compiler.trace import trace

def _f32(*shape):
    return np.zeros(shape, np.float32)


def test_no_graphless_row_claims_lowering_without_evidence() -> None:
    offenders = sorted(
        name for name, entry in pc.all_primitive_coverages().items()
        if entry.graph_name is None
        and entry.contract_status["lowering_rule"] == "complete"
        and not pc._has_graph_lowering(name, None)
    )
    assert not offenders, (
        f"Graph-less rows claim lowering_rule=complete with no Graph IR op: "
        f"{offenders}"
    )


@pytest.mark.parametrize(
    "name", ["wasserstein_distance", "ctc_loss", "triplet_loss",
             "contrastive_loss", "vmap", "switch", "max_pool",
             # Their `*_structured_compute_compiled` lanes hand-build an
             # artifact and dispatch it straight back to the numpy reference
             # (runtime.py `_STRUCTURED_COMPUTE_OPS`), which is not a lowering.
             "image_resize", "interpolate", "edm_precondition", "edm_loss_weight"],
)
def test_graphless_reference_rows_stay_open_with_reason(name: str) -> None:
    entry = pc.coverage_for(name)
    assert entry.graph_name is None
    assert graph_name_for(name) is None
    assert entry.contract_status["lowering_rule"] in {"partial", "planned"}
    assert entry.metadata["lowering_rule_reason"] == pc.GRAPHLESS_LOWERING_REASON
    # The sibling metadata claim moves with it.
    assert entry.metadata["graph_ir_lowering"] == "stub_required"


def test_catalog_backed_planned_rows_keep_lowering() -> None:
    """GA4/EBM `_planned` rows carry graph_name=None but the op_catalog has
    their OpSpec; that is evidence, so they stay closed."""
    for name in ("clifford_geometric_product", "ebm_inner_step"):
        entry = pc.coverage_for(name)
        assert entry.graph_name is None
        assert graph_name_for(name) is not None
        assert entry.contract_status["lowering_rule"] == "complete"
        assert "lowering_rule_reason" not in entry.metadata


def _trace_fori(x):
    return ts.control.fori_loop(0, 3, lambda i, c: ts.ops.add(c, c), x)


def _trace_cond(pred, x):
    return ts.control.cond(pred, lambda a: ts.ops.add(a, a), lambda a: a, x)


def _trace_while(x):
    return ts.control.while_loop(
        lambda c: c, lambda c: ts.ops.add(c, c), x, max_steps=4)


def _trace_scan(init, xs):
    def body(c, xt):
        c2 = ts.ops.add(c, xt)
        return c2, c2
    return ts.control.scan(body, init, xs)


_TRACERS = {
    "fori_loop": (_trace_fori, (_f32(4),)),
    "cond": (_trace_cond, (_f32(1), _f32(4))),
    "while_loop": (_trace_while, (_f32(4),)),
    "scan": (_trace_scan, (_f32(1, 4), _f32(5, 1, 4))),
}


def test_traced_lowering_table_is_fully_exercised() -> None:
    assert set(_TRACERS) == set(pc._TRACED_GRAPH_LOWERINGS)


@pytest.mark.parametrize("name", sorted(pc._TRACED_GRAPH_LOWERINGS))
def test_traced_lowering_table_matches_tracer(name: str) -> None:
    fn, args = _TRACERS[name]
    ops = [op.op_name for op in trace(fn, *args).body]
    assert pc._TRACED_GRAPH_LOWERINGS[name] in ops, ops
    assert pc.coverage_for(name).contract_status["lowering_rule"] == "complete"


def test_pipeline_legality_refuses_graphless_reference() -> None:
    """The consumer that was being told `complete`."""
    refused = pipeline_gates._eval_legality("x86", "wasserstein_distance")
    assert refused.status == pipeline_gates.STATUS_FAIL
    assert "lowering_rule" in refused.detail
    admitted = pipeline_gates._eval_legality("x86", "scan")
    assert admitted.status == pipeline_gates.STATUS_PASS


_REPO = Path(__file__).resolve().parents[2]

#: Inputs that make each producer emit the row's op.
_PRODUCER_ARGS = {
    "ebm_energy": dict(shape=(2, 4), manifold="euclidean"),
    "ebm_sphere_langevin_step": dict(shape=(2, 4), manifold="sphere"),
    "ebm_bivector_langevin_step": dict(shape=(2, 8), manifold="bivector"),
}


def _producer(path: str):
    module, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def test_dialect_lowering_table_is_fully_exercised() -> None:
    assert set(_PRODUCER_ARGS) == set(pc._DIALECT_GRAPH_LOWERINGS)


@pytest.mark.parametrize("name", sorted(pc._DIALECT_GRAPH_LOWERINGS))
def test_dialect_lowering_table_names_a_real_producer(name: str) -> None:
    """Evidence is emission, not declaration: the named producer must put the
    op -- with this row's semantic attribute -- into the IR it builds."""
    op_spec, producer = pc._DIALECT_GRAPH_LOWERINGS[name]
    op, *selector = op_spec.split(" ", 1)
    text = _producer(producer)(
        **_PRODUCER_ARGS[name], eta=0.1, temperature=1.0, steps=2)
    assert f'"{op}"' in text, (name, op)
    if selector:
        attr, value = selector[0].split("=", 1)
        assert f"{attr} = {value}" in text, (name, selector)
    entry = pc.coverage_for(name)
    assert entry.contract_status["lowering_rule"] == "complete"
    assert "lowering_rule_reason" not in entry.metadata


@pytest.fixture(scope="module")
def ods_tiers() -> dict[str, str]:
    """Each dialect op's tier from the ODS consumer audit (Decision #30: reuse
    it, do not hand-roll a walker). The registry module itself is excluded so
    its own table strings cannot vouch for the ops it lists."""
    from tessera.compiler import ods_consumer_audit as audit

    ops = audit.declared_ops()
    corpus = audit.build_corpus(
        exclude=[_REPO / "python/tessera/compiler/primitive_coverage.py",
                 Path(__file__).resolve()],
        names={op.full_name for op in ops})
    tiers = audit.classify(ops, corpus)
    return {op.full_name: tiers[op.key] for op in ops}


@pytest.mark.parametrize("name", sorted(pc._DIALECT_GRAPH_LOWERINGS))
def test_dialect_lowering_ops_are_compiler_tier(name: str, ods_tiers) -> None:
    op = pc._DIALECT_GRAPH_LOWERINGS[name][0].split(" ", 1)[0]
    assert ods_tiers[op] == "compiler", (name, op, ods_tiers[op])


# (dialect, mnemonic) rather than the dotted name: this test never builds the
# op, and spelling it whole would make `test_ods_op_has_consumer`'s scan count
# this file as a fixture reference and move `integral` off `unreferenced`.
@pytest.mark.parametrize("name,dialect,mnemonic", [
    ("ebm_partition_exact", "tessera_ebm", "partition_z"),
    ("ebm_partition_monte_carlo", "tessera_ebm", "partition_z"),
    ("ebm_partition_ais", "tessera_ebm", "partition_z"),
    ("clifford_integral", "tessera_clifford", "integral"),
    ("ebm_decode_init", "tessera_ebm", "decode_init"),
])
def test_declared_but_unproduced_dialect_op_does_not_admit_lowering(
        name: str, dialect: str, mnemonic: str, ods_tiers) -> None:
    """An op that exists only in ODS (fixture_only / unreferenced), or is read
    by a pass (PipelineCandidates reads decode_init) but emitted by nothing, is
    not evidence a row lowers."""
    op = f"{dialect}.{mnemonic}"
    if mnemonic != "decode_init":
        assert ods_tiers[op] in {"fixture_only", "unreferenced"}, (op, ods_tiers[op])
    assert name not in pc._DIALECT_GRAPH_LOWERINGS
    entry = pc.coverage_for(name)
    assert entry.contract_status["lowering_rule"] in {"partial", "planned"}
    assert entry.metadata["lowering_rule_reason"] == pc.GRAPHLESS_LOWERING_REASON
    refused = pipeline_gates._eval_legality("x86", name)
    assert refused.status == pipeline_gates.STATUS_FAIL
