"""S-series #2 (2026-06-02) — long-tail batching/transpose closure guard.

The long-tail batching axis was closed for the textbook-batchable families
(collective / recurrent / state_space / linalg decomposition+solver / sparse /
segment_reduce). Lock those completes, and lock that the genuinely-pending
ones (moe / moe_transport / state_update — cross-device routing or kv-cache
write under vmap) stay partial, so a future category-table edit can't silently
over- or under-claim.

The transpose axis was then closed to **zero open** on the linear-vs-nonlinear
principle: a linear-transpose rule applies only to *linear* primitives. Linear
families (sparse spmm/sddmm/bsmm, moe_transport gather/scatter adjoints,
segment_reduce, tri_solve, avg_pool) are `complete`; nonlinear families
(optimizers, recurrent cells, linalg *decomposition*, ebm energy/sampling, moe
routing, max/min/adaptive pool) are `not_applicable` — their backward is the
registered VJP, not a linear-transpose dual. Lock both sides so a future edit
can't quietly flip a nonlinear op to a (false) `complete` transpose rule.
"""

from __future__ import annotations

from tessera.compiler import primitive_coverage as pc

_COVS = pc.all_primitive_coverages()


def _bk(name: str) -> str:
    return _COVS[name].contract_status.get("batching_rule")


def _tr(name: str) -> str:
    return _COVS[name].contract_status.get("transpose_rule")


def test_batchable_families_are_complete():
    # collective (vmap orthogonal to mesh axis), recurrent (per-sequence),
    # state_space (Mamba batches on B), batched linalg, sparse, segment_reduce.
    for name in (
        "collective_permute", "all_reduce", "all_gather",
        "simple_rnn_cell", "gru_cell", "bidirectional_scan",
        "selective_ssm",
        "cholesky", "qr", "svd", "tri_solve",
        "spmm_coo", "spmm_csr", "sddmm", "bsmm",
        "segment_reduce",
    ):
        assert _bk(name) == "complete", f"{name} batching_rule should be complete"


def test_cross_device_routing_stays_partial():
    # Honest: token routing / all_to_all transport under vmap + kv-cache write
    # per batch are genuinely pending (mesh-aware), not closed.
    for name in ("moe", "moe_dispatch", "moe_combine", "online_softmax_state"):
        assert _bk(name) == "partial", f"{name} batching_rule should stay partial"


def test_transpose_axis_has_zero_open():
    open_status = {"partial", "planned"}
    offenders = [
        n
        for n, c in _COVS.items()
        if c.contract_status.get("transpose_rule") in open_status
    ]
    assert not offenders, f"transpose_rule should be fully closed, open: {offenders}"


def test_linear_primitives_have_transpose_complete():
    # Genuinely-linear maps: the transpose dual is well-defined.
    # sparse spmm/sddmm/bsmm (linear in dense operand), moe_transport
    # dispatch/combine (gather/scatter adjoints of each other),
    # segment_reduce (transpose = segment-broadcast), tri_solve (linear in
    # RHS), avg_pool (uniform upsample / window-size).
    for name in (
        "spmm_coo", "spmm_csr", "sddmm", "bsmm",
        "moe_dispatch", "moe_combine",
        "segment_reduce", "tri_solve", "avg_pool",
    ):
        assert _tr(name) == "complete", f"{name} transpose_rule should be complete"


def test_nonlinear_primitives_have_transpose_not_applicable():
    # Nonlinear maps: a linear-transpose rule does not apply; backward is the
    # registered VJP. Optimizers, recurrent cells, linalg *decomposition*,
    # EBM energy/sampling, top-level MoE routing, max/min/adaptive pool.
    for name in (
        "adam", "adamw", "sgd", "lion", "lamb", "muon", "nesterov",
        "gru_cell", "simple_rnn_cell", "bidirectional_scan",
        "cholesky", "qr", "svd",
        "ebm_energy", "ebm_partition_exact", "ebm_self_verify",
        "ebm_sphere_langevin_step", "ebm_bivector_langevin_step",
        "moe",
        "max_pool", "min_pool", "adaptive_pool",
    ):
        assert _tr(name) == "not_applicable", (
            f"{name} transpose_rule should be not_applicable (nonlinear)"
        )
