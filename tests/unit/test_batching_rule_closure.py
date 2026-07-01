"""S-series #2+ closeout — long-tail batching/transpose closure guard.

The long-tail batching axis is closed for textbook-batchable families and for
the local state/MoE transport lanes. Mesh/device ownership remains under
``sharding_rule`` and ``backend_kernel`` rather than ``batching_rule``.

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


def test_state_and_moe_batching_is_closed_locally():
    # vmap adds an ordinary leading data axis. MoE routing/transport and cache
    # state updates preserve that axis locally; distributed mesh proof remains
    # tracked by sharding_rule/backend_kernel.
    for name in (
        "cache_commit",
        "cache_rollback",
        "moe",
        "moe_dispatch",
        "moe_combine",
        "online_softmax_state",
    ):
        assert _bk(name) == "complete", f"{name} batching_rule should be complete"


def test_batching_axis_has_zero_open():
    open_status = {"partial", "planned"}
    offenders = [
        n
        for n, c in _COVS.items()
        if c.contract_status.get("batching_rule") in open_status
    ]
    assert not offenders, f"batching_rule should be fully closed, open: {offenders}"


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
