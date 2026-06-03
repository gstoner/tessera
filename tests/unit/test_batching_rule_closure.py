"""S-series #2 (2026-06-02) — batching_rule closure guard.

The long-tail batching axis was closed for the textbook-batchable families
(collective / recurrent / state_space / linalg decomposition+solver / sparse /
segment_reduce). Lock those completes, and lock that the genuinely-pending
ones (moe / moe_transport / state_update — cross-device routing or kv-cache
write under vmap) stay partial, so a future category-table edit can't silently
over- or under-claim.
"""

from __future__ import annotations

from tessera.compiler import primitive_coverage as pc

_COVS = pc.all_primitive_coverages()


def _bk(name: str) -> str:
    return _COVS[name].contract_status.get("batching_rule")


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
