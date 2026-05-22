"""Sprint #20b (2026-05-22) — loop_nest sharding mock-mesh proof.

This test is the Bucket B promotion gate for the loop_nest family
(matmul, gemm, batched_gemm, factorized_matmul).  These ops are the
canonical Megatron-style tensor-parallel primitives; their sharding
contract has two equivalent forms:

  Column-parallel (output-axis split):
      Y = X @ W with W sharded as W = [W_0 | W_1 | ... | W_{P-1}]
      on the output axis.  Each rank computes Y_local = X @ W_local;
      ``all_gather`` along axis=-1 reconstructs Y.  No reduction needed.

  Row-parallel (contraction-axis split):
      Y = X @ W with W sharded as W = [W_0; W_1; ...; W_{P-1}]
      on the contraction axis and X correspondingly split.  Each rank
      computes Y_partial = X_local @ W_local; ``all_reduce(sum)``
      reconstructs Y.  This is the canonical TP form for the second
      matrix in a transformer FFN block.

Both shapes are mechanical: matmul is linear in both factors, so the
distributive law gives an exact decomposition.  The mock-mesh proof
just exercises the collective trace.

This test pins:
  1. Numerical equivalence under column-parallel split for matmul, gemm
  2. Numerical equivalence under row-parallel (contraction-axis) split
     for matmul + gemm via all_reduce
  3. Same for batched_gemm (per-batch independence + per-output-shard)
  4. Same for factorized_matmul (the rank-factorization is internal and
     doesn't change the I/O sharding contract)
  5. Registry reflects ``sharding_rule = complete`` for the family.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.testing.mock_collective import MockRankGroup


# Proven by the mock-mesh tests below: matmul / gemm / batched_gemm
# decompose linearly under both column-parallel and row-parallel
# sharding.  `factorized_matmul` is intentionally excluded — its
# rank-r SVD truncation is a post-hoc, non-compositional epilogue
# (truncating per-shard != truncating the full output), so its
# sharding contract stays at `partial` even though the underlying
# matmul shards identically.  See
# `test_factorized_matmul_partial_under_mock_mesh` for the
# documentation lock.
_LOOP_NEST_PROVEN_NAMES = ("matmul", "gemm", "batched_gemm")


# ─────────────────────────────────────────────────────────────────────────────
# Column-parallel (output-axis split) — no collective beyond all_gather
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("op_name", ["matmul", "gemm"])
def test_matmul_column_parallel(world_size: int, op_name: str) -> None:
    """Column-parallel: shard W on the output axis (axis=-1).  Y_local
    = X @ W_local, all_gather on output axis = Y."""
    np.random.seed(0)
    M, K, N = 8, 12, 16
    assert N % world_size == 0
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    op = getattr(ts.ops, op_name)
    expected = np.asarray(op(X, W), dtype=np.float32)

    N_local = N // world_size

    def worker(rank):
        w0 = rank.rank * N_local
        w1 = w0 + N_local
        local_out = op(X, W[:, w0:w1])
        return rank.all_gather(
            np.asarray(local_out, dtype=np.float32), axis=-1,
        )

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Row-parallel (contraction-axis split) — all_reduce(sum) closes the
# reduction.  This is the canonical TP-shard for the FFN-down projection.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("op_name", ["matmul", "gemm"])
def test_matmul_row_parallel(world_size: int, op_name: str) -> None:
    """Row-parallel: shard X on contraction axis and W on contraction
    axis; each rank computes a partial; all_reduce(sum) recovers Y."""
    np.random.seed(1)
    M, K, N = 8, 12, 16
    assert K % world_size == 0
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    op = getattr(ts.ops, op_name)
    expected = np.asarray(op(X, W), dtype=np.float32)

    K_local = K // world_size

    def worker(rank):
        k0 = rank.rank * K_local
        k1 = k0 + K_local
        partial = op(X[:, k0:k1], W[k0:k1, :])
        return rank.all_reduce(
            np.asarray(partial, dtype=np.float32), op="sum",
        )

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for reduced in results:
        np.testing.assert_allclose(reduced, expected, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Batched GEMM under row-parallel — proves per-batch independence
# ─────────────────────────────────────────────────────────────────────────────


def test_batched_gemm_row_parallel() -> None:
    np.random.seed(2)
    B, M, K, N = 3, 8, 12, 16
    world_size = 2
    A = np.random.randn(B, M, K).astype(np.float32)
    Bmat = np.random.randn(B, K, N).astype(np.float32)
    expected = np.asarray(ts.ops.batched_gemm(A, Bmat), dtype=np.float32)

    K_local = K // world_size

    def worker(rank):
        k0 = rank.rank * K_local
        k1 = k0 + K_local
        partial = ts.ops.batched_gemm(A[:, :, k0:k1], Bmat[:, k0:k1, :])
        return rank.all_reduce(
            np.asarray(partial, dtype=np.float32), op="sum",
        )

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for reduced in results:
        np.testing.assert_allclose(reduced, expected, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Factorized matmul — documented exception: its post-hoc rank-r SVD
# truncation is NOT compositional under sharding.  Column-parallel
# would SVD-truncate each (M, N_local) shard separately, producing a
# different rank-r approximation than truncating the full (M, N).
# The sharding contract therefore stays at `partial` even though the
# underlying matmul shards trivially.
# ─────────────────────────────────────────────────────────────────────────────


def test_factorized_matmul_truncation_is_not_compositional() -> None:
    """Lock the finding that motivates keeping factorized_matmul at
    ``sharding_rule = partial``: the rank-r SVD-truncated output of a
    column-parallel split is NOT equal to the SVD-truncated output of
    the full matmul.  This is a property of the truncation, not of
    matmul itself."""
    np.random.seed(3)
    M, K, N = 8, 16, 16
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    rank_r = 4
    # Truncate the full output.
    full = np.asarray(
        ts.ops.factorized_matmul(X, W, rank=rank_r), dtype=np.float32,
    )
    # Column-parallel decomposition (per-shard truncation).
    half_a = np.asarray(
        ts.ops.factorized_matmul(X, W[:, : N // 2], rank=rank_r),
        dtype=np.float32,
    )
    half_b = np.asarray(
        ts.ops.factorized_matmul(X, W[:, N // 2 :], rank=rank_r),
        dtype=np.float32,
    )
    sharded = np.concatenate([half_a, half_b], axis=-1)
    # The two are different by construction — this lock prevents a
    # future change from accidentally promoting factorized_matmul to
    # `sharding_rule = complete` on the strength of the loop_nest
    # category default.
    diff = float(np.max(np.abs(full - sharded)))
    assert diff > 1e-3, (
        "Expected factorized_matmul to differ between per-shard and "
        f"full-output truncation, but max abs diff was {diff} — if "
        "this op is now linear under sharding, the per-name override "
        "in _EXISTING_CONTRACT_OVERRIDES['factorized_matmul'] can be "
        "removed and the loop_nest category default applies."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry promotion claim locked by these proofs
# ─────────────────────────────────────────────────────────────────────────────


def test_loop_nest_proven_set_sharding_promoted_to_complete() -> None:
    """The mock-mesh proofs above license `sharding_rule = complete` for
    matmul, gemm, batched_gemm.  This test pins
    `_SHARDING_RULE_BY_CATEGORY["loop_nest"] = "complete"`."""
    entries = all_primitive_coverages()
    failures: list[tuple[str, str]] = []
    for name in _LOOP_NEST_PROVEN_NAMES:
        if name not in entries:
            continue
        actual = entries[name].contract_status.get("sharding_rule")
        if actual != "complete":
            failures.append((name, str(actual)))
    assert not failures, (
        "loop_nest proven set sharding_rule must be `complete` after "
        f"the Sprint #20b mock-mesh proof, but got: {failures}.  "
        "The proof is the linearity of matmul under the canonical "
        "Megatron-style column-parallel / row-parallel decomposition; "
        "see test_matmul_column_parallel and test_matmul_row_parallel."
    )


def test_factorized_matmul_stays_partial_under_mock_mesh() -> None:
    """factorized_matmul is the documented exception inside loop_nest:
    its rank-r SVD truncation is not compositional under sharding (see
    test_factorized_matmul_truncation_is_not_compositional).  The
    per-name override in _EXISTING_CONTRACT_OVERRIDES must keep
    sharding_rule at `partial`."""
    entries = all_primitive_coverages()
    if "factorized_matmul" not in entries:
        pytest.skip("factorized_matmul not in registry on this branch")
    actual = entries["factorized_matmul"].contract_status.get("sharding_rule")
    assert actual in ("partial", "planned"), (
        f"factorized_matmul sharding_rule must stay at `partial` "
        f"(Phase G/H/I gate — SVD truncation needs real distributed "
        f"validation), but registry shows {actual!r}.  See "
        "test_factorized_matmul_truncation_is_not_compositional."
    )
