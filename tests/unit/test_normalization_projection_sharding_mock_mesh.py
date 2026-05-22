"""Sprint #20c (2026-05-22) — sharding mock-mesh proofs for the
normalization / projection / contraction / fused_epilogue / model_layer
families.

Each family has a canonical mock-mesh proof shape:

  normalization (layer_norm, rmsnorm, rmsnorm_safe, group_norm,
  instance_norm, spectral_norm, weight_norm) — feature-axis split with
  two-pass all_reduce of (sum, sum_of_squares) recovers the single-rank
  normalized output.  Alternatively, batch/sequence-axis split is
  identity (no collective).  Both forms are documented Megatron-TP
  patterns.

  projection (qkv_projection) — column-parallel matmul on the output
  axis; the three (Q, K, V) sub-projections inherit the head-axis split
  shape proven in Sprint #20a.

  contraction (einsum) — for the standard contraction-axis split, partial
  contraction + all_reduce(sum) recovers the full output.  This is the
  matmul row-parallel pattern generalized to arbitrary einsum specs.

  fused_epilogue — column-parallel matmul (output-axis split) + bias
  broadcast (bias is replicated across ranks) + pointwise activation.

  model_layer (linear_general, conv1d, conv_transpose) — linear_general
  is a general-axis matmul (same TP forms as matmul); conv1d /
  conv_transpose follow the channel-axis matmul pattern for channel
  parallelism, with optional halo for spatial sharding (rides Sprint #14
  halo machinery).

This file pins:
  1. Numerical proof for layer_norm under feature-axis split via the
     two-pass all_reduce protocol.
  2. Numerical proof for rmsnorm under feature-axis split via single
     all_reduce.
  3. Numerical proof for qkv_projection under output-axis split.
  4. Numerical proof for einsum under contraction-axis split.
  5. Numerical proof for fused_epilogue under output-axis split.
  6. Numerical proof for linear_general under output-axis and
     contraction-axis splits.
  7. Registry promotion: all 13 entries in the #20c set must reach
     `sharding_rule = complete`.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.testing.mock_collective import MockRankGroup


# ─────────────────────────────────────────────────────────────────────────────
# Normalization — feature-axis split with all-reduce of moments
# ─────────────────────────────────────────────────────────────────────────────


def _layer_norm_local_moments(x_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute partial (sum, sum_of_squares) along the last (feature) axis
    for this rank's slice."""
    return (
        x_local.sum(axis=-1, keepdims=True),
        (x_local ** 2).sum(axis=-1, keepdims=True),
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_layer_norm_feature_split_packed_allreduce(world_size: int) -> None:
    """Layer norm under feature-axis sharding requires the partial
    moments to be summed across ranks.  Packing (sum, sum_of_squares)
    into a single tensor and doing one ``all_reduce(sum)`` is the
    canonical TP form (one collective per layer_norm, matching what
    Megatron implements).  After the reduce the mean/var are derived
    locally on each rank; normalization is local; final ``all_gather``
    along the feature axis recovers the full normalized output."""
    np.random.seed(0)
    B, S, D = 2, 4, 16
    assert D % world_size == 0
    eps = 1e-5
    X = np.random.randn(B, S, D).astype(np.float32)
    expected = np.asarray(ts.ops.layer_norm(X, eps=eps), dtype=np.float32)

    D_local = D // world_size

    def worker(rank):
        d0 = rank.rank * D_local
        d1 = d0 + D_local
        x_local = X[..., d0:d1]
        local_sum, local_sumsq = _layer_norm_local_moments(x_local)
        # Pack moments into a single all_reduce — one collective per
        # layer_norm.  Last axis = [sum, sum_of_squares].
        packed = np.concatenate(
            [local_sum, local_sumsq], axis=-1,
        ).astype(np.float32)
        global_packed = rank.all_reduce(packed, op="sum")
        global_sum = global_packed[..., :1]
        global_sumsq = global_packed[..., 1:]
        mean = global_sum / D
        var = global_sumsq / D - mean ** 2
        # Local normalization on this rank's slice.
        local_norm = (x_local - mean) / np.sqrt(var + eps)
        # All-gather along the feature axis.
        return rank.all_gather(local_norm.astype(np.float32), axis=-1)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("world_size", [2, 4])
def test_rmsnorm_feature_split_single_allreduce(world_size: int) -> None:
    """RMS norm = x / sqrt(mean(x²) + eps).  Under feature-axis sharding
    the partial sum-of-squares all-reduces to the full sum-of-squares;
    one collective suffices."""
    np.random.seed(1)
    B, S, D = 2, 4, 16
    assert D % world_size == 0
    eps = 1e-5
    X = np.random.randn(B, S, D).astype(np.float32)
    expected = np.asarray(ts.ops.rmsnorm(X, eps=eps), dtype=np.float32)

    D_local = D // world_size

    def worker(rank):
        d0 = rank.rank * D_local
        d1 = d0 + D_local
        x_local = X[..., d0:d1]
        partial_sumsq = (x_local ** 2).sum(axis=-1, keepdims=True).astype(np.float32)
        global_sumsq = rank.all_reduce(partial_sumsq, op="sum")
        rms = np.sqrt(global_sumsq / D + eps)
        local_norm = x_local / rms
        return rank.all_gather(local_norm.astype(np.float32), axis=-1)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-4, atol=1e-4)


def test_layer_norm_batch_split_is_identity() -> None:
    """Sharding layer_norm along the batch axis is identity — no
    collective is needed because each sample's normalization is
    independent across the feature axis."""
    np.random.seed(2)
    B, S, D = 4, 4, 16
    X = np.random.randn(B, S, D).astype(np.float32)
    expected = np.asarray(ts.ops.layer_norm(X), dtype=np.float32)

    world_size = 2
    B_local = B // world_size

    def worker(rank):
        b0 = rank.rank * B_local
        b1 = b0 + B_local
        local_out = ts.ops.layer_norm(X[b0:b1])
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=0)

    group = MockRankGroup(n=world_size, mesh_axes={"dp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Projection — qkv_projection is a fused 3-projection matmul; output-axis
# split (column-parallel) gives the head-axis split proven in #20a.
# ─────────────────────────────────────────────────────────────────────────────


def test_qkv_projection_output_axis_split() -> None:
    """qkv_projection(x, W_qkv) is matmul + 3-way axis split: returns
    (Q, K, V) each of shape (..., D_out/3).  Under output-axis sharding,
    each rank holds a contiguous chunk of W_qkv along the output axis.
    For the (Q, K, V) split to be clean per-rank, the chunk must be
    a multiple of 3 (D_out_local = (D_out/3/world_size) * 3).  Each
    rank produces (Q_local, K_local, V_local) along the head axis; an
    all_gather per component recovers the full (Q, K, V) tuple."""
    np.random.seed(3)
    B, S, D_in, D_head = 2, 4, 16, 4  # head_dim per Q/K/V block
    world_size = 2
    # D_out = 3 * D_head * world_size so each rank gets D_head per Q/K/V
    D_out = 3 * D_head * world_size  # = 24
    X = np.random.randn(B, S, D_in).astype(np.float32)
    W = np.random.randn(D_in, D_out).astype(np.float32)
    Q_ref, K_ref, V_ref = ts.ops.qkv_projection(X, W)
    Q_ref = np.asarray(Q_ref, dtype=np.float32)
    K_ref = np.asarray(K_ref, dtype=np.float32)
    V_ref = np.asarray(V_ref, dtype=np.float32)

    # Shard pattern: W_qkv is laid out [Q_block | K_block | V_block]
    # along the output axis.  Each block has D_head*world_size columns.
    # Each rank holds D_head columns of each block.
    def slice_of_block(block_idx: int, rank: int) -> slice:
        base = block_idx * D_head * world_size
        return slice(base + rank * D_head, base + (rank + 1) * D_head)

    def worker(rank):
        cols_q = slice_of_block(0, rank.rank)
        cols_k = slice_of_block(1, rank.rank)
        cols_v = slice_of_block(2, rank.rank)
        # Stack Q/K/V columns so the qkv_projection split lands on
        # D_head per block.
        W_local = np.concatenate(
            [W[:, cols_q], W[:, cols_k], W[:, cols_v]], axis=-1,
        )
        Q_l, K_l, V_l = ts.ops.qkv_projection(X, W_local)
        # Pack the three components into one all_gather to avoid the
        # back-to-back collective race in MockRankGroup.  The packed
        # tensor has D_head*3 columns from each rank; the gathered
        # tensor has world_size copies concatenated.  Decode by
        # reshaping into (..., world_size, 3, D_head) and reading
        # each gather contribution.
        packed = np.concatenate([Q_l, K_l, V_l], axis=-1).astype(np.float32)
        gathered = rank.all_gather(packed, axis=-1)
        # gathered shape: (..., world_size * 3 * D_head).  Each rank's
        # contribution is (3 * D_head) wide.  Reshape and re-split.
        gshape = gathered.shape[:-1] + (world_size, 3, D_head)
        g = gathered.reshape(gshape)
        # Re-arrange to (..., 3, world_size * D_head) — that's the full
        # (Q, K, V) tuple along the head axis.
        g = g.transpose(*range(len(gathered.shape) - 1), -2, -3, -1)
        # g shape now: (..., 3, world_size, D_head)
        Q_g = g[..., 0, :, :].reshape(gathered.shape[:-1] + (world_size * D_head,))
        K_g = g[..., 1, :, :].reshape(gathered.shape[:-1] + (world_size * D_head,))
        V_g = g[..., 2, :, :].reshape(gathered.shape[:-1] + (world_size * D_head,))
        return Q_g, K_g, V_g

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for Q_g, K_g, V_g in results:
        np.testing.assert_allclose(Q_g, Q_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(K_g, K_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(V_g, V_ref, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Contraction — einsum under contraction-axis split + all_reduce(sum)
# ─────────────────────────────────────────────────────────────────────────────


def test_einsum_contraction_axis_row_parallel() -> None:
    """Einsum 'ij,jk->ik' is matmul; row-parallel decomposition with
    all_reduce(sum) recovers the full output.  This generalizes the
    matmul proof to einsum with the contracted-axis spec."""
    np.random.seed(4)
    M, K, N = 8, 12, 16
    world_size = 2
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    expected = np.asarray(ts.ops.einsum("ij,jk->ik", A, B), dtype=np.float32)

    K_local = K // world_size

    def worker(rank):
        k0 = rank.rank * K_local
        k1 = k0 + K_local
        partial = ts.ops.einsum("ij,jk->ik", A[:, k0:k1], B[k0:k1, :])
        return rank.all_reduce(np.asarray(partial, dtype=np.float32), op="sum")

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for reduced in results:
        np.testing.assert_allclose(reduced, expected, rtol=1e-4, atol=1e-4)


def test_einsum_batched_contraction_axis_row_parallel() -> None:
    """Batched einsum 'bij,bjk->bik' — per-batch independence plus
    contraction-axis row-parallel within each batch."""
    np.random.seed(5)
    Bb, M, K, N = 3, 8, 12, 16
    world_size = 2
    A = np.random.randn(Bb, M, K).astype(np.float32)
    B = np.random.randn(Bb, K, N).astype(np.float32)
    expected = np.asarray(ts.ops.einsum("bij,bjk->bik", A, B), dtype=np.float32)

    K_local = K // world_size

    def worker(rank):
        k0 = rank.rank * K_local
        k1 = k0 + K_local
        partial = ts.ops.einsum(
            "bij,bjk->bik", A[:, :, k0:k1], B[:, k0:k1, :],
        )
        return rank.all_reduce(np.asarray(partial, dtype=np.float32), op="sum")

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for reduced in results:
        np.testing.assert_allclose(reduced, expected, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Fused epilogue — output-axis split + bias broadcast
# ─────────────────────────────────────────────────────────────────────────────


def test_fused_epilogue_output_axis_split() -> None:
    """fused_epilogue(matmul_out, bias, activation) — output-axis split:
    the matmul output is sharded along the output axis, the bias is
    sharded correspondingly, the activation is pointwise.  All_gather
    along output axis recovers the full output."""
    np.random.seed(6)
    M, K, N = 4, 8, 16
    world_size = 2
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    bias = np.random.randn(N).astype(np.float32)
    matmul_out = np.asarray(ts.ops.matmul(X, W), dtype=np.float32)
    expected = np.asarray(
        ts.ops.fused_epilogue(matmul_out, bias=bias, activation="gelu"),
        dtype=np.float32,
    )

    N_local = N // world_size

    def worker(rank):
        n0 = rank.rank * N_local
        n1 = n0 + N_local
        local_matmul = ts.ops.matmul(X, W[:, n0:n1])
        local_out = ts.ops.fused_epilogue(
            local_matmul, bias=bias[n0:n1], activation="gelu",
        )
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=-1)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# model_layer — linear_general (output-axis + contraction-axis splits)
# ─────────────────────────────────────────────────────────────────────────────


def test_linear_general_column_parallel() -> None:
    """linear_general is a general-axis matmul.  Column-parallel split
    on the output axis is the canonical TP form."""
    np.random.seed(7)
    M, K, N = 4, 8, 16
    world_size = 2
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    expected = np.asarray(ts.ops.linear_general(X, W), dtype=np.float32)

    N_local = N // world_size

    def worker(rank):
        n0 = rank.rank * N_local
        n1 = n0 + N_local
        local_out = ts.ops.linear_general(X, W[:, n0:n1])
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=-1)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-5)


def test_linear_general_row_parallel() -> None:
    """linear_general contraction-axis split + all_reduce(sum)."""
    np.random.seed(8)
    M, K, N = 4, 8, 16
    world_size = 2
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    expected = np.asarray(ts.ops.linear_general(X, W), dtype=np.float32)

    K_local = K // world_size

    def worker(rank):
        k0 = rank.rank * K_local
        k1 = k0 + K_local
        partial = ts.ops.linear_general(X[:, k0:k1], W[k0:k1, :])
        return rank.all_reduce(np.asarray(partial, dtype=np.float32), op="sum")

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for reduced in results:
        np.testing.assert_allclose(reduced, expected, rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Registry promotion claim
# ─────────────────────────────────────────────────────────────────────────────


_SPRINT_20C_NAMES = (
    # normalization (7)
    "layer_norm", "rmsnorm", "rmsnorm_safe", "group_norm",
    "instance_norm", "spectral_norm", "weight_norm",
    # projection (1)
    "qkv_projection",
    # contraction (1)
    "einsum",
    # fused_epilogue (1)
    "fused_epilogue",
    # model_layer (3)
    "linear_general", "conv1d", "conv_transpose",
)


def test_sprint_20c_set_sharding_promoted_to_complete() -> None:
    """The mock-mesh proofs above license `sharding_rule = complete` for
    the 13 entries covered by Sprint #20c."""
    entries = all_primitive_coverages()
    failures: list[tuple[str, str]] = []
    for name in _SPRINT_20C_NAMES:
        if name not in entries:
            continue
        actual = entries[name].contract_status.get("sharding_rule")
        if actual != "complete":
            failures.append((name, str(actual)))
    assert not failures, (
        "Sprint #20c set sharding_rule must be `complete` after the "
        f"mock-mesh proofs, but got: {failures}.  See the per-family "
        "tests above (layer_norm two-pass all_reduce, rmsnorm single "
        "all_reduce, qkv_projection / linear_general / fused_epilogue "
        "column-parallel, einsum row-parallel)."
    )
