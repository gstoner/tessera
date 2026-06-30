"""Sprint #20d (2026-05-22) — sharding mock-mesh proofs for
segment_reduce + sparse-CSR (spmm_csr, sddmm, bsmm).

The GA differential family (clifford_codiff, clifford_ext_deriv,
clifford_vec_deriv, clifford_integral) inherits the Sprint #14 stencil
halo proof — they implement the same halo-exchange pattern on Clifford
multivector fields.  The category promotion of `geometric_algebra`
(below) covers them; no separate numerical test is needed because the
halo machinery is already proven by the stencil category proof in
tests/unit/test_halo_execution_lane.py.

Proof shapes here:

  segment_reduce (1) — per-rank local segment_reduce on its row slice,
  pad to the full segment-id space, all_reduce(sum) recovers the global
  result.  Identical on all ranks.

  sparse-CSR (3: spmm_csr, sddmm, bsmm) — row-parallel: shard A by
  rows, replicate B, each rank computes its row chunk; all_gather along
  the row axis recovers the full output.

  GA differential (4) — documented as inheriting the stencil halo
  proof; covered by registry promotion and a registry-level assertion.

The 4 ebm sampling ops + the GA pointwise ops are out of scope for
Sprint #20d (separately closed by #19 and #20e).

Bucket C entries that stay at `partial`:
  - spmm_coo (hash-shard requires real distributed execution)
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.primitive_coverage import all_primitive_coverages
from tessera.testing.mock_collective import MockRankGroup


# ─────────────────────────────────────────────────────────────────────────────
# segment_reduce — per-rank local + all_reduce
# ─────────────────────────────────────────────────────────────────────────────


def test_segment_reduce_row_split_with_padded_allreduce() -> None:
    """segment_reduce(x, seg_ids) sums rows of x grouped by seg_id.
    Under row-axis sharding, each rank holds a slice of (x, seg_ids).
    The local segment_reduce produces a partial per-segment sum; padding
    to the full segment-id space and all_reduce(sum) recovers the
    global result.  All ranks see the same final output."""
    np.random.seed(0)
    N, F = 12, 8
    num_segments = 4
    x = np.random.randn(N, F).astype(np.float32)
    seg = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3], dtype=np.int64)
    expected = np.asarray(ts.ops.segment_reduce(x, seg, op="sum"), dtype=np.float32)

    world_size = 2
    N_local = N // world_size

    def worker(rank):
        n0 = rank.rank * N_local
        n1 = n0 + N_local
        x_local = x[n0:n1]
        seg_local = seg[n0:n1]
        # Per-rank local segment_reduce; ts.ops.segment_reduce sizes the
        # output by max(seg)+1 from the local seg_ids, which may be
        # smaller than num_segments.  Compute the partial and pad to
        # (num_segments, F) so the all_reduce shape is identical
        # across ranks.
        partial = np.zeros((num_segments, F), dtype=np.float32)
        for row, sid in zip(x_local, seg_local):
            partial[sid] += row
        result = rank.all_reduce(partial, op="sum")
        return result

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for reduced in results:
        np.testing.assert_allclose(reduced, expected, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Sparse CSR row-parallel — shard A by rows, replicate B, all_gather
# ─────────────────────────────────────────────────────────────────────────────


def _build_csr(M: int, K: int, density: float, seed: int):
    """Build a random sparse matrix in CSR format (indptr, indices,
    values, shape)."""
    rng = np.random.default_rng(seed)
    dense = (rng.random((M, K)) < density).astype(np.float32) * rng.standard_normal((M, K)).astype(np.float32)
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    for row in dense:
        nz = np.nonzero(row)[0]
        indices.extend(int(i) for i in nz)
        values.extend(float(row[i]) for i in nz)
        indptr.append(len(indices))
    return (
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
        (M, K),
    ), dense


def _slice_csr_rows(csr, row_start: int, row_end: int):
    """Return a new CSR triple covering rows [row_start, row_end)."""
    indptr, indices, values, shape = csr
    new_indptr = (indptr[row_start: row_end + 1] - indptr[row_start]).astype(np.int64)
    start_nnz = int(indptr[row_start])
    end_nnz = int(indptr[row_end])
    new_indices = indices[start_nnz:end_nnz]
    new_values = values[start_nnz:end_nnz]
    return (new_indptr, new_indices, new_values, (row_end - row_start, shape[1]))


def test_spmm_csr_row_parallel() -> None:
    """SpMM (CSR sparse * dense) is row-parallel under row sharding:
    shard A by rows, replicate B, each rank computes its row chunk;
    all_gather along the row axis recovers the full output."""
    np.random.seed(0)
    M, K, N = 8, 12, 16
    world_size = 2
    csr, _dense = _build_csr(M, K, density=0.3, seed=42)
    B = np.random.randn(K, N).astype(np.float32)
    expected = np.asarray(ts.ops.spmm_csr(csr, B), dtype=np.float32)

    M_local = M // world_size

    def worker(rank):
        m0 = rank.rank * M_local
        m1 = m0 + M_local
        csr_local = _slice_csr_rows(csr, m0, m1)
        local_out = ts.ops.spmm_csr(csr_local, B)
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=0)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-5)


def test_sddmm_row_parallel() -> None:
    """SDDMM = (A @ B) * mask under row sharding: shard A and the mask
    on the row axis; B replicated.  Each rank computes its row chunk;
    all_gather recovers the full output."""
    np.random.seed(1)
    M, K, N = 8, 12, 16
    world_size = 2
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    mask = (np.random.random((M, N)) > 0.5).astype(np.float32)
    expected = np.asarray(ts.ops.sddmm(A, B, mask), dtype=np.float32)

    M_local = M // world_size

    def worker(rank):
        m0 = rank.rank * M_local
        m1 = m0 + M_local
        local_out = ts.ops.sddmm(A[m0:m1], B, mask[m0:m1])
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=0)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-5)


def test_bsmm_row_parallel() -> None:
    """Block-sparse matmul under row sharding follows the same shape
    as spmm_csr."""
    np.random.seed(2)
    M, K, N = 8, 12, 16
    world_size = 2
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)
    expected = np.asarray(ts.ops.bsmm(X, W), dtype=np.float32)

    M_local = M // world_size

    def worker(rank):
        m0 = rank.rank * M_local
        m1 = m0 + M_local
        local_out = ts.ops.bsmm(X[m0:m1], W)
        return rank.all_gather(np.asarray(local_out, dtype=np.float32), axis=0)

    group = MockRankGroup(n=world_size, mesh_axes={"tp": world_size})
    results = group.run(worker)
    for gathered in results:
        np.testing.assert_allclose(gathered, expected, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Registry promotion claims
# ─────────────────────────────────────────────────────────────────────────────


_SPRINT_20D_PROMOTED_NAMES = (
    # segment_reduce (1)
    "segment_reduce",
    # sparse-CSR (3)
    "spmm_csr", "sddmm", "bsmm",
    # GA differential (4) — inherit Sprint #14 halo proof
    "clifford_codiff", "clifford_vec_deriv",
    "clifford_ext_deriv", "clifford_integral",
)

# Bucket C — stays at partial pending real-hardware validation.
_SPRINT_20D_BUCKET_C = (
    "spmm_coo",  # hash-shard requires real distributed execution
)


def test_sprint_20d_promoted_set_sharding_complete() -> None:
    """segment_reduce + sparse-CSR + GA differential reach
    `sharding_rule = complete` after Sprint #20d.  The GA differential
    ops inherit the halo proof shipped in Sprint #14 (see
    tests/unit/test_halo_execution_lane.py); the others are proven by
    the row-parallel tests above."""
    entries = all_primitive_coverages()
    failures: list[tuple[str, str]] = []
    for name in _SPRINT_20D_PROMOTED_NAMES:
        if name not in entries:
            continue
        actual = entries[name].contract_status.get("sharding_rule")
        if actual != "complete":
            failures.append((name, str(actual)))
    assert not failures, (
        "Sprint #20d promoted set sharding_rule must be `complete` "
        f"after the mock-mesh proofs + halo inheritance, but got: "
        f"{failures}.  See test_segment_reduce_row_split_with_padded_"
        "allreduce, test_spmm_csr_row_parallel, test_sddmm_row_parallel,"
        " test_bsmm_row_parallel; GA differential rides Sprint #14."
    )


def test_spmm_coo_stays_partial_phase_g_gate() -> None:
    """spmm_coo's hash-shard requires real distributed execution to
    validate (the COO hash-partition pattern must be exercised on a
    real fabric to verify hash collisions are handled).  Bucket C."""
    entries = all_primitive_coverages()
    if "spmm_coo" not in entries:
        pytest.skip("spmm_coo not in registry on this branch")
    actual = entries["spmm_coo"].contract_status.get("sharding_rule")
    assert actual in ("partial", "planned"), (
        f"spmm_coo sharding_rule must stay at partial/planned (Phase "
        f"G/H gate — hash-sharded COO needs real distributed "
        f"execution), got {actual!r}."
    )
