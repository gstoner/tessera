"""Multi-sequence block-paged MLA cache — vLLM-style paged attention (2026-05-30).

`tessera.cache.MLABlockPagedCache` manages many concurrent sequences over a
single physical block pool with per-sequence block tables, on-demand allocation,
and free-on-finish. These tests validate the memory manager (allocation / free /
reuse / non-contiguous block tables / pool exhaustion) and cross-check decode
correctness against the already-validated single-sequence `MLAPagedDecoder`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import (MLABlockPagedCache, MLABlockPagedCacheError,
                           MLAPagedDecoder)


def _weights(H=4, dn=16, dr=8, dv=16, Dl=32, seed=0):
    rng = np.random.RandomState(seed)
    Wuk = (rng.randn(H, Dl, dn) * 0.3).astype(np.float32)
    Wuv = (rng.randn(H, Dl, dv) * 0.3).astype(np.float32)
    Wuk_t = np.ascontiguousarray(np.swapaxes(Wuk, 1, 2))
    return Wuk_t, Wuv, (H, dn, dr, dv, Dl)


def _pool(num_blocks=8, block_size=4, **kw):
    Wuk_t, Wuv, (H, dn, dr, dv, Dl) = _weights(**kw)
    return MLABlockPagedCache(num_heads=H, nope_dim=dn, rope_dim=dr, v_dim=dv,
                              latent_dim=Dl, Wuk_t=Wuk_t, Wuv=Wuv,
                              num_blocks=num_blocks, block_size=block_size), \
        Wuk_t, Wuv, (H, dn, dr, dv, Dl)


def _ref_decoder(Wuk_t, Wuv, dims, max_seq=256):
    H, dn, dr, dv, Dl = dims
    return MLAPagedDecoder(num_heads=H, nope_dim=dn, rope_dim=dr, v_dim=dv,
                           latent_dim=Dl, Wuk_t=Wuk_t, Wuv=Wuv, max_seq=max_seq)


def test_block_allocation_grows_with_length():
    pool, _, _, (H, dn, dr, dv, Dl) = _pool(num_blocks=8, block_size=4)
    rng = np.random.RandomState(1)
    pool.add_sequence("a")
    assert pool.num_used_blocks == 0
    # 4 tokens -> exactly 1 block
    pool.append("a", rng.randn(4, Dl).astype(np.float32),
                rng.randn(4, dr).astype(np.float32))
    assert pool.num_used_blocks == 1 and len(pool.block_table("a")) == 1
    # 1 more token -> spills into a 2nd block
    pool.append("a", rng.randn(1, Dl).astype(np.float32),
                rng.randn(1, dr).astype(np.float32))
    assert pool.num_used_blocks == 2 and pool.sequence_length("a") == 5


def test_free_returns_blocks_and_reuses_them():
    pool, _, _, (H, dn, dr, dv, Dl) = _pool(num_blocks=4, block_size=4)
    rng = np.random.RandomState(2)
    pool.add_sequence("a")
    pool.append("a", rng.randn(8, Dl).astype(np.float32),
                rng.randn(8, dr).astype(np.float32))   # 2 blocks
    assert pool.num_free_blocks == 2
    used_blocks = set(pool.block_table("a"))
    pool.free_sequence("a")
    assert pool.num_free_blocks == 4 and pool.num_sequences == 0
    # a new sequence reclaims the freed pages
    pool.add_sequence("b")
    pool.append("b", rng.randn(8, Dl).astype(np.float32),
                rng.randn(8, dr).astype(np.float32))
    assert pool.num_free_blocks == 2
    assert set(pool.block_table("b")) == used_blocks  # same physical pages reused


def test_non_contiguous_block_tables():
    """Interleaving appends across two sequences gives each a non-contiguous
    set of physical blocks — the block table indirection must still decode
    correctly."""
    pool, Wuk_t, Wuv, dims = _pool(num_blocks=8, block_size=2)
    H, dn, dr, dv, Dl = dims
    rng = np.random.RandomState(3)
    pool.add_sequence("a")
    pool.add_sequence("b")
    a_c, a_r, b_c, b_r = [], [], [], []
    for _ in range(3):  # interleave so blocks alternate a,b,a,b,...
        ca, ra = rng.randn(2, Dl).astype(np.float32), rng.randn(2, dr).astype(np.float32)
        cb, rb = rng.randn(2, Dl).astype(np.float32), rng.randn(2, dr).astype(np.float32)
        pool.append("a", ca, ra); a_c.append(ca); a_r.append(ra)
        pool.append("b", cb, rb); b_c.append(cb); b_r.append(rb)
    bt_a, bt_b = pool.block_table("a"), pool.block_table("b")
    # a's blocks are not a contiguous run (b's blocks are interspersed)
    assert bt_a != list(range(bt_a[0], bt_a[0] + len(bt_a)))
    assert set(bt_a).isdisjoint(bt_b)

    # decode a matches a fresh single-seq decoder fed a's tokens
    ref = _ref_decoder(Wuk_t, Wuv, dims)
    ref.append(np.concatenate(a_c), np.concatenate(a_r))
    qn, qr = rng.randn(H, dn).astype(np.float32), rng.randn(H, dr).astype(np.float32)
    np.testing.assert_allclose(pool.decode("a", qn, qr), ref.decode(qn, qr),
                               rtol=1e-4, atol=1e-4)


def test_concurrent_ragged_decode_batch():
    """Three concurrent sequences of different lengths decode in one batch; each
    must match an independent single-sequence decoder."""
    pool, Wuk_t, Wuv, dims = _pool(num_blocks=32, block_size=4)
    H, dn, dr, dv, Dl = dims
    rng = np.random.RandomState(4)
    lengths = {"x": 3, "y": 9, "z": 16}
    refs, queries = {}, {}
    for sid, L in lengths.items():
        pool.add_sequence(sid)
        c = rng.randn(L, Dl).astype(np.float32)
        r = rng.randn(L, dr).astype(np.float32)
        pool.append(sid, c, r)
        ref = _ref_decoder(Wuk_t, Wuv, dims)
        ref.append(c, r)
        refs[sid] = ref
        queries[sid] = (rng.randn(H, dn).astype(np.float32),
                        rng.randn(H, dr).astype(np.float32))

    outs = pool.decode_batch(queries)
    assert set(outs) == set(lengths)
    for sid, (qn, qr) in queries.items():
        assert outs[sid].shape == (H, dv)
        np.testing.assert_allclose(outs[sid], refs[sid].decode(qn, qr),
                                   rtol=1e-4, atol=1e-4)


def test_incremental_decode_loop_matches_single_seq():
    """Token-by-token decode of two concurrent sequences tracks two independent
    single-sequence decoders step for step."""
    pool, Wuk_t, Wuv, dims = _pool(num_blocks=32, block_size=4)
    H, dn, dr, dv, Dl = dims
    rng = np.random.RandomState(5)
    refA, refB = _ref_decoder(Wuk_t, Wuv, dims), _ref_decoder(Wuk_t, Wuv, dims)
    pool.add_sequence("A"); pool.add_sequence("B")
    for step in range(6):
        for sid, ref in (("A", refA), ("B", refB)):
            c, r = rng.randn(1, Dl).astype(np.float32), rng.randn(1, dr).astype(np.float32)
            pool.append(sid, c, r)
            ref.append(c, r)
            qn, qr = rng.randn(H, dn).astype(np.float32), rng.randn(H, dr).astype(np.float32)
            np.testing.assert_allclose(pool.decode(sid, qn, qr),
                                       ref.decode(qn, qr), rtol=1e-4, atol=1e-4)


def test_pool_exhaustion_then_recovery():
    pool, _, _, (H, dn, dr, dv, Dl) = _pool(num_blocks=2, block_size=4)
    rng = np.random.RandomState(6)
    pool.add_sequence("a")
    pool.append("a", rng.randn(8, Dl).astype(np.float32),
                rng.randn(8, dr).astype(np.float32))   # uses both blocks
    assert pool.num_free_blocks == 0
    pool.add_sequence("b")
    with pytest.raises(MLABlockPagedCacheError):
        pool.append("b", rng.randn(1, Dl).astype(np.float32),
                    rng.randn(1, dr).astype(np.float32))
    # free a and retry — now it fits
    pool.free_sequence("a")
    pool.append("b", rng.randn(1, Dl).astype(np.float32),
                rng.randn(1, dr).astype(np.float32))
    assert pool.sequence_length("b") == 1


def test_trim_sequence_rolls_back_tail_and_frees_blocks():
    pool, Wuk_t, Wuv, dims = _pool(num_blocks=6, block_size=4)
    H, dn, dr, dv, Dl = dims
    rng = np.random.RandomState(30)
    pool.add_sequence("a")
    c = rng.randn(10, Dl).astype(np.float32)
    r = rng.randn(10, dr).astype(np.float32)
    pool.append("a", c, r)
    assert pool.is_trimmable("a")
    assert pool.sequence_length("a") == 10
    assert pool.num_used_blocks == 3

    pool.trim("a", 5)
    assert pool.sequence_length("a") == 5
    assert pool.num_used_blocks == 2
    assert len(pool.block_table("a")) == 2

    ref = _ref_decoder(Wuk_t, Wuv, dims)
    ref.append(c[:5], r[:5])
    qn = rng.randn(H, dn).astype(np.float32)
    qr = rng.randn(H, dr).astype(np.float32)
    np.testing.assert_allclose(pool.decode("a", qn, qr), ref.decode(qn, qr),
                               rtol=1e-4, atol=1e-4)


def test_trim_sequence_oversized_clears_and_reuses_all_blocks():
    pool, _, _, (H, dn, dr, dv, Dl) = _pool(num_blocks=3, block_size=2)
    rng = np.random.RandomState(31)
    pool.add_sequence("a")
    pool.append("a", rng.randn(5, Dl).astype(np.float32),
                rng.randn(5, dr).astype(np.float32))
    assert pool.num_free_blocks == 0
    pool.trim("a", 99)
    assert pool.sequence_length("a") == 0
    assert pool.block_table("a") == []
    assert pool.num_free_blocks == 3


def test_mla_paged_decoder_trim_preserves_prefix_decode():
    Wuk_t, Wuv, dims = _weights(seed=32)
    H, dn, dr, dv, Dl = dims
    rng = np.random.RandomState(32)
    dec = _ref_decoder(Wuk_t, Wuv, dims, max_seq=16)
    c = rng.randn(7, Dl).astype(np.float32)
    r = rng.randn(7, dr).astype(np.float32)
    dec.append(c, r)
    assert dec.is_trimmable()
    dec.trim(3)
    assert dec.current_seq == 4

    ref = _ref_decoder(Wuk_t, Wuv, dims, max_seq=16)
    ref.append(c[:4], r[:4])
    qn = rng.randn(H, dn).astype(np.float32)
    qr = rng.randn(H, dr).astype(np.float32)
    np.testing.assert_allclose(dec.decode(qn, qr), ref.decode(qn, qr),
                               rtol=1e-5, atol=1e-5)


def test_utilization_and_footprint():
    pool, _, _, (H, dn, dr, dv, Dl) = _pool(num_blocks=10, block_size=4,
                                            H=128, dn=128, dr=64, dv=128, Dl=512)
    rng = np.random.RandomState(7)
    pool.add_sequence("a")
    pool.append("a", rng.randn(8, Dl).astype(np.float32),
                rng.randn(8, dr).astype(np.float32))
    assert pool.num_used_blocks == 2
    assert abs(pool.utilization - 0.2) < 1e-9
    # latent + rope per token vs explicit per-head K/V
    explicit = H * (dn + dr + dv) * 4
    assert pool.cache_bytes_per_token() == (512 + 64) * 4
    assert explicit / pool.cache_bytes_per_token() > 8.0


def test_decode_batch_same_length_grouped():
    """Same-length sequences are dispatched together (B>1); the batched result
    must match per-sequence single decodes exactly."""
    pool, Wuk_t, Wuv, dims = _pool(num_blocks=32, block_size=4)
    H, dn, dr, dv, Dl = dims
    rng = np.random.RandomState(20)
    L = 7
    queries = {}
    for sid in ("s0", "s1", "s2", "s3"):          # all length 7 -> one group
        pool.add_sequence(sid)
        pool.append(sid, rng.randn(L, Dl).astype(np.float32),
                    rng.randn(L, dr).astype(np.float32))
        queries[sid] = (rng.randn(H, dn).astype(np.float32),
                        rng.randn(H, dr).astype(np.float32))
    batched = pool.decode_batch(queries)
    for sid, (qn, qr) in queries.items():
        single = pool.decode(sid, qn, qr)         # per-sequence path
        np.testing.assert_allclose(batched[sid], single, rtol=1e-5, atol=1e-5)


def test_decode_batch_mixed_lengths_grouping():
    """Mixed lengths form multiple groups; every sequence still matches its own
    single decode regardless of group size (including singletons)."""
    pool, Wuk_t, Wuv, dims = _pool(num_blocks=64, block_size=4)
    H, dn, dr, dv, Dl = dims
    rng = np.random.RandomState(21)
    # lengths: two at 5, three at 8, one at 3
    spec = {"a": 5, "b": 5, "c": 8, "d": 8, "e": 8, "f": 3}
    queries = {}
    for sid, L in spec.items():
        pool.add_sequence(sid)
        pool.append(sid, rng.randn(L, Dl).astype(np.float32),
                    rng.randn(L, dr).astype(np.float32))
        queries[sid] = (rng.randn(H, dn).astype(np.float32),
                        rng.randn(H, dr).astype(np.float32))
    batched = pool.decode_batch(queries)
    assert set(batched) == set(spec)
    for sid, (qn, qr) in queries.items():
        np.testing.assert_allclose(batched[sid], pool.decode(sid, qn, qr),
                                   rtol=1e-5, atol=1e-5)


def test_absorb_decode_batch_matches_per_seq():
    """The batched helper equals stacking single-sequence decodes."""
    from tessera.cache.mla_paged import absorb_decode_batch, absorb_decode_one
    rng = np.random.RandomState(22)
    G, H, dn, dr, dv, Dl, S = 3, 4, 16, 8, 16, 32, 6
    Wuk_t = (rng.randn(H, dn, Dl) * 0.3).astype(np.float32)
    Wuv = (rng.randn(H, Dl, dv) * 0.3).astype(np.float32)
    qn = (rng.randn(G, H, dn) * 0.3).astype(np.float32)
    qr = (rng.randn(G, H, dr) * 0.3).astype(np.float32)
    ckv = (rng.randn(G, S, Dl) * 0.3).astype(np.float32)
    kr = (rng.randn(G, S, dr) * 0.3).astype(np.float32)
    key_pos, q_pos = np.arange(S), S - 1
    batched = absorb_decode_batch(qn, qr, ckv, kr, Wuk_t, Wuv, key_pos, q_pos,
                                  10000.0, "interleaved")
    for g in range(G):
        single = absorb_decode_one(qn[g], qr[g], ckv[g], kr[g], Wuk_t, Wuv,
                                   key_pos, q_pos, 10000.0, "interleaved")
        np.testing.assert_allclose(batched[g], single, rtol=1e-5, atol=1e-5)


def test_sequence_lifecycle_errors():
    pool, _, _, (H, dn, dr, dv, Dl) = _pool()
    with pytest.raises(MLABlockPagedCacheError):
        pool.decode("ghost", np.zeros((H, dn), np.float32),
                    np.zeros((H, dr), np.float32))
    pool.add_sequence("a")
    with pytest.raises(MLABlockPagedCacheError):
        pool.add_sequence("a")                 # duplicate
    with pytest.raises(MLABlockPagedCacheError):
        pool.decode("a", np.zeros((H, dn), np.float32),
                    np.zeros((H, dr), np.float32))  # empty
    with pytest.raises(MLABlockPagedCacheError):
        pool.free_sequence("nope")
