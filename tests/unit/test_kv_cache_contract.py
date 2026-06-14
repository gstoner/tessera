"""Direct contract tests for the KV-cache state primitives
(``kv_cache_append`` / ``kv_cache_read`` / ``kv_cache_prune``).

These promote the KV-cache ops out of ``structural_only`` (they are NOT
metadata wrappers — they carry real numerical state behavior): append↔read
roundtrip, recency-windowed prune, explicit eviction, sliding-window
``auto_evict`` overflow, and quantized-storage roundtrip.  This is the long-
context "don't reprocess history" substrate the RULER/LongMemEval benchmarks
stress; correctness here is an aliasing/state property.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


def _chunk(n, h=2, d=4, scale=1.0, base=0.0):
    rng = np.random.default_rng(int(base * 1000) + n * 7 + h)
    k = (rng.standard_normal((n, h, d)) * scale + base).astype(np.float32)
    v = (rng.standard_normal((n, h, d)) * scale + base).astype(np.float32)
    return k, v


# ── append ↔ read roundtrip ──────────────────────────────────────────────────


def test_append_then_read_returns_written_tokens():
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
    k, v = _chunk(5)
    c.append(k, v)
    assert c.current_seq == 5
    k_back, v_back = c.read(0, 5)
    assert np.allclose(k_back, k, atol=1e-6)
    assert np.allclose(v_back, v, atol=1e-6)


def test_append_is_incremental_no_reprocessing():
    # decode pattern: append one token at a time, history stays put
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
    written_k = []
    for t in range(8):
        k, v = _chunk(1, base=float(t))
        c.append(k, v)
        written_k.append(k)
        # each step only adds one token; prior tokens are unchanged
        k_so_far, _ = c.read(0, c.current_seq)
        assert np.allclose(k_so_far, np.concatenate(written_k), atol=1e-6)
    assert c.current_seq == 8


def test_single_arg_read_is_one_token():
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8)
    k, v = _chunk(3)
    c.append(k, v)
    k1, v1 = c.read(1)                       # [1, 2)
    assert k1.shape[0] == 1
    assert np.allclose(k1[0], k[1], atol=1e-6)


def test_read_out_of_range_raises():
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8)
    c.append(*_chunk(3))
    with pytest.raises(IndexError):
        c.read(0, 5)                          # end past current_seq


# ── prune (recency window) ───────────────────────────────────────────────────


def test_prune_keeps_trailing_window():
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
    k, v = _chunk(10)
    c.append(k, v)
    c.prune(max_entries=4)                    # keep last 4
    assert c.current_seq == 4
    k_back, v_back = c.read(0, 4)
    assert np.allclose(k_back, k[6:], atol=1e-6)   # the newest 4 survive
    assert np.allclose(v_back, v[6:], atol=1e-6)


def test_prune_noop_when_under_budget():
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
    c.append(*_chunk(3))
    c.prune(max_entries=10)
    assert c.current_seq == 3


# ── explicit eviction ────────────────────────────────────────────────────────


def test_evict_oldest_shifts_window():
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
    k, v = _chunk(6)
    c.append(k, v)
    c.evict_oldest(2)
    assert c.current_seq == 4
    k_back, _ = c.read(0, 4)
    assert np.allclose(k_back, k[2:], atol=1e-6)


# ── sliding-window auto_evict on overflow ────────────────────────────────────


def test_auto_evict_provides_sliding_window():
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=4, auto_evict=True)
    k, v = _chunk(3, base=1.0)
    c.append(k, v)                            # seq=3
    k2, v2 = _chunk(3, base=2.0)
    c.append(k2, v2)                          # overflow → keep last 4 of 6
    assert c.current_seq == 4
    k_win, _ = c.read(0, 4)
    expected = np.concatenate([k, k2])[-4:]   # newest 4 tokens
    assert np.allclose(k_win, expected, atol=1e-6)


def test_overflow_without_auto_evict_raises():
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=4)
    c.append(*_chunk(3))
    with pytest.raises(ValueError):
        c.append(*_chunk(3))


# ── quantized storage roundtrip ──────────────────────────────────────────────


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_append_read_roundtrip(bits):
    c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16, quantize_bits=bits)
    k, v = _chunk(6, scale=0.5)
    c.append(k, v)
    k_back, v_back = c.read(0, 6)             # dequantized on egress
    # lossy storage — assert close at a quant-appropriate tolerance, not exact
    tol = 0.1 if bits >= 8 else 0.35
    assert np.abs(k_back - k).max() < tol
    assert np.abs(v_back - v).max() < tol
    assert k_back.dtype == np.float32         # opaque at the read boundary
