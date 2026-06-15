"""Device-resident memory bank — read-residency for resident_state_handle.

The bank's keys stay on-device across reads (one upload); each read scores the
query against the resident bank without re-uploading it. Metamorphic check:
resident reads == reference reads. Traffic check: residency uploads the bank
once, recompute would re-upload it every read.
"""

from __future__ import annotations

import numpy as np

from tessera.resident_bank import ResidentBank


def _bank(seed: int, n: int = 64, d: int = 32, vd: int = 4):
    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((n, d)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    vals = rng.standard_normal((n, vd)).astype(np.float32)
    return keys, vals


def test_resident_read_matches_reference():
    keys, vals = _bank(0)
    with ResidentBank(keys, vals) as bank:
        for i in range(5):
            q = keys[i]                         # self-query → recovers entry i
            v, idx, scores = bank.read(q, top_k=1)
            assert idx[0] == i
            assert np.allclose(v, vals[i], atol=1e-3)


def test_resident_top_k_is_descending():
    keys, vals = _bank(1)
    with ResidentBank(keys, vals) as bank:
        _v, idx, scores = bank.read(keys[3], top_k=5)
        assert idx[0] == 3                       # self is the top match
        assert np.all(np.diff(scores) <= 1e-5)   # descending


def test_metamorphic_resident_equals_recompute():
    # the resident bank must produce identical reads to a from-scratch numpy bank
    keys, vals = _bank(2, n=128)
    with ResidentBank(keys, vals) as bank:
        rng = np.random.default_rng(99)
        for _ in range(8):
            q = rng.standard_normal((keys.shape[1],)).astype(np.float32)
            _v, idx, scores = bank.read(q, top_k=3)
            ref_scores = q @ keys.T
            ref_idx = np.argsort(ref_scores)[::-1][:3]
            assert np.array_equal(idx, ref_idx)


def test_traffic_reduction_bank_uploads_once():
    keys, vals = _bank(3, n=256)
    with ResidentBank(keys, vals) as bank:
        for i in range(16):
            bank.read(keys[i], top_k=1)
        tel = bank.telemetry()
        assert tel["reads"] == 16
        assert tel["bank_uploads"] == 1
        # recompute re-uploads the bank every read; residency uploads it once
        assert tel["recompute_upload_bytes"] == 16 * bank.bank_bytes
        assert tel["resident_upload_bytes"] < tel["recompute_upload_bytes"]
        assert tel["upload_reduction_x"] > 10.0          # ~16× for a 256-row bank


def test_empty_and_shape_guards():
    import pytest

    with pytest.raises(ValueError):
        ResidentBank(np.zeros((4, 8), np.float32), np.zeros((3, 2), np.float32))
    bank = ResidentBank(np.zeros((0, 8), np.float32), np.zeros((0, 2), np.float32))
    with pytest.raises(ValueError):
        bank.read(np.zeros((8,), np.float32))
    bank.free()


# ── append-residency (kv_cache_append_read) ──────────────────────────────────


def test_append_then_read_decode_loop():
    # start empty, append one (key, value) per step, read it back each step —
    # the decode-loop pattern. The bank is never re-uploaded.
    d, vd, T = 16, 4, 32
    keys, vals = _bank(10, n=T, d=d, vd=vd)
    bank = ResidentBank(np.zeros((0, d), np.float32), np.zeros((0, vd), np.float32),
                        capacity=T)
    for t in range(T):
        idx = bank.append(keys[t], vals[t])
        assert idx == t
        v, ridx, _s = bank.read(keys[t], top_k=1)      # the just-appended entry
        assert ridx[0] == t
        assert np.allclose(v, vals[t], atol=1e-3)
    assert bank.current_seq == T
    bank.free()


def test_append_metamorphically_equals_recompute():
    # appending into a resident bank == rebuilding the whole bank from scratch
    # each step (the recompute path) — residency changes cost, never output.
    d, vd, T = 16, 4, 24
    keys, vals = _bank(11, n=T, d=d, vd=vd)
    bank = ResidentBank(np.zeros((0, d), np.float32), np.zeros((0, vd), np.float32),
                        capacity=T)
    rng = np.random.default_rng(7)
    for t in range(T):
        bank.append(keys[t], vals[t])
        q = rng.standard_normal((d,)).astype(np.float32)
        _v, idx, _s = bank.read(q, top_k=2)
        ref_scores = keys[: t + 1] @ q                  # recompute over keys[:t+1]
        ref_idx = np.argsort(ref_scores)[::-1][:2]
        assert np.array_equal(idx, ref_idx)
    bank.free()


def test_append_traffic_is_per_entry_not_per_bank():
    d, T = 16, 64
    keys, vals = _bank(12, n=T, d=d, vd=2)
    bank = ResidentBank(np.zeros((0, d), np.float32), np.zeros((0, 2), np.float32),
                        capacity=T)
    for t in range(T):
        bank.append(keys[t], vals[t])
        bank.read(keys[t], top_k=1)
    tel = bank.telemetry()
    assert tel["appendable"] is True
    # each append uploaded exactly one key (O(d)), never the whole bank
    assert tel["append_bytes"] == T * d * 4
    # recompute would re-send the growing bank every read: Σ t = T(T+1)/2 keys
    assert tel["recompute_upload_bytes"] == (T * (T + 1) // 2) * d * 4
    assert tel["upload_reduction_x"] > 5.0


def test_fixed_bank_rejects_append():
    import pytest

    keys, vals = _bank(13)
    bank = ResidentBank(keys, vals)                     # no capacity → fixed
    with pytest.raises(ValueError):
        bank.append(keys[0], vals[0])
    bank.free()


def test_append_past_capacity_raises():
    import pytest

    keys, vals = _bank(14, n=4)
    bank = ResidentBank(keys, vals, capacity=4)          # already full
    with pytest.raises(ValueError):
        bank.append(keys[0], vals[0])
    bank.free()
