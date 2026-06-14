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
