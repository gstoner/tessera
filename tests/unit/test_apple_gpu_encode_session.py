"""Apple GPU R2 — command-buffer batching (one commit per op-chain).

`AppleGPUEncodeSession` encodes a chain of device-resident ops into a single
command buffer and commits + waits once, instead of one synchronous run (and
one CPU↔GPU sync) per op. The critical correctness question is whether Metal's
automatic hazard tracking orders a later op that reads an earlier op's output
buffer within the same command buffer — these tests verify it does, by checking
multi-op chains against numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R
from tessera.runtime import DeviceTensor, AppleGPUEncodeSession


def _require():
    s = AppleGPUEncodeSession()
    if not s.available:
        s.commit()
        pytest.skip("encode-session ABI unavailable")
    return s


def test_single_op_encoded():
    s = _require()
    rng = np.random.RandomState(0)
    A = rng.randn(2, 4, 5).astype(np.float32)
    B = rng.randn(2, 5, 6).astype(np.float32)
    da, db = DeviceTensor.from_numpy(A), DeviceTensor.from_numpy(B)
    with s:
        c = s.bmm(da, db)
        assert c is not None
    # valid only after commit (the `with` exit)
    np.testing.assert_allclose(c.numpy(),
                               np.matmul(A.astype(np.float64), B.astype(np.float64)),
                               rtol=1e-4, atol=1e-4)


def test_dependent_chain_one_commit():
    """C = bmm(A,B); D = bmm(C,E) — both encoded into ONE command buffer. D is
    correct only if Metal ordered the second bmm after the first (hazard
    tracking on C's buffer)."""
    s = _require()
    rng = np.random.RandomState(1)
    A = rng.randn(2, 4, 5).astype(np.float32)
    B = rng.randn(2, 5, 6).astype(np.float32)
    E = rng.randn(2, 6, 3).astype(np.float32)
    da, db, de = (DeviceTensor.from_numpy(x) for x in (A, B, E))
    with s:
        c = s.bmm(da, db)
        d = s.bmm(c, de)
        assert c is not None and d is not None
    ref = np.matmul(np.matmul(A.astype(np.float64), B.astype(np.float64)),
                    E.astype(np.float64))
    np.testing.assert_allclose(d.numpy(), ref, rtol=1e-4, atol=1e-4)


def test_deep_dependent_chain():
    """A 5-deep dependent chain in a single command buffer / single commit."""
    s = _require()
    rng = np.random.RandomState(2)
    mats = [rng.randn(1, 8, 8).astype(np.float32) for _ in range(6)]
    dts = [DeviceTensor.from_numpy(m) for m in mats]
    with s:
        acc = dts[0]
        for nxt in dts[1:]:
            acc = s.bmm(acc, nxt)
            assert acc is not None
    ref = mats[0].astype(np.float64)
    for m in mats[1:]:
        ref = np.matmul(ref, m.astype(np.float64))
    np.testing.assert_allclose(acc.numpy(), ref, rtol=1e-3, atol=1e-3)


def test_long_chain_auto_flushes_across_command_buffers():
    """A chain longer than a single Metal command buffer can hold must not
    crash. A single MTLCommandBuffer has finite command space; encoding an
    unbounded op-chain into one buffer overflows it and segfaults at commit
    (encodeSignalEvent / _reserveKernelCommandBufferSpace). The session
    auto-flushes across several command buffers — correctness is preserved
    (each flush commits + waits before the next buffer reads the result) and
    only the commit count grows."""
    s = _require()
    rng = np.random.RandomState(7)
    n = 50  # past the default 24-op per-command-buffer ceiling
    # scale by 1/sqrt(dim) so the deep product stays finite (numerics aside,
    # this exercises the cross-command-buffer ordering, not overflow).
    mats = [(rng.randn(1, 8, 8) / np.sqrt(8)).astype(np.float32) for _ in range(n + 1)]
    dts = [DeviceTensor.from_numpy(m) for m in mats]
    with s:
        acc = dts[0]
        for nxt in dts[1:]:
            acc = s.bmm(acc, nxt)
            assert acc is not None
    assert s.commits > 1, "a 50-op chain must span multiple command buffers"
    ref = mats[0].astype(np.float64)
    for m in mats[1:]:
        ref = np.matmul(ref, m.astype(np.float64))
    np.testing.assert_allclose(acc.numpy(), ref, rtol=1e-3, atol=1e-3)


def test_explicit_flush_threshold_is_correct():
    """A small ``max_ops_per_cb`` forces an auto-flush mid-chain; the result
    must still match numpy (the committed buffer's outputs are valid before the
    next buffer reads them) and the commit count must reflect the flushes."""
    s = AppleGPUEncodeSession(max_ops_per_cb=4)
    if not s.available:
        s.commit()
        pytest.skip("encode-session ABI unavailable")
    rng = np.random.RandomState(11)
    n = 10  # -> flush after 4 and 8 ops, final commit -> 3 command buffers
    mats = [(rng.randn(1, 6, 6) / np.sqrt(6)).astype(np.float32) for _ in range(n + 1)]
    dts = [DeviceTensor.from_numpy(m) for m in mats]
    with s:
        acc = dts[0]
        for nxt in dts[1:]:
            acc = s.bmm(acc, nxt)
    assert s.commits == 3
    ref = mats[0].astype(np.float64)
    for m in mats[1:]:
        ref = np.matmul(ref, m.astype(np.float64))
    np.testing.assert_allclose(acc.numpy(), ref, rtol=1e-3, atol=1e-3)


def test_independent_ops_one_commit():
    """Two independent bmms batched into one command buffer; both correct."""
    s = _require()
    rng = np.random.RandomState(3)
    A1, B1 = rng.randn(2, 3, 4).astype(np.float32), rng.randn(2, 4, 5).astype(np.float32)
    A2, B2 = rng.randn(1, 6, 2).astype(np.float32), rng.randn(1, 2, 7).astype(np.float32)
    da1, db1, da2, db2 = (DeviceTensor.from_numpy(x) for x in (A1, B1, A2, B2))
    with s:
        o1 = s.bmm(da1, db1)
        o2 = s.bmm(da2, db2)
    np.testing.assert_allclose(o1.numpy(),
                               np.matmul(A1.astype(np.float64), B1.astype(np.float64)),
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(o2.numpy(),
                               np.matmul(A2.astype(np.float64), B2.astype(np.float64)),
                               rtol=1e-4, atol=1e-4)


def test_matches_unbatched_device_bmm():
    """Encoded (one commit) and per-op device bmm (R1, one sync each) agree."""
    s = _require()
    rng = np.random.RandomState(4)
    A = rng.randn(3, 5, 5).astype(np.float32)
    B = rng.randn(3, 5, 5).astype(np.float32)
    da, db = DeviceTensor.from_numpy(A), DeviceTensor.from_numpy(B)
    # per-op (R1)
    r1 = R._apple_gpu_bmm_device(da, db)
    # batched (R2)
    with s:
        r2 = s.bmm(da, db)
    np.testing.assert_allclose(r2.numpy(), r1.numpy(), rtol=1e-5, atol=1e-5)


def test_explicit_commit_then_read():
    s = _require()
    rng = np.random.RandomState(5)
    A = rng.randn(1, 4, 4).astype(np.float32)
    B = rng.randn(1, 4, 4).astype(np.float32)
    da, db = DeviceTensor.from_numpy(A), DeviceTensor.from_numpy(B)
    c = s.bmm(da, db)
    s.commit()
    np.testing.assert_allclose(c.numpy(),
                               np.matmul(A.astype(np.float64), B.astype(np.float64)),
                               rtol=1e-4, atol=1e-4)


def test_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    for name in ("ts_enc_begin", "ts_enc_commit_wait",
                 "tessera_apple_gpu_bmm_dev_f32_enc"):
        assert hasattr(rt, name), name
