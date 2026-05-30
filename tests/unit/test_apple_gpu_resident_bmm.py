"""Apple GPU R1 — device-resident bmm (op-to-op residency).

`runtime._apple_gpu_bmm_device` consumes and produces `DeviceTensor` handles:
the inputs' shared buffers are used in place (no host upload) and the MPSGraph
result is written straight into the output buffer (no readback). So a chain of
device-resident ops keeps its intermediates on-GPU — the mechanism that lets the
decode loop stop round-tripping activations. Validated against numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R
from tessera.runtime import DeviceTensor


def _require():
    if R._apple_gpu_bmm_dev_f32() is None:
        pytest.skip("device-resident bmm unavailable")


def _ref_bmm(A, B):
    if B.shape[0] == 1 and A.shape[0] != 1:
        B = np.broadcast_to(B, (A.shape[0], B.shape[1], B.shape[2]))
    return np.matmul(A.astype(np.float64), B.astype(np.float64))


def test_device_bmm_matches_numpy():
    _require()
    rng = np.random.RandomState(0)
    A = rng.randn(3, 4, 5).astype(np.float32)
    B = rng.randn(3, 5, 6).astype(np.float32)
    da, db = DeviceTensor.from_numpy(A), DeviceTensor.from_numpy(B)
    out = R._apple_gpu_bmm_device(da, db)
    assert out is not None and out.shape == (3, 4, 6)
    np.testing.assert_allclose(out.numpy(), _ref_bmm(A, B), rtol=1e-4, atol=1e-4)
    for t in (da, db, out):
        t.free()


def test_device_bmm_broadcast_B():
    _require()
    rng = np.random.RandomState(1)
    A = rng.randn(4, 3, 8).astype(np.float32)
    B = rng.randn(1, 8, 7).astype(np.float32)   # shared across batch
    da, db = DeviceTensor.from_numpy(A), DeviceTensor.from_numpy(B)
    out = R._apple_gpu_bmm_device(da, db)
    assert out is not None and out.shape == (4, 3, 7)
    np.testing.assert_allclose(out.numpy(), _ref_bmm(A, B), rtol=1e-4, atol=1e-4)


def test_chain_keeps_intermediate_resident():
    """C = bmm(A, B); D = bmm(C, E). The intermediate C is consumed by the
    second bmm directly as a DeviceTensor — it is never materialized to host
    (we only call .numpy() on the final D)."""
    _require()
    rng = np.random.RandomState(2)
    A = rng.randn(2, 4, 5).astype(np.float32)
    B = rng.randn(2, 5, 6).astype(np.float32)
    E = rng.randn(2, 6, 3).astype(np.float32)
    da, db, de = (DeviceTensor.from_numpy(x) for x in (A, B, E))

    C = R._apple_gpu_bmm_device(da, db)          # resident intermediate
    assert C is not None
    D = R._apple_gpu_bmm_device(C, de)           # consumes C without a readback
    assert D is not None and D.shape == (2, 4, 3)

    ref = np.matmul(_ref_bmm(A, B), E.astype(np.float64))
    np.testing.assert_allclose(D.numpy(), ref, rtol=1e-4, atol=1e-4)
    for t in (da, db, de, C, D):
        t.free()


def test_resident_output_feeds_host_only_at_end():
    """A 3-deep chain; only the final output is read back to host."""
    _require()
    rng = np.random.RandomState(3)
    mats = [rng.randn(1, 8, 8).astype(np.float32) for _ in range(4)]
    dts = [DeviceTensor.from_numpy(m) for m in mats]
    acc = dts[0]
    for nxt in dts[1:]:
        acc = R._apple_gpu_bmm_device(acc, nxt)
        assert acc is not None
    ref = mats[0].astype(np.float64)
    for m in mats[1:]:
        ref = np.matmul(ref, m.astype(np.float64))
    np.testing.assert_allclose(acc.numpy(), ref, rtol=1e-3, atol=1e-3)


def test_device_bmm_rejects_non_f32():
    _require()
    a = DeviceTensor.from_numpy(np.zeros((2, 3, 4), np.float16))
    b = DeviceTensor.from_numpy(np.zeros((2, 4, 5), np.float16))
    assert R._apple_gpu_bmm_device(a, b) is None


def test_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_bmm_dev_f32")
    assert R._apple_gpu_bmm_dev_f32() is not None
