"""Apple GPU R0 — persistent device-tensor handle (GPU-resident activations).

`runtime.DeviceTensor` wraps one shared (unified-memory) Metal buffer; on Apple
Silicon `.numpy()` is a zero-copy view over the same bytes the GPU sees, so
after the one-time `from_numpy` copy there are no further host↔device copies.
On non-Apple hosts the handle is host-memory-backed, so the surface is portable.
This is the foundation for R1–R3 (op-to-op residency + command-buffer batching).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R
from tessera.runtime import DeviceTensor


def _require_api():
    if R._apple_gpu_devtensor_api() is None:
        pytest.skip("device-tensor ABI unavailable")


def test_roundtrip_f32():
    _require_api()
    rng = np.random.RandomState(0)
    a = rng.randn(4, 8).astype(np.float32)
    dt = DeviceTensor.from_numpy(a)
    assert dt is not None
    assert dt.shape == (4, 8) and dt.dtype == np.float32
    np.testing.assert_array_equal(dt.numpy(), a)
    dt.free()


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.int64])
def test_roundtrip_dtypes(dtype):
    _require_api()
    rng = np.random.RandomState(1)
    a = (rng.randn(3, 5) * 4).astype(dtype)
    dt = DeviceTensor.from_numpy(a)
    assert dt is not None and dt.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(dt.numpy(), a)
    dt.free()


def test_numpy_view_is_zero_copy_coherent():
    """The .numpy() view aliases the shared storage: a host write through the
    view is visible on a fresh view (same backing buffer, no copy)."""
    _require_api()
    a = np.zeros((2, 4), np.float32)
    dt = DeviceTensor.from_numpy(a)
    view = dt.numpy()
    view[0, 0] = 42.0
    view[1, 3] = -7.0
    fresh = dt.numpy()
    assert fresh[0, 0] == 42.0 and fresh[1, 3] == -7.0
    dt.free()


def test_copy_to_host_outlives_free():
    _require_api()
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    dt = DeviceTensor.from_numpy(a)
    host = dt.copy_to_host()
    dt.free()
    np.testing.assert_array_equal(host, a)  # still valid after free


def test_empty_then_fill():
    _require_api()
    dt = DeviceTensor.empty((2, 3), np.float32)
    assert dt is not None
    v = dt.numpy()
    v[:] = 5.0
    np.testing.assert_array_equal(dt.numpy(), np.full((2, 3), 5.0, np.float32))
    dt.free()


def test_nbytes():
    _require_api()
    dt = DeviceTensor.from_numpy(np.zeros((4, 8), np.float32))
    assert dt.nbytes == 4 * 8 * 4
    dt.free()


def test_use_after_free_raises():
    _require_api()
    dt = DeviceTensor.from_numpy(np.zeros(4, np.float32))
    dt.free()
    with pytest.raises(RuntimeError):
        dt.numpy()


def test_double_free_is_safe():
    _require_api()
    dt = DeviceTensor.from_numpy(np.zeros(4, np.float32))
    dt.free()
    dt.free()  # no crash


def test_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    for name in ("ts_dev_alloc", "ts_dev_contents", "ts_dev_nbytes",
                 "ts_dev_upload", "ts_dev_download", "ts_dev_free",
                 "ts_dev_is_metal"):
        assert hasattr(rt, name), name


def test_many_alloc_free_no_leak_smoke():
    _require_api()
    for i in range(200):
        dt = DeviceTensor.from_numpy(np.full((16, 16), i, np.float32))
        assert dt.numpy()[0, 0] == i
        dt.free()
