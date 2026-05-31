"""Apple GPU interop escape hatches — raw MTLDevice / MTLCommandQueue / MTLBuffer
handles (cf. Mojo's metal_device(ctx)).

These hand back the *same* Metal objects Tessera's runtime uses, as integer
pointers, so external Metal/MPS code can compose with Tessera's resident buffers.
All return 0 off Metal. Tessera owns their lifetime.
"""

import numpy as np

import tessera.runtime as R


def _on_metal() -> bool:
    return R.DeviceTensor.is_metal()


def test_device_handle_nonzero_and_stable_on_metal():
    d1 = R.apple_gpu_device_handle()
    d2 = R.apple_gpu_device_handle()
    assert d1 == d2                      # process-wide singleton -> stable pointer
    if _on_metal():
        assert d1 != 0
    else:
        assert d1 == 0


def test_command_queue_handle():
    q = R.apple_gpu_command_queue_handle()
    assert q == R.apple_gpu_command_queue_handle()
    if _on_metal():
        assert q != 0
        # device and queue are distinct objects.
        assert q != R.apple_gpu_device_handle()


def test_device_tensor_mtl_buffer():
    dt = R.DeviceTensor.from_numpy(np.arange(16, dtype=np.float32))
    if dt is None:
        return
    try:
        b1 = dt.mtl_buffer()
        b2 = dt.mtl_buffer()
        assert b1 == b2                  # same buffer object across calls
        if _on_metal():
            assert b1 != 0
    finally:
        dt.free()
    # After free, the accessor must not hand back a dangling pointer.
    assert dt.mtl_buffer() == 0


def test_mtl_buffer_distinct_per_tensor():
    a = R.DeviceTensor.from_numpy(np.zeros(8, np.float32))
    b = R.DeviceTensor.from_numpy(np.zeros(8, np.float32))
    if a is None or b is None:
        return
    try:
        if _on_metal():
            assert a.mtl_buffer() != 0 and b.mtl_buffer() != 0
            assert a.mtl_buffer() != b.mtl_buffer()
    finally:
        a.free()
        b.free()
