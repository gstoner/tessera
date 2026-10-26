"""SM120-SAVED-LSE-MIXED-BACKWARD-20261026: mixed Q vs K/V reverse kernels with
pruned, non-compact gradient outputs. An inactive gradient buffer is still
physically present and zero-filled, so its stores must use that role's own
storage; guard bytes catch a store of the wrong width."""
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_saved_lse_half_backward_kernel import NativeKernel, compare, reference
from tests.unit.test_saved_lse_half_backward import source

pytestmark = pytest.mark.hardware_nvidia

DTYPES = {"f16": np.dtype(np.float16), "bf16": np.dtype(ml_dtypes.bfloat16), "f32": np.dtype(np.float32)}


@pytest.mark.parametrize("storage,kv_storage", [
    ("f32", "f16"), ("f32", "bf16"), ("f16", "f32"), ("bf16", "f32"), ("f16", "bf16"), ("bf16", "f16")])
@pytest.mark.parametrize("wrt", [(0, 2), (0, 1), (1, 2), (0,)])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("dims", [(1, 2, 1, 3, 5, 4, 3), (2, 2, 2, 1, 129, 8, 6)])
def test_mixed_reverse_zero_fills_inactive_gradients_in_their_own_storage(storage, kv_storage, wrt, saved, dims):
    import tessera.runtime as rt
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype, kv_dtype = DTYPES[storage], DTYPES[kv_storage]
    rng = np.random.default_rng(20261026)
    b, hq, hkv, sq, sk, d, dv = dims
    q = rng.normal(0, .4, (b, hq, sq, d)).astype(dtype)
    k = rng.normal(0, .4, (b, hkv, sk, d)).astype(kv_dtype)
    v = rng.normal(0, .4, (b, hkv, sk, dv)).astype(kv_dtype)
    do = rng.normal(0, .4, (b, hq, sq, dv)).astype(dtype)
    row_seed = rng.normal(0, .2, (b, hq, sq)).astype(np.float32) if saved else np.zeros((b, hq, sq), np.float32)
    primal, expected = reference(q, k, v, do, None, row_seed, causal=True)
    original = source(storage, saved=saved, dims=dims, kv_storage=kv_storage).replace(
        'tessera.autodiff = "reverse"',
        f'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [{", ".join(map(str, wrt))}]')
    with NvidiaDeviceSession() as session:
        roots = [session.upload(x) for x in (q, k, v)]
        guards = []

        def guarded(shape, dt):
            host = np.full(int(np.prod(shape))+16, 37, dtype=dt)
            arena = session.upload(host)
            guards.append(arena)
            return arena.view(8*host.dtype.itemsize, shape, host.dtype)
        output = guarded((b, hq, sq, dv), dtype)
        lse = guarded((b, hq, sq), np.float32)
        forward = NativeKernel(original, "forward")
        backward = None
        try:
            forward.launch(session, roots+[output, lse], b*hq*sq*dv)
            assert session.synchronize() == 0
            backward = NativeKernel(original, "backward", " prune-checkpoint-gradients=true")
            assert "gradient_activity" in backward.tile and 'gradient_output = "compact_v1"' not in backward.tile
            assert f'kv_storage = "{kv_storage}"' in backward.tile
            outputs = [guarded(x.shape, dtype if index == 0 else kv_dtype) for index, x in enumerate(expected)]
            reverse_inputs = [session.upload(do), *roots, output, lse]
            if saved:
                reverse_inputs.append(session.upload(row_seed))
            backward.launch(session, reverse_inputs+outputs, sum(x.size for x in expected))
            assert session.synchronize() == 0
            for role, (result, oracle) in enumerate(zip(outputs, expected, strict=True)):
                if role in wrt:
                    compare(result.numpy(), oracle)
                else:
                    np.testing.assert_array_equal(result.numpy(), np.zeros(oracle.shape, result.dtype))
            for arena in guards:
                host = arena.numpy()
                np.testing.assert_array_equal(host[:8], np.full(8, 37, host.dtype))
                np.testing.assert_array_equal(host[-8:], np.full(8, 37, host.dtype))
        finally:
            assert session.synchronize() == 0
            if backward is not None:
                backward.close()
            forward.close()
