"""Owning SM120 residual producer proof; public program integration remains separate."""
import ctypes as ct

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_saved_lse_half_backward_kernel import NativeKernel, compare, reference
from tests.unit.test_saved_lse_half_backward import source

pytestmark = pytest.mark.hardware_nvidia


class RowKernel(NativeKernel):
    def launch(self, session, buffers, elements):
        values = [ct.c_void_p(buffer.ptr) for buffer in buffers]
        values += [ct.c_int64(x) for x in self.dims]
        parameters = (ct.c_void_p*len(values))(*(ct.addressof(value) for value in values))
        self.check(self.driver.cuLaunchKernel(self.function, (elements+31)//32, 1, 1,
                   32, 1, 1, 0, ct.c_void_p(session.stream), parameters, None))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("bias,seeded", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dims", [(1, 2, 1, 3, 5, 4, 3), (2, 4, 2, 1, 129, 8, 6), (1, 2, 1, 7, 3, 4, 5)])
@pytest.mark.parametrize("causal", [False, True])
def test_native_row_producer_matches_independent_probability_oracle(storage, wide, bias, seeded, dims, causal):
    import tessera.runtime as rt
    assert rt._nvidia_device_name() == "sm_120", "exact SM120 proof lane requires a live owning GPU"
    dt = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    out_dt = np.dtype(np.float32) if wide else dt
    b, hq, hkv, sq, sk, d, dv = dims
    original = source(storage, bias=bias, saved=seeded, output="f32" if wide else storage,
                      dims=dims, causal=causal)
    rng = np.random.default_rng(20261016)
    with NvidiaDeviceSession() as session:
        forward = NativeKernel(original, "forward")
        correction = None
        try:
            correction = RowKernel(original, "row_correction")
            # Reuse exactly the compiled images with changed roots and seeds.
            for _ in range(2):
                q = rng.normal(0, .4, (b, hq, sq, d)).astype(dt)
                k = rng.normal(0, .4, (b, hkv, sk, d)).astype(dt)
                v = rng.normal(0, .4, (b, hkv, sk, dv)).astype(dt)
                do = rng.normal(0, .4, (b, hq, sq, dv)).astype(out_dt)
                bias_host = rng.normal(0, .1, (1, hq, 1, sk)).astype(np.float32) if bias else None
                seed = rng.normal(0, .2, (b, hq, sq)).astype(np.float32) if seeded else np.zeros((b, hq, sq), np.float32)
                primal, _ = reference(q, k, v, do, bias_host, seed, causal=causal)
                roots = [session.upload(x) for x in (q, k, v)]
                bias_buffer = session.upload(bias_host) if bias else None
                output = session.empty((b, hq, sq, dv), out_dt)
                lse = session.empty((b, hq, sq), np.float32)
                forward.launch(session, roots+([bias_buffer] if bias else [])+[output, lse], b*hq*sq*dv)
                assert session.synchronize() == 0
                compare(lse.numpy(), primal[1])
                arena = session.upload(np.full(b*hq*sq+16, 37, np.float32))
                row = arena.view(8*4, (b, hq, sq), np.dtype(np.float32))
                inputs = [session.upload(do), *roots]+([bias_buffer] if bias else [])+[lse]
                if seeded:
                    inputs.append(session.upload(seed))
                correction.launch(session, inputs+[row], b*hq*sq)
                assert session.synchronize() == 0
                # Independent FP64 oracle uses the actual saved FP32 LSE.
                keys = np.repeat(k.astype(np.float64), hq//hkv, axis=1)
                vals = np.repeat(v.astype(np.float64), hq//hkv, axis=1)
                scores = q.astype(np.float64) @ keys.swapaxes(-1, -2) / np.sqrt(d)
                if bias:
                    scores += bias_host
                if causal:
                    mask = np.arange(sk)[None, :] <= np.arange(sq)[:, None]+max(sk-sq, 0)
                    scores = np.where(mask, scores, -np.inf)
                probabilities = np.exp(scores-lse.numpy().astype(np.float64)[..., None])
                dp = do.astype(np.float64) @ vals.swapaxes(-1, -2)
                expected = (probabilities*dp).sum(-1)-seed
                np.testing.assert_allclose(row.numpy(), expected, atol=3e-6, rtol=3e-5)
                guarded = arena.numpy()
                np.testing.assert_array_equal(guarded[:8], np.full(8, 37, np.float32))
                np.testing.assert_array_equal(guarded[-8:], np.full(8, 37, np.float32))
        finally:
            assert session.synchronize() == 0
            if correction is not None:
                correction.close()
            forward.close()
