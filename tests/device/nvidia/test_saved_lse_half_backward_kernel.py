"""Owning-device native half reverse kernels; public typed owner is a separate gate."""
from __future__ import annotations

import ctypes as ct
import re

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_saved_lse_half_backward import source

pytestmark = pytest.mark.hardware_nvidia


class NativeKernel:
    def __init__(self, source_ir, role):
        tool = find_tessera_opt()
        assert tool is not None
        graph = run_tessera_opt(tool, source_ir, "--tessera-autodiff-paired=checkpoint-product="+role)
        schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
        tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
        self.entry = re.search(r"llvm.func @([^ (]+)", tile)[1]
        self.dims = tuple(map(int, re.search(r"\bshape = array<i64: ([^>]+)>", tile)[1].split(",")))
        self.target, self.ptx, *_ = _compile_tile_ir(tile, self.entry)
        self.graph, self.schedule, self.tile = graph, schedule, tile
        self.driver = ct.CDLL("libcuda.so.1")
        self.driver.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
        self.driver.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint]*7 + [ct.c_void_p]*3
        self.driver.cuModuleUnload.argtypes = [ct.c_void_p]
        self.module, self.function = ct.c_void_p(), ct.c_void_p()
        self.check(self.driver.cuModuleLoadData(ct.byref(self.module), ct.create_string_buffer(self.ptx.encode())))
        try:
            self.check(self.driver.cuModuleGetFunction(ct.byref(self.function), self.module, self.entry.encode()))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        assert status == 0, f"CUDA driver status {status}"

    def launch(self, session, buffers, elements):
        values = [ct.c_void_p(buffer.ptr) for buffer in buffers]
        values += [ct.c_int64(x) for x in self.dims]
        parameters = (ct.c_void_p*len(values))(*(ct.addressof(value) for value in values))
        self.check(self.driver.cuLaunchKernel(self.function, (elements+127)//128, 1, 1,
                   128, 1, 1, 0, ct.c_void_p(session.stream), parameters, None))

    def close(self):
        if self.module.value:
            self.check(self.driver.cuModuleUnload(self.module))
            self.module = ct.c_void_p()


def reference(q, k, v, do, bias, row_seed, *, causal=True):
    q, k, v, do = (x.astype(np.float64) for x in (q, k, v, do))
    b, hq, sq, d = q.shape
    hkv, sk, dv = k.shape[1], k.shape[2], v.shape[-1]
    groups = hq//hkv
    scale = 1/np.sqrt(d)
    keys, values = np.repeat(k, groups, axis=1), np.repeat(v, groups, axis=1)
    scores = (q @ keys.swapaxes(-1, -2))*scale
    if bias is not None:
        scores += bias
    if causal:
        mask = np.arange(sk)[None, :] <= np.arange(sq)[:, None]+max(sk-sq, 0)
        scores = np.where(mask, scores, -np.inf)
    maximum = scores.max(-1, keepdims=True)
    ex = np.exp(scores-maximum)
    denominator = ex.sum(-1, keepdims=True)
    p = ex/denominator
    output = p @ values
    lse = maximum[..., 0]+np.log(denominator[..., 0])
    dp = do @ values.swapaxes(-1, -2)
    ds = p*(dp-(p*dp).sum(-1, keepdims=True)+row_seed[..., None])
    dq = scale*ds @ keys
    dk = (scale*ds.swapaxes(-1, -2) @ q).reshape(b, hkv, groups, sk, d).sum(2)
    dvalue = (p.swapaxes(-1, -2) @ do).reshape(b, hkv, groups, sk, dv).sum(2)
    gradients = (dq, dk, dvalue)
    if bias is not None:
        axes = tuple(i for i, (physical, logical) in enumerate(zip(bias.shape, ds.shape, strict=True)) if physical == 1 and logical != 1)
        gradients += (ds.sum(axes, keepdims=True),)
    return (output, lse), gradients


def compare(actual, expected):
    rounded = expected.astype(actual.dtype)
    if actual.dtype == np.float32:
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
    else:
        center = rounded.astype(np.float64)
        upper = np.nextafter(rounded, actual.dtype.type(np.inf)).astype(np.float64)
        lower = np.nextafter(rounded, actual.dtype.type(-np.inf)).astype(np.float64)
        assert np.isfinite(actual).all()
        assert np.all(np.abs(actual.astype(np.float64)-center) <= np.maximum(upper-center, center-lower)+3e-6)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("wide_output", [False, True])
@pytest.mark.parametrize("dims", [(1, 2, 1, 3, 5, 4, 3), (2, 2, 2, 1, 129, 8, 6), (1, 2, 1, 7, 3, 4, 5)])
@pytest.mark.parametrize("causal", [False, True])
def test_original_half_reverse_kernels_match_fp64_with_guarded_stores(storage, bias, saved, wide_output, dims, causal):
    import tessera.runtime as rt
    if rt._nvidia_device_name() != "sm_120":
        pytest.skip("owning RTX5070/SM120 required")
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    output_dtype = np.dtype(np.float32) if wide_output else dtype
    rng = np.random.default_rng(20261016)
    b, hq, hkv, sq, sk, d, dv = dims
    q = rng.normal(0, .4, (b, hq, sq, d)).astype(dtype)
    k = rng.normal(0, .4, (b, hkv, sk, d)).astype(dtype)
    v = rng.normal(0, .4, (b, hkv, sk, dv)).astype(dtype)
    do = rng.normal(0, .4, (b, hq, sq, dv)).astype(output_dtype)
    bias_host = rng.normal(0, .1, (1, hq, 1, sk)).astype(np.float32) if bias else None
    row_seed = rng.normal(0, .2, (b, hq, sq)).astype(np.float32) if saved else np.zeros((b, hq, sq), np.float32)
    primal, expected = reference(q, k, v, do, bias_host, row_seed, causal=causal)
    original = source(storage, bias=bias, saved=saved, output="f32" if wide_output else storage, dims=dims, causal=causal)
    with NvidiaDeviceSession() as session:
        roots = [session.upload(x) for x in (q, k, v)]
        bias_buffer = session.upload(bias_host) if bias else None
        guards = []
        def guarded(shape, dt):
            host = np.full(int(np.prod(shape))+16, 37, dtype=dt)
            arena = session.upload(host)
            guards.append(arena)
            return arena.view(8*host.dtype.itemsize, shape, host.dtype)
        output = guarded((b, hq, sq, dv), output_dtype)
        lse = guarded((b, hq, sq), np.float32)
        forward = NativeKernel(original, "forward")
        backward = None
        try:
            forward.launch(session, roots+([bias_buffer] if bias else [])+[output, lse], output.shape[0]*output.shape[1]*output.shape[2]*output.shape[3])
            assert session.synchronize() == 0
            compare(output.numpy(), primal[0]); compare(lse.numpy(), primal[1])
            outputs = [guarded(x.shape, np.float32 if index == 3 else dtype) for index, x in enumerate(expected)]
            reverse_inputs = [session.upload(do), *roots, output]
            if bias:
                reverse_inputs.append(bias_buffer)
            reverse_inputs.append(lse)
            if saved:
                reverse_inputs.append(session.upload(row_seed))
            backward = NativeKernel(original, "backward")
            backward.launch(session, reverse_inputs+outputs, sum(x.size for x in expected))
            assert session.synchronize() == 0
            for result, oracle in zip(outputs, expected, strict=True):
                compare(result.numpy(), oracle)
            # The half residual policy must recompute the FP32 row correction.
            # A caller-visible rounded O must never enter this derivative sum.
            poison = np.full(output.shape, np.nan, output_dtype)
            assert session.lib.tessera_nvidia_device_upload(ct.c_void_p(output.ptr), ct.c_void_p(poison.ctypes.data), poison.nbytes, ct.c_void_p(session.stream)) == 0
            backward.launch(session, reverse_inputs+outputs, sum(x.size for x in expected))
            assert session.synchronize() == 0
            for result, oracle in zip(outputs, expected, strict=True):
                compare(result.numpy(), oracle)
            for arena in guards:
                host = arena.numpy()
                np.testing.assert_array_equal(host[:8], np.full(8, 37, host.dtype))
                np.testing.assert_array_equal(host[-8:], np.full(8, 37, host.dtype))
        finally:
            assert session.synchronize() == 0
            if backward is not None:
                backward.close()
            forward.close()
