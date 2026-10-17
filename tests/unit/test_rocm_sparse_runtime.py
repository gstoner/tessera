"""Artifact/status/worker ownership and separately gated gfx1201 binding proof."""
from dataclasses import replace
import multiprocessing as mp
import os
import time

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.rocm_sparse_logical import sparse_logical_schedule_ir

from tessera.compiler.rocm_sparse_runtime import SparseMatmulPackage, compile_sparse_matmul

# `rocm_isa` is imported inside each test on purpose. `rocm_sparse_runtime`
# runs its kernels in a multiprocessing worker, and under spawn/forkserver the
# child RE-IMPORTS this module to unpickle the target. The child's sys.path does
# not resolve `tests._support`, so a module-level import kills the worker at
# startup and the parent sees EOFError instead of the refusal under test.
# Hoisting this to module scope for tidiness cost a full-sweep failure on macOS
# that passed in isolation, in the whole ROCm subset, and in CI (2026-09-19).


def package():
    value = SparseMatmulPackage((16,16,32),'float16',sparse_logical_schedule_ir(16,16,32,'float16'),
                               'lowered',b'image','0'*64)
    return replace(value,digest=value._digest())


def success_worker(connection, package, a, b, device):
    connection.send(('result',np.zeros((16,16),np.float32)))
    connection.close()


def stall_worker(*args):
    time.sleep(60)


def invalid_worker(connection,*args):
    connection.send(('invalid',))
    connection.close()


def test_sparse_identity_and_inputs_refuse_before_process_creation(monkeypatch):
    monkeypatch.setattr(mp,'get_context',lambda *args: pytest.fail('spawn before validation'))
    a,b = np.zeros((16,32),np.float16),np.zeros((32,16),np.float16)
    with pytest.raises(ValueError,match='identity'):
        replace(package(),image=b'changed').run(a,b)
    for bad in (a.astype(np.float32),a[:8],np.full(a.shape,np.nan,np.float16)):
        with pytest.raises(ValueError,match='finite matching'):
            package().run(bad,b)
    for device in (True,-1,2**31):
        with pytest.raises(ValueError,match='ordinal'):
            package().run(a,b,device=device)


def test_sparse_timeout_confirms_worker_death(monkeypatch):
    from tessera.compiler import rocm_sparse_runtime as runtime
    monkeypatch.setattr(runtime,'_worker',stall_worker)
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(TimeoutError,match='deadline'):
        package().run(np.zeros((16,32),np.float16),np.zeros((32,16),np.float16),timeout_seconds=.2)
    assert {p.pid for p in mp.active_children()} == before
    assert not runtime._UNCERTAIN


def test_sparse_invalid_status_exposes_no_result_and_does_not_leak(monkeypatch):
    from tessera.compiler import rocm_sparse_runtime as runtime
    monkeypatch.setattr(runtime,'_worker',invalid_worker)
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(ValueError,match='no output exposed'):
        package().run(np.zeros((16,32),np.float16),np.zeros((32,16),np.float16))
    assert {p.pid for p in mp.active_children()} == before
    assert not runtime._UNCERTAIN
    monkeypatch.setattr(runtime,'_worker',success_worker)
    np.testing.assert_array_equal(package().run(np.zeros((16,32),np.float16),np.zeros((32,16),np.float16)),0)


@pytest.mark.skipif(os.environ.get('TESSERA_GFX1201_DEVICE_PROOF') != '1',reason='owning gfx1201 proof')
@pytest.mark.parametrize('dtype',[np.float16,ml_dtypes.bfloat16])
@pytest.mark.parametrize('shape',[(16,16,32),(64,48,128)])
@pytest.mark.parametrize('low_acc',[False,True])
def test_sparse_public_runtime_binding(dtype,shape,low_acc,tmp_path):
    from tessera import runtime as rt
    assert rt._rocm_live_arch() == 'gfx1201'
    m,n,k = shape
    compiled = compile_sparse_matmul(m,n,k,dtype=np.dtype(dtype).name,
        accum=("f16" if dtype == np.float16 else "bf16") if low_acc else "f32")
    from tests._support import rocm_isa  # deliberately local -- see note at top
    output_type = ("f16" if dtype == np.float16 else "bf16") if low_acc else "f32"
    storage = "f16" if dtype == np.float16 else "bf16"
    # The accumulator is what this row selects, so the other two accumulator
    # widths are forbidden: `in` alone would pass on an f32-accumulating kernel
    # when a reduced-precision one was asked for.
    rocm_isa.assert_selected(compiled.image, chip="gfx1201",
        pattern=r"v_swmmac_\w+",
        require=f"v_swmmac_{output_type}_16x16x32_{storage}",
        forbid=tuple(f"v_swmmac_{other}_16x16x32_{storage}"
                     for other in ("f32", "f16", "bf16") if other != output_type),
        what=f"{np.dtype(dtype).name} accum={output_type}")
    rng = np.random.default_rng(16)
    a = (rng.integers(-4,5,size=(m,k))/4).astype(dtype)
    a.reshape(m,k//4,4)[:,:,2:] = 0
    b = (rng.integers(-4,5,size=(k,n))/4).astype(dtype)
    np.testing.assert_array_equal(compiled.run(a,b),a.astype(np.float32)@b.astype(np.float32))
    a[-1,-4:] = 1
    with pytest.raises(ValueError,match='invalid 2:4'):
        compiled.run(a,b)
    a[-1,-2:] = 0
    np.testing.assert_array_equal(compiled.run(a,b),a.astype(np.float32)@b.astype(np.float32))


def test_mixed_fp8_operand_identity_and_preflight(monkeypatch):
    monkeypatch.setattr(mp,'get_context',lambda *args: pytest.fail('spawn before validation'))
    source = sparse_logical_schedule_ir(16,16,32,'float8_e4m3fn',rhs_dtype='float8_e5m2')
    p = SparseMatmulPackage((16,16,32),'float8_e4m3fn',source,'lowered',b'image','0'*64,
                           rhs_dtype='float8_e5m2')
    p = replace(p,digest=p._digest())
    p.validate()
    a = np.zeros((16,32),ml_dtypes.float8_e4m3fn)
    b = np.zeros((32,16),ml_dtypes.float8_e5m2)
    assert p._inputs(a,b)[1].dtype == b.dtype
    with pytest.raises(ValueError,match='finite matching'):
        p.run(a,b.astype(ml_dtypes.float8_e4m3fn))
    with pytest.raises(ValueError,match='Schedule'):
        replace(p,rhs_dtype='float8_e4m3fn').validate()
    for lhs,rhs in [('float16','bfloat16'),('int8','float8_e5m2'),('float8_e4m3fn','float16')]:
        with pytest.raises(ValueError,match='mixed sparse'):
            sparse_logical_schedule_ir(16,16,32,lhs,rhs_dtype=rhs)


@pytest.mark.parametrize('dtype,low,high',[('int8',-8,7),('uint8',0,15)])
def test_int4_range_and_identity_preflight(dtype,low,high,monkeypatch):
    monkeypatch.setattr(mp,'get_context',lambda *args: pytest.fail('spawn before validation'))
    source = sparse_logical_schedule_ir(16,16,32,dtype,accum='i32',integer_bits=4)
    p = SparseMatmulPackage((16,16,32),dtype,source,'lowered',b'image','0'*64,accum='i32',integer_bits=4)
    p = replace(p,digest=p._digest())
    p.validate()
    a,b = np.full((16,32),low,dtype),np.full((32,16),high,dtype)
    p._inputs(a,b)
    b[0,0] = high+1
    with pytest.raises(ValueError,match='declared range'): p.run(a,b)
    with pytest.raises(ValueError,match='Schedule'): replace(p,integer_bits=8).validate()
    for width in (True,0,3,16):
        with pytest.raises(ValueError,match='integer width'):
            sparse_logical_schedule_ir(16,16,32,dtype,accum='i32',integer_bits=width)


class _ProtocolHip:
    """Host-only byte buffers; this mock proves lifecycle, not GPU arithmetic."""

    def __init__(self):
        self.buffers = {}
        self.order = []
        self.stream_closed = False
        self.released = False

    def __getattr__(self, name):
        if name in {"hipSetDevice", "hipStreamSynchronize"}:
            return lambda *args: 0
        if name in {"hipModuleLoadData", "hipModuleGetFunction", "hipStreamCreateWithFlags"}:
            def create(pointer, *args):
                pointer._obj.value = 1
                return 0
            return create
        raise AttributeError(name)

    def hipMalloc(self, pointer, size):
        import ctypes as ct
        buffer = ct.create_string_buffer(size)
        address = ct.addressof(buffer)
        self.buffers[address] = buffer
        self.order.append(address)
        pointer._obj.value = address
        return 0

    def hipMemcpyAsync(self, dst, src, size, kind, stream):
        import ctypes as ct
        ct.memmove(dst, src, size)
        return 0

    def hipMemsetAsync(self, dst, value, size, stream):
        import ctypes as ct
        ct.memset(dst, value, size)
        return 0

    def hipModuleLaunchKernel(self, *args):
        import ctypes as ct
        status = self.buffers[self.order[3]]
        words = ct.cast(ct.addressof(status), ct.POINTER(ct.c_int32))
        for index in range(ct.sizeof(status) // 4):
            words[index] = 1
        return 0

    def hipFree(self, pointer):
        del self.buffers[pointer.value]
        return 0

    def hipStreamDestroy(self, stream):
        assert not self.buffers
        self.stream_closed = True
        return 0

    def hipModuleUnload(self, module):
        assert self.stream_closed
        self.released = True
        return 0


def _worker_with_blocked_late_shutdown(connection, package, a, b, device):
    import atexit
    from tessera import runtime as rt
    from tessera.compiler import rocm_sparse_runtime as sparse
    hip = _ProtocolHip()
    rt._load_hip_for_launch = lambda: hip
    rt._rocm_live_arch = lambda: "gfx1201"
    atexit.register(time.sleep, 60)

    class CheckedDelivery:
        def send(self, value):
            assert hip.released
            connection.send(value)

        def close(self):
            connection.close()

    sparse._worker(CheckedDelivery(), package, a, b, device)


def test_sparse_checked_worker_exit_avoids_late_finalizers(monkeypatch):
    """Host-free: checked mock cleanup must finish without interpreter shutdown."""
    from tessera.compiler import rocm_sparse_runtime as runtime
    monkeypatch.setattr(runtime, "_worker", _worker_with_blocked_late_shutdown)
    before = {process.pid for process in mp.active_children()}
    result = package().run(
        np.zeros((16, 32), np.float16), np.zeros((32, 16), np.float16),
        timeout_seconds=5)
    assert result.shape == (16, 16) and result.dtype == np.float32
    assert {process.pid for process in mp.active_children()} == before
    assert not runtime._UNCERTAIN
