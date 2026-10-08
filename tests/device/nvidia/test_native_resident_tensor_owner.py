"""Exact SM120 native resident sequence and caller-allocation lifetime proof."""
import ctypes as ct
import numpy as np
import pytest
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs, layer_lhs, softmax_lhs, rms_lhs_fused, layer_lhs_fused,
    softmax_lhs_fused, _storage, _oracle)
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_nvidia_matmul import HostView

pytestmark = pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="owning SM120 required")


@pytest.mark.parametrize("kind", ["rmsnorm", "layernorm", "softmax"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_resident_sequence_numerics_and_reuse(kind, dtype, fused, order, dynamic, monkeypatch):
    from tessera import runtime as rt
    rng = np.random.default_rng(120710)
    storage = _storage(dtype)
    source = (rng.normal(size=(32,64)) * .2).astype(storage)
    rhs = np.array(rng.normal(size=(64,24)) * .2, dtype=storage, order=order)
    bias = (rng.normal(size=24) * .2).astype(np.float32)
    residual = (rng.normal(size=(32,24)) * .2).astype(np.float32)
    fn = ({"rmsnorm": rms_lhs_fused, "layernorm": layer_lhs_fused, "softmax": softmax_lhs_fused}
          if fused else {"rmsnorm": rms_lhs, "layernorm": layer_lhs, "softmax": softmax_lhs})[kind]
    args = (source,rhs,bias,residual) if fused else (source,rhs)
    program = fn.compile_native_lhs_matmul(*args, dynamic_axes=("M","N","K") if dynamic else (),
        rhs_storage_order="row_major" if order == "C" else "col_major").edge
    def forbidden(*args, **kwargs):
        raise AssertionError("resident sequence escaped to Python kernel dispatch")
    monkeypatch.setattr(rt, "launch", forbidden)
    active = [(32,64,24), (7,35,19), (1,1,1)] if dynamic else [(32,64,24)] * 3
    with NvidiaDeviceSession() as session:
        with program.prepare_resident() as owner:
            for i,(m,k,n) in enumerate(active):
                x = np.ascontiguousarray((source[:m,:k].astype(np.float32)*(i+1)/2).astype(storage))
                b = np.array(rhs[:k,:n], order=order)
                ab, ar = np.ascontiguousarray(bias[:n]), np.ascontiguousarray(residual[:m,:n])
                dx = session.upload(x)
                db = session.upload(b, layout="row_major" if order == "C" else "col_major")
                edge = session.empty((m,k),storage)
                output = session.empty((m,n),np.float16 if fused else np.float32)
                buffers = [dx,db]
                if fused:
                    buffers.extend([session.upload(ab),session.upload(ar)])
                buffers.append(output)
                receipts = owner.invoke(buffers,edge,stream=session.stream)
                actual = output.numpy()
                np.testing.assert_allclose(actual,_oracle(x,b,kind,ab if fused else None,ar if fused else None),
                                           rtol=.015,atol=.015)
                assert all(r["native_call_binding"] == "prepared_cpp_resident_tensor_matmul" for r in receipts)
        with pytest.raises(ValueError,match="closed"):
            owner.invoke(buffers,edge,stream=session.stream)
    # The public resident convenience path now uses this same native sequence.
    with program.execute_resident(source,rhs,bias=bias if fused else None,
                                  residual=residual if fused else None) as result:
        np.testing.assert_allclose(result.output.numpy(),_oracle(source,rhs,kind,
            bias if fused else None,residual if fused else None),rtol=.015,atol=.015)


def _views(buffers):
    views = (HostView * len(buffers))()
    for v,b in zip(views,buffers,strict=True):
        interface = b.__cuda_array_interface__
        v.data,v.bytes,v.rank = b.ptr,b.nbytes,len(b.shape)
        v.dtype = {"float16":2,"float32":1,"bfloat16":3}[b.dtype.name]
        strides = interface["strides"] or (b.shape[1]*b.dtype.itemsize,b.dtype.itemsize)
        for axis in range(v.rank):
            v.shape[axis],v.strides[axis] = b.shape[axis],strides[axis]
    return views


@pytest.mark.parametrize("corruption",["alias","bytes","pitch","extent","pointer"])
def test_native_resident_gate_preserves_output_and_recovers(corruption):
    x=np.ones((32,64),np.float16)
    b=np.ones((64,24),np.float16)
    program=rms_lhs.compile_native_lhs_matmul(x,b).edge
    with NvidiaDeviceSession() as session, program.prepare_resident() as owner:
        dx,db=session.upload(x),session.upload(b)
        output=session.upload(np.full((32,24),-123,np.float32))
        edge=session.empty((32,64),np.float16)
        session.synchronize()
        views=_views([dx,db,output,edge])
        if corruption=="alias": views[2].data=views[0].data
        elif corruption=="bytes": views[1].bytes-=2
        elif corruption=="pitch": views[0].strides[0]+=2
        elif corruption=="extent": views[0].shape[0]+=1
        else: views[0].data=16
        assert owner.lib.tessera_nvidia_matmul_invoke_resident(
            owner.handle,views,4,ct.c_void_p(session.stream)) != 0
        np.testing.assert_array_equal(output.numpy(),-123)
        owner.invoke([dx,db,output],edge,stream=session.stream)
        np.testing.assert_allclose(output.numpy(),_oracle(x,b,"rmsnorm"),rtol=.015,atol=.015)


def test_readonly_inputs_and_independent_receipts():
    x=np.ones((32,64),np.float16);b=np.ones((64,24),np.float16)
    program=rms_lhs.compile_native_lhs_matmul(x,b).edge
    class ReadOnly:
        def __init__(self, buffer):self.buffer=buffer;self.dtype=buffer.dtype
        @property
        def __cuda_array_interface__(self):
            interface=dict(self.buffer.__cuda_array_interface__)
            interface["data"]=(self.buffer.ptr,True)
            return interface
    with NvidiaDeviceSession() as session, program.prepare_resident() as owner:
        dx,db=session.upload(x),session.upload(b)
        edge=session.empty(x.shape,np.float16)
        output=session.empty((32,24),np.float32)
        receipts=owner.invoke([ReadOnly(dx),ReadOnly(db),output],edge,stream=session.stream)
        receipts[0]["native_call_binding"]="caller edited copy"
        next_receipts=owner.invoke([dx,db,output],edge,stream=session.stream)
        assert next_receipts[0]["native_call_binding"]=="prepared_cpp_resident_tensor_matmul"
        np.testing.assert_allclose(output.numpy(),_oracle(x,b,"rmsnorm"),rtol=.015,atol=.015)
        with pytest.raises(ValueError,match="writable output"):
            owner.invoke([dx,db,ReadOnly(output)],edge,stream=session.stream)


@pytest.mark.parametrize("foreign",["current","stream","allocation"])
def test_resident_contexts_preserve_output_and_recover(foreign):
    x=np.ones((32,64),np.float16);b=np.ones((64,24),np.float16)
    program=rms_lhs.compile_native_lhs_matmul(x,b).edge
    cuda=ct.CDLL("libcuda.so.1")
    cuda.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
    cuda.cuCtxCreate_v2.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint,ct.c_int]
    cuda.cuCtxSetCurrent.argtypes=[ct.c_void_p]
    cuda.cuCtxDestroy_v2.argtypes=[ct.c_void_p]
    cuda.cuStreamCreate.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint]
    cuda.cuStreamDestroy_v2.argtypes=[ct.c_void_p]
    cuda.cuMemAlloc_v2.argtypes=[ct.POINTER(ct.c_uint64),ct.c_size_t]
    cuda.cuMemFree_v2.argtypes=[ct.c_uint64]
    with NvidiaDeviceSession() as session, program.prepare_resident() as owner:
        dx,db=session.upload(x),session.upload(b)
        edge=session.empty(x.shape,np.float16)
        output=session.upload(np.full((32,24),-123,np.float32))
        session.synchronize()
        views=_views([dx,db,output,edge])
        original,other,stream=ct.c_void_p(),ct.c_void_p(),ct.c_void_p()
        allocation=ct.c_uint64()
        assert cuda.cuCtxGetCurrent(ct.byref(original))==0
        assert cuda.cuCtxCreate_v2(ct.byref(other),0,0)==0
        try:
            assert cuda.cuStreamCreate(ct.byref(stream),1)==0
            assert cuda.cuMemAlloc_v2(ct.byref(allocation),dx.nbytes)==0
            if foreign!="current":assert cuda.cuCtxSetCurrent(original)==0
            if foreign=="allocation":views[0].data=allocation.value
            launch_stream=stream if foreign=="stream" else ct.c_void_p(session.stream)
            assert owner.lib.tessera_nvidia_matmul_invoke_resident(
                owner.handle,views,4,launch_stream)!=0
        finally:
            assert cuda.cuCtxSetCurrent(other)==0
            if allocation.value:assert cuda.cuMemFree_v2(allocation)==0
            if stream.value:assert cuda.cuStreamDestroy_v2(stream)==0
            assert cuda.cuCtxSetCurrent(original)==0
            assert cuda.cuCtxDestroy_v2(other)==0
        np.testing.assert_array_equal(output.numpy(),-123)
        owner.invoke([dx,db,output],edge,stream=session.stream)
        np.testing.assert_allclose(output.numpy(),_oracle(x,b,"rmsnorm"),rtol=.015,atol=.015)
