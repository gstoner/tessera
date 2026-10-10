"""Exact-device pitched RHS proof for one native resident tensor owner."""
import numpy as np
import pytest
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,
    _storage,_oracle)
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="exact SM120 host required")


class PitchedRhs:
    def __init__(self,allocation,shape,strides,offset=0):
        self.allocation=allocation
        self.shape=shape
        self.strides=strides
        self.offset=offset
        self.dtype=allocation.dtype
        self.tessera_layout="strided"

    @property
    def __cuda_array_interface__(self):
        interface=dict(self.allocation.__cuda_array_interface__)
        interface.update(shape=self.shape,strides=self.strides,
                         data=(self.allocation.ptr+self.offset,False))
        return interface


@pytest.mark.parametrize("kind",["rmsnorm","layernorm","softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("fused",[False,True])
@pytest.mark.parametrize("order",["C","F"])
@pytest.mark.parametrize("offset",[False,True])
def test_padded_rhs_single_native_sequence(kind,dtype,fused,order,offset,monkeypatch):
    from tessera import runtime as rt
    rng=np.random.default_rng(120712)
    storage=_storage(dtype);m,k,n=17,35,19
    source=(rng.normal(size=(33,49))*.2).astype(storage)
    right=(rng.normal(size=(49,23))*.2).astype(storage)
    bias=(rng.normal(size=23)*.2).astype(np.float32)
    residual=(rng.normal(size=(33,23))*.2).astype(np.float32)
    fn=({"rmsnorm":rms_lhs_fused,"layernorm":layer_lhs_fused,"softmax":softmax_lhs_fused}
        if fused else {"rmsnorm":rms_lhs,"layernorm":layer_lhs,"softmax":softmax_lhs})[kind]
    args=(source,right,bias,residual) if fused else (source,right)
    program=fn.compile_native_lhs_matmul(*args,dynamic_axes=("M","N","K"),
        rhs_storage_order="row_major" if order=="C" else "col_major").edge
    x=np.ascontiguousarray(source[:m,:k])
    logical=np.ascontiguousarray(right[:k,:n])
    width=np.dtype(storage).itemsize
    if order=="C":
        backing=np.full((k+int(offset),n+5),7,dtype=storage,order="C")
        backing[int(offset):int(offset)+k,:n]=logical
        strides=((n+5)*width,width)
        pointer_offset=int(offset)*(n+5)*width
    else:
        backing=np.full((k+5,n+int(offset)),7,dtype=storage,order="F")
        backing[:k,int(offset):int(offset)+n]=logical
        strides=(width,(k+5)*width)
        pointer_offset=int(offset)*(k+5)*width
    def forbidden(*args,**kwargs):
        raise AssertionError("pitched resident sequence escaped to Python launch")
    monkeypatch.setattr(rt,"launch",forbidden)
    with NvidiaDeviceSession() as session,program.prepare_resident() as owner:
        ds=session.upload(x);allocation=session.upload(backing,layout="row_major" if order=="C" else "col_major")
        rhs=PitchedRhs(allocation,(k,n),strides,pointer_offset)
        edge=session.empty((m,k),storage)
        output=session.empty((m,n),np.float16 if fused else np.float32)
        ab,ar=np.ascontiguousarray(bias[:n]),np.ascontiguousarray(residual[:m,:n])
        buffers=[ds,rhs]
        if fused:buffers.extend([session.upload(ab),session.upload(ar)])
        buffers.append(output)
        for _ in range(3):
            receipts=owner.invoke(buffers,edge,stream=session.stream)
            np.testing.assert_allclose(output.numpy(),_oracle(x,logical,kind,
                ab if fused else None,ar if fused else None),rtol=.015,atol=.015)
            assert all(r["native_call_binding"]=="prepared_cpp_resident_tensor_matmul" for r in receipts)
        np.testing.assert_array_equal(allocation.numpy(),backing)


@pytest.mark.parametrize("corruption",["short","negative","unaligned","minor","capacity","overflow","offset"])
def test_rhs_pitch_gate_preserves_output_and_recovers(corruption):
    x=np.ones((17,35),np.float16);b=np.ones((35,19),np.float16)
    program=rms_lhs.compile_native_lhs_matmul(x,b,dynamic_axes=("M","N","K"),
                                            rhs_storage_order="row_major").edge
    backing=np.ones((35,24),np.float16)
    with NvidiaDeviceSession() as session,program.prepare_resident() as owner:
        dx=session.upload(x);allocation=session.upload(backing)
        rhs=PitchedRhs(allocation,(35,19),(48,2))
        edge=session.empty(x.shape,np.float16)
        output=session.upload(np.full((17,19),-123,np.float32))
        if corruption=="short":rhs.strides=(36,2)
        elif corruption=="negative":rhs.strides=(-48,2)
        elif corruption=="unaligned":rhs.strides=(49,2)
        elif corruption=="minor":rhs.strides=(48,4)
        elif corruption=="capacity":rhs.strides=(4096,2)
        elif corruption=="overflow":rhs.strides=(2**63-2,2)
        else:rhs.offset=allocation.nbytes-2
        with pytest.raises(RuntimeError,match="pitch|capacity|span"):
            owner.invoke([dx,rhs,output],edge,stream=session.stream)
        np.testing.assert_array_equal(output.numpy(),-123)
        rhs.strides=(48,2);rhs.offset=0
        owner.invoke([dx,rhs,output],edge,stream=session.stream)
        np.testing.assert_allclose(output.numpy(),_oracle(x,b,"rmsnorm"),rtol=.015,atol=.015)
