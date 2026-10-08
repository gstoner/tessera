"""Changed values and active-shape replay using one pitched native module owner."""
import ctypes as ct
import numpy as np
import pytest
from tests.device.nvidia.test_native_padded_resident_tensor_owner import PitchedRhs
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,
    _storage,_oracle)
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="exact SM120 required")


@pytest.mark.parametrize("kind",["rmsnorm","layernorm","softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("fused",[False,True])
@pytest.mark.parametrize("order",["C","F"])
def test_changed_pitched_rhs_values_and_shapes(kind,dtype,fused,order,monkeypatch):
    from tessera import runtime as rt
    rng=np.random.default_rng(120713);storage=_storage(dtype)
    x=(rng.normal(size=(33,49))*.2).astype(storage)
    b=(rng.normal(size=(49,23))*.2).astype(storage)
    bias=(rng.normal(size=23)*.2).astype(np.float32)
    residual=(rng.normal(size=(33,23))*.2).astype(np.float32)
    fn=({"rmsnorm":rms_lhs_fused,"layernorm":layer_lhs_fused,"softmax":softmax_lhs_fused}
        if fused else {"rmsnorm":rms_lhs,"layernorm":layer_lhs,"softmax":softmax_lhs})[kind]
    args=(x,b,bias,residual) if fused else (x,b)
    program=fn.compile_native_lhs_matmul(*args,dynamic_axes=("M","N","K"),
        rhs_storage_order="row_major" if order=="C" else "col_major").edge
    width=np.dtype(storage).itemsize
    backing=np.zeros((49,30) if order=="C" else (56,23),dtype=storage,order=order)
    strides=(30*width,width) if order=="C" else (width,56*width)
    def forbidden(*args,**kwargs):
        raise AssertionError("pitched owner replay escaped to Python descriptor launch")
    monkeypatch.setattr(rt,"launch",forbidden)
    with NvidiaDeviceSession() as session,program.prepare_resident() as owner:
        allocation=session.upload(backing,layout="row_major" if order=="C" else "col_major")
        for iteration,(m,k,n) in enumerate(((17,35,19),(1,1,1),(33,49,23))):
            changed=(b[:k,:n].astype(np.float32)*(iteration+1)/2-.1).astype(storage)
            backing.fill(7);backing[:k,:n]=changed
            # Upload preserves the existing padded allocation. The host array
            # remains live through the native synchronous completion.
            assert session.lib.tessera_nvidia_device_upload(
                ct.c_void_p(allocation.ptr),ct.c_void_p(backing.ctypes.data),
                allocation.nbytes,ct.c_void_p(session.stream))==0
            source=np.ascontiguousarray(x[:m,:k])
            dx=session.upload(source)
            rhs=PitchedRhs(allocation,(k,n),strides)
            edge=session.empty((m,k),storage)
            output=session.empty((m,n),np.float16 if fused else np.float32)
            ab,ar=np.ascontiguousarray(bias[:n]),np.ascontiguousarray(residual[:m,:n])
            buffers=[dx,rhs]
            if fused:buffers.extend([session.upload(ab),session.upload(ar)])
            buffers.append(output)
            owner.invoke(buffers,edge,stream=session.stream)
            np.testing.assert_allclose(output.numpy(),_oracle(source,changed,kind,
                ab if fused else None,ar if fused else None),rtol=.015,atol=.015)
            np.testing.assert_array_equal(allocation.numpy(),backing)
