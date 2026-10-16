"""Versioned native row owner proof; public portable-program integration is separate."""
import ctypes as ct
from functools import lru_cache
import os
import re

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.device.nvidia.test_ordered_resident_tensor_dag import queue_delayed_upload
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare, reference
from tests.unit.test_saved_lse_half_backward import source

pytestmark=pytest.mark.hardware_nvidia
P,S,I,U,L,F=ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_uint64,ct.c_int64,ct.c_float


@lru_cache(maxsize=96)
def images(storage="f16",wide=False,bias=True,seeded=True,active=None,threads=128,
           dims=(1,4,2,7,11,8,6),legacy=False):
    active=active or tuple(range(3+int(bias)))
    text=source(storage,bias=bias,saved=seeded,output="f32" if wide else storage,
                dims=dims)
    text=text.replace('tessera.autodiff = "reverse"',
        'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = ['+', '.join(map(str,active))+']')
    tool=find_tessera_opt();assert tool is not None
    result=[]
    for role in ("forward","backward" if legacy else "backward_rows","row_correction"):
        if role=="row_correction" and (legacy or set(active)=={2}):
            result.append(None);continue
        flags="--tessera-autodiff-paired=checkpoint-product="+role
        if role in ("backward_rows","backward"):
            flags+=f" prune-checkpoint-gradients=true compact-checkpoint-gradients=true compact-checkpoint-threads={threads}"
        graph=run_tessera_opt(tool,text,flags)
        schedule=run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
        tile=run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
        entry=re.search(r"llvm.func @([^ (]+)",tile)[1]
        _,ptx,*_=_compile_tile_ir(tile,entry)
        result.append((entry,ptx.encode()))
    return tuple(result)


def values(storage,wide,bias,seeded,seed=987,dims=(1,4,2,7,11,8,6)):
    dt=np.dtype(np.float16 if storage=="f16" else ml_dtypes.bfloat16)
    out=np.dtype(np.float32) if wide else dt
    rng=np.random.default_rng(seed)
    b,hq,hkv,sq,sk,d,dv=dims
    q=rng.normal(0,.4,(b,hq,sq,d)).astype(dt)
    k=rng.normal(0,.4,(b,hkv,sk,d)).astype(dt)
    v=rng.normal(0,.4,(b,hkv,sk,dv)).astype(dt)
    do=rng.normal(0,.4,(b,hq,sq,dv)).astype(out)
    bias_value=rng.normal(0,.1,(1,hq,1,sk)).astype(np.float32) if bias else None
    row_seed=rng.normal(0,.2,(b,hq,sq)).astype(np.float32) if seeded else np.zeros((b,hq,sq),np.float32)
    _,expected=reference(q,k,v,do,bias_value,row_seed)
    inputs=(q,k,v)+((bias_value,) if bias else ())+(do,)+((row_seed,) if seeded else ())
    return inputs,expected


class RowOwner:
    def __init__(self,storage="f16",wide=False,bias=True,seeded=True,active=None,threads=128,compiled=None,
                 dims=(1,4,2,7,11,8,6),legacy=False):
        self.active=active or tuple(range(3+int(bias)))
        self.dtype=np.dtype(np.float16 if storage=="f16" else ml_dtypes.bfloat16)
        self.legacy=legacy
        self.lib=ct.CDLL(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"])
        self.lib.tessera_nvidia_attention_vjp_last_error.restype=ct.c_char_p
        self.lib.tessera_nvidia_attention_vjp_rows_prepare.argtypes=[P,S,ct.c_char_p]*3+[
            ct.POINTER(L),ct.POINTER(L),ct.POINTER(I),ct.POINTER(I),S,I,I,I,ct.POINTER(U)]
        self.lib.tessera_nvidia_attention_vjp_rows_prepare.restype=I
        self.lib.tessera_nvidia_attention_vjp_rows_invoke.argtypes=[U,ct.POINTER(P),ct.POINTER(S),S,
            ct.POINTER(P),ct.POINTER(S),S,ct.POINTER(F),S]
        self.lib.tessera_nvidia_attention_vjp_rows_invoke.restype=I
        self.lib.tessera_nvidia_attention_vjp_rows_invoke_resident_ordered.argtypes=[U,ct.POINTER(P),ct.POINTER(S),S,
            ct.POINTER(U),S,ct.POINTER(P),ct.POINTER(S),S,ct.POINTER(F),S]
        self.lib.tessera_nvidia_attention_vjp_rows_invoke_resident_ordered.restype=I
        self.lib.tessera_nvidia_attention_vjp_rows_info.argtypes=[U,ct.POINTER(S),ct.POINTER(S)]
        self.lib.tessera_nvidia_attention_vjp_rows_info.restype=I
        self.lib.tessera_nvidia_attention_vjp_close.argtypes=[U]
        self.lib.tessera_nvidia_attention_vjp_close.restype=I
        self.lib.tessera_nvidia_attention_vjp_invoke.argtypes=[U,ct.POINTER(P),ct.POINTER(S),S,
            ct.POINTER(P),ct.POINTER(S),S,ct.POINTER(F)]
        self.lib.tessera_nvidia_attention_vjp_invoke.restype=I
        self.lib.tessera_nvidia_attention_vjp_prepare_typed.argtypes=[P,S,ct.c_char_p]*2+[
            ct.POINTER(L),ct.POINTER(L),ct.POINTER(I),ct.POINTER(I),S,I,I,I,ct.POINTER(U)]
        self.lib.tessera_nvidia_attention_vjp_prepare_typed.restype=I
        self.lib.tessera_nvidia_attention_vjp_last_device_window.argtypes=[U,ct.POINTER(F)]
        self.lib.tessera_nvidia_attention_vjp_last_device_window.restype=I
        selected=compiled or images(storage,wide,bias,seeded,self.active,threads,dims,legacy)
        arguments=[];self.keep=[]
        for image in (selected[:2] if legacy else selected):
            if image is None:
                arguments.extend((None,0,None));continue
            entry,payload=image
            buffer=ct.create_string_buffer(payload);self.keep.append(buffer)
            arguments.extend((buffer,len(payload),entry.encode()))
        b,hq,hkv,sq,sk,d,dv=dims
        dims=(L*7)(*dims)
        physical_bias=(L*4)(1,hq,1,sk) if bias else None
        mapping=(I*(3+int(bias)))(*range(3+int(bias)))
        roles=(I*len(self.active))(*self.active)
        handle=U()
        codes={"f16":2,"bf16":3}
        self.handle=0
        prepare=self.lib.tessera_nvidia_attention_vjp_prepare_typed if legacy else self.lib.tessera_nvidia_attention_vjp_rows_prepare
        status=prepare(*arguments,dims,physical_bias,mapping,
            roles,len(self.active),codes[storage],1 if wide else codes[storage],int(seeded),ct.byref(handle))
        self.check(status);self.handle=handle.value

    def check(self,status):
        if status:
            raise RuntimeError(self.lib.tessera_nvidia_attention_vjp_last_error().decode())

    def info(self):
        size,stages=S(),S()
        self.check(self.lib.tessera_nvidia_attention_vjp_rows_info(self.handle,ct.byref(size),ct.byref(stages)))
        return size.value,stages.value

    def invoke(self,inputs,expected,*,resident=False,timing_count=4,legacy=False,check=True):
        outputs=tuple(np.full(expected[i].shape,79,np.float32 if i==3 else self.dtype) for i in self.active)
        pointers=(P*len(inputs))(*(x.ptr if resident else x.ctypes.data for x in inputs))
        sizes=(S*len(inputs))(*(x.nbytes for x in inputs))
        destinations=(P*len(outputs))(*(x.ctypes.data for x in outputs))
        lengths=(S*len(outputs))(*(x.nbytes for x in outputs))
        guard=np.full(8,-987,np.float32)
        times=ct.cast(guard.ctypes.data+8,ct.POINTER(F))
        if legacy or self.legacy:
            status=self.lib.tessera_nvidia_attention_vjp_invoke(self.handle,pointers,sizes,len(inputs),
                destinations,lengths,len(outputs),times)
        elif resident:
            streams=(U*len(inputs))(*(x.__cuda_array_interface__["stream"] for x in inputs))
            status=self.lib.tessera_nvidia_attention_vjp_rows_invoke_resident_ordered(self.handle,pointers,sizes,len(inputs),
                streams,len(inputs),destinations,lengths,len(outputs),times,timing_count)
        else:
            status=self.lib.tessera_nvidia_attention_vjp_rows_invoke(self.handle,pointers,sizes,len(inputs),
                destinations,lengths,len(outputs),times,timing_count)
        if check:self.check(status)
        return outputs,guard,status

    def window(self):
        result=F()
        self.check(self.lib.tessera_nvidia_attention_vjp_last_device_window(self.handle,ct.byref(result)))
        return result.value

    def close(self):
        if self.handle:
            self.check(self.lib.tessera_nvidia_attention_vjp_close(self.handle));self.handle=0


def require_device():
    import tessera.runtime as rt
    assert rt._nvidia_device_name()=="sm_120", "exact SM120 proof requires owning GPU"


def check_result(owner,outputs,times,expected):
    for actual,role in zip(outputs,owner.active,strict=True):compare(actual,expected[role])
    np.testing.assert_array_equal(times[:2],np.full(2,-987,np.float32))
    np.testing.assert_array_equal(times[-2:],np.full(2,-987,np.float32))
    assert np.isfinite(times[2:6]).all() and (times[2:6]>=0).all()
    assert times[5]>=max(times[2:5])


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("wide",[False,True])
@pytest.mark.parametrize("bias,seeded",[(False,False),(False,True),(True,False),(True,True)])
def test_native_owner_executes_three_members_and_reuses_private_frame(storage,wide,bias,seeded):
    require_device()
    with NvidiaDeviceSession():
        active=tuple(reversed(range(3+int(bias))))
        owner=RowOwner(storage,wide,bias,seeded,active)
        try:
            assert owner.info()==(1*4*7*4,3)
            for seed in (987,1294):
                inputs,expected=values(storage,wide,bias,seeded,seed)
                outputs,times,_=owner.invoke(inputs,expected)
                check_result(owner,outputs,times,expected)
                assert times[3]>0
        finally:owner.close()


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("wide",[False,True])
def test_value_only_owner_omits_row_module_stage_and_workspace(storage,wide):
    require_device()
    with NvidiaDeviceSession():
        owner=RowOwner(storage,wide,True,True,(2,))
        try:
            assert owner.info()==(0,2)
            inputs,expected=values(storage,wide,True,True)
            outputs,times,_=owner.invoke(inputs,expected)
            check_result(owner,outputs,times,expected)
            assert times[3]==0
        finally:owner.close()


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("active",[(0,),(1,),(3,0),(2,0)])
@pytest.mark.parametrize("threads",[64,128])
def test_owner_compact_role_order_and_thread_geometry(storage,active,threads):
    require_device()
    with NvidiaDeviceSession():
        owner=RowOwner(storage,True,True,True,active,threads)
        try:
            inputs,expected=values(storage,True,True,True)
            outputs,times,_=owner.invoke(inputs,expected)
            check_result(owner,outputs,times,expected)
        finally:owner.close()


@pytest.mark.parametrize("capacity,legacy",[(0,False),(2,False),(3,False),(5,False),(2,True)])
def test_timing_capacity_and_legacy_entry_refuse_before_writes(capacity,legacy):
    require_device()
    with NvidiaDeviceSession():
        owner=RowOwner()
        try:
            inputs,expected=values("f16",False,True,True)
            outputs,times,status=owner.invoke(inputs,expected,timing_count=capacity,legacy=legacy,check=False)
            assert status!=0
            np.testing.assert_array_equal(times,np.full(8,-987,np.float32))
            for output in outputs:np.testing.assert_array_equal(output,np.full(output.shape,79,output.dtype))
            outputs,times,_=owner.invoke(inputs,expected)
            check_result(owner,outputs,times,expected)
        finally:owner.close()


def test_resident_pending_writes_are_ordered_before_private_snapshots():
    require_device()
    with NvidiaDeviceSession() as first,NvidiaDeviceSession() as second:
        owner=RowOwner("bf16",True,True,True,(3,0,1,2))
        try:
            inputs,expected=values("bf16",True,True,True)
            sessions=[first if i%2==0 else second for i in range(len(inputs))]
            buffers=[session.upload(np.zeros_like(host)) for session,host in zip(sessions,inputs,strict=True)]
            assert first.synchronize()==0 and second.synchronize()==0
            completed=[]
            callbacks=[queue_delayed_upload(session,buffer,host,session.stream,completed)
                       for session,buffer,host in zip(sessions,buffers,inputs,strict=True)]
            outputs,times,_=owner.invoke(buffers,expected,resident=True)
            assert len(completed)==len(inputs) and len(callbacks)==len(inputs)
            check_result(owner,outputs,times,expected)
            for buffer,host in zip(buffers,inputs,strict=True):np.testing.assert_array_equal(buffer.numpy(),host)
        finally:owner.close()


def test_actual_compiled_row_parameter_count_is_checked():
    require_device()
    with NvidiaDeviceSession():
        compiled=list(images())
        entry,payload=compiled[2]
        text=payload.decode()
        pattern=r'(\.visible \.entry '+re.escape(entry)+r'\([\s\S]*?)(\n\))'
        changed,count=re.subn(pattern,r'\1,\n\t.param .u64 unexpected_extra\2',text,count=1)
        assert count==1 and changed!=text
        compiled[2]=(entry,changed.encode())
        with pytest.raises(RuntimeError,match="parameter count differs"):
            RowOwner(compiled=compiled)
        owner=RowOwner()
        try:
            inputs,expected=values("f16",False,True,True)
            outputs,times,_=owner.invoke(inputs,expected)
            check_result(owner,outputs,times,expected)
        finally:owner.close()


@pytest.mark.parametrize("legacy",[False,True])
def test_device_window_requires_completed_call_and_matches_same_interval(legacy):
    require_device()
    with NvidiaDeviceSession():
        owner=RowOwner(legacy=legacy)
        try:
            with pytest.raises(RuntimeError,match="no completed device window"):
                owner.window()
            inputs,expected=values("f16",False,True,True)
            outputs,times,_=owner.invoke(inputs,expected)
            for actual,role in zip(outputs,owner.active,strict=True):compare(actual,expected[role])
            window=owner.window()
            assert window>0
            if not legacy:
                assert window==times[5]
            else:
                assert window>=max(times[2:4])
                np.testing.assert_array_equal(times[4:],np.full(4,-987,np.float32))
        finally:owner.close()
