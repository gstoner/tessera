"""Storage checks for resident saved-LSE generations without a GPU."""
import ctypes as ct
from types import SimpleNamespace
import ml_dtypes
import numpy as np
import pytest
from tessera.compiler.native_device_tape import _Buffer
from tessera.compiler.resident_attention import ResidentAttentionTape,_ReadOnly


@pytest.mark.parametrize('storage,dtype,bytes_per_element',[
    ('fp16',np.float16,2),('bf16',ml_dtypes.bfloat16,2),('fp32',np.float32,4)])
def test_owned_storage_sizes_and_readonly_semantic_dtype(storage,dtype,bytes_per_element):
    allocated=[]
    def alloc(dest,size):
        dest._obj.value=256;allocated.append(size);return 0
    frame=SimpleNamespace(closed=False,buffers=[],alloc=alloc,check=lambda status:None)
    buffer=_Buffer(frame,(2,3),storage)
    result=_ReadOnly(buffer)
    assert buffer.nbytes==6*bytes_per_element and allocated==[buffer.nbytes]
    assert result.dtype==np.dtype(dtype)
    assert result.__cuda_array_interface__['data']==(256,True)
    frame.closed=True
    with pytest.raises(ValueError,match='closed'):
        _=result.__cuda_array_interface__


def test_bf16_void_storage_requires_semantic_bfloat_tag_before_driver_access():
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame._attribute=lambda *args:pytest.fail('invalid BF16 reached driver')
    value=SimpleNamespace(dtype=np.dtype('V2'),__cuda_array_interface__={
        'version':3,'shape':(2,3),'typestr':'<V2','data':(256,False),'strides':None})
    with pytest.raises(ValueError,match='semantic dtype'):
        frame._resident(value,(2,3),storage='bf16')


def test_storage_roles_are_forwarded_before_any_stream_wait():
    frame=ResidentAttentionTape.__new__(ResidentAttentionTape)
    frame.asynchronous=False
    seen=[]
    def resident(value,shape,*,producer_streams,storage):
        seen.append(storage);producer_streams.append(None);return 256
    frame._resident=resident
    assert frame._resident_group((1,2,3),((1,),)*3,dtypes=('fp16','bf16','fp32'))==[256]*3
    assert seen==['fp16','bf16','fp32']


@pytest.mark.parametrize('storage',[np.float16,ml_dtypes.bfloat16,np.float32])
def test_reference_download_preserves_semantic_numeric_dtype(monkeypatch,storage):
    from benchmarks.nvidia import benchmark_jit_attention_vjp as benchmark
    expected=np.array([.25,-.5,2.],dtype=storage)
    class Copy:
        def __call__(self,destination,source,size):
            ct.memmove(destination,source,size)
            return 0
    monkeypatch.setattr(benchmark.ctypes,'CDLL',lambda name:SimpleNamespace(cuMemcpyDtoH_v2=Copy()))
    value=SimpleNamespace(dtype=expected.dtype,__cuda_array_interface__={
        'shape':expected.shape,'typestr':expected.dtype.str,'data':(expected.ctypes.data,True)})
    actual=benchmark.download(SimpleNamespace(synchronize=lambda:0),value)
    assert actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)
