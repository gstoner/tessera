"""Bounded native row producer/consumer execution; public packaging is separate."""
import ctypes as ct
import re

import ml_dtypes
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler.attention_shape_contract import attention_dimensions,physical_attention_bias_shape,DYNAMIC_DIM
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_saved_lse_half_backward_kernel import NativeKernel,compare,reference
from tests.unit.test_saved_lse_half_backward import source

pytestmark=pytest.mark.hardware_nvidia


class BoundedKernel(NativeKernel):
    def __init__(self,text,role):
        super().__init__(text,role)
        self.role=role
        bounds=re.search(r'\bshape_bounds = array<i64: ([^>]+)>',self.tile)
        assert bounds is not None
        self.bounds=tuple(map(int,bounds[1].split(',')))
        bias=re.search(r'\bbias_shape = array<i64: ([^>]+)>',self.tile)
        self.bias=tuple(map(int,bias[1].split(','))) if bias else ()

    def launch(self,session,buffers,elements,dims):
        dims=attention_dimensions(self.dims,self.bounds,dims)
        scalars=dims
        if self.role in ('forward','backward') and DYNAMIC_DIM in self.bias:
            scalars+=physical_attention_bias_shape(dims,self.bias)
        values=[ct.c_void_p(buffer.ptr) for buffer in buffers]
        values+=[ct.c_int64(value) for value in scalars]
        parameters=(ct.c_void_p*len(values))(*(ct.addressof(value) for value in values))
        threads=32 if self.role=='row_correction' else 128
        self.check(self.driver.cuLaunchKernel(self.function,(elements+threads-1)//threads,1,1,
            threads,1,1,0,ct.c_void_p(session.stream),parameters,None))


@pytest.mark.parametrize('storage',['f16','bf16'])
@pytest.mark.parametrize('wide',[False,True])
@pytest.mark.parametrize('bias_kind',['none','query','key','dense'])
def test_bounded_rows_reuse_images_and_read_explicit_residual(storage,wide,bias_kind):
    assert rt._nvidia_device_name()=='sm_120'
    dtype=np.dtype(np.float16 if storage=='f16' else ml_dtypes.bfloat16)
    out_dtype=np.dtype(np.float32) if wide else dtype
    trace=(1,4) if bias_kind=='query' else (3,1) if bias_kind=='key' else (3,4)
    biased=bias_kind!='none';sq,sk=trace
    text=source(storage,bias=biased,saved=True,output='f32' if wide else storage,
        dims=(1,2,1,sq,sk,8,6)).replace('tessera.arch = "sm_120"',
        'tessera.arch = "sm_120", tessera.attention_sequence_bounds = array<i64: 9, 11>')
    if bias_kind in ('key','dense'):
        text=text.replace(f'tensor<1x2x1x{sk}xf32>',f'tensor<1x2x{sq}x{1 if bias_kind=="key" else sk}xf32>')
    with NvidiaDeviceSession() as session:
        kernels=[]
        try:
            for role in ('forward','row_correction','backward_rows','backward'):
                kernels.append(BoundedKernel(text,role))
            forward,producer,consumer,inline=kernels
            for sq,sk in (trace,(3,5),(9,11),(9,1)):
                dims=(1,2,1,sq,sk,8,6)
                rng=np.random.default_rng(202610176+sq*11+sk)
                q=rng.normal(0,.2,(1,2,sq,8)).astype(dtype)
                k=rng.normal(0,.2,(1,1,sk,8)).astype(dtype)
                v=rng.normal(0,.2,(1,1,sk,6)).astype(dtype)
                do=rng.normal(0,.1,(1,2,sq,6)).astype(out_dtype)
                seed=rng.normal(0,.1,(1,2,sq)).astype(np.float32)
                shape=(1,2,1 if bias_kind=='query' else sq,1 if bias_kind=='key' else sk)
                bias=rng.normal(0,.1,shape).astype(np.float32) if biased else None
                expected_primal,expected=reference(q,k,v,do,bias,seed)
                roots=[session.upload(x) for x in (q,k,v)]
                dbias=[session.upload(bias)] if biased else []
                ddo,dseed=session.upload(do),session.upload(seed)
                output=session.empty(do.shape,out_dtype);lse=session.empty(seed.shape,np.float32)
                guards=[]
                def guarded(shape,dtype):
                    arena=session.upload(np.full(int(np.prod(shape))+16,37,dtype));guards.append(arena)
                    return arena.view(8*np.dtype(dtype).itemsize,shape,np.dtype(dtype))
                row=guarded(seed.shape,np.float32)
                outputs=[guarded(value.shape,np.float32 if i==3 else dtype) for i,value in enumerate(expected)]
                controls=[session.empty(value.shape,np.float32 if i==3 else dtype) for i,value in enumerate(expected)]
                forward.launch(session,roots+dbias+[output,lse],do.size,dims)
                producer.launch(session,[ddo,*roots]+dbias+[lse,dseed,row],seed.size,dims)
                count=sum(value.size for value in expected)
                consumer.launch(session,[ddo,*roots]+dbias+[lse,row]+outputs,count,dims)
                inline.launch(session,[ddo,*roots,output]+dbias+[lse,dseed]+controls,count,dims)
                assert session.synchronize()==0
                for value,wanted in zip((output,lse),expected_primal,strict=True):compare(value.numpy(),wanted)
                for actual,control,wanted in zip(outputs,controls,expected,strict=True):
                    compare(actual.numpy(),wanted)
                    np.testing.assert_array_equal(actual.numpy(),control.numpy())
                changed=session.upload(row.numpy()+np.float32(.125))
                consumer.launch(session,[ddo,*roots]+dbias+[lse,changed]+outputs,count,dims)
                assert session.synchronize()==0
                _,changed_expected=reference(q,k,v,do,bias,seed-.125)
                for actual,wanted in zip(outputs,changed_expected,strict=True):compare(actual.numpy(),wanted)
                np.testing.assert_array_equal(outputs[2].numpy(),controls[2].numpy())
                for arena in guards:
                    data=arena.numpy()
                    np.testing.assert_array_equal(data[:8],np.full(8,37,data.dtype))
                    np.testing.assert_array_equal(data[-8:],np.full(8,37,data.dtype))
        finally:
            assert session.synchronize()==0
            for kernel in reversed(kernels):kernel.close()
