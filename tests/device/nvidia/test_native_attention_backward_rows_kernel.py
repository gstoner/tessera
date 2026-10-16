"""Exact-device producer -> external-row backward proof, separate from public ownership."""
import ctypes as ct

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_native_attention_row_correction_kernel import RowKernel
from tests.device.nvidia.test_saved_lse_half_backward_kernel import NativeKernel, compare, reference
from tests.unit.test_saved_lse_half_backward import source

pytestmark = pytest.mark.hardware_nvidia


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("bias,seeded", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dims", [(1,2,1,3,5,4,3), (2,4,2,1,129,8,6), (1,2,1,7,3,4,5)])
@pytest.mark.parametrize("causal", [False, True])
def test_native_row_residual_drives_backward_and_matches_inline(storage, wide, bias, seeded, dims, causal):
    import tessera.runtime as rt
    assert rt._nvidia_device_name() == "sm_120", "exact SM120 lane requires its owning GPU"
    dt = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    out_dt = np.dtype(np.float32) if wide else dt
    b,hq,hkv,sq,sk,d,dv = dims
    original = source(storage, bias=bias, saved=seeded, output="f32" if wide else storage,
                      dims=dims, causal=causal)
    rng = np.random.default_rng(20261016)
    with NvidiaDeviceSession() as session:
        kernels = []
        try:
            forward = NativeKernel(original, "forward"); kernels.append(forward)
            producer = RowKernel(original, "row_correction"); kernels.append(producer)
            consumer = NativeKernel(original, "backward_rows"); kernels.append(consumer)
            inline = NativeKernel(original, "backward"); kernels.append(inline)
            for iteration in range(2):
                q = rng.normal(0,.4,(b,hq,sq,d)).astype(dt)
                k = rng.normal(0,.4,(b,hkv,sk,d)).astype(dt)
                v = rng.normal(0,.4,(b,hkv,sk,dv)).astype(dt)
                do = rng.normal(0,.4,(b,hq,sq,dv)).astype(out_dt)
                bias_host = rng.normal(0,.1,(1,hq,1,sk)).astype(np.float32) if bias else None
                seed = rng.normal(0,.2,(b,hq,sq)).astype(np.float32) if seeded else np.zeros((b,hq,sq),np.float32)
                _, expected = reference(q,k,v,do,bias_host,seed,causal=causal)
                guards = []
                def guarded(shape,dtype):
                    arena = session.upload(np.full(int(np.prod(shape))+16,37,dtype))
                    guards.append(arena)
                    return arena.view(8*np.dtype(dtype).itemsize,shape,np.dtype(dtype))
                roots = [session.upload(x) for x in (q,k,v)]
                bias_buffer = session.upload(bias_host) if bias else None
                output = session.empty((b,hq,sq,dv),out_dt)
                lse = session.empty((b,hq,sq),np.float32)
                row = guarded((b,hq,sq),np.float32)
                outputs = [guarded(x.shape,np.float32 if i==3 else dt) for i,x in enumerate(expected)]
                controls = [session.empty(x.shape,np.float32 if i==3 else dt) for i,x in enumerate(expected)]
                do_buffer = session.upload(do)
                seed_buffer = session.upload(seed) if seeded else None
                bias_list = [bias_buffer] if bias else []
                forward.launch(session, roots+bias_list+[output,lse], b*hq*sq*dv)
                producer.launch(session, [do_buffer,*roots]+bias_list+[lse]+([seed_buffer] if seeded else [])+[row],b*hq*sq)
                # Both kernels are ordered on the native stream without a host
                # synchronization between producer and residual consumer.
                consumer.launch(session, [do_buffer,*roots]+bias_list+[lse,row]+outputs,sum(x.size for x in expected))
                inline.launch(session, [do_buffer,*roots,output]+bias_list+[lse]+([seed_buffer] if seeded else [])+controls,sum(x.size for x in expected))
                assert session.synchronize()==0
                for actual,control,oracle in zip(outputs,controls,expected,strict=True):
                    compare(actual.numpy(),oracle)
                    np.testing.assert_array_equal(actual.numpy(),control.numpy())
                if iteration==1:
                    # A changed explicit residual must change Q/K/bias; this
                    # proves the consumer reads it instead of recomputing delta.
                    changed_row = session.upload(row.numpy()+np.float32(.125))
                    consumer.launch(session,[do_buffer,*roots]+bias_list+[lse,changed_row]+outputs,sum(x.size for x in expected))
                    assert session.synchronize()==0
                    _, changed = reference(q,k,v,do,bias_host,seed-.125,causal=causal)
                    for actual,oracle in zip(outputs,changed,strict=True):
                        compare(actual.numpy(),oracle)
                    np.testing.assert_array_equal(outputs[2].numpy(),controls[2].numpy())
                for arena in guards:
                    data=arena.numpy()
                    np.testing.assert_array_equal(data[:8],np.full(8,37,data.dtype))
                    np.testing.assert_array_equal(data[-8:],np.full(8,37,data.dtype))
        finally:
            assert session.synchronize()==0
            for kernel in reversed(kernels):
                kernel.close()


class CompactConsumer(NativeKernel):
    def __init__(self, text, threads):
        self.threads = threads
        super().__init__(text, "backward_rows prune-checkpoint-gradients=true "
            f"compact-checkpoint-gradients=true compact-checkpoint-threads={threads}")

    def launch(self, session, buffers, elements):
        values = [ct.c_void_p(buffer.ptr) for buffer in buffers]
        values += [ct.c_int64(x) for x in self.dims]
        parameters = (ct.c_void_p*len(values))(*(ct.addressof(value) for value in values))
        self.check(self.driver.cuLaunchKernel(self.function, (elements+self.threads-1)//self.threads,
                   1, 1, self.threads, 1, 1, 0, ct.c_void_p(session.stream), parameters, None))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("active", [(0,), (2,), (3,0)])
@pytest.mark.parametrize("threads", [64,128])
def test_compact_row_consumer_requested_roles_and_value_only_elision(storage,wide,active,threads):
    import tessera.runtime as rt
    assert rt._nvidia_device_name()=="sm_120", "exact SM120 proof requires owning GPU"
    dt=np.dtype(np.float16 if storage=="f16" else ml_dtypes.bfloat16)
    out_dt=np.dtype(np.float32) if wide else dt
    dims=(1,4,2,7,11,4,3)
    b,hq,hkv,sq,sk,d,dv=dims
    text=source(storage,bias=True,saved=True,output="f32" if wide else storage,dims=dims)
    text=text.replace('tessera.autodiff = "reverse"',
        'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = [' + ', '.join(map(str,active)) + ']')
    rng=np.random.default_rng(20261016)
    with NvidiaDeviceSession() as session:
        kernels=[]
        try:
            forward=NativeKernel(text,"forward"); kernels.append(forward)
            producer=RowKernel(text,"row_correction"); kernels.append(producer)
            consumer=CompactConsumer(text,threads); kernels.append(consumer)
            q=rng.normal(0,.4,(b,hq,sq,d)).astype(dt)
            k=rng.normal(0,.4,(b,hkv,sk,d)).astype(dt)
            v=rng.normal(0,.4,(b,hkv,sk,dv)).astype(dt)
            do=rng.normal(0,.4,(b,hq,sq,dv)).astype(out_dt)
            bias=rng.normal(0,.1,(1,hq,1,sk)).astype(np.float32)
            seed=rng.normal(0,.2,(b,hq,sq)).astype(np.float32)
            _,expected=reference(q,k,v,do,bias,seed)
            roots=[session.upload(x) for x in (q,k,v)]
            bias_buffer=session.upload(bias)
            output=session.empty((b,hq,sq,dv),out_dt)
            lse=session.empty((b,hq,sq),np.float32)
            row=session.upload(np.full((b,hq,sq),np.nan,np.float32))
            do_buffer=session.upload(do)
            selected=sorted(active)
            arenas=[session.upload(np.full(expected[i].size+16,37,np.float32 if i==3 else dt)) for i in selected]
            outputs=[arena.view(8*arena.dtype.itemsize,expected[i].shape,arena.dtype) for arena,i in zip(arenas,selected,strict=True)]
            forward.launch(session,roots+[bias_buffer,output,lse],b*hq*sq*dv)
            if active!=(2,):
                seed_buffer=session.upload(seed)
                producer.launch(session,[do_buffer,*roots,bias_buffer,lse,seed_buffer,row],b*hq*sq)
            # Value-only backward deliberately omits the producer and receives
            # poisoned row storage; dV depends only on saved probabilities.
            consumer.launch(session,[do_buffer,*roots,bias_buffer,lse,row]+outputs,sum(expected[i].size for i in selected))
            assert session.synchronize()==0
            for actual,i in zip(outputs,selected,strict=True):
                compare(actual.numpy(),expected[i])
            for arena in arenas:
                data=arena.numpy()
                np.testing.assert_array_equal(data[:8],np.full(8,37,data.dtype))
                np.testing.assert_array_equal(data[-8:],np.full(8,37,data.dtype))
            if active==(2,):
                assert np.isnan(row.numpy()).all()
        finally:
            assert session.synchronize()==0
            for kernel in reversed(kernels):
                kernel.close()
