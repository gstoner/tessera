"""Typed bounded attention capture: correctness-gated host and device windows."""
from __future__ import annotations

import ctypes as ct
import hashlib
import json
import lzma
import os
import subprocess
import time
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import tessera as ts
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare, reference


class GraphWindows:
    """Diagnostic dispatch of the exact functions/ABI already owned by the tape."""
    def __init__(self,tape):
        self.tape=tape
        self.driver=ct.CDLL('libcuda.so.1')
        p=ct.c_void_p
        signatures={
            'cuStreamBeginCapture':([p,ct.c_int]),'cuStreamEndCapture':([p,ct.POINTER(p)]),
            'cuGraphInstantiateWithFlags':([ct.POINTER(p),p,ct.c_uint64]),
            'cuGraphLaunch':([p,p]),'cuGraphExecDestroy':([p]),'cuGraphDestroy':([p]),
            'cuEventCreate':([ct.POINTER(p),ct.c_uint]),'cuEventRecord':([p,p]),
            'cuEventElapsedTime':([ct.POINTER(ct.c_float),p,p]),'cuEventDestroy_v2':([p]),
        }
        for name,args in signatures.items():
            fn=getattr(self.driver,name);fn.argtypes=args;fn.restype=ct.c_int

    def measure(self,backward,buffers,repeats=32,windows=3):
        d=self.driver;t=self.tape
        graph,executable,begin,end=(ct.c_void_p() for _ in range(4))
        capturing=False
        try:
            t.check(t.sync())
            t.check(d.cuStreamBeginCapture(t._stream,1));capturing=True
            for _ in range(repeats):t._execute(backward,buffers)
            status=d.cuStreamEndCapture(t._stream,ct.byref(graph));capturing=False;t.check(status)
            t.check(d.cuGraphInstantiateWithFlags(ct.byref(executable),graph,0))
            t.check(d.cuEventCreate(ct.byref(begin),0));t.check(d.cuEventCreate(ct.byref(end),0))
            t.check(d.cuGraphLaunch(executable,t._stream));t.check(t.sync())
            samples=[]
            for _ in range(windows):
                t.check(d.cuEventRecord(begin,t._stream));t.check(d.cuGraphLaunch(executable,t._stream))
                t.check(d.cuEventRecord(end,t._stream));t.check(t.sync())
                elapsed=ct.c_float();t.check(d.cuEventElapsedTime(ct.byref(elapsed),begin,end))
                samples.append(elapsed.value/repeats)
            return samples
        finally:
            if capturing:d.cuStreamEndCapture(t._stream,ct.byref(graph))
            t.check(t.sync())
            for handle,name in ((end,'cuEventDestroy_v2'),(begin,'cuEventDestroy_v2'),
                                (executable,'cuGraphExecDestroy'),(graph,'cuGraphDestroy')):
                if handle.value:t.check(getattr(d,name)(handle))


def run(output):
    if rt._nvidia_device_name()!='sm_120':raise RuntimeError('requires owning SM120')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap','--format=csv,noheader'],text=True).strip()
    for storage in (np.float16,ml_dtypes.bfloat16):
        for dims in ((1,2,1,3,5,4,3),(1,4,2,16,33,16,8),(1,4,2,32,129,32,16)):
            b,hq,hkv,sq,sk,d,dv=dims
            rng=np.random.default_rng(202610175)
            roots=[rng.normal(0,.2,shape).astype(storage) for shape in ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv))]
            do=rng.normal(0,.1,(b,hq,sq,dv)).astype(storage)
            seed=rng.normal(0,.1,(b,hq,sq)).astype(np.float32)
            expected,gradients=reference(roots[0],roots[1],roots[2],do,None,seed)
            fn=ts.jit(target='nvidia_sm120',autodiff='reverse',wrt=('q','k','v'))(function(True,False,True))
            start=time.perf_counter()
            program=fn.compile_native_attention_vjp(*roots,compiler=os.environ['TESSERA_OPT'],sequence_bounds=(sq+2,sk+3))
            compile_ms=(time.perf_counter()-start)*1000
            with NvidiaDeviceSession() as session:
                resident=[session.upload(value) for value in roots]
                seeds=(session.upload(do),session.upload(seed))
                with program.capture(*resident,asynchronous=True) as frame:
                    frame.wait_on(session.stream)
                    for actual,wanted in zip(frame.primal,expected,strict=True):compare(download(session,actual),wanted)
                    got=frame.backward(seeds);frame.wait_on(session.stream)
                    for actual,wanted in zip(got,gradients,strict=True):compare(download(session,actual),wanted)
                    tape=frame._frame
                    clock=GraphWindows(tape)
                    forward=clock.measure(False,tape._saved)
                    external=[SimpleNamespace(pointer=ct.c_void_p(value.ptr)) for value in seeds]
                    reverse_buffers=[external[0],*tape._saved[:4],tape._saved[4],external[1],*(value._buffer for value in got)]
                    backward=clock.measure(True,reverse_buffers)
                    frame.wait_on(session.stream)
                    for actual,wanted in zip(frame.primal,expected,strict=True):compare(download(session,actual),wanted)
                    for actual,wanted in zip(got,gradients,strict=True):compare(download(session,actual),wanted)
                    host_backward=[]
                    for _ in range(3):
                        start=time.perf_counter();held=frame.backward(seeds);frame.synchronize()
                        host_backward.append((time.perf_counter()-start)*1000)
                    frame.wait_on(session.stream)
                    for actual,wanted in zip(held,gradients,strict=True):compare(download(session,actual),wanted)
                host_forward=[]
                for _ in range(3):
                    start=time.perf_counter();fresh=program.capture(*resident,asynchronous=True);fresh.synchronize()
                    host_forward.append((time.perf_counter()-start)*1000);fresh.close()
            text=program.to_json();name=str(np.dtype(storage))+'-'+'x'.join(map(str,dims))
            payload=text.encode();(output/(name+'.program.json.xz')).write_bytes(lzma.compress(payload))
            record={'shape':dims,'storage':str(np.dtype(storage)),'device':device,
                'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'recorder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'compiler_sha256':hashlib.sha256(Path(os.environ['TESSERA_OPT']).read_bytes()).hexdigest(),
                'program_digest':program.program_digest,'program_sha256':hashlib.sha256(payload).hexdigest(),
                'forward_image_sha256':hashlib.sha256(program.pair.forward.image.payload).hexdigest(),
                'backward_image_sha256':hashlib.sha256(program.pair.backward.image.payload).hexdigest(),
                'forward_schedule_digest':program.pair.forward.descriptor.provenance['schedule_digest'],
                'backward_schedule_digest':program.pair.backward.descriptor.provenance['schedule_digest'],
                'correctness':'forward and reverse independent oracle before and after profiling',
                'compile_ms':compile_ms,'forward_dispatch_samples_ms':forward,'backward_dispatch_samples_ms':backward,
                'forward_capture_host_samples_ms':host_forward,'backward_owner_host_samples_ms':host_backward,
                'timing_scope':{'dispatch':'32 exact native kernels captured on the owned stream; averaged per dispatch; includes CUDA graph dispatch overhead; excludes copies and owner preparation',
                    'forward_host':'completed capture with private snapshots, allocations and module load; excludes compilation, root upload and close',
                    'backward_host':'completed reverse with seed copies and new gradient allocations; excludes external seed upload and readback'},
                'selector_promotion':False}
            for label in ('forward_dispatch','backward_dispatch','forward_capture_host','backward_owner_host'):
                record[label+'_median_ms']=median(record[label+'_samples_ms'])
            (output/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps({key:record[key] for key in ('shape','storage','forward_dispatch_median_ms','backward_dispatch_median_ms','forward_capture_host_median_ms','backward_owner_host_median_ms')}),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
