#!/usr/bin/env python3
"""JIT trace -> automatic Q/K product -> resident CUDA finite-difference proof."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time
import statistics
import subprocess
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import tessera as ts  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402


def function(wrt):
    @ts.jit(target='nvidia_sm120',autodiff='forward',wrt=wrt)
    def attention(q,k,v):
        return ts.ops.flash_attn(q,k,v,causal=True)
    return attention


def timed_native_product(device,package,raw,rows):
    """Time a preloaded kernel; measure checked frame wall time separately."""
    P=ct.c_void_p
    start,end=P(),P()
    with package.bind() as bound:
        args=bound._arguments(tuple(raw))
        shared=int(bound._size(*args))
        if shared<0:
            raise RuntimeError("native scratch sizer refused the product")
        pointers=(P*len(args))(*(ct.cast(ct.pointer(a),P) for a in args))
        def launch():
            device.check(device.launch(bound._function,rows,1,1,128,1,1,shared,None,pointers,None))
        resources={}
        for name,attribute in (("registers",4),("local_bytes",3)):
            value=ct.c_int()
            device.check(device.attribute(ct.byref(value),attribute,bound._function))
            resources[name]=value.value
        resources["dynamic_shared_bytes"]=shared
        resources["block_threads"]=128
        active=ct.c_int()
        device.check(device.occupancy(ct.byref(active),bound._function,128,shared))
        resources["active_blocks_per_sm"]=active.value
        try:
            device.check(device.event_create(ct.byref(start),0))
            device.check(device.event_create(ct.byref(end),0))
            for _ in range(3):
                launch()
            device.check(device.sync())
            samples=[]
            for _ in range(5):
                device.check(device.event_record(start,None))
                for _ in range(64):
                    launch()
                device.check(device.event_record(end,None))
                device.check(device.event_sync(end))
                elapsed=ct.c_float()
                device.check(device.event_elapsed(ct.byref(elapsed),start,end))
                samples.append(elapsed.value/64)
            return dict(device_dispatch_samples_ms=samples,
                        device_dispatch_median_ms=statistics.median(samples),
                        resources=resources)
        finally:
            for event in (start,end):
                if event:
                    device.check(device.event_destroy(event))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-value-only',action='store_true')
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=Device('nvidia')
    gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
        '--query-gpu=name,uuid,compute_cap,driver_version','--format=csv,noheader'],text=True).strip()
    if "12.0" not in gpu or "RTX 5070" not in gpu:
        raise RuntimeError("recorder requires the owning RTX 5070 / sm_120")
    artifact_dir=args.output.parent/'artifacts'
    artifact_dir.mkdir(exist_ok=True)
    rows=[]
    for sk in (5,129):
        cases=(('q',),('k',),('q','k'),('k','q'),('q','k','v'))
        if args.include_value_only:
            cases=(*cases,('v',))
        for wrt in cases:
            rng=np.random.default_rng(119)
            values=[rng.normal(size=shape).astype(np.float32)*.2 for shape in ((1,2,3,4),(1,1,sk,4),(1,1,sk,3))]
            directions=[rng.normal(size=v.shape).astype(np.float32)*.1 if name in wrt else np.zeros_like(v) for name,v in zip(('q','k','v'),values,strict=True)]
            program=function(wrt).compile_native_attention_jvp(*values,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'))
            pointers=[]
            def upload(v):
                p=ct.c_void_p(); device.check(device.alloc(ct.byref(p),v.nbytes)); pointers.append(p)
                device.check(device.htod(p,v.ctypes.data,v.nbytes))
                return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=v.shape,typestr=v.dtype.str,data=(p.value,False)))
            def download(v):
                spec=v.__cuda_array_interface__; out=np.empty(spec['shape'],np.float32)
                device.check(device.dtoh(out.ctypes.data,ct.c_void_p(spec['data'][0]),out.nbytes)); return out
            def reference(q,k,v):
                score=.5*(q@np.swapaxes(k,-1,-2))
                mask=np.arange(sk)[None,:]<=np.arange(3)[:,None]+max(sk-3,0)
                score=np.where(mask,score,-np.inf)
                prob=np.exp(score-score.max(axis=-1,keepdims=True));prob/=prob.sum(axis=-1,keepdims=True)
                return prob@v
            try:
                with program.capture(*(upload(v) for v in values)) as frame:
                    active={name:upload(v) for name,v in zip(('q','k','v'),directions,strict=True) if name in wrt}
                    result=frame.jvp(*(active[name] for name in wrt))
                    step=1e-4
                    plus=reference(*(v.astype(np.float64)+step*d for v,d in zip(values,directions,strict=True)))
                    minus=reference(*(v.astype(np.float64)-step*d for v,d in zip(values,directions,strict=True)))
                    np.testing.assert_allclose(download(result),(plus-minus)/(2*step),atol=3e-5,rtol=3e-5)
                    np.testing.assert_allclose(download(frame.primal),reference(*(v.astype(np.float64) for v in values)),atol=3e-5,rtol=3e-5)
                    native=frame._frame
                    slots={**frame._zeros,**{i:active[name] for i,name in enumerate(('q','k','v')) if name in active}}
                    raw=[buf.pointer.value for buf in native._saved]
                    raw.extend(slots[i].__cuda_array_interface__['data'][0] for i in range(3))
                    raw.extend((result.__cuda_array_interface__['data'][0],128))
                    timing=timed_native_product(device,program.tangent,raw,6)
                    np.testing.assert_allclose(download(result),(plus-minus)/(2*step),atol=3e-5,rtol=3e-5)
                    wall=[]
                    for _ in range(5):
                        begin=time.perf_counter()
                        repeated=frame.jvp(*(active[name] for name in wrt))
                        wall.append((time.perf_counter()-begin)*1000)
                        np.testing.assert_allclose(download(repeated),(plus-minus)/(2*step),atol=3e-5,rtol=3e-5)
                    timing.update(checked_jvp_wall_samples_ms=wall,checked_jvp_wall_median_ms=statistics.median(wall))
                    label=str(sk)+'_'+'_'.join(wrt)
                    (artifact_dir/(label+'.mlir')).write_text(program.tangent.arena_ir)
                    (artifact_dir/(label+'.image')).write_bytes(program.tangent.image)

                rows.append(dict(sk=sk,wrt=wrt,package=program.tangent.binding_digest,forward=program.pair.contract_digest,**timing))
            finally:
                for p in pointers: device.check(device.free(p))
    args.output.write_text(json.dumps(dict(rows=rows,backend='nvidia_sm120',device=gpu,timing_scope='preloaded raw kernel CUDA-event dispatch window; checked allocating JVP host wall measured separately',
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'python/tessera/compiler/jit.py', 'python/tessera/compiler/graph_ir.py',
            'python/tessera/compiler/native_attention_program.py', 'python/tessera/compiler/native_attention_jvp.py',
            'python/tessera/compiler/resident_attention.py', 'python/tessera/compiler/nvidia_native.py',
            'src/compiler/programming_model/lib/NativeAttentionJvp.h', 'src/compiler/programming_model/lib/PMPasses.cpp')},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(len(rows),'JIT attention program cases passed')

if __name__=='__main__': main()
