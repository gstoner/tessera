#!/usr/bin/env python3
"""JIT trace -> automatic Q/K product -> resident CUDA finite-difference proof."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=Device('nvidia')
    rows=[]
    for sk in (5,129):
        for wrt in (('q',),('k',),('q','k'),('k','q'),('q','k','v')):
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
                rows.append(dict(sk=sk,wrt=wrt,package=program.tangent.binding_digest,forward=program.pair.contract_digest))
            finally:
                for p in pointers: device.check(device.free(p))
    args.output.write_text(json.dumps(dict(rows=rows,backend='nvidia_sm120',
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'python/tessera/compiler/jit.py', 'python/tessera/compiler/graph_ir.py',
            'python/tessera/compiler/native_attention_program.py', 'python/tessera/compiler/native_attention_jvp.py',
            'python/tessera/compiler/resident_attention.py', 'python/tessera/compiler/nvidia_native.py')},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(len(rows),'JIT attention program cases passed')

if __name__=='__main__': main()
