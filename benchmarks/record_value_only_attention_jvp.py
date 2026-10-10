"""Saved-LSE V-only linearity proof with finite, Inf and NaN primal V."""
import argparse
import ctypes as ct
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tessera as ts
from benchmarks.record_device_ring_protocol import Device


def function(causal):
    @ts.jit(target='nvidia_sm120',autodiff='forward',wrt=('v',))
    def attention(q,k,v):
        return ts.ops.flash_attn(q,k,v,causal=causal)
    return attention


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
        '--query-gpu=name,uuid,compute_cap,driver_version','--format=csv,noheader'],text=True).strip()
    if len(gpu.splitlines())!=1 or 'RTX 5070' not in gpu or '12.0' not in gpu:
        raise RuntimeError('recorder requires the owning single RTX 5070 / sm_120')
    device=Device('nvidia')
    rows=[]
    for sq,sk in ((3,5),(5,3),(4,4),(3,129)):
        for causal in (False,True):
            rng=np.random.default_rng(915)
            q=rng.normal(size=(1,2,sq,4)).astype(np.float32)*.2
            k=rng.normal(size=(1,1,sk,4)).astype(np.float32)*.2
            v=rng.normal(size=(1,1,sk,3)).astype(np.float32)*.2
            dv=rng.normal(size=v.shape).astype(np.float32)*.1
            score=.5*(q.astype(np.float64)@np.swapaxes(k.astype(np.float64),-1,-2))
            if causal:
                score=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),score,-np.inf)
            weights=np.exp(score-score.max(axis=-1,keepdims=True))
            weights/=weights.sum(axis=-1,keepdims=True)
            expected=weights@dv.astype(np.float64)
            program=function(causal).compile_native_attention_jvp(q,k,v,
                compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'))
            assert 'cooperative_saved_lse_value_linear_v1' in program.tangent.arena_ir
            for mode in ('finite','inf','nan'):
                pointers=[]
                def upload(value):
                    pointer=ct.c_void_p()
                    device.check(device.alloc(ct.byref(pointer),value.nbytes))
                    pointers.append(pointer)
                    device.check(device.htod(pointer,value.ctypes.data,value.nbytes))
                    return SimpleNamespace(__cuda_array_interface__=dict(
                        version=3,shape=value.shape,typestr=value.dtype.str,data=(pointer.value,False)))
                def download(value):
                    interface=value.__cuda_array_interface__
                    out=np.empty(interface['shape'],np.float32)
                    device.check(device.dtoh(out.ctypes.data,ct.c_void_p(interface['data'][0]),out.nbytes))
                    return out
                primal=v if mode=='finite' else np.full_like(v,np.inf if mode=='inf' else np.nan)
                try:
                    with program.capture(upload(q),upload(k),upload(primal)) as frame:
                        first=download(frame.jvp(upload(dv)))
                        twice=download(frame.jvp(upload(dv*2)))
                        np.testing.assert_allclose(first,expected,atol=3e-5,rtol=3e-5)
                        np.testing.assert_allclose(twice,expected*2,atol=3e-5,rtol=3e-5)
                        assert np.isfinite(first).all()
                    rows.append(dict(sq=sq,sk=sk,causal=causal,primal_v=mode,
                        max_abs_error=float(np.max(np.abs(first-expected))),linearity='passed',
                        image_sha256=hashlib.sha256(program.tangent.image).hexdigest(),
                        binding_digest=program.tangent.binding_digest))
                finally:
                    for pointer in pointers:
                        device.check(device.free(pointer))
    args.output.write_text(json.dumps(dict(rows=rows,backend='nvidia_sm120',device=gpu,
        oracle='float64 fixed-QK softmax probabilities times dV; primal V is not a dependency',
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            'src/transforms/lib/AutodiffForwardPass.cpp',
            'src/compiler/programming_model/lib/NativeAttentionJvp.h',
            'python/tessera/compiler/native_attention_program.py')},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(len(rows),'V-only fixed-QK linearity cases passed')


if __name__=='__main__':
    main()
