#!/usr/bin/env python3
"""Exact CUDA proof: fresh paired AD -> native Schedule -> native packages."""
import argparse
import ctypes as ct
from types import SimpleNamespace
import hashlib
import os
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from tessera.compiler.nvidia_native import package_generated_attention_checkpoint_pair  # noqa: E402
from tessera.runtime import RuntimeArtifact, launch  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument("--resident",action="store_true")
    parser.add_argument("--jvp",action="store_true")
    parser.add_argument("--automatic-jvp",action="store_true")
    args=parser.parse_args()
    if args.automatic_jvp and not args.jvp:
        parser.error("--automatic-jvp requires --jvp")
    if args.jvp and not args.resident:
        parser.error("--jvp requires --resident")
    rows=[]
    if args.resident:
        from benchmarks.record_device_ring_protocol import Device
        device=Device('nvidia')
    for sq,sk in (((3,5),(5,3),(4,4),(3,129)) if args.jvp else ((3,5),(5,3),(4,4))):
        for causal in (False,True):
            qtype=f'tensor<1x2x{sq}x4xf32>'
            ktype=f'tensor<1x1x{sk}x4xf32>'
            vtype=f'tensor<1x1x{sk}x3xf32>'
            otype=f'tensor<1x2x{sq}x3xf32>'
            source=f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @attention(%q: {qtype}, %k: {ktype}, %v: {vtype}) -> {otype}
    attributes {{tessera.autodiff = "reverse"}} {{
    %o = tessera.flash_attn %q, %k, %v {{head_dim = 4 : i64, causal = {str(causal).lower()}, dropout_p = 0.0 : f64, operandSegmentSizes = array<i32: 1, 1, 1, 0>}}
      : ({qtype}, {ktype}, {vtype}) -> {otype}
    return %o : {otype}
  }}
}}'''
            pair=package_generated_attention_checkpoint_pair(source,pipeline_name='tessera-nvidia-pipeline-sm120')
            rng=np.random.default_rng(712)
            q=rng.normal(size=(1,2,sq,4)).astype(np.float32)*.2
            k=rng.normal(size=(1,1,sk,4)).astype(np.float32)*.2
            v=rng.normal(size=(1,1,sk,3)).astype(np.float32)*.2
            do=rng.normal(size=(1,2,sq,3)).astype(np.float32)*.2
            scores=.5*(q.astype(np.float64)@np.swapaxes(k.astype(np.float64),-1,-2))
            mask=np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0) if causal else np.ones((sq,sk),bool)
            scores=np.where(mask,scores,-np.inf)
            maximum=np.max(scores,axis=-1,keepdims=True)
            maximum=np.where(np.isfinite(maximum),maximum,0)
            ex=np.where(mask,np.exp(scores-maximum),0)
            denominator=ex.sum(axis=-1,keepdims=True)
            prob=np.divide(ex,denominator,out=np.zeros_like(ex),where=denominator!=0)
            out=prob@v
            with np.errstate(divide='ignore'):
                lse=(maximum+np.log(denominator))[...,0]
            dp=do@np.swapaxes(v,-1,-2)
            ds=prob*(dp-(prob*dp).sum(axis=-1,keepdims=True))
            dq=.5*(ds@k)
            dk=(.5*(np.swapaxes(ds,-1,-2)@q)).sum(axis=1,keepdims=True)
            dv=(np.swapaxes(prob,-1,-2)@do).sum(axis=1,keepdims=True)
            values=dict(q=q,k=k,v=v,dO=do,output=np.empty_like(out,dtype=np.float32),lse=np.empty_like(lse,dtype=np.float32),dq=np.empty_like(q),dk=np.empty_like(k),dv=np.empty_like(v),B=1,Hq=2,Hkv=1,Sq=sq,Sk=sk,D=4,Dv=3)
            jvp_digest=None
            if args.resident:
                allocations=[]
                def upload(value):
                    pointer=ct.c_void_p()
                    device.check(device.alloc(ct.byref(pointer),value.nbytes))
                    allocations.append(pointer)
                    device.check(device.htod(pointer,value.ctypes.data,value.nbytes))
                    return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,typestr=value.dtype.str,data=(pointer.value,False)))
                def download(value):
                    interface=value.__cuda_array_interface__
                    result=np.empty(interface['shape'],np.float32)
                    device.check(device.dtoh(result.ctypes.data,ct.c_void_p(interface['data'][0]),result.nbytes))
                    return result
                frame=None
                try:
                    inputs=[upload(value) for value in (q,k,v)]
                    frame=pair.capture(*inputs)
                    np.testing.assert_allclose(download(frame.primal),out,atol=3e-5,rtol=3e-5)
                    # The original buffers can change after capture. Backward
                    # must use the private Q/K/V and forward-generation LSE.
                    for pointer,value in zip(allocations,(q,k,v),strict=True):
                        changed=np.full_like(value,19)
                        device.check(device.htod(pointer,changed.ctypes.data,changed.nbytes))
                    first=frame.backward(upload(do))
                    second=frame.backward(upload(do*2))
                    for result,twice,reference in zip(first,second,(dq,dk,dv),strict=True):
                        np.testing.assert_allclose(download(result),reference,atol=3e-5,rtol=3e-5)
                        np.testing.assert_allclose(download(twice),reference*2,atol=3e-5,rtol=3e-5)
                    if args.jvp:
                        jvp_digest=frame.prepare_jvp(compiler=Path(os.environ['TESSERA_OPT']),llvm_bin=Path('/usr/lib/llvm-23/bin'),
                            source=source.replace('tessera.autodiff = "reverse"','tessera.autodiff = "forward"') if args.automatic_jvp else None)
                        directions=[rng.normal(size=value.shape).astype(np.float32)*.1 for value in (q,k,v)]
                        for mode in ('q','k','qk','qkv'):
                            qdot,kdot,vdot=[direction if key in mode else np.zeros_like(direction)
                                for key,direction in zip(('q','k','v'),directions,strict=True)]
                            score_dot=.5*(qdot.astype(np.float64)@np.swapaxes(k.astype(np.float64),-1,-2)
                                         +q.astype(np.float64)@np.swapaxes(kdot.astype(np.float64),-1,-2))
                            expected=(prob*score_dot)@v + prob@vdot - out*(prob*score_dot).sum(axis=-1,keepdims=True)
                            def reference(a,b,c):
                                logits=np.where(mask,.5*(a@np.swapaxes(b,-1,-2)),-np.inf)
                                weights=np.exp(logits-logits.max(axis=-1,keepdims=True))
                                return (weights/weights.sum(axis=-1,keepdims=True))@c
                            step=1e-4
                            plus=reference(q.astype(np.float64)+step*qdot,k.astype(np.float64)+step*kdot,v.astype(np.float64)+step*vdot)
                            minus=reference(q.astype(np.float64)-step*qdot,k.astype(np.float64)-step*kdot,v.astype(np.float64)-step*vdot)
                            np.testing.assert_allclose(expected,(plus-minus)/(2*step),atol=1e-8,rtol=1e-5)
                            product=frame.jvp(upload(qdot),upload(kdot),upload(vdot))
                            np.testing.assert_allclose(download(product),expected,atol=3e-5,rtol=3e-5)
                    frame.close()
                    try:
                        first[0].__cuda_array_interface__
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('closed attention frame exposed freed storage')
                finally:
                    if frame is not None:
                        frame.close()
                    for pointer in allocations:
                        device.check(device.free(pointer))
            else:
                for package in (pair.forward,pair.backward):
                    artifact=RuntimeArtifact(tile_ir=package.tile_ir,target_ir=package.target_ir,native_image=package.image,launch_descriptor=package.descriptor,metadata={'target':'nvidia_sm120'})
                    names={b.name for b in package.descriptor.buffers} | {s.name for s in package.descriptor.scalars}
                    result=launch(artifact,{name:values[name] for name in names})
                    if not result['ok']:
                        raise RuntimeError(result)
                for name,reference in [('output',out),('lse',lse),('dq',dq),('dk',dk),('dv',dv)]:
                    np.testing.assert_allclose(values[name],reference,atol=3e-5,rtol=3e-5)
            rows.append(dict(sq=sq,sk=sk,causal=causal,source_sha256=hashlib.sha256(source.encode()).hexdigest(),forward_image=pair.forward.image.image_digest,backward_image=pair.backward.image.image_digest,checkpoint_contract=pair.contract_digest,oracle='3e-5',jvp_package=jvp_digest,jvp_modes=['q','k','qk','qkv'] if args.jvp else []))
    args.output.write_text(json.dumps(dict(rows=rows,backend='nvidia_sm120',recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),resident_lse_tape=args.resident,automatic_jvp=args.automatic_jvp,implementation_sha256={name:hashlib.sha256((ROOT/"python/tessera/compiler"/name).read_bytes()).hexdigest() for name in ("resident_attention.py","nvidia_native.py","native_attention_jvp.py")}),indent=2)+'\n')
    print(len(rows),'generated attention AD cases passed')


if __name__=='__main__':
    main()
