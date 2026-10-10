#!/usr/bin/env python3
"""Fresh-process native attention JVP replay; compiler subprocesses forbidden."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"python")]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram  # noqa: E402

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--program",type=Path,required=True);ap.add_argument("--digest",required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    os.environ["TESSERA_OPT"]="/missing/compiler";os.environ["TESSERA_NVIDIA_OPT"]="/missing/target-compiler"
    def forbidden(*a,**k): raise AssertionError("compiler/process access during replay")
    with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
        program=NativeAttentionJVPProgram.from_json(args.program.read_text(),expected_digest=args.digest)
        device=Device("nvidia");dims=program.pair.forward.descriptor.provenance["shape"];b,hq,hkv,sq,sk,d,dv=dims
        causal=program.pair.forward.descriptor.provenance["causal"]
        scale=program.pair.forward.descriptor.provenance["scale"]
        rng=np.random.default_rng(912)
        values=[rng.normal(size=s).astype(np.float32)*.2 for s in ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv))]
        directions=[rng.normal(size=x.shape).astype(np.float32)*.1 if i in program.active else np.zeros_like(x) for i,x in enumerate(values)]
        def oracle(q,k,v):
            repeated_k=np.repeat(k,hq//hkv,axis=1);repeated_v=np.repeat(v,hq//hkv,axis=1)
            score=scale*(q@np.swapaxes(repeated_k,-1,-2))
            if causal: score=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),score,-np.inf)
            p=np.exp(score-score.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
            return p@repeated_v
        h=1e-4;expected=(oracle(*(x.astype(np.float64)+h*t for x,t in zip(values,directions,strict=True)))-oracle(*(x.astype(np.float64)-h*t for x,t in zip(values,directions,strict=True))))/(2*h)
        pointers=[]
        def upload(x):
            pointer=ct.c_void_p();device.check(device.alloc(ct.byref(pointer),x.nbytes));pointers.append(pointer)
            device.check(device.htod(pointer,x.ctypes.data,x.nbytes))
            return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=x.shape,typestr=x.dtype.str,data=(pointer.value,False)))
        def download(x):
            data=x.__cuda_array_interface__;out=np.empty(data["shape"],np.float32)
            device.check(device.dtoh(out.ctypes.data,ct.c_void_p(data["data"][0]),out.nbytes));return out
        try:
            physical=[upload(x) for x in values];frontend=[None]*3
            for i,index in enumerate(program.input_indices): frontend[index]=physical[i]
            with program.capture(**dict(zip(program.input_names,frontend,strict=True))) as frame:
                np.testing.assert_allclose(download(frame.primal),oracle(*(x.astype(np.float64) for x in values)),atol=3e-5,rtol=3e-5)
                result=frame.jvp(*(upload(directions[i]) for i in program.active))
                np.testing.assert_allclose(download(result),expected,atol=3e-5,rtol=3e-5)
                scaled=frame.jvp(*(upload(2*directions[i]) for i in program.active))
                np.testing.assert_allclose(download(scaled),2*expected,atol=3e-5,rtol=3e-5)
                np.testing.assert_allclose(download(result),expected,atol=3e-5,rtol=3e-5)
                error=float(np.max(np.abs(download(result)-expected)))
        finally:
            for pointer in pointers: device.check(device.free(pointer))
    args.output.write_text(json.dumps(dict(device=gpu,program_digest=args.digest,frontend_indices=program.input_indices,physical_active=program.active,max_abs_error=error,compiler_subprocesses="forbidden",correctness="passed"),indent=2)+"\n")
    print("compiler-free replay passed",args.program,flush=True)
if __name__=="__main__": main()
