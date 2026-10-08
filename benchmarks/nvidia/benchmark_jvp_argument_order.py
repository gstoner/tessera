#!/usr/bin/env python3
"""Native JVP frontend permutations, independent finite differences and timing."""
import argparse
import ctypes as ct
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"python")]
import tessera as ts  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_jit_attention_program import timed_native_product  # noqa: E402

def function(order,causal):
    def qkv(q,k,v): return ts.ops.flash_attn(q,k,v,causal=causal)
    def qvk(q,v,k): return ts.ops.flash_attn(q,k,v,causal=causal)
    def kqv(k,q,v): return ts.ops.flash_attn(q,k,v,causal=causal)
    def kvq(k,v,q): return ts.ops.flash_attn(q,k,v,causal=causal)
    def vqk(v,q,k): return ts.ops.flash_attn(q,k,v,causal=causal)
    def vkq(v,k,q): return ts.ops.flash_attn(q,k,v,causal=causal)
    return {("q","k","v"):qkv,("q","v","k"):qvk,("k","q","v"):kqv,
            ("k","v","q"):kvq,("v","q","k"):vqk,("v","k","q"):vkq}[order]

def run(device,order,wrt,sk,causal,compiler,artifacts):
    rng=np.random.default_rng(912)
    shapes=((1,2,3,4),(1,1,sk,4),(1,1,sk,3))
    values={n:rng.normal(size=s).astype(np.float32)*.2 for n,s in zip(("q","k","v"),shapes,strict=True)}
    directions={n:rng.normal(size=s).astype(np.float32)*.1 if n in wrt else np.zeros(s,np.float32)
                for n,s in zip(("q","k","v"),shapes,strict=True)}
    fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=wrt)(function(order,causal))
    program=fn.compile_native_attention_jvp(*(values[n] for n in order),
                                           compiler=compiler,llvm_bin=Path("/usr/lib/llvm-23/bin"))
    encoded=program.to_json()
    original_digest=program.program_digest
    from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
    program=NativeAttentionJVPProgram.from_json(encoded,expected_digest=original_digest)
    assert program.to_json()==encoded
    expected_mapping=tuple(order.index(n) for n in ("q","k","v"))
    assert program.input_indices==expected_mapping
    assert program.active==tuple(("q","k","v").index(n) for n in wrt)
    pointers=[]
    def upload(x):
        p=ct.c_void_p();device.check(device.alloc(ct.byref(p),x.nbytes));pointers.append(p)
        device.check(device.htod(p,x.ctypes.data,x.nbytes))
        return SimpleNamespace(__cuda_array_interface__={"version":3,"shape":x.shape,"typestr":x.dtype.str,"data":(p.value,False)})
    def download(x):
        spec=x.__cuda_array_interface__;out=np.empty(spec["shape"],np.float32)
        device.check(device.dtoh(out.ctypes.data,ct.c_void_p(spec["data"][0]),out.nbytes))
        return out
    def reference(q,k,v):
        score=.5*(q@np.swapaxes(k,-1,-2))
        if causal:
            mask=np.arange(sk)[None,:]<=np.arange(3)[:,None]+max(sk-3,0)
            score=np.where(mask,score,-np.inf)
        probability=np.exp(score-score.max(axis=-1,keepdims=True));probability/=probability.sum(axis=-1,keepdims=True)
        return probability@v
    step=1e-4
    plus=reference(*(values[n].astype(np.float64)+step*directions[n] for n in ("q","k","v")))
    minus=reference(*(values[n].astype(np.float64)-step*directions[n] for n in ("q","k","v")))
    oracle=(plus-minus)/(2*step)
    try:
        inputs={n:upload(x) for n,x in values.items()}
        tangent={n:upload(directions[n]) for n in wrt}
        with program.capture(**inputs) as frame:
            np.testing.assert_allclose(download(frame.primal),reference(*(values[n].astype(np.float64) for n in ("q","k","v"))),atol=3e-5,rtol=3e-5)
            result=frame.jvp(*(tangent[n] for n in wrt))
            np.testing.assert_allclose(download(result),oracle,atol=3e-5,rtol=3e-5)
            clear=ct.CDLL("libcuda.so.1").cuMemsetD32_v2
            clear.argtypes=[ct.c_uint64,ct.c_uint,ct.c_size_t];clear.restype=ct.c_int
            for n,x in inputs.items():
                device.check(clear(x.__cuda_array_interface__["data"][0],0,values[n].size))
            device.check(device.sync())
            walls=[]
            for multiplier in (2,1):
                scaled={n:upload(directions[n]*multiplier) for n in wrt}
                begin=time.perf_counter_ns()
                repeated=frame.jvp(*(scaled[n] for n in wrt))
                walls.append((time.perf_counter_ns()-begin)/1e6)
                np.testing.assert_allclose(download(repeated),oracle*multiplier,atol=3e-5,rtol=3e-5)
                np.testing.assert_allclose(download(result),oracle,atol=3e-5,rtol=3e-5)
            slots={**frame._zeros,**{("q","k","v").index(n):tangent[n] for n in wrt}}
            native=frame._frame
            raw=[x.pointer.value for x in native._saved]
            raw.extend(slots[i].__cuda_array_interface__["data"][0] for i in range(3))
            raw.extend((result.__cuda_array_interface__["data"][0],128))
            timing=timed_native_product(device,program.tangent,raw,6)
            np.testing.assert_allclose(download(result),oracle,atol=3e-5,rtol=3e-5)
            error=float(np.max(np.abs(download(result)-oracle)))
        try: frame.jvp(*(tangent[n] for n in wrt))
        except ValueError: pass
        else: raise AssertionError("closed JVP frame executed")
        label="".join(order)+"_"+"_".join(wrt)+f"_{sk}_{int(causal)}"
        (artifacts/(label+".mlir")).write_text(program.tangent.arena_ir)
        (artifacts/(label+".image")).write_bytes(program.tangent.image)
        (artifacts/(label+".program.json")).write_text(encoded)
        return {"frontend_order":order,"wrt":wrt,"sk":sk,"causal":causal,"native_mapping":program.input_indices,
                "physical_active":program.active,"max_abs_error":error,"private_capture_mutation_and_repeated_directions":"passed",
                "checked_jvp_wall_samples_ms":walls,"checked_jvp_wall_median_ms":statistics.median(walls),
                "tangent_digest":program.tangent.binding_digest,"forward_digest":program.pair.contract_digest,**timing}
    finally:
        for p in pointers: device.check(device.free(p))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    compiler=Path(__import__("os").environ["TESSERA_OPT"]);device=Device("nvidia")
    artifacts=args.output.parent/"artifacts";artifacts.mkdir(parents=True,exist_ok=True);rows=[]
    for order in itertools.permutations(("q","k","v")):
        for sk,causal in ((5,False),(129,True)):
            for wrt in (("q",),("k",),("v",),("k","q"),("v","q"),("q","k","v")):
                rows.append(run(device,order,wrt,sk,causal,compiler,artifacts));print("verified",order,wrt,sk,causal,flush=True)
    sources=("python/tessera/compiler/native_attention_program.py","python/tessera/compiler/native_attention_jvp.py",
             "python/tessera/compiler/jit.py","src/compiler/programming_model/lib/NativeAttentionJvp.h",
             "src/transforms/lib/AutodiffForwardPass.cpp","python/tessera/compiler/native_attention_jvp_artifact.py","benchmarks/nvidia/benchmark_jvp_argument_order.py")
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    hashes[str(compiler)]=hashlib.sha256(compiler.read_bytes()).hexdigest()
    args.output.write_text(json.dumps({"device":gpu,"architecture":"sm_120","rows":rows,"fingerprints":hashes,
        "timing_scope":"preloaded raw CUDA-event dispatch window; checked allocating JVP wall separate; no throughput or isolated kernel claim",
        "route":"frontend -> native AD -> Graph/Schedule/Tile -> native GPU LLVM/NVVM image"},indent=2)+"\n")
    print("wrote",args.output,flush=True)
if __name__=="__main__": main()
