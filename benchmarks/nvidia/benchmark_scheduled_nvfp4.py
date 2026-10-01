"""Exact SM120 correctness-gated NVFP4 Schedule/Tile timing packet."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, subprocess, time
from pathlib import Path
import numpy as np
from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.runtime import _nvidia_native_descriptor_device_latency, _nvidia_native_descriptor_resources, launch
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_e2e_spine_native import _nvfp4_module, _pack_nvfp4, _decode_e2m1, _decode_ue4m3

def median(xs): return float(statistics.median(xs))
def run_shape(shape, samples, reps, warmup, seed):
    m,n,k=shape
    module=_nvfp4_module(m,n,k)
    bundle=compile_graph_module(module,source_origin="NVIDIA-E2E-1",target="nvidia_sm120",options={"package_native":True},enable_tool_validation=False)
    d,i=bundle.launch_descriptor,bundle.native_image
    if d is None or i is None: raise RuntimeError("Schedule/Tile package did not produce native image")
    artifact=compile_result_from_bundle(bundle,module=module).to_runtime_artifact()
    rng=np.random.default_rng(seed+m+n+k)
    ac=rng.integers(0,16,size=(m,k),dtype=np.uint8); bc=rng.integers(0,16,size=(k,n),dtype=np.uint8)
    sk=(k+15)//16; choices=np.asarray([0x30,0x31,0x33,0x35,0x38,0x3A,0x3D,0x40,0x42],np.uint8)
    sa=np.ascontiguousarray(choices[(np.arange(m)[:,None]+np.arange(sk)[None,:])%choices.size])
    sb=np.ascontiguousarray(choices[(2*np.arange(sk)[:,None]+np.arange(n)[None,:])%choices.size])
    a=_pack_nvfp4(ac,1); b=_pack_nvfp4(bc,0); c=np.zeros((m,n),np.float32)
    args={"a":a,"b":b,"scale_a":sa,"scale_b":sb,"c":c,"M":m,"N":n,"K":k}
    result=launch(artifact,args)
    if not result.get("ok"): raise RuntimeError(f"correctness launch failed: {result}")
    ar=_decode_e2m1(ac)*np.repeat(_decode_ue4m3(sa),16,axis=1)[:,:k]
    br=_decode_e2m1(bc)*np.repeat(_decode_ue4m3(sb),16,axis=0)[:k,:]
    error=float(np.max(np.abs(c-ar@br)))
    if not np.allclose(c,ar@br,rtol=0,atol=2e-3): raise RuntimeError(f"oracle mismatch max_abs_error={error}")
    device=[_nvidia_native_descriptor_device_latency(i,d,args,reps=reps,warmup=warmup) for _ in range(samples)]
    e2e=[]
    for _ in range(samples):
        start=time.perf_counter()
        for _ in range(reps):
            r=launch(artifact,args)
            if not r.get("ok"): raise RuntimeError(f"end-to-end launch failed: {r}")
        e2e.append((time.perf_counter()-start)*1e3/reps)
    return {"shape_mnk":list(shape),"route":"GraphIR->ScheduleIR->TileIR->NVIDIA Target IR->PTX","entry":d.entry_symbol,"abi_id":d.abi_id,"schedule_digest":d.provenance.get("schedule_digest"),"tile_ir_digest":d.provenance.get("tile_ir_digest"),"max_abs_error":error,"correctness":"passed_before_timing","device_event_samples_ms":device,"device_event_median_ms":median(device),"end_to_end_samples_ms":e2e,"end_to_end_median_ms":median(e2e),"resources":_nvidia_native_descriptor_resources(i,d,block_size=128)}
def main():
    p=argparse.ArgumentParser();p.add_argument("--samples",type=int,default=7);p.add_argument("--reps",type=int,default=100);p.add_argument("--warmup",type=int,default=30);p.add_argument("--output",type=Path);a=p.parse_args()
    if not nvidia_cuda_host_ready(): raise SystemExit("exact-device SM120 unavailable")
    rows=[run_shape(s,a.samples,a.reps,a.warmup,120_530) for s in ((16,8,64),(33,19,129),(7,5,31))]
    source_files=("python/tessera/compiler/graph_ir.py","python/tessera/compiler/nvidia_native.py","python/tessera/compiler/scheduled_matmul.py","src/compiler/ir/TesseraOps.td","src/compiler/ir/TesseraOps.cpp","src/compiler/programming_model/lib/PMPasses.cpp","src/compiler/programming_model/ir/ScheduleDialect.cpp","src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp","tests/device/nvidia/test_e2e_spine_native.py","tests/unit/test_nvidia_e2e_spine.py","benchmarks/nvidia/benchmark_scheduled_nvfp4.py")
    source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files}
    build_tools={name:hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest() for name in ("TESSERA_OPT","TESSERA_NVIDIA_OPT") if os.environ.get(name) and Path(os.environ[name]).is_file()}
    revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    packet={"schema":"tessera.nvidia.nvfp4_scheduled_benchmark.v1","work_item":"NVIDIA-NVFP4-SCHEDULE-2026-09","device":"NVIDIA GeForce RTX 5070 (sm_120)","compiler":"LLVM/MLIR 23.1.1","source_revision":revision,"source_sha256":source_hashes,"compiler_binary_sha256":build_tools,"gpu_reported":gpu,"method":{"timing_domains":["cuda_event","end_to_end"],"samples":a.samples,"repetitions":a.reps,"warmup":a.warmup,"end_to_end_definition":"runtime.launch wall time per call, including host binding/staging and synchronization","selector_changed":False},"packets":rows}
    data=json.dumps(packet,indent=2,sort_keys=True)+"\n"
    if a.output:a.output.write_text(data)
    else:print(data,end="")
if __name__=="__main__":main()
