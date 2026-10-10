"""Independent native scaled primal/JVP packages and owning-device baselines."""
import argparse
import ctypes as c
import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import median
import numpy as np
import ml_dtypes
from tessera.compiler.native_scaled_program import (
    NativeScaledProgram, PreparedScaledProgram, package_native_scaled_primal,
    package_native_scaled_jvp)
from tessera import runtime as rt

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def emit(output):
    from tests.unit.test_native_independent_scaled_primal import source
    cases=[]
    for mask,tb,encoded in itertools.product(range(1,16),(False,True),(False,True)):
        cases.append({"mask":mask,"prefixes":[[2,3] if mask&(1<<i) else [] for i in range(4)],
                      "output_prefix":[2,3],"transposeB":tb,"encoded":encoded,"kind":"primal"})
    for prefixes,output_prefix in [
        ([[2,1],[3],[],[1,3]],[2,3]),([[],[],[],[]],[]),
        ([[2,1,3],[],[1,1,3],[2,1,1]],[2,1,3])]:
        cases.append({"mask":None,"prefixes":prefixes,"output_prefix":output_prefix,
                      "transposeB":True,"encoded":False,"kind":"primal"})
    for mask in (1,2,4,8,12,15):
        cases.append({"mask":mask,"prefixes":[[2,3] if mask&(1<<i) else [] for i in range(4)],
                      "output_prefix":[2,3],"transposeB":True,"encoded":False,"kind":"paired_jvp"})
    for i,row in enumerate(cases):
        text=source(tuple(map(tuple,row["prefixes"])),output=tuple(row["output_prefix"]),
                    tb=row["transposeB"],encoded=row["encoded"])
        if row["kind"]=="paired_jvp":
            result="tensor<"+"x".join(map(str,(*row["output_prefix"],3,5)))+"xf32>"
            text=text.replace("-> "+result+" {","-> "+result+
                ' attributes {tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [2,3]} {',1)
            package=package_native_scaled_jvp(text)
        else:
            package=package_native_scaled_primal(text)
        row["package"]=package.to_manifest()
        print("packaged",i+1,"/",len(cases),flush=True)
    paths=["src/compiler/programming_model/lib/PMPasses.cpp",
           "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
           "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
           "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMKernelIdentity.cpp",
           "python/tessera/compiler/native_scaled_program.py",
           "tests/unit/test_native_independent_scaled_primal.py"]
    output.write_text(json.dumps({"compiler_sha256":digest(os.environ["TESSERA_OPT"]),
        "source_sha256":{p:digest(p) for p in paths},"recorder_sha256":digest(__file__),
        "rows":cases},indent=2)+"\n")

def inputs(row,seed):
    rng=np.random.default_rng(seed)
    suffix=((3,64),(5,64) if row["transposeB"] else (64,5),(3,2),
            (2,5) if row["encoded"] else (2,2))
    values=[]
    for i,(prefix,shape) in enumerate(zip(row["prefixes"],suffix,strict=True)):
        dims=(*prefix,*shape)
        if i<2:
            value=rng.uniform(-.5,.5,dims).astype(ml_dtypes.float8_e4m3fn)
        elif row["encoded"]:
            value=rng.integers(126,129,dims,dtype=np.uint8)
        else:
            value=rng.uniform(.3,1.3,dims).astype(np.float32)
        values.append(value)
    if row["kind"]=="paired_jvp":
        values.extend(rng.uniform(-.5,.5,x.shape).astype(np.float32) for x in values[2:4])
    return values

def primal(values,row):
    a,b=values[0].astype(np.float64),values[1].astype(np.float64)
    if row["transposeB"]:b=b.swapaxes(-1,-2)
    sa,sb=values[2:4]
    if row["encoded"]:
        sa=np.ldexp(np.ones(sa.shape),sa.astype(np.int32)-127)
        sb=np.ldexp(np.ones(sb.shape),sb.astype(np.int32)-127)
    else:
        sa,sb=sa.astype(np.float64),sb.astype(np.float64)
    out=np.zeros((*row["output_prefix"],3,5),np.float64)
    scale_n=1 if row["encoded"] else 3
    for g in range(2):
        dot=np.matmul(a[..., :,g*32:(g+1)*32],b[...,g*32:(g+1)*32,:])
        columns=np.take(sb[...,g,:],np.arange(5)//scale_n,axis=-1)
        out+=dot*sa[..., :,g,None]*columns[...,None,:]
    return out

def expected(values,row):
    p=primal(values,row)
    if row["kind"]=="primal":return [p]
    return [p,primal([values[0],values[1],values[4],values[3]],row)+
              primal([values[0],values[1],values[2],values[5]],row)]

def check(actual,wanted):
    maximum=0.
    for got,want in zip(actual,wanted,strict=True):
        np.testing.assert_allclose(got,want,rtol=3e-5,atol=2e-5)
        maximum=max(maximum,float(np.max(np.abs(got-want))))
    return maximum

def measure(package_path,output):
    if rt._rocm_live_arch()!="gfx1201":
        raise RuntimeError("owning gfx1201 is required")
    library=Path(rt._load_rocm_native_movement_runtime()._name)
    hip=c.CDLL("/opt/rocm/lib/libamdhip64.so")
    if hip.hipInit(0):raise RuntimeError("HIP initialization failed")
    name=c.create_string_buffer(256);uuid=(c.c_ubyte*16)()
    if hip.hipDeviceGetName(name,256,0) or hip.hipDeviceGetUuid(c.byref(uuid),0):
        raise RuntimeError("device identity query failed")
    payload=json.loads(package_path.read_text())
    if len(payload["rows"])!=69:raise ValueError("complete 69-case census is required")
    if payload["recorder_sha256"]!=digest(__file__):raise ValueError("recorder source differs")
    rows=[]
    original_run=subprocess.run
    def forbidden(*args,**kwargs):raise AssertionError("serialized primal/JVP replay invoked compiler")
    subprocess.run=forbidden
    try:
        for number,row in enumerate(payload["rows"]):
            package=NativeScaledProgram.from_manifest(row["package"])
            values=inputs(row,1007+number);wanted=expected(values,row);maximum=0.
            with PreparedScaledProgram(package,values,runtime_library=str(library)) as owner:
                old,_=owner.invoke();maximum=check(owner.read(old),wanted)
                values=inputs(row,5007+number);wanted=expected(values,row)
                owner.update(values);generation,_=owner.invoke()
                maximum=max(maximum,check(owner.read(generation),wanted))
                try:owner.read(old)
                except RuntimeError:pass
                else:raise AssertionError("stale native generation accepted")
                events,walls=[],[]
                for _ in range(5):
                    generation,elapsed=owner.invoke(repeats=20,timed=True)
                    events.append(elapsed)
                    maximum=max(maximum,check(owner.read(generation),wanted))
                    start=time.perf_counter()
                    for _ in range(5):
                        owner.update(values);generation,_=owner.invoke();actual=owner.read(generation)
                    walls.append((time.perf_counter()-start)*1e3/5)
                    maximum=max(maximum,check(actual,wanted))
            rows.append({k:v for k,v in row.items() if k!="package"}|{
                "correctness":"passed_before_and_after_each_domain","changed_inputs":"passed",
                "stale_generation":"rejected","max_abs_error":maximum,
                "native_launch_window_samples_ms":events,"native_launch_window_median_ms":median(events),
                "prepared_update_invoke_read_samples_ms":walls,
                "prepared_update_invoke_read_median_ms":median(walls),
                "image_sha256":[hashlib.sha256(image).hexdigest() for image in package.images]})
            print("passed",len(rows),"/ 69",flush=True)
    finally:subprocess.run=original_run
    output.write_text(json.dumps({"architecture":"gfx1201","device":name.value.decode(),
        "device_uuid_raw_hex":bytes(uuid).hex(),"runtime_sha256":digest(library),
        "compiler_sha256":payload["compiler_sha256"],"package_sha256":digest(package_path),
        "recorder_sha256":digest(__file__),"adapter_sha256":digest(
            __import__("tessera.compiler.native_scaled_program",fromlist=[""]).__file__),
        "route":"native Graph program -> Schedule -> Tile -> ROCm Target -> LLVM -> HSACO -> checked HIP owner",
        "timing_scope":"one-member primal or four-member JVP native launch window; separate prepared update/invoke/read wall time",
        "limitations":["static M3 N5 K64 baseline","transposeA remains open","partial K scale groups remain open",
                       "public independent primal/JVP projection remains open","no speedup claim"],"rows":rows},indent=2)+"\n")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.packages:measure(args.packages,args.output)
    else:emit(args.output)
if __name__=="__main__":main()
