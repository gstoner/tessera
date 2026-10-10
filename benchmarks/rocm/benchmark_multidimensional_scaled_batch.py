"""Exact gfx1201 static two-axis batches: native numerics and separate costs."""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from statistics import median
import numpy as np
import tessera as ts
from tessera import runtime
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram,package_native_scaled_jvp
from tests.unit import test_rocm_independent_scaled_batch as shared

def samples(call,repeats=10):
    result=[]
    for _ in range(21):
        start=time.perf_counter()
        for _ in range(repeats):call()
        result.append((time.perf_counter()-start)*1000/repeats)
    return result

def measure(shape,fmt,nk,policy):
    from tests.unit.test_rocm_multidimensional_scaled_batch import nested_case
    fn,values,expected=nested_case(policy,fmt,nk,shape)
    np.testing.assert_allclose(fn(*values),expected,rtol=4e-5,atol=1e-4)
    manifest=fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"]
    package=NativeScaledProgram.from_manifest(manifest)
    programs=[("primal",fn,package,values,expected)]
    lib=runtime._load_rocm_native_movement_runtime()
    rows=[]
    for kind,public,owned,inputs,oracle in programs:
        public_call=(lambda:public(*values)) if kind=="primal" else public
        with PreparedScaledProgram(owned,inputs,runtime_library=lib._name) as owner:
            generation,_=owner.invoke()
            actual=owner.read(generation)
            oracles=(oracle,) if kind=="primal" else oracle
            for got,want in zip(actual,oracles,strict=True):
                np.testing.assert_allclose(got,want,rtol=4e-5,atol=1e-4)
            old=subprocess.run
            try:
                def forbidden(*args,**kwargs):raise AssertionError("warm compiler subprocess")
                subprocess.run=forbidden
                public_call()
                public_ms=samples(public_call)
                def host():
                    owner.update(inputs)
                    generation,_=owner.invoke()
                    owner.read(generation)
                host_ms=samples(host)
                events=[]
                for _ in range(21):
                    _,elapsed=owner.invoke(repeats=30,timed=True)
                    events.append(elapsed)
            finally:subprocess.run=old
        program=json.loads(owned.program_json)
        rows.append({"kind":kind,"batching":policy,"shape_batch_mnk":list(shape),"scale_format":fmt,"rhs_layout":"NK" if nk else "KN",
            "logical_output_shape":list(actual[0].shape),
            "tile_staging":re.findall(r'staging = "(global|lds)"',fn.compile_result.tile_ir),
            "tile_ir_sha256":hashlib.sha256(fn.compile_result.tile_ir.encode()).hexdigest(),"native_steps":len(program["steps"]),
            "member_scalars":[json.loads(m)["scalars"] for m in owned.members_json],
            "member_geometry":[json.loads(m)["geometry"] for m in owned.members_json],
            "public_samples_ms":public_ms,"public_median_ms":median(public_ms),
            "prepared_host_samples_ms":host_ms,"prepared_host_median_ms":median(host_ms),
            "native_sequence_event_samples_ms":events,"native_sequence_event_median_ms":median(events),
            "owner_image_sha256":[hashlib.sha256(i).hexdigest() for i in owned.images],
            "correctness":"independent_float64_block_oracle_before_timing","compiler_subprocess_forbidden":True})
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if runtime._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    rows=[]
    for shape in [(2,3,7,19,256),(2,2,200,129,1536)]:
        for fmt in ["fp32","e8m0"]:
            for nk in [False,True]:
                for policy in ["shared_rhs_rows","independent_rhs","shared_lhs"]:rows.extend(measure(shape,fmt,nk,policy))
    lib=runtime._load_rocm_native_movement_runtime()
    packet={"architecture":"gfx1201","rocminfo":subprocess.run(["/opt/rocm/bin/rocminfo"],text=True,capture_output=True,check=True).stdout,
        "runtime_sha256":hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),"rows":rows,
        "timing":"separate public-call wall, prepared update/invoke/read wall, and native HIP sequence event windows; events include native enqueue gaps and are not isolated kernel cost"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps([{k:r[k] for k in ["kind","batching","shape_batch_mnk","scale_format","rhs_layout","public_median_ms","prepared_host_median_ms","native_sequence_event_median_ms"]} for r in rows],indent=2))
if __name__=="__main__":main()
