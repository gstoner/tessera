"""Same-image counterbalanced serial/batched native output transport."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from unittest.mock import patch
import numpy as np
import tessera as ts
from tessera import runtime
from tessera.autodiff import vmap
from tessera.compiler.native_scaled_program import PreparedScaledProgram,package_native_scaled_primal
from tests.unit.test_public_floating_scaled_reverse import floating_owner
from tests.device.rocm.test_floating_scaled_adjoint import inputs,oracle
from tests.unit.test_native_typed_scaled_vmap import case
from tests.device.rocm.test_public_mapped_scale_vjp import scale_oracle


def reverse_case(kind):
    if kind=="f32_four_gradients":
        values=inputs(False,False)
        owner=floating_owner();raw=values[:4];seed=values[4]
        expected=oracle(values,False,False)
    else:
        scalar,leading,raw,primal=case("independent_rhs","fp32",False,(2,7,19,256))
        roles=("sa",) if kind=="fp8_single_gradient_control" else ("sa","sb")
        owner=vmap(ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=roles)(scalar._fn),
                   in_axes=leading._frontend_batch_axes)
        seed=np.random.default_rng(941).uniform(-.5,.5,primal.shape).astype(np.float32)
        expected=scale_oracle(raw,leading._frontend_batch_axes,seed)[:len(roles)]
    actual=owner.native_backward(*raw,out_cotangents=seed)
    for a,b in zip(actual,expected,strict=True):np.testing.assert_allclose(a,b,rtol=4e-5,atol=3e-5)
    return owner,owner._native_backward_artifact,(*raw,seed),expected,raw,seed


def primal_case(fmt):
    _,owner,values,expected=case("independent_rhs",fmt,False,(2,7,19,256))
    graph=owner._specialized_autodiff_module(values,{})
    graph=replace(graph,module_attrs={**graph.module_attrs,
        "tessera.target":json.dumps("rocm"),"tessera.arch":json.dumps("gfx1201")})
    package=package_native_scaled_primal(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    return None,package,values,(expected,),None,None


def measure(label):
    if label.startswith("mxfp8"):
        owner,package,values,expected,raw,seed=primal_case("e8m0")
    elif label=="fp8_primal_control":
        owner,package,values,expected,raw,seed=primal_case("fp32")
    else:owner,package,values,expected,raw,seed=reverse_case(label)
    def compare(outputs):
        for a,b in zip(outputs,expected,strict=True):np.testing.assert_allclose(a,b,rtol=4e-5,atol=3e-5)
    samples={"serial":[],"batched":[]};public={"serial":[],"batched":[]}
    native=[]
    original=PreparedScaledProgram.read
    def forbidden(*a,**kw):raise AssertionError("warm transport comparison invoked compiler")
    with PreparedScaledProgram(package,values,runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
        if prepared._read_many is None:raise RuntimeError("new native batched-read provider required")
        generation,_=prepared.invoke();compare(prepared.read(generation))
        for round_ in range(9):
            for batched in ((False,True) if round_%2==0 else (True,False)):
                arm="batched" if batched else "serial"
                for _ in range(5):compare(prepared.read(generation,batched=batched))
                start=time.perf_counter()
                for _ in range(50):output=prepared.read(generation,batched=batched)
                samples[arm].append((time.perf_counter()-start)*1e3/50);compare(output)
                if owner is not None:
                    def read_control(self,generation,*,batched=batched):
                        return original(self,generation,batched=batched)
                    with patch.object(PreparedScaledProgram,"read",read_control),patch("subprocess.run",side_effect=forbidden):
                        start=time.perf_counter()
                        for _ in range(20):output=owner.native_backward(*raw,out_cotangents=seed)
                        public[arm].append((time.perf_counter()-start)*1e3/20);compare(output)
            generation,elapsed=prepared.invoke(repeats=128,timed=True);native.append(elapsed)
            compare(prepared.read(generation))
    return {"label":label,"input_shapes":[list(value.shape) for value in values],
        "output_count":len(expected),"bulk_api_applies":len(expected)>1,
        "correctness":"independent_before_and_after_each_rotating_transport/public/window",
        "read_samples_ms":samples,"read_median_ms":{key:median(value) for key,value in samples.items()},
        "read_batched_over_serial":median(samples["batched"])/median(samples["serial"]),
        "public_samples_ms":public,"public_median_ms":{key:median(value) for key,value in public.items() if value},
        "public_batched_over_serial":median(public["batched"])/median(public["serial"]) if owner is not None else None,
        "native_program_samples_ms":native,"native_program_median_ms":median(native),
        "image_sha256":[hashlib.sha256(image).hexdigest() for image in package.images],
        "program_sha256":hashlib.sha256(package.program_json.encode()).hexdigest()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if runtime._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    paths=[Path(os.environ[name]).resolve() for name in ("TESSERA_OPT","TESSERA_ROCM_OPT","TESSERA_ROCM_NATIVE_MOVEMENT_LIB")]
    paths += [Path(path).resolve() for path in (
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.h",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_image_cache.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_nvfp4_runtime.cpp",
        "python/tessera/compiler/native_scaled_program.py","python/tessera/compiler/jit.py",
        "tests/unit/test_native_typed_scaled_vmap.py","tests/unit/test_public_floating_scaled_reverse.py",
        "tests/device/rocm/test_floating_scaled_adjoint.py","tests/device/rocm/test_public_mapped_scale_vjp.py")]
    paths.append(Path(__file__).resolve())
    packet={"architecture":"gfx1201",
        "device_inventory":subprocess.run(["rocminfo"],check=True,capture_output=True,text=True,timeout=30).stdout,
        "identity_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "pinned_mode":os.environ.get("TESSERA_ROCM_PROGRAM_PINNED","automatic"),
        "cases":[measure(kind) for kind in ("f32_four_gradients","fp8_two_gradients",
                    "fp8_single_gradient_control","fp8_primal_control","mxfp8_primal_control")],
        "scope":"nine rotating same-provider/same-image rounds; read-only host transport and completed public native_backward are separate from native program events; single-output arms use the same legacy read symbol"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__=="__main__":main()
