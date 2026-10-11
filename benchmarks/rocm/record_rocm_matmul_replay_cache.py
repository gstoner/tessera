"""Exact-device ROCm matmul/math replay cache A/B with warm native images."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_native as native, scheduled_matmul, rocm_pass_cache as cache
from tessera.compiler import rocm_math_native
from tessera.compiler.graph_ir import IROp, tensor_ir_type
from tests.unit.test_rocm_math_native_package import module as math_graph
from tests.unit.test_scheduled_matmul_consumers import _module as matmul_graph

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(pairs, architecture="gfx1151", family_selection="matmul"):
    arch = rt._rocm_live_arch()
    if architecture not in {"gfx1151", "gfx1201"} or arch != architecture:
        raise RuntimeError(f"Expected live {architecture}, got {arch}")
    info = subprocess.run(["rocminfo"], text=True, capture_output=True, check=True).stdout
    rng = np.random.default_rng(1151)
    rows = []
    profiles = [
        ("matmul_fp16", (16,16,16)), ("matmul_fp16", (17,31,19)),
        ("matmul_fp16", (64,256,64)), ("matmul_fp16", (200,128,64)),
    ]
    if family_selection == "math":
        profiles = [(kind+"_"+storage, shape)
                    for kind in ("sqrt","exp","add","div","cumsum","cummax")
                    for storage in ("f32","f16","bf16")
                    for shape in ((3,17),(64,257))]
    elif family_selection != "matmul":
        raise ValueError("Unknown replay consumer family")
    for family, shape in profiles:
        if family == "matmul_fp16":
            graph = matmul_graph(target="rocm", shape=shape, dtype="fp16")
            artifact = scheduled_matmul.lower_scheduled_matmul(graph, target="rocm_"+architecture)
            package_fn = lambda: native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
        else:
            kind, storage = family.rsplit("_",1)
            graph = math_graph(kind,shape)
            if storage != "f32":
                fn = graph.functions[0]
                result_type = fn.result_types[0]
                operation = fn.body[-1]
                casts = []
                for index, arg in enumerate(fn.args):
                    arg.ir_type = tensor_ir_type(shape, "fp16" if storage=="f16" else "bf16")
                    name = "wide_"+arg.name
                    casts.append(IROp(result=name,op_name="tessera.cast",
                        operands=["%"+arg.name],operand_types=[str(arg.ir_type)],
                        result_type=str(result_type),kwargs={"dtype":"fp32"}))
                    operation.operands[index] = "%"+name
                fn.body = casts+[operation]
            artifact = rocm_math_native.lower_math_graph(graph,"rocm_"+architecture)
            package_fn = lambda: rocm_math_native.package_math_recipe(
                artifact,pipeline_name="tessera-lower-to-rocm")
        prime = package_fn()
        if family == "matmul_fp16":
            m,k,n = shape
            a = (rng.standard_normal((m,k))*.25).astype(np.float16)
            b = (rng.standard_normal((k,n))*.25).astype(np.float16)
            got = np.zeros((m,n),np.float32)
            obj = rt.RuntimeArtifact(metadata={"target":prime.image.target}, native_image=prime.image,
                launch_descriptor=prime.descriptor,tile_ir=prime.tile_ir,target_ir=prime.target_ir)
            receipt = rt.launch(obj, {"buffers":{"a":a,"b":b,"o":got},"scalars":{"M":m,"N":n,"K":k}})
            if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
                raise RuntimeError(receipt)
            expected = a.astype(np.float64) @ b.astype(np.float64)
            atol = 1e-4
        else:
            storage_type = np.float32 if storage=="f32" else np.float16
            if storage=="bf16":
                import ml_dtypes
                storage_type = ml_dtypes.bfloat16
            x = rng.uniform(.125,1.25,shape).astype(storage_type)
            y = rng.uniform(.5,1.5,shape).astype(storage_type)
            got = np.empty(shape,np.float32)
            obj = rt.RuntimeArtifact(
                graph_ir=artifact.graph_ir,schedule_ir=artifact.schedule_ir,
                tile_ir=prime.tile_ir,target_ir=prime.target_ir,
                metadata={"target":prime.image.target,"compiler_path":"rocm_math_native_descriptor"},
                native_image=prime.image,launch_descriptor=prime.descriptor)
            obj = rt.RuntimeArtifact.from_json(obj.to_json())
            info = prime.descriptor.provenance["native_math"]
            buffers = {"a":x,"out":got}
            if kind in {"add","div"}:buffers["b"]=y
            scalars = ({"Rows":info["rows"],"Columns":info["columns"]}
                       if info["family"]=="scan" else {"N":info["elements"]})
            receipt = rt.launch(obj,{"buffers":buffers,"scalars":scalars})
            if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
                raise RuntimeError(receipt)
            xf,yf = x.astype(np.float64),y.astype(np.float64)
            expected = {"sqrt":lambda:np.sqrt(xf),"exp":lambda:np.exp(xf),
                        "add":lambda:xf+yf,"div":lambda:xf/yf,
                        "cumsum":lambda:np.cumsum(xf,axis=-1),
                        "cummax":lambda:np.maximum.accumulate(xf,axis=-1)}[kind]()
            atol = 1e-6
        np.testing.assert_allclose(got,expected,rtol=1e-5,atol=atol)
        identity = lambda p: (p.image.image_digest,hashlib.sha256(p.image.payload).hexdigest(),
                             p.image.compiler_fingerprint,p.image.toolchain_fingerprint,p.descriptor.entry_symbol)
        base = identity(prime)
        samples = {"uncached_replay":[],"cached_replay":[]}
        counts = {a:[] for a in samples}
        for index in range(pairs):
            for arm in (tuple(samples) if index%2==0 else tuple(reversed(samples))):
                if arm=="uncached_replay": cache.clear()
                original = subprocess.run
                calls={"version":0,"other":0}
                def observed(argv,*args,**kwargs):
                    calls["version" if "--version" in argv else "other"]+=1
                    return original(argv,*args,**kwargs)
                subprocess.run=observed
                try:
                    start=time.perf_counter_ns()
                    package=package_fn()
                    elapsed=(time.perf_counter_ns()-start)/1e6
                finally:
                    subprocess.run=original
                if identity(package)!=base: raise AssertionError("cache arm changed native package identity")
                if package.image.compile_state!="warm_cache": raise AssertionError("image was not warm")
                samples[arm].append(elapsed); counts[arm].append(calls)
        if any(c["version"] for c in counts["cached_replay"]):
            raise AssertionError("reused arm still queries tool versions")
        if any(c["version"] for c in counts["uncached_replay"]):
            raise AssertionError("warm image arm unexpectedly queried versions")
        if not all(a["other"] > b["other"] for a,b in zip(counts["uncached_replay"], counts["cached_replay"])):
            raise AssertionError("replay caching did not remove native pass subprocesses")
        medians={a:statistics.median(v) for a,v in samples.items()}
        rows.append({"family":family,"role":family_selection+"_replay_candidate","shape":list(shape),"max_abs_error":float(np.max(abs(got-expected))),
            "correctness":"native_gpu_against_float64_before_timing","samples_ms":samples,
            "subprocess_counts":counts,"medians_ms":medians,
            "uncached_over_cached":medians["uncached_replay"]/medians["cached_replay"],
            "image_digest":base[0],"entry":base[-1],"identity_equal_all_arms":True})
        print(f"passed {family} {shape}",flush=True)
    return {"architecture":arch,"host":platform.node(),"rocminfo":info,
        "measurement":"warm-image package wall time; only native ancestry replay cache is varied",
        "source_sha256":digest(native.__file__),"replay_cache_sha256":digest(cache.__file__),"ancestry_sha256":digest((scheduled_matmul if family_selection=="matmul" else rocm_math_native).__file__),"recorder_sha256":digest(__file__),
        "compiler_sha256":digest(native._tessera_opt()),"rows":rows,
        "limitations":["No kernel speed claim","Owning-host compiler snapshot identified by content hash",
                       "Static "+family_selection+" profiles; no generic cache closure claim"]}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--pairs",type=int,default=7);p.add_argument("--architecture",choices=["gfx1151","gfx1201"],required=True)
    p.add_argument("--family",choices=["matmul","math"],default="matmul")
    args=p.parse_args()
    Path(args.output).write_text(json.dumps(run(args.pairs,args.architecture,args.family),indent=2)+"\n")
