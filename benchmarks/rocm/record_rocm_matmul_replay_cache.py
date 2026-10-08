"""Exact-device ROCm matmul replay cache A/B with warm native images."""
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
from tests.unit.test_rocm_shape_free_cache_key import _module, _package, _launch
from tests.unit.test_scheduled_matmul_consumers import _module as matmul_graph

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(pairs, architecture="gfx1151"):
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
    for family, shape in profiles:
        if family == "matmul_fp16":
            graph = matmul_graph(target="rocm", shape=shape, dtype="fp16")
            artifact = scheduled_matmul.lower_scheduled_matmul(graph, target="rocm_"+architecture)
            package_fn = lambda: native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
        else:
            graph = _module("cache_probe", "tessera.softmax" if family=="softmax" else "tessera.mean",
                shape, "fp32", shape if family=="softmax" else (shape[0],shape[2]),
                "fp32", {"axis":-1} if family=="softmax" else {"axis":1,"keepdims":False})
            artifact, _ = _package(graph, architecture)
            package_fn = lambda: native.package_scheduled_kernel(artifact, pipeline_name="tessera-lower-to-rocm")
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
            x = rng.standard_normal(shape).astype(np.float32)
            got = _launch(artifact, prime, x)
            if family == "softmax":
                e = np.exp(x.astype(np.float64)-x.max(-1,keepdims=True))
                expected = e/e.sum(-1,keepdims=True)
            else:
                expected = x.astype(np.float64).mean(1)
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
        rows.append({"family":family,"role":"matmul_replay_candidate","shape":list(shape),"max_abs_error":float(np.max(abs(got-expected))),
            "correctness":"native_gpu_against_float64_before_timing","samples_ms":samples,
            "subprocess_counts":counts,"medians_ms":medians,
            "uncached_over_cached":medians["uncached_replay"]/medians["cached_replay"],
            "image_digest":base[0],"entry":base[-1],"identity_equal_all_arms":True})
        print(f"passed {family} {shape}",flush=True)
    return {"architecture":arch,"host":platform.node(),"rocminfo":info,
        "measurement":"warm-image package wall time; only native ancestry replay cache is varied",
        "source_sha256":digest(native.__file__),"replay_cache_sha256":digest(cache.__file__),"ancestry_sha256":digest(scheduled_matmul.__file__),"recorder_sha256":digest(__file__),
        "compiler_sha256":digest(native._tessera_opt()),"rows":rows,
        "limitations":["No kernel speed claim","Owning-host compiler snapshot identified by content hash",
                       "Four static fp16 profiles; no generic cache closure claim"]}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--pairs",type=int,default=7);p.add_argument("--architecture",choices=["gfx1151","gfx1201"],required=True)
    args=p.parse_args()
    Path(args.output).write_text(json.dumps(run(args.pairs,args.architecture),indent=2)+"\n")
