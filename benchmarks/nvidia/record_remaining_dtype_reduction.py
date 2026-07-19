"""Record TEST-5 closure evidence for low-precision epilogues and reductions.

Each row retains two disjoint repeated-median runs in CUDA-event and end-to-end
domains, exact image/resource/cache fingerprints, and winner consensus.  The
recorder is evidence-only and never mutates a production selector.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_remaining_dtype_reduction.json"
RESOURCE = ROOT / "benchmarks/baselines/nvidia_sm120_test5_route_resources.json"
POLICY = 0.04
WSL_ACCEPTANCE_MARGIN = 0.0015


def _delta(a: float, b: float) -> float:
    return abs(a - b) / min(a, b) if min(a, b) else 0.0


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _apply_stability_policy(payload: dict) -> dict:
    """Apply the strict gate plus the user-approved WSL rounding margin."""
    for row in payload["rows"]:
        stability = row["stability_fraction"]
        strict = all(value <= POLICY for value in stability.values())
        accepted = all(value <= POLICY + WSL_ACCEPTANCE_MARGIN for value in stability.values())
        row["strict_stable"] = strict
        row["stable"] = accepted
        row["wsl_margin_accepted_domains"] = [
            domain for domain, value in stability.items()
            if POLICY < value <= POLICY + WSL_ACCEPTANCE_MARGIN
        ]
        row["selector_eligible"] = bool(row["winner_consensus"] and strict)
    payload["stability_policy"].update({
        "wsl_acceptance_margin_fraction": WSL_ACCEPTANCE_MARGIN,
        "margin_rows_selector_eligible": False,
    })
    return payload


def _epilogue_module(storage: str, shape: tuple[int, int, int]):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
    from tessera.compiler.primitive_coverage import NumericPolicy
    m, n, k = shape
    graph, ir = {"tf32": ("fp32", "f32"), "fp8_e4m3": ("fp8_e4m3", "f8E4M3FN"),
                 "fp8_e5m2": ("fp8_e5m2", "f8E5M2")}[storage]
    a=IRType(f"tensor<{m}x{k}x{ir}>",(str(m),str(k)),graph)
    b=IRType(f"tensor<{k}x{n}x{ir}>",(str(k),str(n)),graph)
    bias=IRType(f"tensor<{n}xf32>",(str(n),),"fp32")
    out=IRType(f"tensor<{m}x{n}xf32>",(str(m),str(n)),"fp32")
    return GraphIRModule(functions=[GraphIRFunction(name=f"remaining_epilogue_{storage}",
        args=[IRArg("a",a),IRArg("b",b),IRArg("bias",bias)],result_types=[out],
        body=[IROp(result="c",op_name="tessera.matmul",operands=["%a","%b"],
            operand_types=[str(a),str(b)],result_type=str(out),
            kwargs={"bias":"%bias","activation":"gelu"},
            numeric_policy=NumericPolicy(storage="fp32",accum="fp32",math_mode="tf32") if storage=="tf32" else None)],
        return_values=["%c"])])


def _reduction_module(storage: str, kind: str, shape=(16,257,32), axis=1):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
    ir="f16" if storage=="fp16" else "f32"; dims="x".join(map(str,shape))
    out_shape=shape[:axis]+shape[axis+1:]; out_dims="x".join(map(str,out_shape))
    x=IRType(f"tensor<{dims}x{ir}>",tuple(map(str,shape)),storage)
    out=IRType(f"tensor<{out_dims}xf32>",tuple(map(str,out_shape)),"fp32")
    op="tessera.reduce" if kind=="sum" else f"tessera.{kind}"
    return GraphIRModule(functions=[GraphIRFunction(name=f"remaining_reduce_{kind}_{ir}",
        args=[IRArg("x",x)],result_types=[out],body=[IROp(result="o",op_name=op,
        operands=["%x"],operand_types=[str(x)],result_type=str(out),
        kwargs={"axis":axis,"keepdims":False})],return_values=["%o"])])


def _runs(candidates, *, samples, device_reps, e2e_reps, warmup):
    from benchmarks.nvidia._clock_conditioning import condition_sm120

    names=tuple(candidates); device=[{name:[] for name in names} for _ in range(2)]
    e2e=[{name:[] for name in names} for _ in range(2)]
    conditioning=[]
    # Keep the two repeated-median cohorts disjoint while interleaving them in
    # time.  That balances host-managed WSL clock drift without merging the
    # samples or weakening the two-run acceptance rule.
    for sample in range(samples):
        conditioning.append(condition_sm120())
        for run in ((0,1) if sample%2==0 else (1,0)):
            # Candidate order also alternates independently of cohort identity;
            # neither run is permanently biased toward the first/last route.
            order=names if sample%2==0 else names[::-1]
            for name in order:
                # First launch is deliberately excluded.  The following event
                # interval amortizes exactly ``device_reps`` resident launches.
                candidates[name]["device"](1,0)
                device[run][name].append(candidates[name]["device"](device_reps,0))
                candidates[name]["e2e"]()
                start=time.perf_counter()
                for _ in range(e2e_reps): candidates[name]["e2e"]()
                e2e[run][name].append((time.perf_counter()-start)*1e3/e2e_reps)
    runs={name:[{"run":run+1,"device_event_ms":statistics.median(device[run][name]),
        "end_to_end_ms":statistics.median(e2e[run][name]),"device_samples_ms":device[run][name],
        "end_to_end_samples_ms":e2e[run][name]} for run in range(2)] for name in names}
    winners={domain:[min(names,key=lambda name:runs[name][run][domain]) for run in range(2)]
             for domain in ("device_event_ms","end_to_end_ms")}
    return runs,winners,conditioning


def record(samples: int, device_reps: int, e2e_reps: int, warmup: int) -> dict:
    import ml_dtypes
    from tessera import runtime as rt
    from tessera.compiler import nvidia_native
    from tessera.compiler.canonical_compile import compile_result_from_bundle
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler.emit import candidate as candidate_mod
    from tessera.compiler.emit.candidate import OP_FUSED_REGION
    from tessera.compiler.emit.nvidia_cuda import measure_row_reduce_device, run_row_reduce
    from tessera.compiler.fusion import FusedRegion
    device=subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap",
        "--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.strip()
    bridge=rt._load_nvidia_ptx_launch()
    resources=json.loads(RESOURCE.read_text())
    production_reduce=resources["details"].get("generated_row_reduce",[])
    rows=[]
    for storage in ("tf32","fp8_e4m3","fp8_e5m2"):
      for shape in ((1024,1024,1024),(511,769,513)):
        module=_epilogue_module(storage,shape); nvidia_native._cache.clear()
        start=time.perf_counter(); bundle=compile_graph_module(module,source_origin="NVIDIA-E2E-2",
            target="nvidia_sm120",options={"package_native":True},enable_tool_validation=False); cold=(time.perf_counter()-start)*1e3
        start=time.perf_counter(); warm_bundle=compile_graph_module(module,source_origin="NVIDIA-E2E-2",
            target="nvidia_sm120",options={"package_native":True},enable_tool_validation=False); warm=(time.perf_counter()-start)*1e3
        assert bundle.native_image and bundle.launch_descriptor and warm_bundle.native_image
        m,n,k=shape; rng=np.random.default_rng(122000+sum(shape)); raw_a=(rng.standard_normal((m,k))*.1).astype(np.float32); raw_b=(rng.standard_normal((k,n))*.1).astype(np.float32)
        dtype={"tf32":np.float32,"fp8_e4m3":ml_dtypes.float8_e4m3fn,"fp8_e5m2":ml_dtypes.float8_e5m2}[storage]
        a=np.ascontiguousarray(raw_a.astype(dtype)); b=np.asfortranarray(raw_b.astype(dtype)); bias=np.ascontiguousarray((rng.standard_normal(n)*.03).astype(np.float32)); out=np.zeros((m,n),np.float32)
        artifact=compile_result_from_bundle(bundle,module=module).to_runtime_artifact(); bindings={"a":a,"b":b,"bias":bias,"c":out,"M":m,"N":n,"K":k}
        assert rt.launch(artifact,bindings)["ok"]
        ordered=sorted(bundle.launch_descriptor.buffers,key=lambda item:item.ordinal); cbuf=(ctypes.c_void_p*len(ordered))(*(int(bindings[item.name].ctypes.data) for item in ordered)); dims=(ctypes.c_int64*3)(m,n,k)
        def native_device(reps,warmup):
            ms=ctypes.c_float(); rc=bridge.tessera_nvidia_ptx_benchmark(bundle.launch_descriptor.entry_symbol.encode(),cbuf,len(ordered),dims,3,warmup,reps,ctypes.byref(ms))
            if rc: raise RuntimeError(f"native epilogue benchmark rc={rc}")
            return float(ms.value)
        def native_e2e():
            result=rt.launch(artifact,bindings)
            if not result["ok"]: raise RuntimeError(result.get("reason"))
        candidate_storage="f32" if storage=="tf32" else storage
        composed=next(c for c in candidate_mod.candidates_for("nvidia",OP_FUSED_REGION) if c.name==f"nvidia_mma_fused_composed_{'tf32' if storage=='tf32' else storage}")
        region=FusedRegion(epilogue=("bias","gelu"),storage_dtype=candidate_storage)
        def prod_device(reps,warmup):
            value=composed.measure_device_latency(region,raw_a,raw_b,bias,reps=reps,warmup=warmup)
            if value is None: raise RuntimeError("production composed epilogue declined")
            return value
        def prod_e2e():
            _,tag=composed.run(region,raw_a,raw_b,bias)
            if tag!="nvidia_cuda_composed": raise RuntimeError(f"production epilogue declined to {tag}")
        start=time.perf_counter(); prod_e2e(); prod_first_use=(time.perf_counter()-start)*1e3
        start=time.perf_counter(); prod_e2e(); prod_steady_use=(time.perf_counter()-start)*1e3
        candidates={"canonical_tile_fused":{"device":native_device,"e2e":native_e2e},"production_composed":{"device":prod_device,"e2e":prod_e2e}}
        measured,winners,conditioning=_runs(candidates,samples=samples,device_reps=device_reps,e2e_reps=e2e_reps,warmup=warmup)
        native_resource=bundle.native_image.resource_record.to_dict() if bundle.native_image.resource_record else None
        for name in candidates:
            stability={domain:_delta(measured[name][0][domain],measured[name][1][domain]) for domain in ("device_event_ms","end_to_end_ms")}
            evidence=native_resource if name=="canonical_tile_fused" else {"route_fingerprints":resources["routes"].get(composed.name,[]),"details":resources["details"].get(composed.name,[])}
            consensus=all(winners[d]==[name,name] for d in winners)
            compile_evidence=({"cold_ms":cold,"warm_ms":warm,"cold_state":bundle.native_image.compile_state,"warm_state":warm_bundle.native_image.compile_state,"image_digest_reproducible":bundle.native_image.image_digest==warm_bundle.native_image.image_digest} if name=="canonical_tile_fused" else {"state":"resident_nvcc_cache","first_use_ms":prod_first_use,"second_use_ms":prod_steady_use,"first_use_includes_compile_and_cache_fill":True,"prewarmed_before_measurement":True})
            rows.append({"op":"fused_epilogue","storage":storage,"shape":list(shape),"candidate":name,"runs":measured[name],"stability_fraction":stability,"stable":all(v<=POLICY for v in stability.values()),"winner_consensus":consensus,"selector_eligible":consensus and all(v<=POLICY for v in stability.values()),"resources":evidence,"resource_fingerprint":_fingerprint(evidence),"image_digest":bundle.native_image.image_digest if name=="canonical_tile_fused" else None,"compile":compile_evidence,"clock_conditioning_ms":conditioning,"selector_changed":False})
    for storage in ("fp16","fp32"):
      for kind in ("sum","mean","max"):
        module=_reduction_module(storage,kind,shape=(64,1025,64),axis=1); shape=(64,1025,64); axis=1; rng=np.random.default_rng(122500+len(kind)); dtype=np.float16 if storage=="fp16" else np.float32; x=np.ascontiguousarray((rng.standard_normal(shape)*.4).astype(dtype)); moved=np.ascontiguousarray(np.moveaxis(x,axis,-1)).reshape(-1,shape[axis]); output_shape=shape[:axis]+shape[axis+1:]
        packages={}; compile_state={}
        for schedule in ("serial","cooperative_128"):
            nvidia_native._cache.clear()
            start=time.perf_counter(); bundle=compile_graph_module(module,source_origin="NVIDIA-E2E-2",target="nvidia_sm120",options={"package_native":True,"nvidia_reduction_schedule":schedule},enable_tool_validation=False); elapsed=(time.perf_counter()-start)*1e3
            start=time.perf_counter(); warm_bundle=compile_graph_module(module,source_origin="NVIDIA-E2E-2",target="nvidia_sm120",options={"package_native":True,"nvidia_reduction_schedule":schedule},enable_tool_validation=False); warm_elapsed=(time.perf_counter()-start)*1e3
            assert bundle.native_image and bundle.launch_descriptor; artifact=compile_result_from_bundle(bundle,module=module).to_runtime_artifact(); out=np.zeros(output_shape,np.float32); bindings={"x":x,"o":out,"Outer":shape[0],"AxisExtent":shape[axis],"Inner":shape[2]}; assert rt.launch(artifact,bindings)["ok"]
            raw=(ctypes.c_void_p*2)(int(x.ctypes.data),int(out.ctypes.data)); dims=(ctypes.c_int64*3)(shape[0],shape[axis],shape[2])
            def dev(reps,warmup,b=bundle,raw=raw,dims=dims):
                ms=ctypes.c_float(); rc=bridge.tessera_nvidia_ptx_benchmark(b.launch_descriptor.entry_symbol.encode(),raw,2,dims,3,warmup,reps,ctypes.byref(ms));
                if rc: raise RuntimeError(f"reduction benchmark rc={rc}")
                return float(ms.value)
            def e2e(a=artifact,bindings=bindings):
                result=rt.launch(a,bindings)
                if not result["ok"]: raise RuntimeError(result.get("reason"))
            packages[schedule]={"bundle":bundle,"device":dev,"e2e":e2e}; compile_state[schedule]={"cold_ms":elapsed,"warm_ms":warm_elapsed,"cold_state":bundle.native_image.compile_state,"warm_state":warm_bundle.native_image.compile_state,"image_digest_reproducible":bundle.native_image.image_digest==warm_bundle.native_image.image_digest}
        start=time.perf_counter(); run_row_reduce(moved,kind); prod_first_use=(time.perf_counter()-start)*1e3
        start=time.perf_counter(); run_row_reduce(moved,kind); prod_steady_use=(time.perf_counter()-start)*1e3
        candidates={"canonical_serial":packages["serial"],"canonical_cooperative_128":packages["cooperative_128"],"production_cuda_composed":{"device":lambda reps,warmup:measure_row_reduce_device(moved,kind,reps=reps),"e2e":lambda:run_row_reduce(moved,kind)}}
        measured,winners,conditioning=_runs(candidates,samples=samples,device_reps=device_reps,e2e_reps=e2e_reps,warmup=warmup)
        for name,spec in candidates.items():
            stability={domain:_delta(measured[name][0][domain],measured[name][1][domain]) for domain in ("device_event_ms","end_to_end_ms")}; consensus=all(winners[d]==[name,name] for d in winners)
            if name.startswith("canonical"):
                b=spec["bundle"]; evidence=b.native_image.resource_record.to_dict() if b.native_image.resource_record else None; image=b.native_image.image_digest; cache=compile_state[name.removeprefix("canonical_")]
            else:
                evidence={"route_fingerprints":resources["routes"].get("generated_row_reduce",[]),"details":production_reduce}; image=None; cache={"state":"resident_nvcc_cache","first_use_ms":prod_first_use,"second_use_ms":prod_steady_use,"first_use_includes_compile_and_cache_fill":True,"prewarmed_before_measurement":True}
            rows.append({"op":"reduction","storage":storage,"kind":kind,"shape":list(shape),"axis":axis,"candidate":name,"runs":measured[name],"stability_fraction":stability,"stable":all(v<=POLICY for v in stability.values()),"winner_consensus":consensus,"selector_eligible":consensus and all(v<=POLICY for v in stability.values()),"resources":evidence,"resource_fingerprint":_fingerprint(evidence),"image_digest":image,"compile":cache,"clock_conditioning_ms":conditioning,"selector_changed":False})
    return _apply_stability_policy({"schema":"tessera.nvidia.remaining-dtype-reduction.v1","work_item":"NVIDIA-E2E-2","device":device,"stability_policy":{"relative_fraction":POLICY,"runs":2,"required_domains":["device_event_ms","end_to_end_ms"]},"method":{"samples_per_run":samples,"device_repetitions_per_sample":device_reps,"end_to_end_repetitions_per_sample":e2e_reps,"discarded_launches_per_candidate_sample":1,"warmup":warmup,"sampling":"two_disjoint_time_interleaved_cohorts","candidate_order":"rotated_interleaved","clock_conditioning":"sm120 resident 1024^3 TF32 GEMM before every paired cohort sample","compile_accounting":"cold/warm canonical compilation and production first/second use are recorded outside steady-state timing"},"rows":rows,"selector_promotions":[]})


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--samples",type=int,default=9); p.add_argument("--device-reps",type=int,default=50); p.add_argument("--e2e-reps",type=int,default=10); p.add_argument("--warmup",type=int,default=5); p.add_argument("--output",type=Path,default=OUT); p.add_argument("--finalize-existing",action="store_true"); a=p.parse_args(); payload=_apply_stability_policy(json.loads(a.output.read_text())) if a.finalize_existing else record(a.samples,a.device_reps,a.e2e_reps,a.warmup); a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(f"wrote {a.output} ({len(payload['rows'])} rows)"); return 0


if __name__ == "__main__": raise SystemExit(main())
