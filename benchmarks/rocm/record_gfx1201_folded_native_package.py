#!/usr/bin/env python3
"""Matched native-MLIR versus frozen HIP folded package timing on gfx1201.

Device-clock windows include any host dispatch gaps. Public launch wall time
includes host transfers, allocation and module loading. Neither is a profiler
phase attribution or a comparison against Radiance.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"python")]
from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights, package_mxfp4_folded_prefill
from tessera.compiler.rocm_mxfp4_folded_frontend import compile_folded_scaled_matmul
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base
from benchmarks.rocm.ablate_gfx1201_folded_load_schedule import package_engine
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock, _schedule_from
from benchmarks.rocm.folded_graph_windows import FoldedGraphWindows

SOURCES=(
 "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
 "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
 "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMFoldedW4A8Contract.h",
 "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMKernelIdentity.cpp",
 "python/tessera/compiler/rocm_native.py",
 "python/tessera/compiler/rocm_mxfp4_folded_frontend.py",
 "python/tessera/compiler/rocm_mxfp4_folded_carrier.py",
 "python/tessera/compiler/rocm_mxfp4_folded.py",
 "python/tessera/runtime.py",
 "benchmarks/rocm/folded_launch_arguments.py",
 "benchmarks/rocm/record_gfx1201_folded_native_package.py",
 "benchmarks/rocm/folded_graph_windows.py",
 "benchmarks/rocm/ablate_gfx1201_folded_load_schedule.py",
 "benchmarks/rocm/record_gfx1201_mxfp4_folded_load_schedule.py",
)

def public_launch(package,case,inputs,folded):
    output=np.zeros_like(inputs["output"])
    artifact=rt.RuntimeArtifact(metadata={"target":package.image.target},
        native_image=package.image,launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,target_ir=package.target_ir)
    start=time.perf_counter_ns()
    result=rt.launch(artifact,{"buffers":{
        "a":inputs["a"],"b_folded":folded.weight_bytes,"a_scale":inputs["a_scale"],
        "row_reference":folded.row_reference,"output":output},
        "scalars":{"M":case.m,"N":case.n,"K":case.k}})
    elapsed=(time.perf_counter_ns()-start)/1e6
    if not result.get("ok") or result.get("execution_kind")!="native_gpu":
        raise RuntimeError(f"public folded launch failed: {result}")
    return output,elapsed

def compile_for_tool(compiler,a,a_scale,folded,**options):
    # The frontend and native image projector must use the same executable.
    # This recorder is sequential; restore environment even on compile failure.
    previous=os.environ.get("TESSERA_OPT")
    os.environ["TESSERA_OPT"]=str(compiler.resolve())
    try:
        return compile_folded_scaled_matmul(a,a_scale,folded,
            tessera_opt=compiler,allow_approximate=True,**options)
    finally:
        if previous is None:os.environ.pop("TESSERA_OPT",None)
        else:os.environ["TESSERA_OPT"]=previous

def run_case(hip,clock,case,compiler,trials,static_control=False,runtime_k=True,
             reference_compiler=None,hip_graph_windows=False):
    inputs=base._logical_inputs(case)
    folded=prepare_folded_weights(inputs["packed_row_major"],inputs["b_scale"],allow_approximate=True)
    if not folded.lossless:
        raise RuntimeError("matched packet requires lossless folding")
    start=time.perf_counter_ns()
    program=compile_for_tool(compiler,inputs["a"],inputs["a_scale"],folded,
        runtime_k=runtime_k)
    native_compile_ms=(time.perf_counter_ns()-start)/1e6
    schedule=_schedule_from(program.route_receipt)
    start=time.perf_counter_ns()
    control=package_mxfp4_folded_prefill(case.m,case.n,case.k,folded,
        entry="folded_hip_matched_control",allow_approximate=True,schedule=schedule)
    hip_compile_ms=(time.perf_counter_ns()-start)/1e6
    packages={"native_mlir":program.package,"hip_matched_control":control}
    static_compile_ms=None
    if static_control:
        start=time.perf_counter_ns()
        static=compile_for_tool(compiler,inputs["a"],inputs["a_scale"],folded,
            runtime_mn=False,runtime_k=False)
        static_compile_ms=(time.perf_counter_ns()-start)/1e6
        packages["native_static_control"]=static.package
    reference_compile_ms=None
    if reference_compiler is not None:
        start=time.perf_counter_ns()
        reference=compile_for_tool(reference_compiler,inputs["a"],inputs["a_scale"],folded,
            runtime_k=runtime_k)
        reference_compile_ms=(time.perf_counter_ns()-start)/1e6
        if _schedule_from(reference.route_receipt)!=schedule:
            raise RuntimeError("native reference uses a different physical schedule")
        packages["native_reference_control"]=reference.package
    engines=[]
    graph_clock=None
    try:
        for name,package in packages.items():
            engines.append(package_engine(hip,case,inputs,folded,package,name=name,
                metadata={"physical_abi":package.descriptor.provenance.get("kernel_argument_layout","raw_pointer"),
                          "route":package.descriptor.provenance,
                          "compiler_fingerprint":package.image.compiler_fingerprint,
                          "compile_state":package.image.compile_state},
                copies=3))
        outputs={engine.name:engine.output() for engine in engines}
        np.testing.assert_array_equal(outputs["native_mlir"].view(np.uint16),
                                      outputs["hip_matched_control"].view(np.uint16))
        for output in outputs.values():
            np.testing.assert_array_equal(output.view(np.uint16),outputs["native_mlir"].view(np.uint16))
        rows,cols,reference=base._sampled_exact_reference(case,inputs)
        for output in outputs.values():
            np.testing.assert_array_equal(output[np.ix_(rows,cols)],reference)
        public={name:[] for name in packages}
        for trial in range(3):
            names=list(packages)
            if trial%2: names.reverse()
            for name in names:
                output,elapsed=public_launch(packages[name],case,inputs,folded)
                np.testing.assert_array_equal(output.view(np.uint16),outputs[name].view(np.uint16))
                public[name].append(elapsed)
        for engine in engines:
            base._warmup(hip,engine,6)
        estimates=[clock.window(engine,3,bracketed=False)["event_window_ms"]/3 for engine in engines]
        launches=max(12,math.ceil(12/min(estimates)))
        samplers={"ordinary":clock}
        if hip_graph_windows:
            graph_clock=FoldedGraphWindows(clock)
            samplers["hip_graph"]=graph_clock
            # Poison every resident output, then verify only graph-produced data.
            for engine in engines:
                for bracketed in (True,False):
                    for bundle in engine.copies:
                        poison=np.empty_like(outputs[engine.name])
                        poison.view(np.uint8).fill(255)
                        clock._check(hip.hipMemcpy(bundle.device[engine.output_index],
                            poison.ctypes.data_as(ctypes.c_void_p),poison.nbytes,1),"graph output poison")
                    graph_clock.window(engine,12,bracketed=bracketed)
                    for bundle in engine.copies:
                        np.testing.assert_array_equal(bundle.download(engine.output_index).view(np.uint16),
                                                      outputs[engine.name].view(np.uint16))
        samples={scope:{engine.name:[] for engine in engines} for scope in samplers}
        for trial in range(trials):
            order=engines if trial%2==0 else list(reversed(engines))
            scopes=list(samplers)
            if trial%2:scopes.reverse()
            for scope in scopes:
                for engine in order:
                    for bracketed in ((True,False) if trial%2==0 else (False,True)):
                        count=3*math.ceil(launches/3)
                        rejected=[]
                        for attempt in range(5):
                            sample=samplers[scope].window(engine,count,bracketed=bracketed)
                            if not bracketed or sample["device_window_ms"]>=5:
                                break
                            rejected.append(sample)
                            count=3*math.ceil(max(count*2,count*8/max(sample["device_window_ms"],0.001))/3)
                        if bracketed and (sample["device_window_ms"]<5 or sample["device_event_disagreement"]>0.05):
                            raise RuntimeError(f"device-clock witness failed: {sample}")
                        sample["rejected_short_windows"]=rejected
                        samples[scope][engine.name].append(sample)
        # Last timing operation may be graph replay; check all copies without relaunch.
        for engine in engines:
            for bundle in engine.copies:
                np.testing.assert_array_equal(bundle.download(engine.output_index).view(np.uint16),
                                              outputs[engine.name].view(np.uint16))
        result=[]
        for engine in engines:
            windows=samples["ordinary"][engine.name]
            device=[s["device_window_ms"]/s["launches"] for s in windows if s["bracketed"]]
            plain=[s["event_window_ms"]/s["launches"] for s in windows if not s["bracketed"]]
            result.append({"engine":engine.name,"device_window_per_launch_samples_ms":device,
                "device_window_per_launch_median_ms":statistics.median(device),
                "plain_event_per_launch_samples_ms":plain,
                "public_launch_wall_samples_ms":public[engine.name],
                "public_launch_wall_median_ms":statistics.median(public[engine.name]),
                "windows":windows,"image_evidence":engine.metadata,
                "output_sha256":hashlib.sha256(outputs[engine.name].tobytes()).hexdigest()})
        graph_rows=[]
        if hip_graph_windows:
            for engine in engines:
                windows=samples["hip_graph"][engine.name]
                device=[w["device_window_ms"]/w["launches"] for w in windows if w["bracketed"]]
                graph_rows.append({"engine":engine.name,"windows":windows,
                    "device_window_per_launch_samples_ms":device,
                    "device_window_per_launch_median_ms":statistics.median(device),
                    "correctness":"bracketed_and_plain_graphs_all_three_poisoned_outputs_bitwise_matched_before_timing",
                    "graph_over_ordinary_ratio":statistics.median(device)/
                        next(r["device_window_per_launch_median_ms"] for r in result if r["engine"]==engine.name)})
        return {"shape_mnk":[case.m,case.n,case.k],"correctness":"bitwise_matched_and_sampled_independent_oracle",
            "frontend_receipt":program.route_receipt,"fold_lossless":folded.lossless,
            "native_frontend_package_wall_ms":native_compile_ms,"hip_control_package_wall_ms":hip_compile_ms,
            "native_static_frontend_package_wall_ms":static_compile_ms,
            "native_reference_frontend_package_wall_ms":reference_compile_ms,
            "native_over_reference_device_ratio":(
                result[0]["device_window_per_launch_median_ms"]/
                next(row["device_window_per_launch_median_ms"] for row in result
                     if row["engine"]=="native_reference_control")
                if reference_compiler is not None else None),
            "launches_per_window":launches,"rows":result,"graph_rows":graph_rows,
            "native_over_hip_graph_ratio":(graph_rows[0]["device_window_per_launch_median_ms"]/graph_rows[1]["device_window_per_launch_median_ms"] if graph_rows else None),
            "native_over_hip_device_ratio":result[0]["device_window_per_launch_median_ms"]/result[1]["device_window_per_launch_median_ms"],
            "native_over_static_device_ratio":(
                result[0]["device_window_per_launch_median_ms"]/result[2]["device_window_per_launch_median_ms"]
                if static_control else None)}
    finally:
        if graph_clock is not None:graph_clock.close()
        for engine in reversed(engines):engine.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tessera-opt",required=True,type=Path)
    parser.add_argument("--llvm-bin",required=True,type=Path)
    parser.add_argument("--case",action="append",type=base._parse_case)
    parser.add_argument("--native-static-control",action="store_true")
    parser.add_argument("--hip-graph-windows",action="store_true",help="Also capture/replay checked resident packages without repeated host dispatch")
    parser.add_argument("--native-reference-opt",type=Path,
        help="Matching native reference compiler for interleaved schedule comparison")
    parser.add_argument("--native-runtime-k",action=argparse.BooleanOptionalAction,default=True,
        help="Use the native checked full-K64 runtime loop and shape-independent image")
    parser.add_argument("--trials",type=int,default=7)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.trials<3: raise ValueError("at least three alternating trials required")
    if args.native_reference_opt is not None and not args.native_reference_opt.is_file():
        raise FileNotFoundError("native reference compiler does not exist")
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0)!=0:raise RuntimeError("HIP unavailable")
    clock=DeviceClock(hip,args.tessera_opt,args.llvm_bin)
    try:
        rows=[run_case(hip,clock,case,args.tessera_opt,args.trials,args.native_static_control,args.native_runtime_k,args.native_reference_opt,args.hip_graph_windows) for case in
            (args.case or [base.Case("prefill",256,n,5120) for n in (4096,8192,16384)])]
    finally:clock.close()
    packet={"schema":"tessera.rocm.folded_native_package.v1","architecture":rt._rocm_live_arch(),
        "device":base._selected_device_name(hip),"kernel_release":platform.release(),
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "dirty_worktree":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
        "source_sha256":{name:base._sha256(ROOT/name) for name in SOURCES},
        "compiler_sha256":base._sha256(args.tessera_opt),
        "native_reference_compiler_sha256":(
            base._sha256(args.native_reference_opt) if args.native_reference_opt else None),
        "timing_scope":"rotating_three_resident_copies; device windows include dispatch gaps; public launch includes transfers",
        "schedule_match":"same Target-selected keys; native instruction schedule may differ",
        "hip_graph_windows":args.hip_graph_windows,
        "graph_timing_scope":"GPU graph dispatch plus captured kernels and optional markers; capture/instantiate excluded",
        "profiler_counters":False,"radiance_comparison":False,"rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    for row in rows:
        print(row["shape_mnk"],"native/control device ratio",row["native_over_hip_device_ratio"],flush=True)

if __name__=="__main__":main()
