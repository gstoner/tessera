"""Native-owned HIP graph versus direct submission of identical compiled images."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import patch
import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_native_movement import device_identity
from benchmarks.rocm.benchmark_paged_softmax_edge import normalized
from tests.unit.test_native_graph_rocm_movement import build_case

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def record(architecture,family,large,paired,directory,pairs):
    fn,args,reference=build_case(architecture,family,large)
    np.testing.assert_array_equal(fn(*args).view(np.uint32),reference(args).view(np.uint32))
    if paired:
        consumer=ts.jit(target="rocm_"+architecture,native_required=True)(normalized)
        owner=fn.prepare_native_paged_softmax(consumer,*args)
        raw=reference
        def reference(values):
            x=raw(values).astype(np.float64)
            e=np.exp(x-x.max(-1,keepdims=True))
            return e/e.sum(-1,keepdims=True)
        def check(value,expected):np.testing.assert_allclose(value,expected,rtol=3e-5,atol=2e-6)
    else:
        owner=fn.prepare_native_movement(*args)
        def check(value,expected):np.testing.assert_array_equal(value.view(np.uint32),expected.view(np.uint32))
    lineage={}
    for role,jitted in (("producer",fn),("consumer",consumer)) if paired else (("producer",fn),):
        bundle=jitted.compile_bundle
        stages=(bundle.graph,bundle.schedule,bundle.tile,bundle.target_ir,bundle.backend)
        assert bundle.schedule.producer=="tessera-opt.tessera-graph-to-schedule"
        for a,b in zip(stages[:-1],stages[1:],strict=True):assert b.input_digest==a.output_digest
        lineage[role]={stage.level:dict(producer=stage.producer,input_digest=stage.input_digest,
            output_digest=stage.output_digest) for stage in stages}
    expected=reference(args)
    label=family+("_large" if large else "_small")+("_softmax" if paired else "")
    samples={"direct":[],"graph":[]};rejected=[]
    try:
        prepared=time.perf_counter_ns();nodes=owner.capture()
        prepare_ms=(time.perf_counter_ns()-prepared)/1e6
        assert nodes==(2 if paired else 1)
        check(owner.execute(captured=True)[0],expected)
        retained=owner.execute(captured=False)[0]
        changed=list(args);source,index=owner.prepared.input_positions
        changed[source]=np.ascontiguousarray(args[source]*np.float32(-1.25))
        changed[index]=args[index][::-1].copy()
        owner.upload(tuple(changed))
        check(owner.execute(captured=True)[0],reference(tuple(changed)));check(retained,expected)
        owner.upload(args)
        def forbidden(*a,**k):raise AssertionError("compiler during warm native graph benchmark")
        with ExitStack() as guards:
            for symbol in ("subprocess.run","subprocess.Popen","subprocess.check_output"):
                guards.enter_context(patch(symbol,forbidden))
            for captured in (False,True)*3:owner.execute(download=False,captured=captured)
            repetitions=128
            for pair in range(pairs):
                labels=("direct","graph") if pair%2==0 else ("graph","direct")
                while True:
                    window={}
                    for name in labels:
                        before=rt._rocm_native_image_cache_stats()
                        events=[];producer=[];cons=[]
                        start=time.perf_counter_ns()
                        for _ in range(repetitions):
                            value,receipt=owner.execute(download=False,captured=name=="graph")
                            events.append(receipt["kernel_elapsed_ms"])
                            if paired and name=="direct":
                                producer.append(receipt["producer_kernel_elapsed_ms"])
                                cons.append(receipt["consumer_kernel_elapsed_ms"])
                        wall=(time.perf_counter_ns()-start)/1e6;check(value.to_host(),expected)
                        after=rt._rocm_native_image_cache_stats()
                        assert after["loads"]==before["loads"] and after["unloads"]==before["unloads"]
                        window[name]=dict(pair=pair,repetitions=repetitions,host_window_ms=wall,
                            completed_host_ms_per_call=wall/repetitions,device_event_samples_ms=events,
                            device_event_median_ms=statistics.median(events),
                            event_scope="whole_sequence" if name=="graph" else "sum_of_member_intervals",
                            producer_event_samples_ms=producer,consumer_event_samples_ms=cons)
                    if min(w["host_window_ms"] for w in window.values())>=20:break
                    rejected.append(window);repetitions*=2
                    if repetitions>8192:raise RuntimeError("native graph timing windows too short")
                for name in samples:samples[name].append(window[name])
        bundle=fn.compile_bundle
        stages=(bundle.graph,bundle.schedule,bundle.tile,bundle.target_ir,bundle.backend)
        assert bundle.schedule.producer=="tessera-opt.tessera-graph-to-schedule"
        for a,b in zip(stages[:-1],stages[1:],strict=True):assert b.input_digest==a.output_digest
        for stage in stages:(directory/(label+"."+stage.level+".mlir")).write_text(stage.text)
        host={n:statistics.median(w["completed_host_ms_per_call"] for w in rows) for n,rows in samples.items()}
        event={n:statistics.median(w["device_event_median_ms"] for w in rows) for n,rows in samples.items()}
        artifact=fn.runtime_artifact()
        return dict(case=label,architecture=architecture,shape=list(expected.shape),
            correctness="oracle_before_after_every_arm_and_changed_input_replay",graph_kernel_nodes=nodes,
            graph_prepare_ms=prepare_ms,producer_image=artifact.native_image.image_digest,
            producer_entry=artifact.launch_descriptor.entry_symbol,compiler_lineage=lineage,
            consumer_image=owner.consumer.native_image.image_digest if paired else None,
            consumer_entry=owner.consumer.launch_descriptor.entry_symbol if paired else None,
            warm_module_loads=0,warm_compiler_subprocesses="forbidden",samples=samples,
            rejected_short_pairs=rejected,host_medians_ms=host,event_medians_ms=event,
            graph_over_direct_host_ratio=host["graph"]/host["direct"],
            timing_caveat="HIP event dispatch intervals; graph whole sequence and direct member sums have distinct scopes")
    finally:owner.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--architecture",choices=("gfx1151","gfx1201"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pairs",type=int,default=7)
    args=parser.parse_args()
    if not 3<=args.pairs<=31:raise ValueError("pairs must be in [3,31]")
    if rt._rocm_live_arch()!=args.architecture:raise RuntimeError("owning GPU architecture mismatch")
    args.output.mkdir(parents=True,exist_ok=True)
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError('usable owning HIP device required')
    device=device_identity(hip,args.architecture);root=Path.cwd()
    paths=("src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
           "python/tessera/compiler/resident_rocm_movement.py",
           "benchmarks/rocm/benchmark_native_graph_movement.py","tests/unit/test_native_graph_rocm_movement.py")
    native=rt._load_rocm_native_movement_runtime()
    rows=[record(args.architecture,f,l,p,args.output,args.pairs) for f,l,p in (
        ("paged",False,False),("paged",True,False),("dispatched",False,False),
        ("dispatched",True,False),("paged",False,True),("paged",True,True),("full",True,True))
        if args.architecture=="gfx1151" or f!="dispatched"]
    packet=dict(architecture=args.architecture,device=device,python=platform.python_version(),
        source_sha256={p:sha(root/p) for p in paths},
        compiler_sha256={n:sha(os.environ[n]) for n in ("TESSERA_OPT","TESSERA_ROCM_OPT")},
        native_runtime_sha256=sha(native._name),rows=rows,default_changed=False,
        unsupported_profile="gfx1201 prepared MoE dispatch remains unadmitted" if args.architecture=="gfx1201" else None)
    (args.output/"device.json").write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps({r["case"]:r["graph_over_direct_host_ratio"] for r in rows},indent=2))
if __name__=="__main__":main()
