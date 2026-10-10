"""Native-owned paged read -> softmax, with both compiler packages retained."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_native_movement import device_identity
from benchmarks.rocm.benchmark_captured_movement import paged_full
from tests.unit.test_public_movement_frontend import paged,inputs

def normalized(x):
    return ts.ops.softmax(x,axis=-1)

def oracle(pages,table,full):
    gathered=pages[table].reshape(-1,*pages.shape[2:])
    x=(gathered if full else gathered[1:6]).astype(np.float64)
    y=np.exp(x-x.max(axis=-1,keepdims=True))
    return y/y.sum(axis=-1,keepdims=True)

def record(arch,case,directory):
    full=case=="full"
    if full:
        rng=np.random.default_rng(61008)
        pages=rng.normal(size=(32,16,8,128)).astype(np.float32)
        table=rng.integers(0,32,64,dtype=np.int32)
        args=(pages,table)
    else:
        args,_=inputs("paged",case=="large")
    fn=ts.jit(target="rocm_"+arch,native_required=True)(paged_full if full else paged)
    consumer=ts.jit(target="rocm_"+arch,native_required=True)(normalized)
    owner=fn.prepare_native_paged_softmax(consumer,*args)
    artifact=fn.runtime_artifact()
    expected=oracle(*args,full)
    def check(value):
        np.testing.assert_allclose(value,expected,rtol=3e-5,atol=2e-6)
    try:
        receipt=rt.launch(artifact,{"resident_movement":owner})
        assert receipt["ok"] and receipt["native_call_binding"]=="resident_cpp_paged_softmax",receipt
        check(receipt["output"])
        error=float(np.max(np.abs(receipt["output"]-expected)))
        retained=receipt["output"]
        previous=owner.execute(download=False)[0]
        owner.execute(download=False)
        try:previous.to_host()
        except RuntimeError as exc:assert "rc=10" in str(exc)
        else:raise AssertionError("stale edge output was readable")
        changed=(np.ascontiguousarray(args[0]*np.float32(-1.25)),args[1][::-1].copy())
        owner.upload(changed)
        mutated=owner.execute()[0]
        np.testing.assert_allclose(mutated,oracle(*changed,full),rtol=3e-5,atol=2e-6)
        check(retained)
        owner.upload(args)
        samples={name:[] for name in ("public_host_pair","common_resident_download","resident_only")}
        producer_events=[];consumer_events=[]
        def forbidden(*a,**k):raise AssertionError("compiler subprocess in warm edge")
        with ExitStack() as guards:
            for symbol in ("subprocess.run","subprocess.Popen","subprocess.check_output"):
                guards.enter_context(patch(symbol,forbidden))
            for trial in range(9):
                labels=list(samples);labels=labels[trial%3:]+labels[:trial%3]
                if trial%2:labels.reverse()
                for name in labels:
                    before=rt._rocm_native_image_cache_stats()
                    start=time.perf_counter_ns()
                    if name=="public_host_pair":value=consumer(fn(*args))
                    elif name=="common_resident_download":
                        receipt=rt.launch(artifact,{"resident_movement":owner})
                        assert receipt["ok"],receipt
                        value=receipt["output"]
                    else:
                        value,receipt=owner.execute(download=False)
                    samples[name].append((time.perf_counter_ns()-start)/1e6)
                    after=rt._rocm_native_image_cache_stats()
                    assert after["loads"]==before["loads"] and after["unloads"]==before["unloads"]
                    if name!="public_host_pair":
                        producer_events.append(receipt["producer_kernel_elapsed_ms"])
                        consumer_events.append(receipt["consumer_kernel_elapsed_ms"])
                    if name=="resident_only":value=value.to_host()
                    check(value)
        for label,jitted in (("producer",fn),("consumer",consumer)):
            bundle=jitted.compile_bundle
            stages=(bundle.graph,bundle.schedule,bundle.tile,bundle.target_ir,bundle.backend)
            assert bundle.schedule.producer=="tessera-opt.tessera-graph-to-schedule"
            for a,b in zip(stages[:-1],stages[1:],strict=True):assert b.input_digest==a.output_digest
            for name,stage in zip(("graph","schedule","tile","target","backend"),stages,strict=True):
                (directory/(case+"."+label+"."+name+".mlir")).write_text(stage.text)
            (directory/(case+"."+label+".hsaco")).write_bytes(jitted.runtime_artifact().native_image.payload)
        medians={k:statistics.median(v) for k,v in samples.items()}
        return dict(case=case,shape=list(expected.shape),max_abs_error=error,
            correctness="independent_float64_oracle_and_rebinding",stale_generation_rejected=True,
            warm_compiler_subprocesses="forbidden",samples_ms=samples,medians_ms=medians,
            producer_event_samples_ms=producer_events,consumer_event_samples_ms=consumer_events,
            producer_event_median_ms=statistics.median(producer_events),
            consumer_event_median_ms=statistics.median(consumer_events),
            common_over_host_pair=medians["common_resident_download"]/medians["public_host_pair"],
            producer_entry=artifact.launch_descriptor.entry_symbol,
            consumer_entry=owner.consumer.launch_descriptor.entry_symbol,
            producer_image_digest=artifact.native_image.image_digest,
            consumer_image_digest=owner.consumer.native_image.image_digest)
    finally:owner.close()

def standalone(arch):
    rng=np.random.default_rng(61009)
    fn=ts.jit(target="rocm_"+arch,native_required=True)(normalized)
    rows=[]
    for shape in ((13,),(9,17),(5,3,29)):
        x=rng.normal(size=shape).astype(np.float32)
        y=np.exp(x.astype(np.float64)-x.max(axis=-1,keepdims=True))
        expected=y/y.sum(axis=-1,keepdims=True)
        actual=fn(x)
        assert fn.execution_kind=="native_gpu"
        receipt=fn._native_descriptor_last_receipt
        assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=2e-6)
        artifact=fn.runtime_artifact()
        def forbidden(*a,**k):raise AssertionError("compiler subprocess in warm standalone softmax")
        with patch("subprocess.Popen",forbidden),patch("subprocess.run",forbidden):
            np.testing.assert_allclose(fn(x),expected,rtol=3e-5,atol=2e-6)
        assert fn.runtime_artifact() is artifact
        bundle=fn.compile_bundle
        stages=(bundle.graph,bundle.schedule,bundle.tile,bundle.target_ir,bundle.backend)
        for a,b in zip(stages[:-1],stages[1:],strict=True):assert b.input_digest==a.output_digest
        rows.append(dict(shape=list(shape),max_abs_error=float(np.max(np.abs(actual-expected))),
                         entry=artifact.launch_descriptor.entry_symbol,
                         image_digest=artifact.native_image.image_digest,execution_kind="native_gpu"))
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--architecture",choices=("gfx1151","gfx1201"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError("owning HIP device required")
    identity=device_identity(hip,args.architecture)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    unary=standalone(args.architecture)
    rows=[record(args.architecture,c,args.output.parent) for c in ("small","large","full")]
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    paths=("python/tessera/compiler/resident_rocm_movement.py","python/tessera/compiler/jit.py",
        "python/tessera/runtime.py","python/tessera/compiler/capabilities.py","python/tessera/compiler/driver.py","src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
        "benchmarks/rocm/benchmark_paged_softmax_edge.py")
    args.output.write_text(json.dumps(dict(device=identity,rows=rows,standalone_softmax=unary,
        compiler_sha256=hashlib.sha256(Path(find_tessera_opt()).read_bytes()).hexdigest(),
        runtime_library_sha256=hashlib.sha256(Path(rt._load_rocm_native_movement_runtime()._name).read_bytes()).hexdigest(),
        fingerprints={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        contract="two typed frontend packages; native-owned static f32 intermediate; synchronous private stream",
        timing_scope="warm input-resident arms exclude upload; download separately identified; HIP per-stage event windows may include host feed"),indent=2)+"\n")
if __name__=="__main__":main()
