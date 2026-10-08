"""Owning-device native resident movement and checked common-runtime replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_native_movement import device_identity
from benchmarks.rocm.benchmark_captured_movement import paged_full
from tests.unit.test_public_movement_frontend import paged,dispatched,inputs

def record(arch,family,large,directory):
    if family=="full":
        rng=np.random.default_rng(61007)
        pages=rng.normal(size=(32,16,8,128)).astype(np.float32)
        table=rng.integers(0,32,64,dtype=np.int32)
        args=(pages,table);source=paged_full
    else:
        args,_=inputs(family,large)
        source=paged if family=="paged" else dispatched
    values=(args[1][args[0][0]] if family=="dispatched" else
            args[0][args[1][0],0 if family=="full" else 1])
    values.view(np.uint32).flat[:4]=[0x7fc12345,0x80000000,0x7f800000,0xff800000]
    def oracle(ordered):
        if family=="dispatched":return ordered[1][ordered[0]]
        x,idx=ordered
        result=x[idx].reshape(-1,*x.shape[2:])
        return result if family=="full" else result[1:6]
    expected=oracle(args)
    fn=ts.jit(target="rocm_"+arch,native_required=True)(source)
    np.testing.assert_array_equal(fn(*args).view(np.uint32),expected.view(np.uint32))
    artifact=fn.runtime_artifact()
    calls=list(fn._native_prepared_movement_calls.values())
    assert len(calls)==1
    prepared=calls[0]
    uninitialized=prepared.resident()
    try:
        try:uninitialized.execute()
        except RuntimeError as exc:assert "rc=10" in str(exc)
        else:raise AssertionError("resident movement ran before upload")
    finally:uninitialized.close()
    owner=fn.prepare_native_movement(*args)
    try:
        owner.upload(args)
        receipt=rt.launch(artifact,{"resident_movement":owner})
        assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
        assert receipt["native_call_binding"]=="resident_cpp_movement"
        np.testing.assert_array_equal(receipt["output"].view(np.uint32),expected.view(np.uint32))
        retained=receipt["output"]
        previous=owner.execute(download=False)[0]
        owner.execute(download=False)
        try:previous.to_host()
        except RuntimeError as exc:assert "rc=10" in str(exc)
        else:raise AssertionError("retired resident generation was readable")
        bad=list(args)
        index_position=prepared.input_positions[1]
        bad[index_position]=bad[index_position].copy()
        bad[index_position][0]=-1
        try:owner.upload(bad)
        except RuntimeError as exc:assert "rc=1" in str(exc)
        else:raise AssertionError("resident index bounds were not checked")
        np.testing.assert_array_equal(owner.execute()[0].view(np.uint32),expected.view(np.uint32))
        assert not rt.launch(artifact,{"resident_movement":owner},stream=1)["ok"]
        changed=list(args)
        source_position=prepared.input_positions[0]
        changed[source_position]=changed[source_position].view(np.uint32).__xor__(
            np.uint32(0x80000000)).view(np.float32)
        changed[index_position]=changed[index_position][::-1].copy()
        owner.upload(changed)
        np.testing.assert_array_equal(owner.execute()[0].view(np.uint32),oracle(changed).view(np.uint32))
        np.testing.assert_array_equal(retained.view(np.uint32),expected.view(np.uint32))
        owner.upload(args)
        samples={name:[] for name in ("public_host","common_resident_download","direct_resident_download","resident_only")}
        events=[]
        functions={
            "public_host":lambda:fn(*args),
            "common_resident_download":lambda:rt.launch(artifact,{"resident_movement":owner}),
            "direct_resident_download":lambda:owner.execute(),
            "resident_only":lambda:owner.execute(download=False)}
        def forbidden(*a,**k):
            raise AssertionError("compiler subprocess during warm resident movement")
        with ExitStack() as guards:
            for symbol in ("subprocess.run","subprocess.Popen","subprocess.check_output"):
                guards.enter_context(patch(symbol,forbidden))
            for trial in range(9):
                labels=list(functions);labels=labels[trial%4:]+labels[:trial%4]
                if trial%2:labels.reverse()
                for name in labels:
                    before=rt._rocm_native_image_cache_stats()
                    start=time.perf_counter_ns()
                    result=functions[name]()
                    samples[name].append((time.perf_counter_ns()-start)/1e6)
                    after=rt._rocm_native_image_cache_stats()
                    assert after["loads"]==before["loads"] and after["unloads"]==before["unloads"]
                    if name=="public_host":value=result
                    elif name=="common_resident_download":
                        assert result["ok"] and result["native_call_binding"]=="resident_cpp_movement"
                        value=result["output"]
                    else:
                        value,receipt=result
                        events.append(receipt["kernel_elapsed_ms"])
                        if name=="resident_only":value=value.to_host()
                    np.testing.assert_array_equal(value.view(np.uint32),expected.view(np.uint32))
        label=family+("_large" if large else "_small")
        bundle=fn.compile_bundle
        stages=(bundle.graph,bundle.schedule,bundle.tile,bundle.target_ir,bundle.backend)
        assert bundle.schedule.producer=="tessera-opt.tessera-graph-to-schedule"
        for a,b in zip(stages[:-1],stages[1:],strict=True):assert b.input_digest==a.output_digest
        for name,stage in zip(("graph","schedule","tile","target","backend"),stages,strict=True):
            (directory/(label+"."+name+".mlir")).write_text(stage.text)
        (directory/(label+".hsaco")).write_bytes(artifact.native_image.payload)
        medians={name:statistics.median(values) for name,values in samples.items()}
        return dict(family=family,large=large,correctness="bit_exact_rebind_and_retained_host_outputs",
                    stale_generation_rejected=True,bad_indices_rejected=True,warm_compiler_subprocesses="forbidden",
                    entry=artifact.launch_descriptor.entry_symbol,image_digest=artifact.native_image.image_digest,
                    samples_ms=samples,medians_ms=medians,native_kernel_event_samples_ms=events,
                    native_kernel_event_median_ms=statistics.median(events),
                    common_resident_over_public=medians["common_resident_download"]/medians["public_host"])
    finally:owner.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--architecture",choices=("gfx1151","gfx1201"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError("usable owning HIP device required")
    identity=device_identity(hip,args.architecture)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    families=("paged","full","dispatched") if args.architecture=="gfx1151" else ("paged","full")
    rows=[record(args.architecture,f,large,args.output.parent) for f in families
          for large in ((False,True) if f!="full" else (True,))]
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    compiler=Path(find_tessera_opt())
    paths=("python/tessera/compiler/resident_rocm_movement.py",
        "python/tessera/compiler/prepared_rocm_movement.py","python/tessera/compiler/jit.py","python/tessera/runtime.py",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
        "benchmarks/rocm/benchmark_resident_movement.py")
    args.output.write_text(json.dumps(dict(device=identity,rows=rows,
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
        runtime_library_sha256=hashlib.sha256(Path(rt._load_rocm_native_movement_runtime()._name).read_bytes()).hexdigest(),
        fingerprints={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        timing_scope="separate synchronous host wall arms and native one-kernel HIP event windows; compilation/uploads excluded from resident warm arms"),indent=2)+"\n")
if __name__=="__main__":main()
