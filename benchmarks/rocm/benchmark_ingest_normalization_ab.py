"""Paired native converter A/B; complete Graph packages, identical inputs."""
from pathlib import Path
from contextlib import ExitStack
import argparse
import ctypes as C
import hashlib
import json
import os
from statistics import median
import time

import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_resident import package_resident_nvfp4_matmul
from tessera.compiler.rocm_nvfp4_ingest import nvfp4_requantization_policy
from benchmarks.rocm.benchmark_resident_nvfp4 import sha
from benchmarks.rocm import benchmark_gfx1201_three_formats as formats
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle

ROOT=Path(__file__).resolve().parents[2]


def paired(args,offsets,converted,stored,verify,options):
    codes,scales,globals_,a,a_scale=args
    m,k=a.shape;n=codes.shape[0]
    programs={}
    for label,compiler in (("baseline",options.baseline),("reciprocal",options.compiler)):
        os.environ["TESSERA_OPT"]=str(compiler.resolve())
        programs[label]=package_resident_nvfp4_matmul(m,n,k,offsets,
            numeric_policy=nvfp4_requantization_policy(),approximate_policy="explicit_allow")
    # Identical untouched native storage/consumer leaves isolate the converter.
    for leaf in ("storage","consumer"):
        x=getattr(programs["baseline"],leaf)
        y=getattr(programs["reciprocal"],leaf)
        xp=x.native if leaf=="storage" else x.package
        yp=y.native if leaf=="storage" else y.package
        assert xp.image.image_digest==yp.image.image_digest,leaf
    assert programs["baseline"].ingest.native.image.image_digest!=programs["reciprocal"].ingest.native.image.image_digest
    stages=("converter","combined")
    windows={label:{stage:[] for stage in stages} for label in programs}
    walls={label:[] for label in programs}
    output_hashes={}
    with ExitStack() as stack:
        sessions={label:stack.enter_context(program.session(*args)) for label,program in programs.items()}
        stats={}
        for label,session in sessions.items():
            session.run_combined()
            output=session.read_output();verify(output)
            output_hashes[label]=sha(output)
            diagnostic=session.diagnostics()
            for name,wanted in zip(("packed","exponents","stats"),converted):
                if name=="stats":
                    np.testing.assert_allclose(diagnostic[name],wanted,rtol=1e-12,atol=1e-30)
                else:
                    np.testing.assert_array_equal(diagnostic[name],wanted)
            np.testing.assert_array_equal(diagnostic["fragment"],stored[0])
            np.testing.assert_array_equal(diagnostic["plane"],stored[1])
            stats[label]=diagnostic["stats"]
        assert output_hashes["baseline"]==output_hashes["reciprocal"]
        np.testing.assert_array_equal(stats["baseline"],stats["reciprocal"])
        repeats={}
        for stage in stages:
            estimates=[s.measure_graph(stage,samples=1,repeats=1)[0]["window_ms"] for s in sessions.values()]
            repeats[stage]=max(8,min(256,int(4/max(estimates))))
            for s in sessions.values():
                s.measure_graph(stage,samples=1,repeats=repeats[stage])
        for trial in range(options.windows):
            labels=list(sessions)
            if trial%2:labels.reverse()
            for stage in (stages if trial%2==0 else stages[::-1]):
                for label in labels:
                    windows[label][stage].extend(sessions[label].measure_graph(
                        stage,samples=1,repeats=repeats[stage]))
        for session in sessions.values():
            verify(session.read_output())
            diagnostic=session.diagnostics()
            np.testing.assert_array_equal(diagnostic["packed"],converted[0])
            np.testing.assert_array_equal(diagnostic["exponents"],converted[1])
    for trial in range(options.windows):
        labels=list(programs)
        if trial%2:labels.reverse()
        for label in labels:
            start=time.perf_counter_ns()
            with programs[label].session(*args) as session:
                session.run_combined();output=session.read_output()
            walls[label].append((time.perf_counter_ns()-start)/1e6)
            verify(output)
            assert sha(output)==output_hashes[label]
    medians={label:{stage:median(w["per_iteration_ms"] for w in values)
                   for stage,values in row.items()} for label,row in windows.items()}
    return {"shape_mnk":[m,n,k],"row_offsets":list(offsets),
        "input_sha256":dict(zip(("codes","scales","globals","a","a_scale"),map(sha,args))),
        "output_sha256":output_hashes,"correctness":"independent oracle; bitwise A/B bytes, exponent, stats and final output",
        "graph_windows":windows,"graph_median_ms":medians,
        "checked_wall_samples_ms":walls,"checked_wall_median_ms":{label:median(v) for label,v in walls.items()},
        "converter_speedup":medians["baseline"]["converter"]/medians["reciprocal"]["converter"],
        "combined_speedup":medians["baseline"]["combined"]/medians["reciprocal"]["combined"],
        "program_receipts":{label:program.receipt for label,program in programs.items()},
        "measurement_order":"alternate baseline/candidate and reverse stage order each trial",
        "scope":"resident HIP graph windows include GPU dispatch; walls include allocation/upload/readback/cleanup"}


def record(options):
    if options.windows<3:raise ValueError("at least three paired windows")
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("exact gfx1201 required")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError("HIP unavailable")
    ordinal=C.c_int();name=C.create_string_buffer(256)
    hip.hipGetDevice.argtypes=[C.POINTER(C.c_int)]
    hip.hipDeviceGetName.argtypes=[C.c_char_p,C.c_int,C.c_int]
    if hip.hipGetDevice(C.byref(ordinal)) or hip.hipDeviceGetName(name,len(name),ordinal.value) or not name.value:
        raise RuntimeError("actual device identity required")
    packet={"schema":"tessera.rocm.ingest_reciprocal_ab.v1","architecture":"gfx1201",
        "device":name.value.decode(),"device_ordinal":ordinal.value,"selector_promotion":False,
        "compiler_sha256":{label:hashlib.sha256(path.read_bytes()).hexdigest()
                           for label,path in (("baseline",options.baseline),("reciprocal",options.compiler))},
        "compiler_leaf_source_sha256":{
            "baseline":hashlib.sha256(options.baseline_source.read_bytes()).hexdigest(),
            "reciprocal":hashlib.sha256((ROOT/"src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/NativeNVFP4Ingest.h").read_bytes()).hexdigest()},
        "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"rows":[]}
    def save(row):
        packet["rows"].append(row)
        options.output.parent.mkdir(parents=True,exist_ok=True)
        options.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
        print("verified paired",row["shape_mnk"],"speedup",row["converter_speedup"],flush=True)
    for shape in ((128,32,256),(257,80,1024),(256,64,64)):
        args,offsets,converted,stored,expected=inputs_and_oracle(*shape)
        save(paired(args,offsets,converted,stored,
            lambda out:np.testing.assert_allclose(out.astype(np.float64),expected.astype(np.float64),rtol=1/128,atol=1e-5),
            options))
    if options.checkpoint:
        from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint
        from tessera.compiler.rocm_nvfp4_ingest import reference_nvfp4_requantize
        from tessera.compiler.rocm_mxfp4_storage import reference_mxfp4_folded_storage,MXFP4_STORAGE_CONTRACT
        from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
        loaded=[checkpoint._load_projection(name) for name in (
            "model.layers.0.mlp.gate_proj.weight","model.layers.0.mlp.up_proj.weight")]
        projections=[item["projection"] for item in loaded]
        packet["source_checkpoints"]=[item["source"] for item in loaded]
        codes=np.concatenate([p.packed_codes for p in projections])
        scales=np.concatenate([p.e4m3_scales for p in projections])
        globals_=np.array([p.global_scale for p in projections],np.float64)
        offsets=[0,projections[0].packed_codes.shape[0],codes.shape[0]]
        k=codes.shape[1]*2
        values=np.random.default_rng(527).normal(size=(256,k)).astype(np.float32)
        a_scale=np.maximum(np.max(np.abs(values),axis=1)/448,np.finfo(np.float32).tiny).astype(np.float32)
        a=np.ascontiguousarray((values/a_scale[:,None]).astype(ml_dtypes.float8_e4m3fn).view(np.uint8))
        converted=reference_nvfp4_requantize(codes,scales,globals_,row_offsets=offsets,numeric_policy=nvfp4_requantization_policy())
        stored=reference_mxfp4_folded_storage(*converted[:2],storage_contract=MXFP4_STORAGE_CONTRACT)
        folded=prepare_folded_weights(*converted[:2],allow_approximate=True)
        da=a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*a_scale[:,None]
        db=folded.weight_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*np.where(
            folded.row_reference==0,0.,np.exp2(folded.row_reference.astype(np.int16)-127))[:,None]
        ideal=da@db.T;absolute=np.abs(da)@np.abs(db).T
        save(paired((codes,scales,globals_,a,a_scale),offsets,converted,stored,
            lambda out:formats.verify(out,ideal,absolute,k),options))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--baseline-source",type=Path,required=True)
    parser.add_argument("--compiler",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--windows",type=int,default=5)
    parser.add_argument("--checkpoint",action="store_true")
    record(parser.parse_args())
