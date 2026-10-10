"""Resident conversion/storage/matmul correctness and ownership gates."""
from dataclasses import replace
import os

import ml_dtypes
import numpy as np
import pytest

from tessera.compiler.rocm_mxfp4 import folded_weights
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera.compiler.rocm_mxfp4_storage import (
    MXFP4_STORAGE_CONTRACT,reference_mxfp4_folded_storage,
)
from tessera.compiler.rocm_nvfp4_ingest import (
    nvfp4_requantization_policy,reference_nvfp4_requantize,
)
from tessera.compiler.rocm_nvfp4_resident import package_resident_nvfp4_matmul

device=pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",
    reason="requires matching compiler and exact gfx1201")

def inputs_and_oracle(m,n,k):
    rng=np.random.default_rng(120105+n+k+m)
    codes=rng.integers(0,256,(n,k//2),np.uint8)
    scales=rng.choice(np.array([0,.03125,.125,1,2,6]),(n,k//16)).astype(ml_dtypes.float8_e4m3fn)
    scales[0,:]=0
    globals_=np.array([.5,2.],np.float64)
    a_values=rng.integers(-4,5,(m,k)).astype(ml_dtypes.float8_e4m3fn)
    a=a_values.view(np.uint8).copy()
    a_scale=np.exp2(rng.integers(-1,2,m)).astype(np.float32)
    offsets=[0,n//2,n]
    converted=reference_nvfp4_requantize(codes,scales,globals_,
        row_offsets=offsets,numeric_policy=nvfp4_requantization_policy())
    stored=reference_mxfp4_folded_storage(*converted[:2],storage_contract=MXFP4_STORAGE_CONTRACT)
    folded=prepare_folded_weights(*converted[:2],allow_approximate=True)
    weights=folded_weights(folded)
    expected=((a_values.astype(np.float64)*a_scale[:,None]) @ weights.astype(np.float64).T
        ).astype(ml_dtypes.bfloat16).astype(np.float32)
    return (codes,scales,globals_,a,a_scale),offsets,converted,stored,expected

@pytest.fixture(scope="module",params=[(128,32,256),(257,80,1024),(256,64,64)])
def compiled(request):
    if os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1":
        pytest.skip("matching exact gfx1201")
    m,n,k=request.param
    args,offsets,converted,stored,expected=inputs_and_oracle(m,n,k)
    program=package_resident_nvfp4_matmul(m,n,k,offsets,
        numeric_policy=nvfp4_requantization_policy(),approximate_policy="explicit_allow")
    return program,args,converted,stored,expected

def test_policies_require_explicit_loss_acceptance_before_compiler():
    with pytest.raises(ValueError,match="explicit conversion"):
        package_resident_nvfp4_matmul(128,32,256,[0,16,32],
            numeric_policy=nvfp4_requantization_policy(),approximate_policy="exact")
    with pytest.raises(ValueError,match="explicit conversion"):
        package_resident_nvfp4_matmul(128,32,256,[0,16,32],
            numeric_policy={"execution_mode":"approximate"},approximate_policy="explicit_allow")

@device
def test_resident_three_stage_without_host_conversion(compiled,monkeypatch):
    from tessera.compiler import rocm_nvfp4_ingest as conversion
    from tessera.compiler import rocm_mxfp4_storage as storage
    from tessera.compiler import rocm_mxfp4_packed_folded as packed
    program,args,converted,stored,expected=compiled
    def forbidden(*a,**kw):
        pytest.fail("resident chain ran host weight conversion/preparation")
    monkeypatch.setattr(conversion,"reference_nvfp4_requantize",forbidden)
    monkeypatch.setattr(storage,"reference_mxfp4_folded_storage",forbidden)
    monkeypatch.setattr(packed,"prepare_packed_folded_payload",forbidden)
    monkeypatch.setattr(packed,"prepare_folded_weights",forbidden)
    with program.session(*args) as session:
        with pytest.raises(RuntimeError,match="successful ingest"):
            session.launch_matmul()
        with pytest.raises(RuntimeError,match="successful matmul"):
            session.read_output()
        session.run_combined()
        actual=session.read_output().astype(np.float32)
        np.testing.assert_allclose(actual,expected,rtol=.008,atol=.015625)
        diagnostics=session.diagnostics()
        for name,wanted in zip(("packed","exponents","stats"),converted):
            if name=="stats":
                np.testing.assert_allclose(diagnostics[name],wanted,rtol=1e-13,atol=1e-30)
            else:
                np.testing.assert_array_equal(diagnostics[name],wanted)
        np.testing.assert_array_equal(diagnostics["fragment"],stored[0])
        np.testing.assert_array_equal(diagnostics["plane"],stored[1])
        for stage in ("converter","storage","consumer","combined"):
            events=session.measure(stage,samples=2,repeats=3)
            assert len(events)==2 and all(value>0 for value in events)
            replay=session.measure_graph(stage,samples=2,repeats=16)
            assert len(replay)==2 and all(value["per_iteration_ms"]>0 for value in replay)
            assert all(value["graph_nodes"]==16*(3 if stage=="combined" else 1) for value in replay)
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
        pointers={name:pointer.value for name,pointer in session._buffers.items()}
        session.update_activations(args[3],args[4]*np.float32(.5))
        session.launch_matmul()
        np.testing.assert_allclose(session.read_output().astype(np.float32),
            (expected*.5).astype(ml_dtypes.bfloat16).astype(np.float32),rtol=.008,atol=.015625)
        assert pointers=={name:pointer.value for name,pointer in session._buffers.items()}
        np.testing.assert_array_equal(session.diagnostics()["plane"],stored[1])
        assert len(session._buffers)==11
        assert len({p.value for p in session._buffers.values()})==11
    assert session._closed and not session._buffers and not session._modules and not session._graphs
    session.close()
    with pytest.raises(RuntimeError,match="closed"):
        session.run_combined()
    receipt=program.receipt
    assert len(receipt["component_image_digests"])==3
    assert "weight_sha256" not in program.consumer.package.descriptor.provenance
    assert "fold_lossless" not in program.consumer.package.descriptor.provenance

@device
def test_resident_snapshots_and_failure_before_hip(compiled,monkeypatch):
    from tessera import runtime as rt
    program,args,converted,stored,expected=compiled
    owned=[a.copy() for a in args]
    with program.session(*owned) as session:
        for value in owned:
            value[:]=0
        session.run_combined()
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
        np.testing.assert_array_equal(session.diagnostics()["packed"],converted[0])
    def forbidden(*a,**kw):
        pytest.fail("bad program/input reached HIP loading")
    monkeypatch.setattr(rt,"_load_hip_for_launch",forbidden)
    bad=list(args);bad[-1]=np.full_like(args[-1],np.nan)
    with pytest.raises(ValueError,match="finite"):
        program.session(*bad)
    bad=list(args);bad[0]=args[0][:,:-1]
    with pytest.raises(ValueError,match="dtype/shape"):
        program.session(*bad)
    native=program.consumer.package
    changed=replace(program.consumer,package=replace(native,
        descriptor=replace(native.descriptor,provenance={
            **native.descriptor.provenance,"kernel_argument_layout":"compact"})))
    with pytest.raises(ValueError,match="descriptor changed"):
        replace(program,consumer=changed).session(*args)
    with pytest.raises(ValueError,match="Graph semantics"):
        replace(program,consumer=replace(program.consumer,graph_ir=program.consumer.graph_ir+" ")).session(*args)


@device
def test_failed_activation_upload_and_close_retain_owners(compiled,monkeypatch):
    program,args,converted,stored,expected=compiled
    session=program.session(*args)
    try:
        session.run_combined()
        session.synchronize()
        hip=session._hip
        copy=hip.hipMemcpyAsync
        count=0
        def fail_second_copy(*values):
            nonlocal count
            count+=1
            return 1 if count==2 else copy(*values)
        with monkeypatch.context() as patch:
            patch.setattr(hip,"hipMemcpyAsync",fail_second_copy)
            with pytest.raises(RuntimeError,match="HIP status"):
                session.update_activations(args[3],args[4]*np.float32(.5))
        assert not session._activations_ready and len(session._buffers)==11
        with pytest.raises(RuntimeError,match="successful upload"):
            session.launch_matmul()
        session.update_activations(args[3],args[4])
        session.launch_matmul()
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
        with monkeypatch.context() as patch:
            patch.setattr(hip,"hipStreamSynchronize",lambda *a:1)
            with pytest.raises(RuntimeError,match="HIP status"):
                session.close()
        # Completion failed: no image/allocation owner was released.
        assert not session._closed and len(session._buffers)==11 and len(session._modules)==3
    finally:
        session.close()
    assert session._closed and not session._buffers and not session._modules


def test_quality_reference_uses_original_activation_and_nk_weights():
    from benchmarks.rocm.benchmark_resident_nvfp4 import _source_quality,sha
    rng=np.random.default_rng(1025)
    source_a=rng.normal(size=(3,7)).astype(np.float32)
    source_b=rng.normal(size=(5,7)).astype(np.float32)
    output=source_a.astype(np.float64) @ source_b.astype(np.float64).T
    result=_source_quality(output,source_b,source_a.astype(np.float64)*.5,source_b,source_a)
    assert result["native_output_vs_source"]["relative_rms"]==0
    assert result["native_output_vs_dequantized_activation_bf16_weights"]["relative_rms"]==pytest.approx(1)
    assert result["source_activation_sha256"]==sha(source_a)
    assert result["source_weight_sha256"]==sha(source_b)
    with pytest.raises(ValueError,match="matched source"):
        _source_quality(output,source_b.T,source_a,source_b,source_a)


def test_portable_schema_rejects_invalid_shapes_and_duplicate_json_before_devices():
    from tessera.compiler.rocm_nvfp4_resident import NVFP4ResidentProgram
    with pytest.raises(ValueError,match="duplicate"):
        NVFP4ResidentProgram.from_json('{"schema":1,"schema":2}')
    with pytest.raises(ValueError,match="nonfinite"):
        NVFP4ResidentProgram.from_json('{"schema":NaN}')
    for shape in ([True,32,256],[128,32.0,256],None):
        with pytest.raises(ValueError,match="integer M/N/K"):
            NVFP4ResidentProgram.from_dict({
                "schema":"tessera.rocm.nvfp4_resident_program.v1",
                "shape_mnk":shape,"stages":[],"program_digest":""})


@device
def test_portable_resident_replay_without_compiler_or_host_kernels(compiled,monkeypatch):
    from tessera.compiler.rocm_nvfp4_resident import NVFP4ResidentProgram
    from tessera.compiler import scheduled_matmul,rocm_native,rocm_nvfp4_resident
    from tessera.compiler import rocm_nvfp4_ingest,rocm_mxfp4_storage
    program,args,converted,stored,expected=compiled
    encoded=program.to_json()
    def forbidden(*a,**kw):
        pytest.fail("portable replay invoked a compiler or host weight kernel")
    monkeypatch.setattr(scheduled_matmul,"run_tessera_opt",forbidden)
    monkeypatch.setattr(rocm_nvfp4_resident,"run_tessera_opt",forbidden)
    monkeypatch.setattr(rocm_native,"_compile_native_tile_ir",forbidden)
    monkeypatch.setattr(rocm_nvfp4_ingest,"reference_nvfp4_requantize",forbidden)
    monkeypatch.setattr(rocm_mxfp4_storage,"reference_mxfp4_folded_storage",forbidden)
    restored=NVFP4ResidentProgram.from_json(encoded)
    assert restored.to_json()==encoded
    assert restored.receipt==program.receipt
    exported=program.to_dict()
    isolated=NVFP4ResidentProgram.from_dict(exported)
    exported["stages"][0]["descriptor"]["provenance"]["row_offsets"][0]=999
    # Exported metadata and restored metadata each own their nested lists.
    assert program.to_json()==encoded
    assert isolated.to_json()==encoded
    with restored.session(*args) as session:
        session.run_combined()
        np.testing.assert_allclose(session.read_output().astype(np.float32),
            expected,rtol=.008,atol=.015625)
        np.testing.assert_array_equal(session.diagnostics()["packed"],converted[0])
        np.testing.assert_array_equal(session.diagnostics()["plane"],stored[1])


@device
def test_portable_mutations_rejected_before_hip(compiled,monkeypatch):
    from copy import deepcopy
    import hashlib
    from tessera import runtime as rt
    from tessera.compiler.rocm_nvfp4_resident import NVFP4ResidentProgram,_program_digest
    program,args,*_=compiled
    def forbidden(*a,**kw):
        pytest.fail("invalid portable program reached HIP")
    monkeypatch.setattr(rt,"_load_hip_for_launch",forbidden)
    original=program.to_dict()
    def reseal(data):
        identity={key:data[key] for key in ("schema","shape_mnk","stages")}
        data["program_digest"]=_program_digest(identity)
        return data
    bad=deepcopy(original);bad["shape_mnk"][0]+=1
    with pytest.raises(ValueError,match="digest"):
        NVFP4ResidentProgram.from_dict(bad)
    bad=deepcopy(original);bad["stages"].reverse()
    with pytest.raises(ValueError):
        NVFP4ResidentProgram.from_dict(reseal(bad))
    bad=deepcopy(original);bad["shape_mnk"][1]+=16
    with pytest.raises(ValueError):
        NVFP4ResidentProgram.from_dict(reseal(bad))
    bad=deepcopy(original)
    descriptor=program.consumer.package.descriptor
    bad["stages"][2]["descriptor"]=replace(descriptor,
        geometry=replace(descriptor.geometry,workgroup=(128,1,1))).to_dict()
    with pytest.raises(ValueError):
        NVFP4ResidentProgram.from_dict(reseal(bad))
    bad=deepcopy(original);bad["stages"][0]["image"]["payload_b64"]="AAAA"
    with pytest.raises(ValueError):
        NVFP4ResidentProgram.from_dict(reseal(bad))
    # Resealing metadata cannot relabel the converter's retained Graph semantics.
    bad=deepcopy(original)
    stage=bad["stages"][0]
    stage["graph_ir"]=stage["graph_ir"].replace("tessera.nvfp4_requantize","tessera.unrelated")
    stage["descriptor"]=replace(program.ingest.native.descriptor,provenance={
        **program.ingest.native.descriptor.provenance,
        "graph_digest":hashlib.sha256(stage["graph_ir"].encode()).hexdigest()}).to_dict()
    with pytest.raises(ValueError,match="topology or semantic policy"):
        NVFP4ResidentProgram.from_dict(reseal(bad))
    bad=deepcopy(original);bad["stages"][1]["extra"]="ignored?"
    with pytest.raises(ValueError,match="stage schema"):
        NVFP4ResidentProgram.from_dict(reseal(bad))


@device
def test_portable_resident_fresh_process_replay(compiled,tmp_path):
    import subprocess
    import sys
    program,args,converted,stored,expected=compiled
    program_path=tmp_path/"program.json"
    input_path=tmp_path/"inputs.npz"
    output_path=tmp_path/"output.npz"
    program_path.write_text(program.to_json())
    # NPZ uses raw scale storage because NumPy does not preserve custom dtype metadata.
    np.savez(input_path,codes=args[0],scales=args[1].view(np.uint8),
             globals=args[2],a=args[3],a_scale=args[4])
    script="""
import sys
import numpy as np
import ml_dtypes
from tessera.compiler.rocm_nvfp4_resident import NVFP4ResidentProgram
from tessera.compiler import scheduled_matmul, rocm_native
def forbidden(*a, **k):
    raise AssertionError("fresh portable replay invoked compiler")
scheduled_matmul.run_tessera_opt=forbidden
rocm_native._compile_native_tile_ir=forbidden
from pathlib import Path
program=NVFP4ResidentProgram.from_json(Path(sys.argv[1]).read_text())
with np.load(sys.argv[2],allow_pickle=False) as x:
    with program.session(x["codes"],x["scales"].view(ml_dtypes.float8_e4m3fn),
                         x["globals"],x["a"],x["a_scale"]) as session:
        session.run_combined()
        np.savez(sys.argv[3],output=session.read_output().astype(np.float32),
                 packed=session.diagnostics()["packed"])
"""
    environment={**os.environ,"TESSERA_OPT":"/nonexistent/portable-replay-needs-no-compiler"}
    result=subprocess.run([sys.executable,"-c",script,str(program_path),str(input_path),str(output_path)],
                          env=environment,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    with np.load(output_path,allow_pickle=False) as output:
        np.testing.assert_allclose(output["output"],expected,rtol=.008,atol=.015625)
        np.testing.assert_array_equal(output["packed"],converted[0])

@device
def test_projected_resident_consumer_reuses_image_across_static_graphs():
    from tessera.compiler.rocm_nvfp4_resident import package_resident_packed_consumer
    left=package_resident_packed_consumer(128,32,256)
    right=package_resident_packed_consumer(200,80,256)
    assert left.package.image.image_digest==right.package.image.image_digest
    assert left.package.image.payload==right.package.image.payload
    assert left.graph_ir!=right.graph_ir
    assert left.schedule_ir!=right.schedule_ir
    assert left.package.descriptor.shape_guards!=right.package.descriptor.shape_guards
    for consumer in (left,right):
        consumer.validate()
        p=consumer.package
        assert p.descriptor.provenance["image_shape_policy"]=="runtime_mn_fixed_k"
        authored=p.descriptor.provenance["authored_target_ir"]
        assert "runtime_mn" not in authored and "runtime_mn" in p.target_ir


@device
def test_static_resident_consumer_remains_portable():
    from tessera.compiler.rocm_nvfp4_resident import NVFP4ResidentProgram
    args,offsets,converted,stored,expected=inputs_and_oracle(128,32,256)
    program=package_resident_nvfp4_matmul(128,32,256,offsets,
        numeric_policy=nvfp4_requantization_policy(),approximate_policy="explicit_allow",
        runtime_mn=False)
    assert program.consumer.package.descriptor.provenance["image_shape_policy"]=="static_mnk"
    restored=NVFP4ResidentProgram.from_json(program.to_json())
    with restored.session(*args) as session:
        session.run_combined()
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)


@device
def test_authored_projection_attribute_mutation_is_rejected(compiled):
    import hashlib
    program,*_=compiled
    consumer=program.consumer
    package=consumer.package
    provenance=dict(package.descriptor.provenance)
    authored=provenance["authored_target_ir"].replace('stage_k = 64 : i64','stage_k = 32 : i64')
    provenance.update(authored_target_ir=authored,
                      authored_target_ir_sha256=hashlib.sha256(authored.encode()).hexdigest())
    changed=replace(consumer,package=replace(package,
        descriptor=replace(package.descriptor,provenance=provenance)))
    with pytest.raises(ValueError,match="projected Target contract"):
        changed.validate()
