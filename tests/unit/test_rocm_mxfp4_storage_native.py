"""Lossless Graph-originated storage bridge and native device parity."""
from dataclasses import replace
import os
import numpy as np
import pytest
from tessera.compiler.rocm_mxfp4_storage import MXFP4_STORAGE_CONTRACT,reference_mxfp4_folded_storage
from tessera.compiler.rocm_mxfp4_storage_native import build_mxfp4_storage_graph,package_mxfp4_storage_graph,execute_mxfp4_storage

@pytest.mark.parametrize("n,k",[(16,64),(32,256),(64,1024)])
def test_catalog_bridge_shapes_and_frontend_without_oracle(n,k,monkeypatch):
    import tessera as ts
    from tessera.compiler.trace import trace,to_graph_ir_module
    from tessera.compiler import rocm_mxfp4_storage as reference
    def function(codes,exponents):
        return ts.ops.mxfp4_folded_storage(codes,exponents,storage_contract=MXFP4_STORAGE_CONTRACT)
    def forbidden(*a,**kw):
        pytest.fail("storage tracing executed the host oracle")
    monkeypatch.setattr(reference,"reference_mxfp4_folded_storage",forbidden)
    module=to_graph_ir_module(trace(function,np.zeros((n,k//2),np.uint8),
        np.zeros((k//32,n),np.uint8)),target="rocm_gfx1201")
    assert list(map(str,module.functions[0].result_types))==[
        f"tensor<{n}x{k//2}xui8>",f"tensor<{k//32+1}x{n}xui8>"]

@pytest.mark.parametrize("n,k",[(15,64),(16,32),(0,64)])
def test_shape_contract_rejects_unsupported_envelope(n,k):
    with pytest.raises(ValueError):
        build_mxfp4_storage_graph(n,k)

@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="matching exact gfx1201")
@pytest.mark.parametrize("n,k",[(16,64),(32,256),(64,1024)])
def test_native_bridge_is_bitwise_lossless(n,k,monkeypatch):
    from tessera.compiler import rocm_mxfp4_storage as reference
    package=package_mxfp4_storage_graph(build_mxfp4_storage_graph(n,k))
    rng=np.random.default_rng(120132)
    codes=rng.integers(0,256,(n,k//2),np.uint8)
    # Byte-storage transformation preserves even reserved scale bit patterns.
    exponents=rng.integers(0,256,(k//32,n),np.uint8)
    exponents[:,0]=0
    exponents[:,1]=127
    expected=reference_mxfp4_folded_storage(codes,exponents,storage_contract=MXFP4_STORAGE_CONTRACT)
    before=(codes.copy(),exponents.copy())
    def forbidden(*a,**kw):
        pytest.fail("native storage bridge ran the host oracle")
    monkeypatch.setattr(reference,"reference_mxfp4_folded_storage",forbidden)
    events=[]
    actual=execute_mxfp4_storage(package,codes,exponents,event_samples=events)
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(codes,before[0])
    np.testing.assert_array_equal(exponents,before[1])
    assert len(events)==3 and all(v>0 for v in events)
    assert "tile.mxfp4_folded_storage_kernel" in package.native.tile_ir
    assert "mxfp4_folded_storage" in package.native.target_ir
    assert package.native.descriptor.provenance["lossy_steps"]==[]
    with pytest.raises(ValueError):
        replace(package,graph_ir=package.graph_ir+" ").validate()

@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="matching exact gfx1201")
def test_native_bridge_rejects_changed_policy_before_codegen():
    module=build_mxfp4_storage_graph(16,64)
    module.functions[0].body[0].kwargs["storage_contract"]="different"
    with pytest.raises(RuntimeError,match="lossless layout contract"):
        package_mxfp4_storage_graph(module)


import tessera as ts

@ts.jit(target="rocm_gfx1201")
def storage_jit(codes,exponents):
    return ts.ops.mxfp4_folded_storage(codes,exponents,storage_contract=MXFP4_STORAGE_CONTRACT)

@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="matching exact gfx1201")
def test_storage_jit_and_portable_runtime_without_oracle(monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import rocm_mxfp4_storage as reference
    rng=np.random.default_rng(120133)
    codes=rng.integers(0,256,(16,32),np.uint8)
    exponents=rng.integers(0,256,(2,16),np.uint8)
    expected=reference_mxfp4_folded_storage(codes,exponents,storage_contract=MXFP4_STORAGE_CONTRACT)
    def forbidden(*a,**kw):
        pytest.fail("native JIT ran the storage oracle")
    monkeypatch.setattr(reference,"reference_mxfp4_folded_storage",forbidden)
    result=storage_jit(codes,exponents)
    artifact=storage_jit.runtime_artifact()
    assert artifact.metadata["canonical_executable"]
    assert storage_jit.execution_kind=="native_gpu"
    assert "schedule.artifact" in artifact.schedule_ir
    for actual,wanted in zip(result,expected):
        np.testing.assert_array_equal(actual,wanted)
    digest=artifact.native_image.image_digest
    warm=storage_jit(exponents=exponents,codes=codes)
    assert storage_jit.runtime_artifact().native_image.image_digest==digest
    for actual,wanted in zip(warm,expected):
        np.testing.assert_array_equal(actual,wanted)
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    arrays=[codes,exponents,*[np.empty_like(x) for x in expected]]
    names=[b.name for b in sorted(restored.launch_descriptor.buffers,key=lambda b:b.ordinal)]
    receipt=rt.launch(restored,{"buffers":dict(zip(names,arrays)),"scalars":{}})
    assert receipt.get("ok"),receipt
    for actual,wanted in zip(arrays[2:],expected):
        np.testing.assert_array_equal(actual,wanted)
    aliased=[codes,exponents,codes,np.empty_like(expected[1])]
    receipt=rt.launch(restored,{"buffers":dict(zip(names,aliased)),"scalars":{}})
    assert not receipt["ok"] and "alias" in receipt["reason"]
    changed=replace(restored,graph_ir=restored.graph_ir+" ")
    receipt=rt.launch(changed,{"buffers":{},"scalars":{}})
    assert not receipt["ok"] and receipt["diagnostic_code"]=="E_LAUNCH_BINDING_MISMATCH"


@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="matching exact gfx1201")
def test_raw_storage_submit_checks_complete_abi_before_hip(monkeypatch):
    from tessera.compiler import rocm_mxfp4_storage_native as native
    from tessera.compiler.native_artifact import LaunchGeometry
    package=package_mxfp4_storage_graph(build_mxfp4_storage_graph(16,64))
    desc=package.native.descriptor
    arrays=[np.zeros((16,32),np.uint8),np.zeros((2,16),np.uint8),
        np.empty((16,32),np.uint8),np.empty((3,16),np.uint8)]
    buffers=dict(zip((b.name for b in sorted(desc.buffers,key=lambda b:b.ordinal)),arrays))
    def forbidden(*args,**kwargs):
        pytest.fail("changed descriptor reached HIP")
    monkeypatch.setattr(native,"_execute_arrays",forbidden)
    altered=(
        replace(desc,geometry=LaunchGeometry(grid=(3,1,1),workgroup=(256,1,1))),
        replace(desc,buffers=tuple(replace(b,direction="input") if b.ordinal==2 else b
            for b in desc.buffers)),
        replace(desc,provenance={**desc.provenance,"lossy_steps":["requantize"]}),
        replace(desc,ordering=replace(desc.ordering,ordered_submission=False)),
    )
    for changed in altered:
        with pytest.raises(ValueError,match="descriptor differs"):
            native.submit_mxfp4_storage(package.native.image,changed,buffers,{},None)


@pytest.mark.skipif(os.getenv("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="matching exact gfx1201")
def test_storage_schedule_replay_rejects_changed_contract():
    from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
    graph=build_mxfp4_storage_graph(16,64).to_mlir(target="rocm_gfx1201",canonical=True)
    tool=find_tessera_opt()
    schedule=run_tessera_opt(tool,graph,"--tessera-graph-to-schedule")
    for before,after in (
        ('ownership = "private_outputs_distinct_readonly_inputs"','ownership = "shared_outputs"'),
        ('scale_storage = "legacy_e8m0_bits"','scale_storage = "e4m3_bits"'),
        ('layout = "row_major"','layout = "col_major"'),
    ):
        assert before in schedule
        with pytest.raises(RuntimeError,match="contract changed after hashing"):
            run_tessera_opt(tool,schedule.replace(before,after),"--tessera-schedule-to-tile")
